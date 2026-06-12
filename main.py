import logging
import os
import time
from collections.abc import Generator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Form, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

import config
from config import (
    ADMIN_PASSWORD,
    CANDIDATE_MULTIPLIER,
    DEFAULT_K,
    ENABLE_LEGACY_ALS_FALLBACK,
    HISTORY_LIMIT_DEFAULT,
    HISTORY_LIMIT_MAX,
    JWT_ALGO,
    JWT_EXPIRE_MIN,
    JWT_SECRET,
)
from db import RequestHistory, SessionLocal, init_db
from ranker_store import RankerStore
from schemas import (
    DemoStoryResponse,
    DemoUserResponse,
    DemoUserSummary,
    ModelInfoResponse,
    RichRecommendationResponse,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy ALS artifacts (loaded only when ENABLE_LEGACY_ALS_FALLBACK=True)
# ---------------------------------------------------------------------------
_als_model = None
_als_user_id_to_idx: dict = {}
_als_idx_to_item_id: dict = {}
_als_purchased_by_user_idx: dict = {}
_als_popular_items: list[int] = []


def _try_load_als() -> bool:
    """Attempt to load ALS artifacts. Returns True if successful."""
    global _als_model, _als_user_id_to_idx, _als_idx_to_item_id
    global _als_purchased_by_user_idx, _als_popular_items
    try:
        import implicit
        _als_model = implicit.cpu.als.AlternatingLeastSquares.load(config.ALS_MODEL_PATH)
        user_map = pd.read_parquet(config.ALS_USER_MAPPING_PATH)
        _als_user_id_to_idx = dict(zip(user_map.user_id, user_map.user_idx, strict=True))
        item_map = pd.read_parquet(config.ALS_ITEM_MAPPING_PATH)
        _als_idx_to_item_id = dict(zip(item_map.item_idx, item_map.item_id, strict=True))
        _als_purchased_by_user_idx = np.load(
            config.ALS_PURCHASED_PATH, allow_pickle=True
        ).item()
        _als_popular_items = (
            pd.read_parquet(config.POPULAR_ITEMS_PATH).item_id.astype(int).tolist()
        )
        log.info("Legacy ALS fallback loaded successfully")
        return True
    except Exception as exc:
        log.warning("Legacy ALS fallback load failed: %s", exc)
        return False


def _als_recommend(user_id: int, k: int) -> tuple[bool, list[int]]:
    if user_id not in _als_user_id_to_idx:
        return True, _als_popular_items[:k]
    u_idx = _als_user_id_to_idx[user_id]
    item_idxs, _ = _als_model.recommend(
        userid=u_idx,
        user_items=None,
        N=k * CANDIDATE_MULTIPLIER,
        filter_already_liked_items=False,
    )
    purchased = set(_als_purchased_by_user_idx.get(u_idx, []))
    filtered = [i for i in item_idxs if i not in purchased][:k]
    recs = [int(_als_idx_to_item_id[i]) for i in filtered]
    if len(recs) < k:
        seen = set(recs)
        for pid in _als_popular_items:
            if pid not in seen:
                recs.append(pid)
            if len(recs) == k:
                break
    return False, recs


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()

    # Load PRD ranker
    try:
        application.state.ranker = RankerStore.load(config)
        log.info("RankerStore loaded OK")
    except Exception as exc:
        log.error("RankerStore load FAILED: %s", exc)
        application.state.ranker = None

    # Optionally load legacy ALS fallback
    if ENABLE_LEGACY_ALS_FALLBACK:
        _try_load_als()

    yield

    # Cleanup (nothing persistent to close)
    application.state.ranker = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ozon Apparel Recommendation Service",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "inference", "description": "Recommendation inference"},
        {"name": "demo", "description": "Rich demo endpoints (PRD reranker)"},
        {"name": "auth", "description": "Authentication and authorization"},
        {"name": "admin", "description": "Administrative and debug endpoints"},
        {"name": "monitoring", "description": "Health checks and service metrics"},
    ],
)

security = HTTPBearer()


def get_db() -> Generator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_ranker() -> RankerStore:
    ranker: RankerStore | None = app.state.ranker
    if ranker is None:
        if ENABLE_LEGACY_ALS_FALLBACK:
            raise HTTPException(
                503, "PRD ranker unavailable and legacy ALS fallback is enabled but failed to load"
            )
        raise HTTPException(
            503,
            "PRD ranker not available — run scripts/build_api_demo_store.py "
            "and ensure model artifacts exist, or set ENABLE_LEGACY_ALS_FALLBACK=true",
        )
    return ranker


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def create_admin_token() -> str:
    payload = {
        "role": "admin",
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MIN),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def require_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> None:
    try:
        payload = jwt.decode(
            credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGO]
        )
    except JWTError:
        raise HTTPException(401, "invalid token") from None
    if payload.get("role") != "admin":
        raise HTTPException(403, "forbidden")


# ---------------------------------------------------------------------------
# Pydantic models (kept here for /forward backward compat)
# ---------------------------------------------------------------------------

class ForwardRequest(BaseModel):
    user_id: int = Field(..., ge=0, example=3423970)
    k: int = Field(DEFAULT_K, ge=1, le=100, example=20)


class ForwardResponse(BaseModel):
    user_id: int
    recommendations: list[int]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


# ---------------------------------------------------------------------------
# DB write helper (shared by /forward)
# ---------------------------------------------------------------------------

def _log_request(
    db: Session,
    *,
    endpoint: str,
    status: int,
    user_id: int,
    requested_k: int,
    is_cold: bool | None,
    duration_ms: float,
    inference_ms: float | None,
    recs_count: int | None,
    error_type: str | None,
    error: str | None,
) -> None:
    try:
        obj = RequestHistory(
            timestamp=datetime.utcnow().isoformat(),
            endpoint=endpoint,
            status_code=status,
            user_id=user_id,
            requested_k=requested_k,
            is_cold=int(is_cold) if is_cold is not None else None,
            processing_ms=duration_ms,
            inference_ms=inference_ms,
            recs_count=recs_count,
            error_type=error_type,
            error=error,
            db_write_ms=0.0,
            db_error=0,
        )
        db.add(obj)
        db.commit()
        obj.db_write_ms = 0.0
        obj.db_error = 0
        db.commit()
    except Exception:
        db.rollback()


# ---------------------------------------------------------------------------
# /forward  (backward-compatible)
# ---------------------------------------------------------------------------

@app.post("/forward", response_model=ForwardResponse, tags=["inference"])
def forward(payload: ForwardRequest, db: Session = Depends(get_db)):
    start = time.perf_counter()
    status = 200
    error: str | None = None
    error_type: str | None = None
    inference_ms: float | None = None
    recs_count: int | None = None
    is_cold: bool | None = None

    try:
        inf_start = time.perf_counter()

        ranker: RankerStore | None = app.state.ranker
        if ranker is not None:
            # PRD path
            is_cold, recs_dicts = ranker.score_user(payload.user_id, payload.k)
            recs = [r["item_id"] for r in recs_dicts]
        elif ENABLE_LEGACY_ALS_FALLBACK and _als_model is not None:
            # Legacy ALS fallback
            is_cold, recs = _als_recommend(payload.user_id, payload.k)
        else:
            raise HTTPException(
                503,
                "PRD ranker not available. "
                "Set ENABLE_LEGACY_ALS_FALLBACK=true to use legacy ALS, "
                "or run scripts/build_api_demo_store.py.",
            )

        inference_ms = (time.perf_counter() - inf_start) * 1000
        recs_count = len(recs)
        return {"user_id": payload.user_id, "recommendations": recs}

    except HTTPException:
        raise
    except Exception as exc:
        status = 500
        error = str(exc)
        error_type = "internal_error"
        raise HTTPException(500, "internal error") from None
    finally:
        duration = (time.perf_counter() - start) * 1000
        _log_request(
            db,
            endpoint="/forward",
            status=status,
            user_id=payload.user_id,
            requested_k=payload.k,
            is_cold=is_cold,
            duration_ms=duration,
            inference_ms=inference_ms,
            recs_count=recs_count,
            error_type=error_type,
            error=error,
        )


# ---------------------------------------------------------------------------
# /recommendations/{user_id}  — rich response with metadata
# ---------------------------------------------------------------------------

@app.get(
    "/recommendations/{user_id}",
    response_model=RichRecommendationResponse,
    tags=["demo"],
)
def recommendations(
    user_id: int,
    k: int = Query(20, ge=1, le=512),
    ranker: RankerStore = Depends(get_ranker),
):
    is_cold, recs = ranker.score_user(user_id, k)
    return {
        "user_id": user_id,
        "model": ranker.get_model_info().get("model_name", "prd_reranker"),
        "is_cold": is_cold,
        "recommendations": recs,
    }


# ---------------------------------------------------------------------------
# /demo/users
# ---------------------------------------------------------------------------

@app.get("/demo/users", response_model=list[DemoUserSummary], tags=["demo"])
def demo_users(
    limit: int = Query(20, ge=1, le=100),
    min_hits: int | None = Query(1, ge=0),
    ranker: RankerStore = Depends(get_ranker),
):
    return ranker.get_demo_users(limit=limit, min_hits=min_hits)


# ---------------------------------------------------------------------------
# /demo/user/{user_id}
# ---------------------------------------------------------------------------

@app.get(
    "/demo/user/{user_id}",
    response_model=DemoUserResponse,
    tags=["demo"],
)
def demo_user(
    user_id: int,
    k: int = Query(20, ge=1, le=512),
    history_limit: int = Query(20, ge=1, le=200),
    ranker: RankerStore = Depends(get_ranker),
):
    result = ranker.get_demo_user_full(user_id=user_id, k=k, history_limit=history_limit)
    return result


# ---------------------------------------------------------------------------
# /demo/story/{user_id}  — aggregated view-model for Streamlit UI
# ---------------------------------------------------------------------------

@app.get(
    "/demo/story/{user_id}",
    response_model=DemoStoryResponse,
    tags=["demo"],
)
def demo_story(
    user_id: int,
    k: int = Query(20, ge=1, le=100),
    history_limit: int = Query(20, ge=1, le=200),
    analysis_k: int = Query(100, ge=1, le=500),
    ranker: RankerStore = Depends(get_ranker),
):
    return ranker.get_demo_story(
        user_id=user_id, k=k, history_limit=history_limit, analysis_k=analysis_k
    )


# ---------------------------------------------------------------------------
# /model/info
# ---------------------------------------------------------------------------

@app.get("/model/info", response_model=ModelInfoResponse, tags=["demo"])
def model_info(ranker: RankerStore = Depends(get_ranker)):
    info = ranker.get_model_info()
    return {
        "model_name": info.get("model_name", "unknown"),
        "model_uri": info.get("model_uri"),
        "prd_run_id": info.get("prd_run_id"),
        "feature_count": info.get("feature_count", 0),
        "candidate_store": info.get("candidate_store", ""),
        "metrics": info.get("metrics", {}),
        "baseline": info.get("baseline", {}),
        "description": info.get("description"),
        "artifact_status": info.get("artifact_status", {}),
    }


# ---------------------------------------------------------------------------
# Auth & admin (unchanged)
# ---------------------------------------------------------------------------

@app.post("/login", response_model=TokenResponse, tags=["auth"])
def login(password: str = Form(...)):
    if password != ADMIN_PASSWORD:
        raise HTTPException(403, "forbidden")
    return {"access_token": create_admin_token(), "token_type": "bearer"}


@app.get("/history", dependencies=[Depends(require_admin)], tags=["admin"])
def history(
    limit: int = Query(HISTORY_LIMIT_DEFAULT, ge=1, le=HISTORY_LIMIT_MAX),
    db: Session = Depends(get_db),
):
    try:
        rows = (
            db.query(RequestHistory)
            .order_by(RequestHistory.id.desc())
            .limit(limit)
            .all()
        )
    except OperationalError:
        raise HTTPException(
            500,
            "database is not initialized or schema mismatch: recreate sqlite db (or run migrations)",
        ) from None
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp,
            "endpoint": r.endpoint,
            "status_code": r.status_code,
            "user_id": r.user_id,
            "requested_k": r.requested_k,
            "is_cold": r.is_cold,
            "processing_ms": r.processing_ms,
            "inference_ms": r.inference_ms,
            "recs_count": r.recs_count,
            "db_write_ms": r.db_write_ms,
            "db_error": r.db_error,
            "error_type": r.error_type,
            "error": r.error,
        }
        for r in rows
    ]


@app.post("/admin/rebuild-demo-store", tags=["admin"])
def rebuild_demo_store(
    use_test: bool = Query(True, description="Pass --use-test flag (test_eval candidates)"),
):
    """Start background rebuild of api/demo_store artifacts. Restart uvicorn after completion."""
    import pathlib
    import subprocess
    import sys

    script = str(pathlib.Path(__file__).parent / "scripts" / "build_api_demo_store.py")
    args = [sys.executable, script]
    if use_test:
        args.append("--use-test")
    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(pathlib.Path(__file__).parent),
        )
        return {
            "status": "started",
            "pid": proc.pid,
            "command": " ".join(args),
            "note": "Check server logs for progress. Restart uvicorn to load new data.",
        }
    except Exception as exc:
        raise HTTPException(500, f"Failed to start rebuild: {exc}") from None


@app.delete("/history", dependencies=[Depends(require_admin)], tags=["admin"])
def clear_history(db: Session = Depends(get_db)):
    try:
        db.query(RequestHistory).delete()
        db.commit()
    except OperationalError:
        db.rollback()
        return {"status": "history table does not exist"}
    return {"status": "history cleared"}


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

@app.get("/health", tags=["monitoring"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/health/ready", tags=["monitoring"])
def health_ready(db: Session = Depends(get_db)) -> dict:
    checks: dict[str, str] = {}

    # DB
    try:
        db.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as exc:
        raise HTTPException(503, f"db unavailable: {exc}") from None

    ranker: RankerStore | None = app.state.ranker

    if ranker is None:
        checks["ranker_model"] = "missing"
        checks["feature_cols"] = "missing"
        checks["candidate_store"] = "missing"
        checks["item_metadata"] = "missing"
        checks["user_history"] = "missing"
        checks["future_events"] = "missing"

        if ENABLE_LEGACY_ALS_FALLBACK:
            als_ok = _als_model is not None
            checks["legacy_als"] = "ok" if als_ok else "missing"
            overall = "degraded"
        else:
            checks["legacy_als"] = "disabled"
            raise HTTPException(
                503,
                {
                    "status": "unavailable",
                    "checks": checks,
                    "detail": (
                        "PRD ranker not loaded. "
                        "Run scripts/build_api_demo_store.py or set ENABLE_LEGACY_ALS_FALLBACK=true."
                    ),
                },
            )
    else:
        ranker_checks = ranker.health_checks()
        checks.update(ranker_checks)

        # Legacy ALS status
        if ENABLE_LEGACY_ALS_FALLBACK:
            checks["legacy_als"] = "ok" if _als_model is not None else "missing"
        else:
            checks["legacy_als"] = "disabled"

        # Determine overall status
        critical = ("ranker_model", "feature_cols", "candidate_store")
        if any(checks.get(k, "missing") != "ok" for k in critical):
            raise HTTPException(503, {"status": "unavailable", "checks": checks})

        overall = "ok"

    return {"status": overall, "checks": checks}


@app.get("/stats", dependencies=[Depends(require_admin)], tags=["monitoring"])
def stats(db: Session = Depends(get_db)):
    rows = db.query(RequestHistory).all()

    if not rows:
        return {"count": 0}

    proc = np.array([r.processing_ms for r in rows])

    def q(arr: np.ndarray, p: float) -> float:
        return float(np.quantile(arr, p))

    def quantiles(arr: np.ndarray) -> dict:
        return {
            "mean": float(arr.mean()),
            "p50": q(arr, 0.50),
            "p95": q(arr, 0.95),
            "p99": q(arr, 0.99),
            "sum": float(arr.sum()),
        }

    inf_vals = [r.inference_ms for r in rows if r.inference_ms is not None]
    inference = np.array(inf_vals, dtype=float) if inf_vals else None

    recs_vals = [r.recs_count for r in rows if r.recs_count is not None]
    recs_arr = np.array(recs_vals, dtype=float) if recs_vals else None

    dbw_vals = [r.db_write_ms for r in rows if r.db_write_ms is not None]
    dbw = np.array(dbw_vals, dtype=float) if dbw_vals else None

    cold_vals = [r.is_cold for r in rows if r.is_cold is not None]
    cold_rate = float(np.mean([1 if v else 0 for v in cold_vals])) if cold_vals else None

    db_err_vals = [r.db_error for r in rows if r.db_error is not None]
    db_error_rate = (
        float(np.mean([1 if v else 0 for v in db_err_vals])) if db_err_vals else None
    )

    empty_rate = float(np.mean(recs_arr == 0)) if recs_arr is not None else None

    return {
        "count": len(rows),
        "processing_ms": quantiles(proc),
        "inference_ms": quantiles(inference) if inference is not None else None,
        "recs_count": quantiles(recs_arr) if recs_arr is not None else None,
        "db_write_ms": quantiles(dbw) if dbw is not None else None,
        "cold_rate": cold_rate,
        "db_error_rate": db_error_rate,
        "empty_rate": empty_rate,
    }
