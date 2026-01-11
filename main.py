import os
import time
from collections.abc import Generator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import implicit
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Form, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from config import (
    ADMIN_PASSWORD,
    ALS_ITEM_MAPPING_PATH,
    ALS_MODEL_PATH,
    ALS_PURCHASED_PATH,
    ALS_USER_MAPPING_PATH,
    CANDIDATE_MULTIPLIER,
    DEFAULT_K,
    HISTORY_LIMIT_DEFAULT,
    HISTORY_LIMIT_MAX,
    JWT_ALGO,
    JWT_EXPIRE_MIN,
    JWT_SECRET,
    POPULAR_ITEMS_PATH,
)
from db import RequestHistory, SessionLocal, init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Ozon Apparel Recommendation Service",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "inference", "description": "Recommendation inference"},
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


# Логика авторизации
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
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGO],
        )
    except JWTError:
        raise HTTPException(401, "invalid token") from None

    if payload.get("role") != "admin":
        raise HTTPException(403, "forbidden")


# pydantic
class ForwardRequest(BaseModel):
    user_id: int = Field(..., ge=0, example=3423970)
    k: int = Field(DEFAULT_K, ge=1, example=100)


class ForwardResponse(BaseModel):
    user_id: int
    recommendations: list[int]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


# Загрузка моделей
model = implicit.cpu.als.AlternatingLeastSquares.load(ALS_MODEL_PATH)

user_map = pd.read_parquet(ALS_USER_MAPPING_PATH)
user_id_to_idx = dict(zip(user_map.user_id, user_map.user_idx, strict=True))

item_map = pd.read_parquet(ALS_ITEM_MAPPING_PATH)
idx_to_item_id = dict(zip(item_map.item_idx, item_map.item_id, strict=True))

purchased_by_user_idx = np.load(ALS_PURCHASED_PATH, allow_pickle=True).item()

popular_items = pd.read_parquet(POPULAR_ITEMS_PATH).item_id.astype(int).tolist()


# Инференс
def recommend(user_id: int, k: int) -> tuple[bool, list[int]]:
    if user_id not in user_id_to_idx:
        return True, popular_items[:k]

    u_idx = user_id_to_idx[user_id]

    item_idxs, _ = model.recommend(
        userid=u_idx,
        user_items=None,
        N=k * CANDIDATE_MULTIPLIER,
        filter_already_liked_items=False,
    )

    purchased = set(purchased_by_user_idx.get(u_idx, []))
    filtered = [i for i in item_idxs if i not in purchased][:k]
    recs = [int(idx_to_item_id[i]) for i in filtered]

    if len(recs) < k:
        seen = set(recs)
        for pid in popular_items:
            if pid not in seen:
                recs.append(pid)
            if len(recs) == k:
                break

    return False, recs


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
        is_cold, recs = recommend(payload.user_id, payload.k)
        inference_ms = (time.perf_counter() - inf_start) * 1000
        recs_count = len(recs)

        return {"user_id": payload.user_id, "recommendations": recs}

    except HTTPException:
        raise
    except Exception as e:
        status = 500
        error = str(e)
        error_type = "internal_error"
        raise HTTPException(500, "internal error") from None
    finally:
        duration = (time.perf_counter() - start) * 1000

        db_start = time.perf_counter()
        db_error_flag = 0
        try:
            obj = RequestHistory(
                timestamp=datetime.utcnow().isoformat(),
                endpoint="/forward",
                status_code=status,
                user_id=payload.user_id,
                requested_k=payload.k,
                is_cold=int(is_cold) if is_cold is not None else None,
                processing_ms=duration,
                inference_ms=inference_ms,
                recs_count=recs_count,
                error_type=error_type,
                error=error,
                db_write_ms=0.0,  # заполняем после записи в БД
                db_error=0,
            )
            # измеряем время записи истории (add+commit)
            db.add(obj)
            db.commit()
        except Exception:
            db.rollback()
            db_error_flag = 1
        else:
            # если запись прошла - обновим поля и сделаем второй commit
            try:
                obj.db_write_ms = (time.perf_counter() - db_start) * 1000
                obj.db_error = db_error_flag
                db.commit()
            except Exception:
                db.rollback()


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
        rows = db.query(RequestHistory).order_by(RequestHistory.id.desc()).limit(limit).all()
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


@app.delete("/history", dependencies=[Depends(require_admin)], tags=["admin"])
def clear_history(db: Session = Depends(get_db)):
    try:
        db.query(RequestHistory).delete()
        db.commit()
    except OperationalError:
        db.rollback()
        return {"status": "history table does not exist"}
    return {"status": "history cleared"}


@app.get("/health", tags=["monitoring"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/health/ready", tags=["monitoring"])
def health_ready(db: Session = Depends(get_db)) -> dict:
    # доступность БД
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(503, "db unavailable") from None

    # проверка на загрузку модели и нужных файлов
    if model is None:
        raise HTTPException(503, "model not loaded")
    if not user_id_to_idx or not idx_to_item_id or not popular_items:
        raise HTTPException(503, "artifacts not loaded")

    return {"status": "ok", "checks": {"db": "ok", "model": "ok"}}


@app.get("/stats", dependencies=[Depends(require_admin)], tags=["monitoring"])
def stats(db: Session = Depends(get_db)):
    rows = db.query(RequestHistory).all()

    if not rows:
        return {"count": 0}

    proc = np.array([r.processing_ms for r in rows])

    def q(arr: np.ndarray, p: float) -> float:
        return float(np.quantile(arr, p))

    inf_vals = [r.inference_ms for r in rows if r.inference_ms is not None]
    inference = np.array(inf_vals, dtype=float) if inf_vals else None

    recs_vals = [r.recs_count for r in rows if r.recs_count is not None]
    recs = np.array(recs_vals, dtype=float) if recs_vals else None

    dbw_vals = [r.db_write_ms for r in rows if r.db_write_ms is not None]
    dbw = np.array(dbw_vals, dtype=float) if dbw_vals else None

    def quantiles(arr: np.ndarray) -> dict:
        return {
            "mean": float(arr.mean()),
            "p50": q(arr, 0.50),
            "p95": q(arr, 0.95),
            "p99": q(arr, 0.99),
            "sum": float(arr.sum()),
        }

    cold_vals = [r.is_cold for r in rows if r.is_cold is not None]
    cold_rate = float(np.mean([1 if v else 0 for v in cold_vals])) if cold_vals else None

    db_err_vals = [r.db_error for r in rows if r.db_error is not None]
    db_error_rate = float(np.mean([1 if v else 0 for v in db_err_vals])) if db_err_vals else None

    empty_rate = None
    if recs is not None:
        empty_rate = float(np.mean(recs == 0))

    return {
        "count": len(rows),
        "processing_ms": quantiles(proc),
        "inference_ms": (quantiles(inference) if inference is not None else None),
        "recs_count": (quantiles(recs) if recs is not None else None),
        "db_write_ms": (quantiles(dbw) if dbw is not None else None),
        "cold_rate": cold_rate,
        "db_error_rate": db_error_rate,
        "empty_rate": empty_rate,
    }
