import os
import time
from datetime import datetime, timedelta
from typing import List, Tuple

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import implicit

from fastapi import FastAPI, Form, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from jose import jwt, JWTError

from db import SessionLocal, RequestHistory

from config import (
    # DATABASE_URL,
    ALS_MODEL_PATH,
    ALS_USER_MAPPING_PATH,
    ALS_ITEM_MAPPING_PATH,
    ALS_PURCHASED_PATH,
    POPULAR_ITEMS_PATH,
    DEFAULT_K,
    CANDIDATE_MULTIPLIER,
    JWT_SECRET,
    JWT_ALGO,
    JWT_EXPIRE_MIN,
    ADMIN_PASSWORD,
)


app = FastAPI(title="Ozon Apparel Recommendation Service")
security = HTTPBearer()


# Логика авторизации
def create_admin_token() -> str:
    payload = {
        "role": "admin",
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MIN),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def require_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGO],
        )
    except JWTError:
        raise HTTPException(403, "invalid token")

    if payload.get("role") != "admin":
        raise HTTPException(403, "forbidden")

# pydantic
class ForwardRequest(BaseModel):
    user_id: int = Field(..., example=3423970)
    k: int = Field(DEFAULT_K, example=100)


class ForwardResponse(BaseModel):
    user_id: int
    recommendations: List[int]

# Загрузка моделей
model = implicit.cpu.als.AlternatingLeastSquares.load(ALS_MODEL_PATH)

user_map = pd.read_parquet(ALS_USER_MAPPING_PATH)
user_id_to_idx = dict(zip(user_map.user_id, user_map.user_idx))

item_map = pd.read_parquet(ALS_ITEM_MAPPING_PATH)
idx_to_item_id = dict(zip(item_map.item_idx, item_map.item_id))

purchased_by_user_idx = np.load(ALS_PURCHASED_PATH, allow_pickle=True).item()

popular_items = (pd.read_parquet(POPULAR_ITEMS_PATH).item_id.astype(int).tolist())

# Инференс
def recommend(user_id: int, k: int) -> Tuple[bool, List[int]]:
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


@app.post("/login")
def login(password: str = Form(...)):
    if password != ADMIN_PASSWORD:
        raise HTTPException(403, "forbidden")
    return {"access_token": create_admin_token(), "token_type": "bearer"}


@app.post("/forward", response_model=ForwardResponse)
def forward(payload: ForwardRequest):
    start = time.perf_counter()
    status = 200
    error = None
    is_cold = None

    try:
        is_cold, recs = recommend(payload.user_id, payload.k)
        return ForwardResponse(user_id=payload.user_id, recommendations=recs)

    except Exception:
        status = 403
        error = "модель не смогла обработать данные"
        raise HTTPException(403, error)

    finally:
        duration = (time.perf_counter() - start) * 1000
        db = SessionLocal()
        try:
            db.add(RequestHistory(
                timestamp=datetime.utcnow().isoformat(),
                endpoint="/forward",
                status_code=status,
                user_id=payload.user_id,
                requested_k=payload.k,
                is_cold=int(is_cold) if is_cold is not None else None,
                processing_ms=duration,
                request_bytes=len(payload.json()),
                token_count=len(payload.dict()),
                error=error,
            ))
            db.commit()
        finally:
            db.close()


@app.get("/history", dependencies=[Depends(require_admin)])
def history(limit: int = 100):
    db = SessionLocal()
    try:
        rows = (
            db.query(RequestHistory)
            .order_by(RequestHistory.id.desc())
            .limit(limit)
            .all()
        )
        return [r.__dict__ for r in rows]
    finally:
        db.close()


@app.delete("/history", dependencies=[Depends(require_admin)])
def clear_history():
    db = SessionLocal()
    try:
        db.query(RequestHistory).delete()
        db.commit()
    finally:
        db.close()
    return {"status": "history cleared"}


@app.get("/stats", dependencies=[Depends(require_admin)])
def stats():
    db = SessionLocal()
    rows = db.query(RequestHistory).all()
    db.close()

    if not rows:
        return {"count": 0}

    proc = np.array([r.processing_ms for r in rows])
    q = lambda x: float(np.quantile(proc, x))

    return {
        "count": len(rows),
        "processing_ms": {
            "mean": float(proc.mean()),
            "p50": q(0.50),
            "p95": q(0.95),
            "p99": q(0.99),
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}
