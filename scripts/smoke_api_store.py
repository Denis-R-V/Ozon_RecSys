#!/usr/bin/env python3
"""
Smoke test for api/demo_store/ artifacts and PRD LightGBM model.

Loads best_ranker.joblib, reads one demo user from api_candidates_features.parquet,
runs model.predict(), sorts by score, and prints top-20 with metadata + future_hit flag.

Usage:
    python scripts/smoke_api_store.py [--user-id USER_ID] [--k 20]

    Required env: recsys conda env (lightgbm, pandas, joblib)
    conda activate recsys && python scripts/smoke_api_store.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import warnings

import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models/reranker"
    / "final_reranker_v12_repeat__merged_with_dl_all_repeat_history"
    / "best_ranker.joblib"
)
FEATURE_COLS_PATH = (
    PROJECT_ROOT
    / "models/reranker"
    / "final_reranker_v12_repeat__merged_with_dl_all_repeat_history"
    / "feature_cols.json"
)
DEMO_STORE = PROJECT_ROOT / "api/demo_store"
CANDIDATES_PATH = DEMO_STORE / "api_candidates_features.parquet"
ITEM_META_PATH = DEMO_STORE / "api_item_metadata.parquet"
FUTURE_EVENTS_PATH = DEMO_STORE / "api_future_events.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(path: pathlib.Path, label: str) -> None:
    if not path.exists():
        print(f"[ERROR] {label} not found: {path}")
        print("  -> Run: python scripts/build_api_demo_store.py")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def load_model_and_features() -> tuple:
    print("\n" + "=" * 60)
    print("  smoke_api_store.py . PRD inference smoke test")
    print("=" * 60)

    _check(MODEL_PATH, "best_ranker.joblib")
    _check(FEATURE_COLS_PATH, "feature_cols.json")

    print(f"\n[1] Loading model: {MODEL_PATH.name}")
    t0 = time.perf_counter()
    model = joblib.load(MODEL_PATH)
    print(f"    type        : {type(model).__name__}")
    print(f"    n_features  : {model.n_features_in_}")
    print(f"    load time   : {(time.perf_counter() - t0) * 1000:.0f} ms")

    print(f"\n[2] Loading feature_cols.json")
    fc = json.loads(FEATURE_COLS_PATH.read_text(encoding="utf-8"))
    feature_cols = fc["numeric_feature_cols"]
    print(f"    numeric cols: {len(feature_cols)}")

    # Sanity: model expects exactly len(feature_cols) features
    if model.n_features_in_ != len(feature_cols):
        print(
            f"    [WARN] model.n_features_in_={model.n_features_in_} "
            f"!= len(feature_cols)={len(feature_cols)}"
        )

    return model, feature_cols


def pick_user(user_id: int | None) -> int:
    _check(CANDIDATES_PATH, "api_candidates_features.parquet")

    print(f"\n[3] Loading candidate store: {CANDIDATES_PATH.name}")
    t0 = time.perf_counter()
    cands = pd.read_parquet(CANDIDATES_PATH, columns=["user_id", "label"])
    print(f"    rows   : {len(cands):,}")
    print(f"    users  : {cands['user_id'].nunique()}")
    print(f"    load   : {(time.perf_counter() - t0) * 1000:.0f} ms")

    if user_id is not None:
        if user_id not in cands["user_id"].values:
            print(f"    [WARN] user_id={user_id} not in demo store . picking first with positives")
            user_id = None

    if user_id is None:
        # Prefer a user with >=1 positive
        pos_users = cands[cands["label"] == 1]["user_id"].unique()
        if len(pos_users) == 0:
            user_id = int(cands["user_id"].iloc[0])
            print(f"    No users with positives . using first user: {user_id}")
        else:
            user_id = int(pos_users[0])
            n_pos = int((cands[cands["user_id"] == user_id]["label"]).sum())
            print(f"    Auto-selected user {user_id} ({n_pos} positives in candidate set)")

    return user_id


def score_user(
    user_id: int,
    model,
    feature_cols: list[str],
    k: int,
) -> pd.DataFrame:
    print(f"\n[4] Scoring user_id={user_id}  k={k}")

    t0 = time.perf_counter()
    cands = pd.read_parquet(CANDIDATES_PATH)
    user_df = cands[cands["user_id"] == user_id].copy()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"    Candidate rows : {len(user_df)}")
    print(f"    Load time      : {load_ms:.0f} ms")

    # Feature matrix
    missing_feats = [c for c in feature_cols if c not in user_df.columns]
    if missing_feats:
        print(f"    [WARN] {len(missing_feats)} feature cols missing . filling with 0")
        for c in missing_feats:
            user_df[c] = 0.0

    X = user_df[feature_cols].fillna(0)  # keep as DataFrame — LGBMRanker uses feature names

    t1 = time.perf_counter()
    scores = model.predict(X)
    predict_ms = (time.perf_counter() - t1) * 1000
    print(f"    Score range    : {scores.min():.4f} ... {scores.max():.4f}")
    print(f"    Predict time   : {predict_ms:.0f} ms")

    user_df = user_df.copy()
    user_df["score"] = scores
    user_df = user_df.sort_values("score", ascending=False).reset_index(drop=True)
    user_df["rank"] = range(1, len(user_df) + 1)

    return user_df.head(k)


def enrich_with_metadata(top_df: pd.DataFrame) -> pd.DataFrame:
    if not ITEM_META_PATH.exists():
        print("\n[5] Item metadata not found . skipping enrich")
        return top_df

    meta = pd.read_parquet(ITEM_META_PATH)
    top_df = top_df.merge(meta, on="item_id", how="left")
    return top_df


def attach_future_hits(top_df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    if not FUTURE_EVENTS_PATH.exists():
        top_df["future_hit"] = False
        return top_df

    fut = pd.read_parquet(FUTURE_EVENTS_PATH)
    fut_user = fut[fut["user_id"] == user_id]
    future_item_ids = set(fut_user["item_id"].unique().tolist())
    top_df["future_hit"] = top_df["item_id"].isin(future_item_ids)

    buy_types = ["to_cart", "order", "purchase", "buy"]
    buy_events = fut_user[fut_user["action_type"].str.lower().isin(buy_types)]
    buy_item_ids = set(buy_events["item_id"].unique().tolist())
    top_df["future_purchase"] = top_df["item_id"].isin(buy_item_ids)

    return top_df


def print_results(top_df: pd.DataFrame, user_id: int, k: int) -> None:
    print(f"\n[6] Top-{k} recommendations for user_id={user_id}")
    print("-" * 90)

    has_title = "title" in top_df.columns
    has_brand = "brand" in top_df.columns
    has_cat = "category_name" in top_df.columns

    n_pos_in_top = int(top_df["label"].sum()) if "label" in top_df.columns else 0
    n_future_hit = int(top_df["future_hit"].sum()) if "future_hit" in top_df.columns else 0
    n_future_buy = int(top_df.get("future_purchase", pd.Series([False] * len(top_df))).sum())

    for _, row in top_df.iterrows():
        rk = int(row["rank"])
        iid = int(row["item_id"])
        sc = float(row["score"])
        label = int(row["label"]) if "label" in row else 0
        fhit = bool(row.get("future_hit", False))
        fbuy = bool(row.get("future_purchase", False))

        title = str(row.get("title", ""))[:35] if has_title else ""
        brand = str(row.get("brand", ""))[:15] if has_brand else ""
        cat = str(row.get("category_name", ""))[:20] if has_cat else ""

        flags = []
        if label:
            flags.append("*LABEL")
        if fhit:
            flags.append("future_hit")
        if fbuy:
            flags.append("future_BUY")
        flag_str = "  " + " ".join(flags) if flags else ""

        src_cols = [
            c for c in ["in_als", "in_top_popular", "is_repeat_history_candidate",
                         "in_dl_sasrec_v3", "in_dl_dssm_final", "in_clip_hnsw"]
            if c in row and row[c]
        ]
        src_short = {
            "in_als": "ALS", "in_top_popular": "POP", "is_repeat_history_candidate": "RPT",
            "in_dl_sasrec_v3": "SRECV3", "in_dl_dssm_final": "DSSM", "in_clip_hnsw": "CLIP",
        }
        sources = "+".join(src_short.get(c, c) for c in src_cols)

        print(
            f"  {rk:3d}  {iid:12d}  score={sc:7.3f}  {title:<35}  "
            f"{brand:<15}  {cat:<20}  [{sources}]{flag_str}"
        )

    print("-" * 90)

    # Summary
    recall_k = n_pos_in_top / max(
        int(top_df["label"].sum()) if "label" in top_df.columns else 1, 1
    )
    print(f"\n  Summary for user_id={user_id}:")
    print(f"    candidates scored   : {len(top_df)} (shown top-{k})")
    print(f"    positives in top-{k}  : {n_pos_in_top}")
    print(f"    future hits in top-{k}: {n_future_hit}")
    print(f"    future buys in top-{k}: {n_future_buy}")

    if "label" in top_df.columns:
        cands_total = None
        try:
            all_cands = pd.read_parquet(CANDIDATES_PATH, columns=["user_id", "label"])
            user_cands = all_cands[all_cands["user_id"] == int(top_df["user_id"].iloc[0])]
            cands_total = len(user_cands)
            total_pos = int(user_cands["label"].sum())
            recall_at_k = n_pos_in_top / max(total_pos, 1)
            print(f"    total candidates    : {cands_total}")
            print(f"    total positives     : {total_pos}")
            print(f"    recall@{k:<3}          : {recall_at_k:.4f}")
        except Exception:
            pass


def check_artifacts() -> None:
    """Print artifact availability summary."""
    print("\n[0] Artifact availability check:")
    artifacts = {
        "api_candidates_features.parquet": CANDIDATES_PATH,
        "api_item_metadata.parquet": ITEM_META_PATH,
        "api_future_events.parquet": FUTURE_EVENTS_PATH,
        "api_user_history.parquet": DEMO_STORE / "api_user_history.parquet",
        "api_model_info.json": DEMO_STORE / "api_model_info.json",
        "popular_items.parquet": DEMO_STORE / "popular_items.parquet",
        "best_ranker.joblib": MODEL_PATH,
        "feature_cols.json": FEATURE_COLS_PATH,
    }
    for name, path in artifacts.items():
        status = "OK" if path.exists() else "FAIL MISSING"
        size = f"{path.stat().st_size / 1024 / 1024:.1f} MB" if path.exists() else ""
        print(f"    {status}  {name:<40} {size}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test for api/demo_store/ + PRD model")
    p.add_argument("--user-id", type=int, default=None, help="user_id to score (default: auto)")
    p.add_argument("--k", type=int, default=20, help="top-k to display (default: 20)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    check_artifacts()

    model, feature_cols = load_model_and_features()
    user_id = pick_user(args.user_id)
    top_df = score_user(user_id, model, feature_cols, k=args.k)
    top_df = enrich_with_metadata(top_df)
    top_df = attach_future_hits(top_df, user_id)
    print_results(top_df, user_id, k=args.k)

    print("\n  PRD model smoke test PASSED OK\n")


if __name__ == "__main__":
    main()
