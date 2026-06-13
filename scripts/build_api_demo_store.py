#!/usr/bin/env python3
"""
Build api/demo_store/ artifacts for the FastAPI demo service.

Reads from existing notebook outputs (val_eval dataset, raw item catalog,
tracker interactions) and builds small, pre-extracted Parquet/JSON files
that the API can query without touching the full 135GB val_eval dataset.

Usage:
    python scripts/build_api_demo_store.py [options]

    Required env: recsys conda env (lightgbm, pandas, pyarrow, joblib)
    conda activate recsys && python scripts/build_api_demo_store.py

Outputs (written to api/demo_store/):
    api_candidates_features.parquet  - demo users x candidates with all model features
    api_item_metadata.parquet        - title / brand / type / category per item
    api_user_history.parquet         - recent interactions for demo users
    api_future_events.parquet        - test-period events for demo users
    api_model_info.json              - model card (run_id, metrics, feature count)
    popular_items.parquet            - top-N popular items for cold fallback
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

VAL_EVAL_DIR = (
    PROJECT_ROOT
    / "data/processed/reranker"
    / "final_reranker_v12_repeat__merged_with_dl_all_repeat_history"
    / "dataset/val_eval"
)
TEST_EVAL_DIR = (
    PROJECT_ROOT
    / "data/processed/reranker"
    / "final_reranker_v12_repeat__merged_with_dl_all_repeat_history"
    / "dataset/test"
)
TEST_ORDERS_DIR = (
    PROJECT_ROOT / "data/raw/archive_extra_orders_data/final_apparel_orders_data_07"
)
FEATURE_COLS_PATH = (
    PROJECT_ROOT
    / "models/reranker"
    / "final_reranker_v12_repeat__merged_with_dl_all_repeat_history"
    / "feature_cols.json"
)
ITEMS_DIR = PROJECT_ROOT / "data/raw/ml_ozon_recsys_train_final_apparel_items_data"
CATEGORIES_PATH = (
    PROJECT_ROOT
    / "data/raw/ml_ozon_recsys_train_final_categories_tree"
    / "part-00000-163f9108-23d6-4fd6-98c4-7bcb73ee27e2-c000.snappy.parquet"
)
TRAIN_TRACKER_DIR = PROJECT_ROOT / "data/raw/ml_ozon_recsys_train_final_apparel_tracker_data"
TRAIN_ORDERS_DIR  = PROJECT_ROOT / "data/raw/ml_ozon_recsys_train_final_apparel_orders_data"
TEST_TRACKER_DIR  = PROJECT_ROOT / "data/raw/final_apparel_tracker_data_08_action_widget"

# Order statuses considered as real purchases (exclude canceled/returned)
_POSITIVE_ORDER_STATUSES = {"delivered_orders", "processed_orders", "sent_orders"}
POPULAR_ITEMS_PATH = PROJECT_ROOT / "models/popular_items.parquet"
PRD_RUN_ID_PATH = (
    PROJECT_ROOT
    / "artifacts/reports"
    / "final_reranker_v12_repeat__merged_with_dl_all_repeat_history"
    / "prd_run_id.txt"
)
METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts/reports"
    / "final_reranker_v12_repeat__merged_with_dl_all_repeat_history"
    / "final_reranker_best_val_metrics.parquet"
)

# Source-flag columns present in val_eval
SOURCE_FLAGS = [
    "in_als",
    "in_clip_faiss",
    "in_clip_hnsw",
    "in_dl_bert4rec_small",
    "in_dl_dssm_final",
    "in_dl_sasrec_v3",
    "in_i2i_bm25",
    "in_top_popular",
    "in_user_category_popular",
    "is_repeat_history_candidate",
]

# Extra candidate columns useful for API display
EXTRA_CANDIDATE_COLS = ["catalogid", "cat_l0", "candidate_rank", "rank"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def _quick_collect_item_ids(
    user_ids: set[int],
    tracker_dir: pathlib.Path,
    max_files: int | None = None,
) -> set[int]:
    """Fast scan of tracker dir — reads only user_id + item_id columns."""
    if not tracker_dir.exists():
        return set()
    files = sorted(tracker_dir.rglob("*.parquet"))
    if max_files is not None:
        files = files[:max_files]
    ids: set[int] = set()
    for f in files:
        try:
            chunk = pd.read_parquet(f, columns=["user_id", "item_id"])
            ids.update(
                chunk[chunk["user_id"].isin(user_ids)]["item_id"].astype(int).tolist()
            )
        except Exception:
            pass
    return ids


def _header(msg: str) -> None:
    print(f"\n{'-' * 60}", flush=True)
    print(f"  {msg}", flush=True)
    print(f"{'-' * 60}", flush=True)


def _check_path(p: pathlib.Path, label: str) -> bool:
    if p.exists():
        return True
    print(f"  [SKIP] {label} not found: {p}", flush=True)
    return False


def _extract_attr(attrs: Any, name: str) -> str | None:
    """Extract attribute_value by attribute_name from raw attributes array."""
    if attrs is None:
        return None
    try:
        for a in attrs:
            if isinstance(a, dict) and a.get("attribute_name") == name:
                v = a.get("attribute_value")
                return str(v).strip() if v is not None else None
    except Exception:
        pass
    return None


def _build_catalog_map(categories_path: pathlib.Path) -> dict[int, dict]:
    """Return catalogid -> {category_name, category_path} mapping."""
    if not categories_path.exists():
        _log(f"Categories file not found: {categories_path}")
        return {}

    cats = pd.read_parquet(categories_path)
    result: dict[int, dict] = {}
    for _, row in cats.iterrows():
        cpath = row["catalogpath"]
        if not isinstance(cpath, (list, np.ndarray)):
            continue
        names = [
            e.get("name", "") for e in cpath if isinstance(e, dict) and e.get("id", -1) != -1
        ]
        # catalogpath is ordered leaf -> root; reverse for display
        names = list(reversed(names))
        leaf_name = names[-1] if names else ""
        result[int(row["catalogid"])] = {
            "category_name": leaf_name,
            "category_path": " > ".join(names),
        }
    return result


# ---------------------------------------------------------------------------
# Step 1: feature_cols
# ---------------------------------------------------------------------------

def load_feature_cols() -> list[str]:
    _header("Step 1: Loading feature_cols.json")
    if not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError(f"feature_cols.json not found: {FEATURE_COLS_PATH}")

    fc = json.loads(FEATURE_COLS_PATH.read_text(encoding="utf-8"))
    cols = fc["numeric_feature_cols"]  # LightGBM uses numeric only (240 cols)
    _log(f"numeric_feature_cols: {len(cols)} columns")
    _log(f"  first 3: {cols[:3]}")
    _log(f"  last  3: {cols[-3:]}")
    return cols


# ---------------------------------------------------------------------------
# Step 2: Select demo users from val_eval
# ---------------------------------------------------------------------------

def select_demo_users(
    n_demo_users: int, feature_cols: list[str], source_dir: pathlib.Path | None = None
) -> pd.DataFrame:
    """Scan candidate shards until we have n_demo_users with at least 1 positive."""
    if source_dir is None:
        source_dir = VAL_EVAL_DIR
    _header(f"Step 2: Selecting {n_demo_users} demo users from {source_dir.name}")

    if not source_dir.exists():
        raise FileNotFoundError(f"candidates dir not found: {source_dir}")

    # Columns to load from each shard
    keep_cols = (
        ["user_id", "item_id", "label"]
        + [c for c in feature_cols]
        + [c for c in SOURCE_FLAGS]
        + [c for c in EXTRA_CANDIDATE_COLS]
    )

    selected_users: set[int] = set()
    frames: list[pd.DataFrame] = []

    shards = sorted(source_dir.iterdir())
    t0 = time.perf_counter()

    for shard_dir in shards:
        pq_files = list(shard_dir.rglob("*.parquet"))
        if not pq_files:
            continue

        shard_df = pd.read_parquet(pq_files[0])

        # Resolve available keep_cols (some extra cols might be absent)
        avail = [c for c in keep_cols if c in shard_df.columns]
        missing_feature = [c for c in feature_cols if c not in shard_df.columns]
        if missing_feature:
            _log(f"WARNING: {len(missing_feature)} feature cols missing in shard {shard_dir.name}")

        shard_df = shard_df[avail]

        # Find users with >=1 positive not already selected
        pos_users = shard_df[shard_df["label"] == 1]["user_id"].unique()
        new_users = [u for u in pos_users if u not in selected_users]

        if not new_users:
            continue

        # Take as many as we still need
        need = n_demo_users - len(selected_users)
        batch = new_users[:need]
        selected_users.update(batch)

        batch_df = shard_df[shard_df["user_id"].isin(batch)]
        frames.append(batch_df)

        elapsed = time.perf_counter() - t0
        _log(
            f"shard {shard_dir.name}: +{len(batch)} users  "
            f"(total {len(selected_users)}/{n_demo_users})  {elapsed:.1f}s"
        )

        if len(selected_users) >= n_demo_users:
            break

    if not frames:
        raise RuntimeError(f"No demo users found in {source_dir}. Directory may be empty or have no positives.")

    df = pd.concat(frames, ignore_index=True)
    _log(f"Demo users collected: {df['user_id'].nunique()}")
    _log(f"Total candidate rows : {len(df):,}")
    _log(f"Total positives      : {df['label'].sum():,}")
    _log(
        f"Avg candidates/user  : {len(df) / df['user_id'].nunique():.0f}"
    )

    return df


# ---------------------------------------------------------------------------
# Step 3: api_candidates_features.parquet
# ---------------------------------------------------------------------------

def save_candidates(df: pd.DataFrame, output_dir: pathlib.Path, feature_cols: list[str]) -> None:
    _header("Step 3: Saving api_candidates_features.parquet")

    # Validate that all model feature columns are present
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  [ERROR] {len(missing)} model feature columns missing from candidates:", flush=True)
        for c in missing[:10]:
            print(f"    {c}", flush=True)
        raise ValueError(f"{len(missing)} model feature columns missing . rebuild val_eval")
    else:
        _log(f"All {len(feature_cols)} model feature columns present OK")

    out_path = output_dir / "api_candidates_features.parquet"
    df.to_parquet(out_path, index=False, compression="snappy")

    size_mb = out_path.stat().st_size / 1024 / 1024
    _log(f"Saved: {out_path.name}  {size_mb:.1f} MB")
    _log(f"Shape: {df.shape}")
    _log(f"Users: {df['user_id'].nunique()}, positives: {int(df['label'].sum())}")


# ---------------------------------------------------------------------------
# Step 4: api_item_metadata.parquet
# ---------------------------------------------------------------------------

def build_item_metadata(item_ids: set[int], output_dir: pathlib.Path) -> pd.DataFrame:
    _header("Step 4: Building api_item_metadata.parquet")

    if not _check_path(ITEMS_DIR, "items_dir"):
        _log("Skipping item metadata . items dir not found")
        return pd.DataFrame(columns=["item_id"])

    catalog_map = _build_catalog_map(CATEGORIES_PATH)
    _log(f"Categories loaded: {len(catalog_map)} catalogids")

    # Read all item parts (100 files x ~190MB each . we read 3 cols only)
    item_files = sorted(ITEMS_DIR.rglob("*.parquet"))
    _log(f"Reading {len(item_files)} item parquet files (columns: item_id, itemname, catalogid, attributes)")

    parts: list[pd.DataFrame] = []
    for f in item_files:
        chunk = pd.read_parquet(f, columns=["item_id", "itemname", "catalogid", "attributes"])
        chunk = chunk[chunk["item_id"].isin(item_ids)]
        if len(chunk):
            parts.append(chunk)

    if not parts:
        _log("No matching items found")
        return pd.DataFrame(columns=["item_id", "title", "brand", "item_type", "catalogid"])

    items = pd.concat(parts, ignore_index=True).drop_duplicates("item_id")
    _log(f"Items matched: {len(items):,} / {len(item_ids):,} requested")

    # Extract brand and type from attributes
    items["brand"] = items["attributes"].apply(lambda a: _extract_attr(a, "Brand"))
    items["item_type"] = items["attributes"].apply(lambda a: _extract_attr(a, "Type"))

    # Attach category info
    items["category_name"] = items["catalogid"].map(
        lambda cid: catalog_map.get(int(cid), {}).get("category_name", "")
    )
    items["category_path"] = items["catalogid"].map(
        lambda cid: catalog_map.get(int(cid), {}).get("category_path", "")
    )

    # Clean up
    items = items.rename(columns={"itemname": "title"})[
        ["item_id", "title", "brand", "item_type", "catalogid", "category_name", "category_path"]
    ]

    out_path = output_dir / "api_item_metadata.parquet"
    items.to_parquet(out_path, index=False, compression="snappy")
    size_mb = out_path.stat().st_size / 1024 / 1024
    _log(f"Saved: {out_path.name}  {size_mb:.1f} MB, {len(items)} items")
    return items


# ---------------------------------------------------------------------------
# Step 5: api_user_history.parquet
# ---------------------------------------------------------------------------

def build_user_history(
    demo_user_ids: set[int],
    output_dir: pathlib.Path,
    item_meta: pd.DataFrame,
    max_tracker_files: int,
) -> None:
    _header("Step 5: Building api_user_history.parquet")

    if not _check_path(TRAIN_TRACKER_DIR, "train_tracker_dir"):
        _log("Skipping user history")
        return

    tracker_files = sorted(TRAIN_TRACKER_DIR.rglob("*.parquet"))
    n_files = min(len(tracker_files), max_tracker_files)
    _log(f"Scanning {n_files}/{len(tracker_files)} train tracker files for {len(demo_user_ids)} users")

    parts: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    for i, f in enumerate(tracker_files[:n_files]):
        chunk = pd.read_parquet(
            f, columns=["user_id", "item_id", "timestamp", "action_type"]
        )
        chunk = chunk[chunk["user_id"].isin(demo_user_ids)]
        if len(chunk):
            parts.append(chunk)
        if (i + 1) % 20 == 0:
            _log(f"  {i + 1}/{n_files} files  {time.perf_counter() - t0:.1f}s")

    if not parts:
        _log("No history found for demo users in scanned files")
        pd.DataFrame(
            columns=["user_id", "item_id", "timestamp", "action_type", "title", "category_name"]
        ).to_parquet(output_dir / "api_user_history.parquet", index=False)
        return

    hist = pd.concat(parts, ignore_index=True).sort_values(["user_id", "timestamp"])

    # Step 5b: add train-period purchase orders to history so "до: order" badges work
    order_parts: list[pd.DataFrame] = []
    if TRAIN_ORDERS_DIR.exists():
        order_files = sorted(TRAIN_ORDERS_DIR.rglob("*.parquet"))
        _log(f"Loading train orders from {len(order_files)} files (positive statuses only)")
        for of in order_files:
            try:
                ochunk = pd.read_parquet(
                    of,
                    columns=["user_id", "item_id", "created_timestamp", "last_status"],
                )
                ochunk = ochunk[
                    ochunk["user_id"].isin(demo_user_ids)
                    & ochunk["last_status"].isin(_POSITIVE_ORDER_STATUSES)
                ]
                if len(ochunk):
                    ochunk = ochunk.rename(columns={"created_timestamp": "timestamp"})
                    ochunk["action_type"] = "order"
                    order_parts.append(ochunk[["user_id", "item_id", "timestamp", "action_type"]])
            except Exception as exc:
                _log(f"  Warning: could not read {of.name}: {exc}")
    else:
        _log(f"TRAIN_ORDERS_DIR not found ({TRAIN_ORDERS_DIR}), skipping order history")

    if order_parts:
        orders_df = pd.concat(order_parts, ignore_index=True)
        _log(f"  Found {len(orders_df):,} train order rows for {orders_df['user_id'].nunique()} demo users")
        hist = pd.concat([hist, orders_df], ignore_index=True).sort_values(["user_id", "timestamp"])
    else:
        _log("No train order rows found for demo users")

    # Join item metadata
    if not item_meta.empty and "title" in item_meta.columns:
        hist = hist.merge(
            item_meta[["item_id", "title", "category_name"]],
            on="item_id",
            how="left",
        )

    out_path = output_dir / "api_user_history.parquet"
    hist.to_parquet(out_path, index=False, compression="snappy")
    size_mb = out_path.stat().st_size / 1024 / 1024
    _log(
        f"Saved: {out_path.name}  {size_mb:.1f} MB  "
        f"({len(hist):,} rows, {hist['user_id'].nunique()} users)"
    )

    # Coverage report
    covered = hist["user_id"].nunique()
    _log(f"History coverage: {covered}/{len(demo_user_ids)} demo users have history records")
    if max_tracker_files < len(tracker_files):
        _log(
            f"NOTE: scanned only {max_tracker_files}/{len(tracker_files)} tracker files . "
            "some users may have no history. Increase --max-tracker-files for full coverage."
        )


# ---------------------------------------------------------------------------
# Step 6: api_future_events.parquet
# ---------------------------------------------------------------------------

def build_future_events(
    demo_user_ids: set[int],
    output_dir: pathlib.Path,
    item_meta: pd.DataFrame,
) -> None:
    _header("Step 6: Building api_future_events.parquet (tracker + real orders)")

    parts: list[pd.DataFrame] = []

    # 6a — tracker behavioral events
    if _check_path(TEST_TRACKER_DIR, "test_tracker_dir"):
        test_files = sorted(TEST_TRACKER_DIR.rglob("*.parquet"))
        _log(f"Scanning {len(test_files)} test tracker files for {len(demo_user_ids)} users")
        t0 = time.perf_counter()
        for i, f in enumerate(test_files):
            chunk = pd.read_parquet(
                f, columns=["user_id", "item_id", "timestamp", "action_type"]
            )
            chunk = chunk[chunk["user_id"].isin(demo_user_ids)]
            if len(chunk):
                parts.append(chunk)
            if (i + 1) % 500 == 0:
                _log(f"  {i + 1}/{len(test_files)} files  {time.perf_counter() - t0:.1f}s")
        _log(f"Tracker events collected: {sum(len(p) for p in parts):,}")
    else:
        _log("Skipping tracker events (test_tracker_dir not found)")

    # 6b — real test-period orders (delivered + processed, not canceled)
    if _check_path(TEST_ORDERS_DIR, "test_orders_dir"):
        order_parts: list[pd.DataFrame] = []
        for partition_dir in sorted(TEST_ORDERS_DIR.iterdir()):
            if not partition_dir.is_dir():
                continue
            for f in sorted(partition_dir.glob("*.parquet")):
                chunk = pd.read_parquet(
                    f, columns=["user_id", "item_id", "created_timestamp", "last_status"]
                )
                chunk = chunk[chunk["user_id"].isin(demo_user_ids)]
                chunk = chunk[chunk["last_status"] != "canceled_orders"]
                if len(chunk):
                    order_parts.append(chunk)
        if order_parts:
            orders = pd.concat(order_parts, ignore_index=True)
            orders = orders.rename(columns={"created_timestamp": "timestamp"})
            orders["action_type"] = "order"
            orders = orders[["user_id", "item_id", "timestamp", "action_type"]]
            parts.append(orders)
            _log(f"Real orders added: {len(orders):,} (delivered + processed, {orders['user_id'].nunique()} users)")
        else:
            _log("No non-canceled orders found for demo users")
    else:
        _log("Skipping real orders (test_orders_dir not found)")

    if not parts:
        _log("No future events for demo users")
        pd.DataFrame(
            columns=["user_id", "item_id", "timestamp", "action_type", "title", "category_name"]
        ).to_parquet(output_dir / "api_future_events.parquet", index=False)
        return

    future = pd.concat(parts, ignore_index=True).sort_values(["user_id", "timestamp"])

    if not item_meta.empty and "title" in item_meta.columns:
        future = future.merge(
            item_meta[["item_id", "title", "category_name"]],
            on="item_id",
            how="left",
        )

    out_path = output_dir / "api_future_events.parquet"
    future.to_parquet(out_path, index=False, compression="snappy")
    size_mb = out_path.stat().st_size / 1024 / 1024
    _log(
        f"Saved: {out_path.name}  {size_mb:.1f} MB  "
        f"({len(future):,} rows, {future['user_id'].nunique()} users)"
    )

    future_orders = future[future["action_type"] == "order"]
    _log(f"Real order events: {len(future_orders):,} for {future_orders['user_id'].nunique()} users")


# ---------------------------------------------------------------------------
# Step 6b: split_info.json
# ---------------------------------------------------------------------------

def save_split_info(
    output_dir: pathlib.Path,
    n_demo_users: int,
    max_tracker_files: int,
    use_test: bool = False,
    history_ts_min: str | None = None,
    history_ts_max: str | None = None,
    future_ts_min: str | None = None,
    future_ts_max: str | None = None,
) -> None:
    _header("Step 6b: Saving split_info.json")
    if use_test:
        info: dict = {
            "candidates_source": "test_eval",
            "candidates_path": str(TEST_EVAL_DIR.relative_to(PROJECT_ROOT)),
            "history_source": "train_tracker",
            "history_path": str(TRAIN_TRACKER_DIR.relative_to(PROJECT_ROOT)),
            "future_events_source": "test_tracker_08 + test_orders",
            "future_events_path": str(TEST_TRACKER_DIR.relative_to(PROJECT_ROOT)),
            "label_definition": "test_target_window_positive",
            "ground_truth_note": (
                "MATCH: labels (Jul 2-15 orders) == future_events (test period). "
                "purchase_from_recs = real orders in top-k recommendations."
            ),
            "n_demo_users": n_demo_users,
            "max_tracker_files_scanned": max_tracker_files,
        }
    else:
        info = {
            "candidates_source": "val_eval",
            "candidates_path": str(VAL_EVAL_DIR.relative_to(PROJECT_ROOT)),
            "history_source": "train_tracker",
            "history_path": str(TRAIN_TRACKER_DIR.relative_to(PROJECT_ROOT)),
            "future_events_source": "test_tracker_08",
            "future_events_path": str(TEST_TRACKER_DIR.relative_to(PROJECT_ROOT)),
            "label_definition": "val_target_window_positive",
            "ground_truth_note": (
                "MISMATCH: label (val period) != future_events (test period 08). "
                "Use label_hits_at_k for metrics consistent with /demo/users sidebar."
            ),
            "n_demo_users": n_demo_users,
            "max_tracker_files_scanned": max_tracker_files,
        }
    if history_ts_min:
        info["history_ts_min"] = history_ts_min
    if history_ts_max:
        info["history_ts_max"] = history_ts_max
    if future_ts_min:
        info["future_ts_min"] = future_ts_min
    if future_ts_max:
        info["future_ts_max"] = future_ts_max

    out = output_dir / "split_info.json"
    out.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"Saved: {out.name}")
    _log(f"  ground_truth_note: {info['ground_truth_note']}")


# ---------------------------------------------------------------------------
# Step 7: api_model_info.json
# ---------------------------------------------------------------------------

def build_model_info(feature_cols: list[str], output_dir: pathlib.Path) -> None:
    _header("Step 7: Building api_model_info.json")

    info: dict[str, Any] = {
        "model_name": "2stage_reranker_v12_repeat",
        "model_file": "models/reranker/final_reranker_v12_repeat__merged_with_dl_all_repeat_history/best_ranker.joblib",
        "feature_cols_file": str(FEATURE_COLS_PATH.relative_to(PROJECT_ROOT)),
        "feature_count": len(feature_cols),
        "candidate_store": "api/demo_store/api_candidates_features.parquet",
        "description": "2-stage: retrieval (ALS + TopPop + CLIP + i2i + DL + repeat-history) -> LightGBM LambdaRank",
        "prd_run_id": None,
        "model_uri": None,
        "metrics": {},
        "baseline": {
            "TopPopular_ndcg_at_100": 0.005056,
            "TopPopular_recall_at_100": 0.017450,
            "ALS_ndcg_at_100": 0.004486,
            "CandidateRawRank_ndcg_at_100": 0.029241,
        },
    }

    # Try to load prd_run_id
    if PRD_RUN_ID_PATH.exists():
        info["prd_run_id"] = PRD_RUN_ID_PATH.read_text(encoding="utf-8").strip()
        _log(f"prd_run_id: {info['prd_run_id']}")
        # Derive model_uri from run_id
        info["model_uri"] = f"runs:/{info['prd_run_id']}/model"
    else:
        _log(f"prd_run_id.txt not found at {PRD_RUN_ID_PATH} . using hardcoded values")
        info["prd_run_id"] = "ba67e7e4efd842778a8b5d6dca78d1fe"
        info["model_uri"] = "runs:/ba67e7e4efd842778a8b5d6dca78d1fe/model"

    # Try to load metrics
    # Format: columns [model, k, users_total, hit_rate, recall, precision, ndcg, map, mrr]
    if METRICS_PATH.exists():
        try:
            mdf = pd.read_parquet(METRICS_PATH)
            metrics_dict = {}
            for _, row in mdf.iterrows():
                k_val = int(row["k"]) if "k" in row else 0
                for metric in ("ndcg", "recall", "hit_rate", "precision", "map", "mrr"):
                    if metric in row:
                        key = f"reranker_{metric}_at_{k_val}"
                        metrics_dict[key] = round(float(row[metric]), 6)
            info["metrics"] = metrics_dict
            _log(f"Metrics loaded: {len(metrics_dict)} entries")
            # Show key metric
            ndcg100 = metrics_dict.get("reranker_ndcg_at_100")
            if ndcg100:
                _log(f"  ndcg@100 = {ndcg100}")
        except Exception as e:
            _log(f"Could not load metrics: {e}")
    else:
        _log("Metrics parquet not found - using inline values")
        info["metrics"] = {
            "reranker_ndcg_at_100": 0.087751,
            "reranker_recall_at_100": 0.145960,
        }

    out_path = output_dir / "api_model_info.json"
    out_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Step 8: popular_items.parquet
# ---------------------------------------------------------------------------

def save_popular_items(output_dir: pathlib.Path, top_k: int = 2000) -> None:
    _header("Step 8: Saving popular_items.parquet")

    if not _check_path(POPULAR_ITEMS_PATH, "popular_items"):
        _log("Skipping popular items")
        return

    pop = pd.read_parquet(POPULAR_ITEMS_PATH)
    pop = pop.sort_values("global_pop_rank").head(top_k)[["item_id", "global_pop_rank", "score"]]

    out_path = output_dir / "popular_items.parquet"
    pop.to_parquet(out_path, index=False, compression="snappy")
    size_kb = out_path.stat().st_size / 1024
    _log(f"Saved: {out_path.name}  {size_kb:.0f} KB  ({len(pop)} items)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(output_dir: pathlib.Path) -> None:
    _header("Build Summary")
    total_mb = 0.0
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        total_mb += size_mb
        _log(f"{f.name:<45}  {size_mb:6.1f} MB")
    _log(f"{'TOTAL':<45}  {total_mb:6.1f} MB")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build api/demo_store/ artifacts for FastAPI demo service"
    )
    p.add_argument(
        "--n-demo-users", type=int, default=500,
        help="Number of demo users to select (users with >=1 positive label). Default: 500",
    )
    p.add_argument(
        "--output-dir", type=pathlib.Path,
        default=PROJECT_ROOT / "api/demo_store",
        help="Output directory for demo store artifacts. Default: api/demo_store/",
    )
    p.add_argument(
        "--max-tracker-files", type=int, default=200,
        help=(
            "Max number of train-tracker parquet files to scan for user history. "
            "Full coverage needs all 200 files (~20 GB, ~5 min for 500 users). Default: 200"
        ),
    )
    p.add_argument(
        "--popular-top-k", type=int, default=2000,
        help="Number of popular items to keep as cold fallback. Default: 2000",
    )
    p.add_argument(
        "--skip-history", action="store_true",
        help="Skip building user history (faster build for testing)",
    )
    p.add_argument(
        "--skip-future-events", action="store_true",
        help="Skip building future events (faster build for testing)",
    )
    p.add_argument(
        "--use-test", action="store_true",
        help=(
            "Use test_eval candidates (Jul 2-15 labels) instead of val_eval. "
            "Produces realistic purchase_from_recs metric. Requires dataset/test/ to exist."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()

    source_dir = TEST_EVAL_DIR if args.use_test else VAL_EVAL_DIR

    print("\n" + "=" * 60)
    print("  build_api_demo_store.py")
    print(f"  project root : {PROJECT_ROOT}")
    print(f"  output dir   : {args.output_dir}")
    print(f"  demo users   : {args.n_demo_users}")
    print(f"  candidate src: {'test_eval (Jul 2-15)' if args.use_test else 'val_eval (Jun 18 - Jul 1)'}")
    print("=" * 60)

    # Create output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1
    feature_cols = load_feature_cols()

    # Step 2
    candidates_df = select_demo_users(args.n_demo_users, feature_cols, source_dir=source_dir)
    demo_user_ids: set[int] = set(candidates_df["user_id"].unique().tolist())

    # Step 3
    save_candidates(candidates_df, args.output_dir, feature_cols)

    # Step 4: build metadata for union of candidate + history + future item_ids
    # Quick pre-scan tracker dirs to collect all item_ids (reads only user_id + item_id cols)
    candidate_item_ids = set(candidates_df["item_id"].unique().tolist())
    extra_item_ids: set[int] = set()
    if not args.skip_history:
        _log("Pre-scanning train tracker for item_ids (fast pass)...")
        extra_item_ids |= _quick_collect_item_ids(
            demo_user_ids, TRAIN_TRACKER_DIR, args.max_tracker_files
        )
        _log(f"  History item_ids: {len(extra_item_ids):,}")
    if not args.skip_future_events:
        _log("Pre-scanning test tracker for item_ids (fast pass)...")
        fut_extra = _quick_collect_item_ids(demo_user_ids, TEST_TRACKER_DIR)
        _log(f"  Future item_ids: {len(fut_extra):,}")
        extra_item_ids |= fut_extra
    all_item_ids = candidate_item_ids | extra_item_ids
    _log(f"Total item_ids for metadata: {len(all_item_ids):,} ({len(candidate_item_ids):,} candidates + {len(extra_item_ids):,} from tracker)")
    item_meta = build_item_metadata(all_item_ids, args.output_dir)

    # Step 5
    if not args.skip_history:
        build_user_history(demo_user_ids, args.output_dir, item_meta, args.max_tracker_files)
    else:
        _log("Skipping user history (--skip-history)")

    # Step 6
    if not args.skip_future_events:
        build_future_events(demo_user_ids, args.output_dir, item_meta)
    else:
        _log("Skipping future events (--skip-future-events)")

    # Step 6b: split_info
    # Read timestamps from built artifacts for split_info
    hist_ts_min = hist_ts_max = fut_ts_min = fut_ts_max = None
    try:
        hist_path = args.output_dir / "api_user_history.parquet"
        if hist_path.exists():
            h_ts = pd.read_parquet(hist_path, columns=["timestamp"])["timestamp"]
            hist_ts_min = str(h_ts.min())
            hist_ts_max = str(h_ts.max())
    except Exception:
        pass
    try:
        fut_path = args.output_dir / "api_future_events.parquet"
        if fut_path.exists():
            f_ts = pd.read_parquet(fut_path, columns=["timestamp"])["timestamp"]
            fut_ts_min = str(f_ts.min())
            fut_ts_max = str(f_ts.max())
    except Exception:
        pass
    save_split_info(
        args.output_dir,
        n_demo_users=args.n_demo_users,
        max_tracker_files=args.max_tracker_files,
        use_test=args.use_test,
        history_ts_min=hist_ts_min,
        history_ts_max=hist_ts_max,
        future_ts_min=fut_ts_min,
        future_ts_max=fut_ts_max,
    )

    # Step 7
    build_model_info(feature_cols, args.output_dir)

    # Step 8
    save_popular_items(args.output_dir, args.popular_top_k)

    # Summary
    print_summary(args.output_dir)

    elapsed = time.perf_counter() - t_start
    print(f"  Done in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
