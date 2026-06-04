#!/usr/bin/env python3
"""Build repeat_history candidate source for v12 reranker.

Memory-safe: one user_shard at a time, never holds all users in RAM.
Split windows from FinalRankerConfig (v10/v11 notebooks).

Usage:
    # Smoke test — single shard:
    python scripts/build_repeat_history_candidates.py --split ranker_train --user-shard 0

    # Full split:
    python scripts/build_repeat_history_candidates.py --split val --overwrite

    # Dry run:
    python scripts/build_repeat_history_candidates.py --split ranker_train --dry-run

    # Audit existing output:
    python scripts/build_repeat_history_candidates.py --audit-only
"""
from __future__ import annotations

import argparse
import gc
import sys
from datetime import date
from pathlib import Path

import polars as pl

# ── Split windows ──────────────────────────────────────────────────────────────
# ranker_train/val: same as RepeatHistoryConfig in gen_repeat_history_baseline_notebook.py
# test: full_context_{start,end} from FinalRankerConfig in 5_reranker_ndcg_boost_v11_experiments.ipynb
SPLIT_WINDOWS: dict[str, dict[str, date]] = {
    "ranker_train": {"context_start": date(2025, 4, 18), "context_end": date(2025, 6,  3)},
    "val":          {"context_start": date(2025, 4, 18), "context_end": date(2025, 6, 17)},
    "test":         {"context_start": date(2025, 5,  2), "context_end": date(2025, 7,  1)},
}

SOURCE_NAME = "repeat_history"
RRF_K       = 60
W_ORDERS    = 100.0
W_CART      =  30.0

OUTPUT_COLS = [
    "user_id", "item_id",
    "repeat_history_score", "repeat_history_rank",
    "repeat_recency_score", "repeat_recency_rank",
    "repeat_frequency_score", "repeat_frequency_rank",
    "repeat_cart_orders_score", "repeat_cart_orders_rank",
    "repeat_in_recency", "repeat_in_frequency", "repeat_in_cart_orders",
    "repeat_source",
]


# ── Utilities ──────────────────────────────────────────────────────────────────

def find_project_root(start: Path = Path.cwd()) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "datamart").exists():
            return p
    raise FileNotFoundError("Cannot find project root with data/datamart/")


def _scan(path: Path) -> pl.LazyFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.scan_parquet(str(path / "**" / "*.parquet"), hive_partitioning=True)


def _date_filter(lf: pl.LazyFrame, col: str, start: date, end: date) -> pl.LazyFrame:
    return lf.filter(
        (pl.col(col).cast(pl.Date, strict=False) >= pl.lit(start).cast(pl.Date))
        & (pl.col(col).cast(pl.Date, strict=False) <= pl.lit(end).cast(pl.Date))
    )


def _shard_filter(lf: pl.LazyFrame, shard_id: int, num_shards: int) -> pl.LazyFrame:
    return lf.filter((pl.col("user_id").cast(pl.Int64) % num_shards) == shard_id)


def _add_zero_cols(lf: pl.LazyFrame, cols: list[str]) -> pl.LazyFrame:
    existing = set(lf.collect_schema().names())
    exprs = [pl.lit(0.0, dtype=pl.Float64).alias(c) for c in cols if c not in existing]
    return lf.with_columns(exprs) if exprs else lf


def _topk_lazy(lf: pl.LazyFrame, k: int) -> pl.LazyFrame:
    """sort+group_by+head — no window function over 100M+ rows."""
    return (
        lf.filter(pl.col("score") > 0)
        .sort(["user_id", "score"], descending=[False, True])
        .group_by("user_id", maintain_order=True)
        .agg(pl.col("item_id").head(k), pl.col("score").head(k))
        .explode(["item_id", "score"])
    )


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except Exception:
        return lf.collect()


# ── Component builders ─────────────────────────────────────────────────────────

def build_b1_lf(events_lf: pl.LazyFrame, orders_lf: pl.LazyFrame) -> pl.LazyFrame:
    """B1: score = cnt_orders*100 + cnt_to_cart*30."""
    ev = _add_zero_cols(events_lf, ["cnt_to_cart"])
    od = _add_zero_cols(orders_lf, ["cnt_orders"])
    cart = (
        ev.select(
            pl.col("user_id").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
            (pl.col("cnt_to_cart").cast(pl.Float64).fill_null(0) * W_CART).alias("score"),
        )
        .group_by(["user_id", "item_id"]).agg(pl.col("score").sum())
    )
    orders = (
        od.select(
            pl.col("user_id").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
            (pl.col("cnt_orders").cast(pl.Float64).fill_null(0) * W_ORDERS).alias("score"),
        )
        .group_by(["user_id", "item_id"]).agg(pl.col("score").sum())
    )
    return (
        pl.concat([cart, orders])
        .group_by(["user_id", "item_id"]).agg(pl.col("score").sum())
        .filter(pl.col("score") > 0)
    )


def build_b2_lf(events_lf: pl.LazyFrame) -> pl.LazyFrame:
    """B2: score = epoch_days(last_event) + cnt/1e6."""
    raw_cols = set(events_lf.collect_schema().names())
    cnt_col = "cnt_events" if "cnt_events" in raw_cols else "cnt_page_view"
    lf = _add_zero_cols(events_lf, [cnt_col])
    return (
        lf.select(
            pl.col("user_id").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
            pl.col("event_date").cast(pl.Date),
            pl.col(cnt_col).cast(pl.Float64).fill_null(0).alias("_cnt"),
        )
        .group_by(["user_id", "item_id"])
        .agg(
            pl.col("event_date").max().alias("last_date"),
            pl.col("_cnt").sum().alias("cnt_total"),
        )
        .with_columns(
            (
                pl.col("last_date").cast(pl.Int32).cast(pl.Float64)
                + pl.col("cnt_total") / 1_000_000.0
            ).alias("score")
        )
        .filter(pl.col("score").is_not_null())
        .select(["user_id", "item_id", "score"])
    )


def build_b3_lf(events_lf: pl.LazyFrame) -> pl.LazyFrame:
    """B3: score = cnt_page_view + cnt_view_description + cnt_review_view."""
    lf = _add_zero_cols(events_lf, ["cnt_page_view", "cnt_view_description", "cnt_review_view"])
    return (
        lf.select(
            pl.col("user_id").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
            (
                pl.col("cnt_page_view").cast(pl.Float64).fill_null(0)
                + pl.col("cnt_view_description").cast(pl.Float64).fill_null(0)
                + pl.col("cnt_review_view").cast(pl.Float64).fill_null(0)
            ).alias("score"),
        )
        .group_by(["user_id", "item_id"]).agg(pl.col("score").sum())
        .filter(pl.col("score") > 0)
    )


# ── RRF merge ─────────────────────────────────────────────────────────────────

def rrf_merge(
    b1_df: pl.DataFrame,
    b2_df: pl.DataFrame,
    b3_df: pl.DataFrame,
    top_k: int,
) -> pl.DataFrame:
    """B5 RRF: merge B1/B2/B3 via 1/(rrf_k + rank). Inputs are shard-level topk DataFrames."""

    def _rank(df: pl.DataFrame, score_col: str, rank_col: str) -> pl.DataFrame:
        if df.height == 0:
            return df.with_columns(pl.lit(None, dtype=pl.Int32).alias(rank_col))
        return df.sort(score_col, descending=True).with_columns(
            pl.col(score_col).rank(method="ordinal", descending=True)
            .over("user_id").cast(pl.Int32).alias(rank_col)
        )

    b2r = _rank(b2_df, "score", "b2_rank").rename({"score": "repeat_recency_score"})
    b3r = _rank(b3_df, "score", "b3_rank").rename({"score": "repeat_frequency_score"})
    b1r = _rank(b1_df, "score", "b1_rank").rename({"score": "repeat_cart_orders_score"})

    all_items = pl.concat([
        b2_df.select(["user_id", "item_id"]),
        b3_df.select(["user_id", "item_id"]),
        b1_df.select(["user_id", "item_id"]),
    ]).unique(["user_id", "item_id"])

    result = (
        all_items
        .join(b2r, on=["user_id", "item_id"], how="left")
        .join(b3r, on=["user_id", "item_id"], how="left")
        .join(b1r, on=["user_id", "item_id"], how="left")
        .with_columns([
            (1.0 / (RRF_K + pl.col("b2_rank"))).fill_null(0.0).alias("_rrf2"),
            (1.0 / (RRF_K + pl.col("b3_rank"))).fill_null(0.0).alias("_rrf3"),
            (1.0 / (RRF_K + pl.col("b1_rank"))).fill_null(0.0).alias("_rrf1"),
            pl.col("b2_rank").is_not_null().alias("repeat_in_recency"),
            pl.col("b3_rank").is_not_null().alias("repeat_in_frequency"),
            pl.col("b1_rank").is_not_null().alias("repeat_in_cart_orders"),
        ])
        .with_columns(
            (pl.col("_rrf2") + pl.col("_rrf3") + pl.col("_rrf1")).alias("repeat_history_score")
        )
        .sort(["user_id", "repeat_history_score"], descending=[False, True])
        .group_by("user_id", maintain_order=True)
        .agg([
            pl.col("item_id").head(top_k),
            pl.col("repeat_history_score").head(top_k),
            pl.col("repeat_recency_score").head(top_k),
            pl.col("b2_rank").head(top_k).alias("repeat_recency_rank"),
            pl.col("repeat_frequency_score").head(top_k),
            pl.col("b3_rank").head(top_k).alias("repeat_frequency_rank"),
            pl.col("repeat_cart_orders_score").head(top_k),
            pl.col("b1_rank").head(top_k).alias("repeat_cart_orders_rank"),
            pl.col("repeat_in_recency").head(top_k),
            pl.col("repeat_in_frequency").head(top_k),
            pl.col("repeat_in_cart_orders").head(top_k),
        ])
        .explode([
            "item_id", "repeat_history_score",
            "repeat_recency_score", "repeat_recency_rank",
            "repeat_frequency_score", "repeat_frequency_rank",
            "repeat_cart_orders_score", "repeat_cart_orders_rank",
            "repeat_in_recency", "repeat_in_frequency", "repeat_in_cart_orders",
        ])
        .with_columns([
            pl.col("repeat_history_score").rank(method="ordinal", descending=True)
            .over("user_id").cast(pl.Int32).alias("repeat_history_rank"),
            pl.lit(f"{SOURCE_NAME}_b5_rrf").alias("repeat_source"),
        ])
        .select(OUTPUT_COLS)
    )
    return result


# ── Shard processing ───────────────────────────────────────────────────────────

def process_shard(
    split: str,
    shard_id: int,
    top_k: int,
    num_shards: int,
    out_root: Path,
    events_dir: Path,
    orders_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> int:
    """Process one user shard. Returns rows written (0 on skip/dry-run)."""
    shard_dir = out_root / split / f"user_shard={shard_id}"
    out_file  = shard_dir / f"{SOURCE_NAME}.parquet"

    if out_file.exists() and not overwrite:
        print(f"  shard {shard_id:3d}: SKIP (exists, use --overwrite to rebuild)")
        return 0
    if dry_run:
        win = SPLIT_WINDOWS[split]
        print(f"  shard {shard_id:3d}: DRY-RUN  [{win['context_start']} - {win['context_end']}]  → {out_file}")
        return 0

    win = SPLIT_WINDOWS[split]
    cs, ce = win["context_start"], win["context_end"]

    def ev() -> pl.LazyFrame:
        return _shard_filter(
            _date_filter(_scan(events_dir), "event_date", cs, ce),
            shard_id, num_shards,
        )

    def od() -> pl.LazyFrame:
        return _shard_filter(
            _date_filter(_scan(orders_dir), "created_date", cs, ce),
            shard_id, num_shards,
        )

    b1_df = _collect(_topk_lazy(build_b1_lf(ev(), od()), top_k))
    gc.collect()
    b2_df = _collect(_topk_lazy(build_b2_lf(ev()), top_k))
    gc.collect()
    b3_df = _collect(_topk_lazy(build_b3_lf(ev()), top_k))
    gc.collect()

    result = rrf_merge(b1_df, b2_df, b3_df, top_k)
    del b1_df, b2_df, b3_df
    gc.collect()

    # Validate schema
    missing = set(OUTPUT_COLS) - set(result.columns)
    assert not missing, f"shard {shard_id}: missing columns {missing}"

    # Validate no duplicate (user_id, item_id)
    n_unique = result.select(["user_id", "item_id"]).unique().height
    assert n_unique == result.height, (
        f"shard {shard_id}: {result.height - n_unique} duplicate (user_id, item_id) pairs"
    )

    # Validate rank bound
    max_rank = result["repeat_history_rank"].max()
    assert max_rank is None or max_rank <= top_k, (
        f"shard {shard_id}: max repeat_history_rank={max_rank} > top_k={top_k}"
    )

    shard_dir.mkdir(parents=True, exist_ok=True)
    result.select(OUTPUT_COLS).write_parquet(out_file)
    n = result.height
    print(f"  shard {shard_id:3d}: {n:>7,} rows → user_shard={shard_id}/{SOURCE_NAME}.parquet")
    del result
    gc.collect()
    return n


# ── Audit ──────────────────────────────────────────────────────────────────────

def audit_output(out_root: Path, splits: list[str], num_shards: int) -> None:
    print("=== Audit: repeat_history candidate source ===")
    for sp in splits:
        split_dir = out_root / sp
        if not split_dir.exists():
            print(f"  {sp:15s}: NOT FOUND ({split_dir})")
            continue
        found, total = 0, 0
        for s in range(num_shards):
            f = split_dir / f"user_shard={s}" / f"{SOURCE_NAME}.parquet"
            if f.exists():
                n = pl.scan_parquet(str(f)).select(pl.len().alias("n")).collect()["n"].item()
                total += n
                found += 1
        print(f"  {sp:15s}: {found:3d}/{num_shards} shards   {total:>10,} rows")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build repeat_history candidate source (memory-safe, sharded).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--split",           choices=list(SPLIT_WINDOWS),
                    help="Which split to build (required unless --audit-only)")
    ap.add_argument("--num-user-shards", type=int, default=128, metavar="N",
                    help="Total shards (default 128)")
    ap.add_argument("--user-shard",      type=int, default=None, metavar="S",
                    help="Process a single shard only (smoke test)")
    ap.add_argument("--top-k",           type=int, default=200, metavar="K",
                    help="Max candidates per user (default 200)")
    ap.add_argument("--overwrite",       action="store_true",
                    help="Overwrite existing output shards")
    ap.add_argument("--dry-run",         action="store_true",
                    help="Print what would be done without writing any files")
    ap.add_argument("--audit-only",      action="store_true",
                    help="List existing shards and row counts, then exit")
    args = ap.parse_args()

    root       = find_project_root()
    events_dir = root / "data" / "datamart" / "agg" / "events" / "user_item"
    orders_dir = root / "data" / "datamart" / "agg" / "orders" / "user_item"
    out_root   = root / "data" / "datamart" / "candidates" / SOURCE_NAME

    if args.audit_only:
        audit_splits = [args.split] if args.split else list(SPLIT_WINDOWS)
        audit_output(out_root, audit_splits, args.num_user_shards)
        return 0

    if args.split is None:
        ap.error("--split is required (unless --audit-only)")

    win = SPLIT_WINDOWS[args.split]
    shards: list[int] | range = (
        [args.user_shard] if args.user_shard is not None else range(args.num_user_shards)
    )
    n_shards_to_run = 1 if args.user_shard is not None else args.num_user_shards

    print(f"split={args.split}  shards={n_shards_to_run}  top_k={args.top_k}")
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"context: {win['context_start']} -> {win['context_end']}")
    print(f"output:  {out_root / args.split}")
    if args.dry_run:
        print("(DRY-RUN — no files will be written)")

    total = 0
    for sid in shards:
        total += process_shard(
            args.split, sid, args.top_k, args.num_user_shards,
            out_root, events_dir, orders_dir,
            args.overwrite, args.dry_run,
        )

    if not args.dry_run:
        print(f"\nDone. Total rows written: {total:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
