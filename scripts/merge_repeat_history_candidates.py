#!/usr/bin/env python3
"""Merge repeat_history candidates into existing pool for v12 reranker.

Memory-safe: one user_shard at a time.
Repeat-history candidates take priority on (user_id, item_id) deduplication.

Shard mapping:
  merged_with_dl_all uses N_existing shards (auto-detected, typically 128).
  repeat_history uses N_output shards (--num-user-shards, typically 512).
  For output shard s: read repeat/user_shard=s + existing/user_shard=(s % N_existing)
  then filter existing to user_id % N_output == s.
  Works whenever N_output is divisible by N_existing.

Usage:
    # Smoke test:
    python scripts/merge_repeat_history_candidates.py --split ranker_train --user-shard 0

    # Full split:
    python scripts/merge_repeat_history_candidates.py --split ranker_train --overwrite

    # Audit:
    python scripts/merge_repeat_history_candidates.py --audit-only
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import polars as pl

EXISTING_SOURCE = "merged_with_dl_all"
REPEAT_SOURCE   = "repeat_history"
OUTPUT_SOURCE   = "merged_with_dl_all_repeat_history"

VALID_SPLITS = ["ranker_train", "val", "test"]

REPEAT_COLS = [
    "repeat_history_score", "repeat_history_rank",
    "repeat_recency_score", "repeat_recency_rank",
    "repeat_frequency_score", "repeat_frequency_rank",
    "repeat_cart_orders_score", "repeat_cart_orders_rank",
    "repeat_in_recency", "repeat_in_frequency", "repeat_in_cart_orders",
]


def find_project_root(start: Path = Path.cwd()) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "datamart").exists():
            return p
    raise FileNotFoundError("Cannot find project root with data/datamart/")


def detect_num_shards(split_dir: Path) -> int:
    if not split_dir.exists():
        return 0
    return sum(1 for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("user_shard="))


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except Exception:
        return lf.collect()


def _count_rows_lazy(split_dir: Path) -> int:
    parquets = list(split_dir.rglob("*.parquet"))
    if not parquets:
        return 0
    return (
        pl.scan_parquet(str(split_dir / "**" / "*.parquet"))
        .select(pl.len().alias("n"))
        .collect()["n"]
        .item()
    )


def process_shard(
    split: str,
    shard_id: int,
    num_output_shards: int,
    existing_dir: Path,
    repeat_dir: Path,
    out_root: Path,
    existing_num_shards: int,
    overwrite: bool,
    dry_run: bool,
) -> int:
    shard_dir = out_root / split / f"user_shard={shard_id}"
    out_file  = shard_dir / f"{OUTPUT_SOURCE}.parquet"

    if out_file.exists() and not overwrite:
        print(f"  shard {shard_id:3d}: SKIP (exists, use --overwrite)")
        return 0
    if dry_run:
        print(f"  shard {shard_id:3d}: DRY-RUN → {out_file}")
        return 0

    # ── Read repeat_history shard (exact match: same shard key) ──────────────
    repeat_shard_dir = repeat_dir / split / f"user_shard={shard_id}"
    repeat_parquets  = list(repeat_shard_dir.glob("*.parquet")) if repeat_shard_dir.exists() else []

    if repeat_parquets:
        repeat_raw   = _collect(pl.scan_parquet(str(repeat_shard_dir / "*.parquet")))
        n_repeat_raw = repeat_raw.height
    else:
        repeat_raw   = pl.DataFrame()
        n_repeat_raw = 0

    # ── Read existing shard (mapped: s % N_existing), filter to this sub-shard ─
    existing_shard_id  = shard_id % existing_num_shards
    existing_shard_dir = existing_dir / split / f"user_shard={existing_shard_id}"
    existing_parquets  = list(existing_shard_dir.glob("*.parquet")) if existing_shard_dir.exists() else []

    if existing_parquets:
        existing_raw   = _collect(
            pl.scan_parquet(str(existing_shard_dir / "*.parquet"))
            .filter((pl.col("user_id").cast(pl.Int64) % num_output_shards) == shard_id)
        )
        n_existing_raw = existing_raw.height
    else:
        existing_raw   = pl.DataFrame()
        n_existing_raw = 0

    if n_existing_raw == 0:
        shard_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame().write_parquet(out_file)
        print(f"  shard {shard_id:3d}: 0 rows (no existing users)")
        return 0

    # ── Filter repeat to allowed users (existing pool defines the split universe) ─
    if repeat_raw.height > 0:
        allowed_users = existing_raw.select(pl.col("user_id").cast(pl.Int64)).unique()
        repeat_raw = repeat_raw.join(
            allowed_users, left_on=pl.col("user_id").cast(pl.Int64), right_on="user_id", how="semi"
        )
        n_repeat_raw = repeat_raw.height

    if n_repeat_raw == 0 and n_existing_raw == 0:
        shard_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame().write_parquet(out_file)
        print(f"  shard {shard_id:3d}: 0 rows (no input)")
        return 0

    # ── Feature-preserving 3-way merge ───────────────────────────────────────
    # Part A: items in BOTH existing and repeat  → existing base + repeat cols overlaid
    # Part B: items ONLY in repeat               → new candidates, repeat cols as score/rank
    # Part C: items ONLY in existing             → unchanged, repeat cols null
    # No information lost: existing source flags/scores are preserved for overlapping items.

    repeat_slim = (
        repeat_raw.select(["user_id", "item_id"] + REPEAT_COLS + ["repeat_source"])
        if repeat_raw.height > 0 else pl.DataFrame()
    )

    if repeat_raw.height > 0 and existing_raw.height > 0:
        part_a = (
            existing_raw
            .join(repeat_slim, on=["user_id", "item_id"], how="inner")
            .with_columns(pl.lit(True).alias("is_repeat_history_candidate"))
        )
        repeat_only_keys = (
            repeat_raw.select(["user_id", "item_id"])
            .join(existing_raw.select(["user_id", "item_id"]), on=["user_id", "item_id"], how="anti")
        )
        part_b = (
            repeat_slim
            .join(repeat_only_keys, on=["user_id", "item_id"], how="semi")
            .with_columns([
                pl.lit(True).alias("is_repeat_history_candidate"),
                pl.col("repeat_history_score").alias("score"),
                pl.col("repeat_history_rank").alias("rank"),
                pl.col("repeat_source").alias("source"),
            ])
        )
        part_c = (
            existing_raw
            .join(repeat_raw.select(["user_id", "item_id"]), on=["user_id", "item_id"], how="anti")
            .with_columns(pl.lit(False).alias("is_repeat_history_candidate"))
        )
        n_overlap     = part_a.height
        n_repeat_only = part_b.height
        n_exist_only  = part_c.height
    elif repeat_raw.height > 0:
        part_a, part_c = pl.DataFrame(), pl.DataFrame()
        part_b = repeat_slim.with_columns([
            pl.lit(True).alias("is_repeat_history_candidate"),
            pl.col("repeat_history_score").alias("score"),
            pl.col("repeat_history_rank").alias("rank"),
            pl.col("repeat_source").alias("source"),
        ])
        n_overlap, n_repeat_only, n_exist_only = 0, part_b.height, 0
    else:
        part_a, part_b = pl.DataFrame(), pl.DataFrame()
        part_c = existing_raw.with_columns(pl.lit(False).alias("is_repeat_history_candidate"))
        n_overlap, n_repeat_only, n_exist_only = 0, 0, part_c.height

    del repeat_raw, existing_raw, repeat_slim
    gc.collect()

    result = pl.concat(
        [p for p in [part_a, part_b, part_c] if p.height > 0],
        how="diagonal_relaxed",
    )
    del part_a, part_b, part_c
    gc.collect()

    # ── Validate ──────────────────────────────────────────────────────────────
    n      = result.height
    n_users = result["user_id"].n_unique()
    n_unique_pairs = result.select(["user_id", "item_id"]).unique().height
    assert n_unique_pairs == n, (
        f"shard {shard_id}: {n - n_unique_pairs} duplicate (user_id, item_id) pairs"
    )

    shard_dir.mkdir(parents=True, exist_ok=True)
    result.write_parquet(out_file)
    print(
        f"  shard {shard_id:3d}: {n:>7,} rows  {n_users:>5,} users  "
        f"overlap={n_overlap:,}  repeat_only={n_repeat_only:,}  exist_only={n_exist_only:,}"
    )
    del result
    gc.collect()
    return n


def audit_output(
    out_root: Path,
    repeat_dir: Path,
    existing_dir: Path,
    splits: list[str],
    num_shards: int,
) -> None:
    print("=== Audit: candidate source overview ===")
    for sp in splits:
        print(f"\n  split: {sp}")
        for label, d in [
            ("existing", existing_dir),
            ("repeat",   repeat_dir),
            ("output",   out_root),
        ]:
            sp_dir     = d / sp
            n_shards   = detect_num_shards(sp_dir)
            total_rows = _count_rows_lazy(sp_dir) if sp_dir.exists() else 0
            print(f"    {label:8s}: {n_shards:3d} shards   {total_rows:>12,} rows   {sp_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge repeat_history into existing candidate pool (memory-safe).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--split",           choices=VALID_SPLITS,
                    help="Split to process (required unless --audit-only)")
    ap.add_argument("--num-user-shards", type=int, default=512, metavar="N",
                    help="Output shard count = repeat_history shard count (default 512)")
    ap.add_argument("--user-shard",      type=int, default=None, metavar="S",
                    help="Process a single shard (smoke test)")
    ap.add_argument("--overwrite",       action="store_true",
                    help="Overwrite existing output shards")
    ap.add_argument("--dry-run",         action="store_true",
                    help="Print plan without writing files")
    ap.add_argument("--audit-only",      action="store_true",
                    help="Show existing/repeat/output row counts and exit")
    args = ap.parse_args()

    root         = find_project_root()
    cand_root    = root / "data" / "datamart" / "candidates"
    existing_dir = cand_root / EXISTING_SOURCE
    repeat_dir   = cand_root / REPEAT_SOURCE
    out_root     = cand_root / OUTPUT_SOURCE

    if args.audit_only:
        audit_splits = [args.split] if args.split else VALID_SPLITS
        audit_output(out_root, repeat_dir, existing_dir, audit_splits, args.num_user_shards)
        return 0

    if args.split is None:
        ap.error("--split is required unless --audit-only")

    # Test split guard: disabled if repeat_history/test not built
    if args.split == "test":
        test_repeat_dir = repeat_dir / "test"
        if detect_num_shards(test_repeat_dir) == 0:
            print("ERROR: --split test is disabled — repeat_history/test not built.")
            print("  Build it first: python scripts/build_repeat_history_candidates.py --split test")
            return 1

    existing_num_shards = detect_num_shards(existing_dir / args.split)
    if existing_num_shards == 0:
        print(f"ERROR: no existing shards found at {existing_dir / args.split}")
        return 1

    if args.num_user_shards % existing_num_shards != 0:
        print(
            f"WARNING: num_user_shards={args.num_user_shards} is not divisible by "
            f"existing_num_shards={existing_num_shards}. "
            "Shard mapping will be approximate."
        )

    shards: list[int] | range = (
        [args.user_shard] if args.user_shard is not None else range(args.num_user_shards)
    )
    n_to_run = 1 if args.user_shard is not None else args.num_user_shards

    print(f"split={args.split}  output_shards={n_to_run}  existing_shards={existing_num_shards}")
    print(f"existing: {existing_dir / args.split}")
    print(f"repeat:   {repeat_dir   / args.split}")
    print(f"output:   {out_root     / args.split}")
    if args.dry_run:
        print("(DRY-RUN — no files will be written)")

    total = 0
    for sid in shards:
        total += process_shard(
            args.split, sid, args.num_user_shards,
            existing_dir, repeat_dir, out_root, existing_num_shards,
            args.overwrite, args.dry_run,
        )

    if not args.dry_run:
        print(f"\nDone. Total rows written: {total:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
