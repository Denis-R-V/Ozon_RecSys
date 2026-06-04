#!/usr/bin/env python3
"""
scripts/audit_reranker_integrity.py

P0 диагностика: schema, кандидаты, GT, recall/hit_rate для val/test.

Использование:
    python scripts/audit_reranker_integrity.py --split val --k 100,200
    python scripts/audit_reranker_integrity.py --split both --k 100,700
    python scripts/audit_reranker_integrity.py --split test --sample-users 5000
    python scripts/audit_reranker_integrity.py --split both --k 100 --source-recall
    python scripts/audit_reranker_integrity.py --split val --check-duplicates
    python scripts/audit_reranker_integrity.py --split val --check-duplicates --duplicate-buckets 128

Почему memory-safe:
    - Schema читается через .schema (нулевые данные).
    - Статистика кандидатов: group_by user_id → ~350k строк.
    - GT: collect ~1.4M строк однократно (~22 MB).
    - Recall: inner join candidates ↔ GT → результат ≤ |GT| ≤ 1.4M строк.
    - Dedup hits: group_by(user_id, item_id) → min(rank) перед collect.
    - Дубли: проверяются по бакетам (user_id % N) — 1/N данных за раз.
    - Все тяжёлые scans используют streaming=True.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import polars as pl

# ── paths & constants ─────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parents[1]
DATAMART_DIR = ROOT / "data" / "datamart"
CAND_DIR     = DATAMART_DIR / "candidates" / "merged"
ORDERS_DIR   = DATAMART_DIR / "agg" / "orders" / "user_item"

SPLIT_DATES: dict[str, tuple[date, date]] = {
    "ranker_train": (date(2025, 6, 4),  date(2025, 6, 17)),
    "val":          (date(2025, 6, 18), date(2025, 7, 1)),
    "test":         (date(2025, 7, 2),  date(2025, 7, 15)),
}

# Candidate rank column name candidates — first found wins
RANK_COL_CANDIDATES = ["candidate_rank", "rank", "merged_rank", "final_rank"]

# Date partition column candidates for agg/orders — first found wins
DATE_COL_CANDIDATES = ["date", "created_date", "dt"]

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit candidate/GT integrity per split.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--split", default="both", choices=["val", "test", "ranker_train", "both"],
        help="Which split(s) to audit. 'both' means val + test.",
    )
    p.add_argument(
        "--k", default="100,200",
        help="Comma-separated K values for recall@K (default: 100,200).",
    )
    p.add_argument(
        "--sample-users", type=int, default=None, metavar="N",
        help="Randomly sample N GT users (faster diagnostic, no full candidate scan).",
    )
    p.add_argument(
        "--source-recall", action="store_true",
        help="Compute per-source recall@max(K). Off by default (extra scans per source).",
    )
    p.add_argument(
        "--check-duplicates", action="store_true",
        help="Check for duplicate (user_id, item_id) pairs in candidates. Off by default.",
    )
    p.add_argument(
        "--duplicate-buckets", type=int, default=256, metavar="N",
        help="Number of buckets for memory-safe duplicate check (default: 256).",
    )
    p.add_argument(
        "--quick", action="store_true",
        help=(
            "Quick mode: build GT first, sample users (default 5000 if --sample-users "
            "not set), then scan candidates only for those users. "
            "Skips duplicate check even if --check-duplicates is given."
        ),
    )
    return p.parse_args()

# ── path helpers ──────────────────────────────────────────────────────────────

def check_path(path: Path, description: str) -> None:
    """Raise FileNotFoundError with helpful context if path is missing."""
    if path.exists():
        return
    parent = path.parent
    siblings = sorted(parent.iterdir()) if parent.exists() else []
    msg = f"\nPath not found: {path}\n  ({description})"
    if siblings:
        names = [p.name for p in siblings[:12]]
        msg += f"\n  Found in {parent}:\n    " + "\n    ".join(names)
        if len(siblings) > 12:
            msg += f"\n    ... and {len(siblings) - 12} more"
    else:
        msg += f"\n  Parent directory does not exist either: {parent}"
    raise FileNotFoundError(msg)


def validate_paths(split: str) -> None:
    check_path(ORDERS_DIR, "agg/orders/user_item — pre-aggregated order data")
    check_path(CAND_DIR / split, f"candidates/merged/{split} — merged candidates for split")

# ── scan helpers ──────────────────────────────────────────────────────────────

def scan_candidates(split: str) -> pl.LazyFrame:
    return pl.scan_parquet(str(CAND_DIR / split / "**/*.parquet"))


def _detect_date_col(schema: pl.Schema) -> str:
    """
    Find the date partition column in the orders agg schema.
    Reads only parquet footer metadata — zero rows loaded.
    """
    for col in DATE_COL_CANDIDATES:
        if col in schema:
            return col
    raise ValueError(
        f"No date column found in orders agg schema.\n"
        f"Tried: {DATE_COL_CANDIDATES}\n"
        f"Available columns: {list(schema.keys())}"
    )


def scan_orders_lf(start: date, end: date) -> pl.LazyFrame:
    """Unique (user_id Int64, item_id Int64) with ≥1 order in [start, end]."""
    lf       = pl.scan_parquet(str(ORDERS_DIR / "**/*.parquet"), hive_partitioning=True)
    date_col = _detect_date_col(lf.schema)
    return (
        lf
        .filter(
            (pl.col(date_col).cast(pl.Date, strict=False) >= pl.lit(start))
            & (pl.col(date_col).cast(pl.Date, strict=False) <= pl.lit(end))
            & (pl.col("cnt_orders").fill_null(0) > 0)
        )
        .select(
            pl.col("user_id").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
        )
        .unique()
    )

# ── schema helpers ────────────────────────────────────────────────────────────

def detect_rank_col(schema: pl.Schema, split: str) -> str:
    """Find the first known rank column in schema; raise with full schema on failure."""
    for col in RANK_COL_CANDIDATES:
        if col in schema:
            return col
    raise ValueError(
        f"No rank column found in '{split}' candidates.\n"
        f"Tried: {RANK_COL_CANDIDATES}\n"
        f"Available columns: {list(schema.keys())}"
    )


def is_source_flag_true(col: str) -> pl.Expr:
    """
    Boolean expression: True if source flag is set.
    Handles bool, int (0/1), and null values uniformly.
    """
    return (
        pl.when(pl.col(col).is_null())
        .then(pl.lit(False))
        .otherwise(pl.col(col).cast(pl.Boolean))
    )

# ── display ───────────────────────────────────────────────────────────────────

def sep(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

# ── report sections ───────────────────────────────────────────────────────────

def report_schema(split: str) -> tuple[str, list[str]]:
    """
    Zero data loaded — reads only parquet footer metadata.
    Returns (rank_col, src_flags).
    """
    schema = scan_candidates(split).schema
    rank_col  = detect_rank_col(schema, split)
    src_flags = [c for c in schema if c.startswith("in_")]

    sep(f"Schema — {split}")
    print(f"  user_id dtype    : {schema.get('user_id')}")
    print(f"  item_id dtype    : {schema.get('item_id')}")
    print(f"  rank column      : '{rank_col}'  (dtype: {schema.get(rank_col)})")
    print(f"  source flags     : {src_flags}")
    print(f"  total columns    : {len(schema)}")
    return rank_col, src_flags


def report_candidate_stats(
    split: str,
    rank_col: str,
    sampled_ids: pl.Series | None = None,
) -> pl.DataFrame:
    """
    group_by user_id → ~350k строк.
    242M исходных строк обрабатывает streaming engine; в Python не попадают.
    В режиме --quick: фильтрует candidates до sampled_ids перед scan.
    """
    lf = scan_candidates(split).select("user_id", rank_col)
    if sampled_ids is not None:
        lf = lf.filter(pl.col("user_id").is_in(sampled_ids))
    per_user = (
        lf
        .group_by("user_id")
        .agg(pl.len().alias("n_cands"))
        .collect(streaming=True)
    )
    sep(f"Candidates — {split}")
    print(f"  Total rows        : {per_user['n_cands'].sum():,}")
    print(f"  Unique users      : {len(per_user):,}")
    print(f"  Avg cands/user    : {per_user['n_cands'].mean():.1f}")
    print(f"  Median cands/user : {per_user['n_cands'].median():.1f}")
    print(f"  P95 cands/user    : {per_user['n_cands'].quantile(0.95):.0f}")
    print(f"  Max cands/user    : {per_user['n_cands'].max()}")
    return per_user


def report_duplicates_bucketed(split: str, num_buckets: int) -> None:
    """
    Bucketed duplicate check: каждый bucket = user_id % num_buckets == b.
    Peak memory per bucket ≈ 242M / num_buckets строк (streaming).
    Результат каждого bucket — только строки с cnt > 1 (обычно 0).
    """
    sep(f"Duplicate check — {split}  (buckets={num_buckets})")
    total_dups = 0

    for b in range(num_buckets):
        if b % max(1, num_buckets // 8) == 0:
            print(f"  bucket {b}/{num_buckets}...", end="\r", flush=True)

        n_dups = (
            scan_candidates(split)
            .filter(pl.col("user_id") % num_buckets == b)
            .select("user_id", "item_id")
            .group_by("user_id", "item_id")
            .agg(pl.len().alias("cnt"))
            .filter(pl.col("cnt") > 1)
            .select(pl.len())
            .collect(streaming=True)
            .item()
        )
        total_dups += n_dups

    print(f"  {'OK: 0 duplicate (user_id, item_id) pairs' if total_dups == 0 else f'WARNING: {total_dups:,} duplicates found!':<55}")


def report_gt_stats(split: str, gt_df: pl.DataFrame) -> pl.DataFrame:
    """
    gt_df уже собран (~1.4M строк).
    group_by → ~350k строк; collect OK.
    """
    gt_size = (
        gt_df.lazy()
        .group_by("user_id")
        .agg(pl.len().alias("gt_size"))
        .collect()
    )
    sizes = gt_size["gt_size"]

    sep(f"Ground Truth — {split}")
    print(f"  Users with GT     : {len(gt_size):,}")
    print(f"  Total GT items    : {sizes.sum():,}")
    print(f"  Avg GT size/user  : {sizes.mean():.3f}")
    print(f"  Median GT size    : {sizes.median():.1f}")
    print(f"  P95 GT size       : {sizes.quantile(0.95):.0f}")
    print(f"  P99 GT size       : {sizes.quantile(0.99):.0f}")
    print(f"  Max GT size       : {sizes.max()}")
    print("  GT size distribution (cumulative %):")
    for v in [1, 2, 3, 5, 10, 20]:
        pct = float((sizes <= v).mean()) * 100
        print(f"    ≤ {v:>2}: {pct:5.1f}%")
    return gt_size


def report_coverage(
    split: str,
    cand_per_user: pl.DataFrame,
    gt_size: pl.DataFrame,
) -> None:
    """
    Два ~350k-строчных DF. Все сравнения — через Polars join, без Python-множеств.
    """
    overlap = (
        gt_size.lazy()
        .join(cand_per_user.lazy().select("user_id"), on="user_id", how="inner")
        .select(pl.len())
        .collect()
        .item()
    )
    gt_only = (
        gt_size.lazy()
        .join(cand_per_user.lazy().select("user_id"), on="user_id", how="anti")
        .select(pl.len())
        .collect()
        .item()
    )
    cand_only = (
        cand_per_user.lazy()
        .join(gt_size.lazy().select("user_id"), on="user_id", how="anti")
        .select(pl.len())
        .collect()
        .item()
    )
    sep(f"User Coverage — {split}")
    print(f"  GT users          : {len(gt_size):,}")
    print(f"  Candidate users   : {len(cand_per_user):,}")
    print(f"  GT ∩ candidates   : {overlap:,}")
    print(f"  GT only (no cand) : {gt_only:,}")
    print(f"  Cand only (no GT) : {cand_only:,}")


def report_recall(
    split: str,
    gt_df: pl.DataFrame,
    gt_size: pl.DataFrame,
    rank_col: str,
    ks: list[int],
    sample_users: int | None,
) -> None:
    """
    Memory safety:
      - gt_df      : ~1.4M строк (уже в памяти, ~22 MB)
      - top_k_lf   : scan 242M → streaming filter → n_users * max_K строк
      - inner join : результат ≤ |GT| ≤ 1.4M строк (только hits)
      - dedup hits : group_by(user_id, item_id) → min(rank) → ≤ 1.4M строк
      - hits_df    : collect ≤ 1.4M строк
      - per_user   : ~350k строк

    Macro recall = mean_users(hits_i / gt_size_i)
    Micro recall = sum_users(hits_i) / sum_users(gt_size_i)
    Hit_rate     = mean_users(hits_i > 0)
    """
    if sample_users is not None:
        sampled = gt_size.sample(min(sample_users, len(gt_size)), seed=42)
        sampled_ids = sampled["user_id"]
        gt_df   = gt_df.filter(pl.col("user_id").is_in(sampled_ids))
        gt_size = sampled
        print(f"  [sampling] {len(gt_size):,} GT users")

    max_k = max(ks)

    # Filter candidates to max_k once — streaming, not materialized in Python
    top_k_lf = (
        scan_candidates(split)
        .filter(pl.col(rank_col) <= max_k)
        .select(
            pl.col("user_id").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
            pl.col(rank_col).alias("candidate_rank"),
        )
    )
    if sample_users is not None:
        top_k_lf = top_k_lf.filter(pl.col("user_id").is_in(sampled_ids))

    # Inner join: only candidate rows that hit GT items.
    # Peak: GT hash table in memory (~22 MB) + streaming candidates.
    # Result: ≤ |GT| ≤ 1.4M rows (cannot exceed GT size by definition).
    #
    # Dedup: a GT item may appear in candidates from multiple sources.
    # Take min rank per (user_id, item_id) so each GT item counts at most once.
    hits_df = (
        gt_df.lazy()
        .select("user_id", "item_id")
        .join(top_k_lf, on=["user_id", "item_id"], how="inner")
        .group_by(["user_id", "item_id"])
        .agg(pl.col("candidate_rank").min())
        .collect(streaming=True)        # peak: ≤ 1.4M rows
    )

    sep(f"Recall — {split}")
    print(f"  Total unique hits (@ max_k={max_k}): {len(hits_df):,}")

    for k in sorted(ks):
        # Work from already-collected small hits_df — no re-scan of candidates
        hits_k = (
            hits_df.lazy()
            .filter(pl.col("candidate_rank") <= k)
            .group_by("user_id")
            .agg(pl.len().alias("hits"))
            .collect()
        )

        # Left join: all GT users present, zero hits filled for those with no match
        per_user = (
            gt_size
            .join(hits_k, on="user_id", how="left")
            .with_columns(pl.col("hits").fill_null(0))
            .with_columns(
                (pl.col("hits") / pl.col("gt_size")).alias("recall_per_user"),
                (pl.col("hits") > 0).cast(pl.Float64).alias("hit_flag"),
            )
        )

        macro_recall = float(per_user["recall_per_user"].mean())
        micro_recall = float(per_user["hits"].sum() / per_user["gt_size"].sum())
        hit_rate     = float(per_user["hit_flag"].mean())
        n_with_hits  = int((per_user["hits"] > 0).sum())
        n_users      = len(per_user)

        print(f"\n  @{k}:")
        print(f"    macro_recall  = {macro_recall:.6f}")
        print(f"    micro_recall  = {micro_recall:.6f}")
        print(f"    hit_rate      = {hit_rate:.6f}")
        print(f"    users w/ hits = {n_with_hits:,} / {n_users:,}")

        # Breakdown by GT size: locates which users pull recall down
        print(f"    GT-size breakdown (macro/micro/hit_rate per bucket):")
        buckets = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 999)]
        for lo, hi in buckets:
            filt = pl.col("gt_size").is_between(lo, hi)
            bucket = per_user.filter(filt)
            if len(bucket) == 0:
                continue
            b_macro = float(bucket["recall_per_user"].mean())
            b_micro = float(bucket["hits"].sum() / bucket["gt_size"].sum())
            b_hr    = float(bucket["hit_flag"].mean())
            b_n     = len(bucket)
            b_pct   = b_n / n_users * 100
            label   = f"{lo}" if lo == hi else (f"{lo}–{hi}" if hi < 999 else f"{lo}+")
            print(
                f"      gt={label:<5}: {b_n:>6,} users ({b_pct:4.1f}%) | "
                f"macro={b_macro:.4f}  micro={b_micro:.4f}  hr={b_hr:.4f}"
            )


def report_source_recall(
    split: str,
    gt_df: pl.DataFrame,
    gt_size: pl.DataFrame,
    rank_col: str,
    src_flags: list[str],
    max_k: int,
) -> None:
    """
    Per-source: один inner join на источник.
    Каждый результат ≤ |GT| ≤ 1.4M строк.
    Включается только через --source-recall.
    """
    sep(f"Source Recall @{max_k} — {split}")
    if not src_flags:
        print("  No in_* columns found in candidates schema.")
        return

    for src in src_flags:
        src_lf = (
            scan_candidates(split)
            .filter(
                (pl.col(rank_col) <= max_k)
                & is_source_flag_true(src)
            )
            .select(
                pl.col("user_id").cast(pl.Int64),
                pl.col("item_id").cast(pl.Int64),
            )
        )
        hits = (
            gt_df.lazy()
            .select("user_id", "item_id")
            .join(src_lf, on=["user_id", "item_id"], how="inner")
            .group_by("user_id")
            .agg(pl.len().alias("hits"))
            .collect(streaming=True)
        )
        per_user = (
            gt_size
            .join(hits, on="user_id", how="left")
            .with_columns(pl.col("hits").fill_null(0))
        )
        macro = float((per_user["hits"] / per_user["gt_size"]).mean())
        micro = float(per_user["hits"].sum() / per_user["gt_size"].sum())
        hr    = float((per_user["hits"] > 0).cast(pl.Float64).mean())
        n_src_rows = (
            scan_candidates(split)
            .filter(is_source_flag_true(src))
            .select(pl.len())
            .collect()
            .item()
        )
        print(
            f"  {src:<35}: macro={macro:.6f}  micro={micro:.6f}  "
            f"hr={hr:.6f}  rows={n_src_rows:,}"
        )

# ── orchestration ─────────────────────────────────────────────────────────────

def run_split(split: str, ks: list[int], args: argparse.Namespace) -> None:
    if split not in SPLIT_DATES:
        print(f"Unknown split {split!r}", file=sys.stderr)
        return

    start, end   = SPLIT_DATES[split]
    quick        = args.quick
    sample_users = args.sample_users
    if quick and sample_users is None:
        sample_users = 5000

    print(f"\n{'=' * 60}")
    print(f"SPLIT: {split}  |  GT window: {start} → {end}" + ("  [quick]" if quick else ""))
    print(f"{'=' * 60}")

    validate_paths(split)
    rank_col, src_flags = report_schema(split)

    if quick:
        # Build GT first so we know which users to sample; then filter everything.
        # GT full collect: ~1.4M rows, ~22 MB — always safe.
        gt_df_full   = scan_orders_lf(start, end).collect(streaming=True)
        gt_size_full = (
            gt_df_full.lazy()
            .group_by("user_id")
            .agg(pl.len().alias("gt_size"))
            .collect()
        )
        n_sample    = min(sample_users, len(gt_size_full))
        sampled     = gt_size_full.sample(n_sample, seed=42)
        sampled_ids = sampled["user_id"]
        gt_df       = gt_df_full.filter(pl.col("user_id").is_in(sampled_ids))
        gt_size     = sampled
        print(f"  [quick] sampled {n_sample:,} of {len(gt_size_full):,} GT users")
        del gt_df_full, gt_size_full

        cand_per_user = report_candidate_stats(split, rank_col, sampled_ids=sampled_ids)

        if args.check_duplicates:
            print("  [quick] skipping duplicate check (not supported in quick mode)")

        report_gt_stats(split, gt_df)
        report_coverage(split, cand_per_user, gt_size)
        # gt_df / gt_size already filtered — pass sample_users=None to avoid double-sampling
        report_recall(split, gt_df, gt_size, rank_col, ks, sample_users=None)
    else:
        cand_per_user = report_candidate_stats(split, rank_col)

        if args.check_duplicates:
            report_duplicates_bucketed(split, args.duplicate_buckets)

        # GT: собираем однократно (~1.4M строк, ~22 MB)
        gt_df   = scan_orders_lf(start, end).collect(streaming=True)
        gt_size = report_gt_stats(split, gt_df)

        report_coverage(split, cand_per_user, gt_size)
        report_recall(split, gt_df, gt_size, rank_col, ks, sample_users)

    if args.source_recall:
        report_source_recall(split, gt_df, gt_size, rank_col, src_flags, max(ks))


def main() -> None:
    args   = parse_args()
    ks     = [int(x.strip()) for x in args.k.split(",")]
    splits = ["val", "test"] if args.split == "both" else [args.split]

    for split in splits:
        try:
            run_split(split, ks, args)
        except FileNotFoundError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        except ValueError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
