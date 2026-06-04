#!/usr/bin/env python3
"""Audit old/repeat/new candidate pools for v12 reranker.

GT built ONLY from agg/orders/user_item target window — never from labels.
Memory-safe: one user_shard at a time.
Old pool (merged_with_dl_all) uses 128 shards; new/repeat use 512.

Usage:
    # Smoke test (1 shard):
    python scripts/audit_repeat_candidate_pool.py --split val --user-shard 0

    # Full val audit:
    python scripts/audit_repeat_candidate_pool.py --split val
"""
from __future__ import annotations

import argparse
import gc
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

# ── Config ─────────────────────────────────────────────────────────────────────
SPLITS = {
    "ranker_train": {"target_start": date(2025, 6,  4), "target_end": date(2025, 6, 17)},
    "val":          {"target_start": date(2025, 6, 18), "target_end": date(2025, 7,  1)},
}
METRIC_KS        = (20, 50, 100, 200)
OLD_SOURCE       = "merged_with_dl_all"
REPEAT_SOURCE    = "repeat_history"
NEW_SOURCE       = "merged_with_dl_all_repeat_history"
OLD_SHARDS_DFLT  = 128


# ── Utilities ──────────────────────────────────────────────────────────────────

def find_project_root(start: Path = Path.cwd()) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "datamart").exists():
            return p
    raise FileNotFoundError("data/datamart not found")


def detect_num_shards(split_dir: Path) -> int:
    if not split_dir.exists():
        return 0
    return sum(1 for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("user_shard="))


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except Exception:
        return lf.collect()


def _date_filter(lf: pl.LazyFrame, col: str, start: date, end: date) -> pl.LazyFrame:
    return lf.filter(
        (pl.col(col).cast(pl.Date, strict=False) >= pl.lit(start).cast(pl.Date))
        & (pl.col(col).cast(pl.Date, strict=False) <= pl.lit(end).cast(pl.Date))
    )


# ── GT (built from orders, never labels) ──────────────────────────────────────

def build_gt(agg_dir: Path, target_start: date, target_end: date) -> dict[int, list[int]]:
    """Ground truth: all purchased user-item pairs in the target window."""
    orders_dir = agg_dir / "orders" / "user_item"
    lf = _date_filter(
        pl.scan_parquet(str(orders_dir / "**" / "*.parquet"), hive_partitioning=True),
        "created_date", target_start, target_end,
    )
    df = _collect(
        lf.select(pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64))
        .drop_nulls().unique()
        .group_by("user_id").agg(pl.col("item_id").sort().alias("items"))
    )
    gt = {int(r["user_id"]): [int(x) for x in r["items"]] for r in df.iter_rows(named=True)}
    n_items = sum(len(v) for v in gt.values())
    print(f"  GT: {len(gt):,} users  {n_items:,} items")
    return gt


# ── Pool readers ───────────────────────────────────────────────────────────────

def _read_shard(shard_dir: Path) -> pl.DataFrame:
    parquets = list(shard_dir.glob("*.parquet")) if shard_dir.exists() else []
    if not parquets:
        return pl.DataFrame()
    return pl.scan_parquet(str(shard_dir / "*.parquet")).collect()


def read_old(old_split_dir: Path, shard_id: int, num_out: int, old_shards: int) -> pl.DataFrame:
    old_sid = shard_id % old_shards
    shard_dir = old_split_dir / f"user_shard={old_sid}"
    parquets = list(shard_dir.glob("*.parquet")) if shard_dir.exists() else []
    if not parquets:
        return pl.DataFrame()
    return _collect(
        pl.scan_parquet(str(shard_dir / "*.parquet"))
        .filter((pl.col("user_id").cast(pl.Int64) % num_out) == shard_id)
    )


def read_exact(split_dir: Path, shard_id: int) -> pl.DataFrame:
    return _read_shard(split_dir / f"user_shard={shard_id}")


# ── Basic stats ────────────────────────────────────────────────────────────────

def _basic(df: pl.DataFrame, gt_shard: dict) -> dict:
    if df.height == 0:
        return {"n_rows": 0, "n_users": 0, "n_dups": 0, "cand_counts": [],
                "gt_hit_users": 0, "gt_hit_items": 0}

    n_unique  = df.select(["user_id", "item_id"]).unique().height
    n_dups    = df.height - n_unique
    per_user  = df.group_by("user_id").agg(pl.len().alias("n"))["n"].to_list()

    gt_hit_users = gt_hit_items = 0
    if gt_shard:
        gt_pairs = pl.DataFrame({
            "user_id": [u for u, its in gt_shard.items() for _ in its],
            "item_id": [i for u, its in gt_shard.items() for i in its],
        }, schema={"user_id": pl.Int64, "item_id": pl.Int64})
        hits = df.select(
            pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)
        ).join(gt_pairs, on=["user_id", "item_id"], how="inner")
        gt_hit_items = hits.height
        gt_hit_users = hits["user_id"].n_unique()

    return {
        "n_rows": df.height, "n_users": df["user_id"].n_unique(),
        "n_dups": n_dups, "cand_counts": per_user,
        "gt_hit_users": gt_hit_users, "gt_hit_items": gt_hit_items,
    }


# ── NDCG (repeat pool only, using repeat_history_rank) ────────────────────────

def _ndcg(df: pl.DataFrame, gt_shard: dict) -> dict:
    if df.height == 0 or not gt_shard or "repeat_history_rank" not in df.columns:
        return {f"ndcg_sum@{k}": 0.0 for k in METRIC_KS} | \
               {f"recall_sum@{k}": 0.0 for k in METRIC_KS} | {"hr100_sum": 0.0}

    max_k = max(METRIC_KS)
    log2  = np.array([1.0 / math.log2(i + 2) for i in range(max_k)])

    ndcg_s   = {k: 0.0 for k in METRIC_KS}
    recall_s = {k: 0.0 for k in METRIC_KS}
    hr100    = 0.0

    ranked = (
        df.lazy()
        .filter(
            pl.col("user_id").cast(pl.Int64).is_in(list(gt_shard.keys()))
            & pl.col("repeat_history_rank").is_not_null()
        )
        .sort(["user_id", "repeat_history_rank"])
        .group_by("user_id", maintain_order=True)
        .agg(pl.col("item_id").alias("items"))
        .collect()
    )

    for row in ranked.iter_rows(named=True):
        u = int(row["user_id"])
        if u not in gt_shard:
            continue
        gt_set = set(gt_shard[u])
        n_gt   = len(gt_set)
        items  = [int(x) for x in row["items"]]
        rel    = np.zeros(max_k, dtype=np.float32)
        for i, item in enumerate(items[:max_k]):
            if item in gt_set:
                rel[i] = 1.0
        for k in METRIC_KS:
            dcg   = float(np.dot(rel[:k], log2[:k]))
            ideal = float(np.sum(log2[:min(n_gt, k)]))
            ndcg_s[k]   += dcg / ideal if ideal > 0 else 0.0
            recall_s[k] += float(np.sum(rel[:k])) / n_gt
        if any(item in gt_set for item in items[:100]):
            hr100 += 1.0

    return {f"ndcg_sum@{k}": ndcg_s[k] for k in METRIC_KS} | \
           {f"recall_sum@{k}": recall_s[k] for k in METRIC_KS} | {"hr100_sum": hr100}


# ── New-pool breakdown (shard-level anti-join) ─────────────────────────────────

def _breakdown(new_df: pl.DataFrame, old_df: pl.DataFrame, gt_shard: dict) -> dict:
    empty = {"repeat_only_hit_u": 0, "repeat_only_hit_i": 0,
             "overlap_hit_u": 0,     "overlap_hit_i": 0,
             "exist_only_hit_u": 0,  "exist_only_hit_i": 0}
    if new_df.height == 0 or not gt_shard or "is_repeat_history_candidate" not in new_df.columns:
        return empty

    def _hits(sub: pl.DataFrame) -> tuple[int, int]:
        if sub.height == 0 or not gt_shard:
            return 0, 0
        gt_pairs = pl.DataFrame({
            "user_id": [u for u, its in gt_shard.items() for _ in its],
            "item_id": [i for u, its in gt_shard.items() for i in its],
        }, schema={"user_id": pl.Int64, "item_id": pl.Int64})
        h = sub.select(
            pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)
        ).join(gt_pairs, on=["user_id", "item_id"], how="inner")
        return h["user_id"].n_unique(), h.height

    repeat_rows = new_df.filter(pl.col("is_repeat_history_candidate"))
    exist_rows  = new_df.filter(~pl.col("is_repeat_history_candidate"))

    if old_df.height > 0:
        old_keys    = old_df.select(
            pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)
        )
        overlap     = repeat_rows.select(
            pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)
        ).join(old_keys, on=["user_id", "item_id"], how="semi")
        repeat_only = repeat_rows.select(
            pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)
        ).join(old_keys, on=["user_id", "item_id"], how="anti")
    else:
        overlap     = pl.DataFrame()
        repeat_only = repeat_rows.select(
            pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)
        )

    ro_u, ro_i = _hits(repeat_only)
    ov_u, ov_i = _hits(overlap)
    ex_u, ex_i = _hits(exist_rows)

    return {"repeat_only_hit_u": ro_u, "repeat_only_hit_i": ro_i,
            "overlap_hit_u": ov_u,     "overlap_hit_i": ov_i,
            "exist_only_hit_u": ex_u,  "exist_only_hit_i": ex_i}


# ── Accumulator helpers ────────────────────────────────────────────────────────

def _new_acc(pool: str, split: str) -> dict:
    return {
        "pool": pool, "split": split,
        "n_rows": 0, "n_users": 0, "n_dups": 0, "cand_counts": [],
        "gt_hit_users": 0, "gt_hit_items": 0,
        **{f"ndcg_sum@{k}": 0.0 for k in METRIC_KS},
        **{f"recall_sum@{k}": 0.0 for k in METRIC_KS},
        "hr100_sum": 0.0,
        "repeat_only_hit_u": 0, "repeat_only_hit_i": 0,
        "overlap_hit_u": 0,     "overlap_hit_i": 0,
        "exist_only_hit_u": 0,  "exist_only_hit_i": 0,
    }


def _merge_acc(acc: dict, basic: dict, extra: dict) -> None:
    for k, v in basic.items():
        if k == "cand_counts":
            acc["cand_counts"].extend(v)
        elif k in acc:
            acc[k] += v
    for k, v in extra.items():
        if k in acc:
            acc[k] += v


def _finalize(acc: dict, gt: dict) -> dict:
    n_gt_u = len(gt)
    n_gt_i = sum(len(v) for v in gt.values())
    c = np.array(acc["cand_counts"], dtype=np.float32) if acc["cand_counts"] else np.array([0.0])
    row = {
        "pool": acc["pool"], "split": acc["split"],
        "n_rows": acc["n_rows"], "n_users": acc["n_users"], "n_dups": acc["n_dups"],
        "avg_cands": round(float(c.mean()), 2),
        "p50_cands": round(float(np.percentile(c, 50)), 1),
        "p95_cands": round(float(np.percentile(c, 95)), 1),
        "gt_users": n_gt_u, "gt_items": n_gt_i,
        "coverage": round(acc["n_users"] / n_gt_u, 4) if n_gt_u else 0.0,
        "recall@all": round(acc["gt_hit_items"] / n_gt_i, 6) if n_gt_i else 0.0,
        "hit_rate@all": round(acc["gt_hit_users"] / n_gt_u, 6) if n_gt_u else 0.0,
    }
    for k in METRIC_KS:
        row[f"ndcg@{k}"]   = round(acc[f"ndcg_sum@{k}"]   / n_gt_u, 6) if n_gt_u else 0.0
        row[f"recall@{k}"] = round(acc[f"recall_sum@{k}"] / n_gt_u, 6) if n_gt_u else 0.0
    row["hit_rate@100"] = round(acc["hr100_sum"] / n_gt_u, 6) if n_gt_u else 0.0
    for k in ["repeat_only_hit_u", "repeat_only_hit_i",
              "overlap_hit_u", "overlap_hit_i",
              "exist_only_hit_u", "exist_only_hit_i"]:
        row[k] = acc[k]
    return row


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Audit old/repeat/new candidate pools.")
    ap.add_argument("--split",           choices=list(SPLITS), required=True)
    ap.add_argument("--num-user-shards", type=int, default=512, metavar="N")
    ap.add_argument("--user-shard",      type=int, default=None, metavar="S",
                    help="Single shard (smoke test)")
    args = ap.parse_args()

    root       = find_project_root()
    agg_dir    = root / "data" / "datamart" / "agg"
    cand_root  = root / "data" / "datamart" / "candidates"
    report_dir = root / "artifacts" / "reports" / "repeat_history_candidate_audit"
    report_dir.mkdir(parents=True, exist_ok=True)

    old_split_dir    = cand_root / OLD_SOURCE    / args.split
    repeat_split_dir = cand_root / REPEAT_SOURCE / args.split
    new_split_dir    = cand_root / NEW_SOURCE    / args.split

    old_shards = detect_num_shards(old_split_dir) or OLD_SHARDS_DFLT

    cfg = SPLITS[args.split]
    print(f"\nBuilding GT for {args.split} ({cfg['target_start']} -> {cfg['target_end']})...")
    gt = build_gt(agg_dir, cfg["target_start"], cfg["target_end"])

    n_gt_users = len(gt)
    n_gt_items = sum(len(v) for v in gt.values())
    if args.split == "val":
        assert n_gt_users > 300_000, f"val GT too small: {n_gt_users:,}"
        assert n_gt_items > 1_000_000, f"val GT items too few: {n_gt_items:,}"
    print(f"GT sanity OK: {n_gt_users:,} users  {n_gt_items:,} items")

    shards = [args.user_shard] if args.user_shard is not None else range(args.num_user_shards)
    n_run  = 1 if args.user_shard is not None else args.num_user_shards

    old_acc    = _new_acc("old",    args.split)
    repeat_acc = _new_acc("repeat", args.split)
    new_acc    = _new_acc("new",    args.split)

    print(f"Processing {n_run} shards (old_shards={old_shards})...")

    for shard_id in shards:
        gt_shard = {u: its for u, its in gt.items() if u % args.num_user_shards == shard_id}

        old_df    = read_old(old_split_dir, shard_id, args.num_user_shards, old_shards)
        repeat_df = read_exact(repeat_split_dir, shard_id)
        new_df    = read_exact(new_split_dir, shard_id)

        _merge_acc(old_acc,    _basic(old_df,    gt_shard), {})
        _merge_acc(repeat_acc, _basic(repeat_df, gt_shard), _ndcg(repeat_df, gt_shard))
        _merge_acc(new_acc,    _basic(new_df,    gt_shard), _breakdown(new_df, old_df, gt_shard))

        del old_df, repeat_df, new_df
        gc.collect()

        if shard_id % 64 == 0 or args.user_shard is not None:
            print(f"  shard {shard_id:3d}  gt_users_in_shard={len(gt_shard)}  "
                  f"new_rows_acc={new_acc['n_rows']:,}")

    rows = [_finalize(old_acc, gt), _finalize(repeat_acc, gt), _finalize(new_acc, gt)]

    print(f"\n{'='*70}\nAUDIT: {args.split}\n{'='*70}")
    for r in rows:
        pool = r["pool"]
        print(f"\n[{pool}]")
        print(f"  rows={r['n_rows']:,}  users={r['n_users']:,}  dups={r['n_dups']:,}")
        print(f"  avg/p50/p95 cands: {r['avg_cands']:.1f} / {r['p50_cands']:.0f} / {r['p95_cands']:.0f}")
        print(f"  coverage={r['coverage']:.2%}  recall@all={r['recall@all']:.4f}  hit_rate@all={r['hit_rate@all']:.4f}")
        if pool == "repeat":
            for k in METRIC_KS:
                print(f"  ndcg@{k:<4}={r[f'ndcg@{k}']:.6f}  recall@{k:<4}={r[f'recall@{k}']:.6f}")
            print(f"  hit_rate@100={r['hit_rate@100']:.6f}")
        if pool == "new":
            print(f"  repeat_only hits: u={r['repeat_only_hit_u']:,}  i={r['repeat_only_hit_i']:,}")
            print(f"  overlap     hits: u={r['overlap_hit_u']:,}  i={r['overlap_hit_i']:,}")
            print(f"  exist_only  hits: u={r['exist_only_hit_u']:,}  i={r['exist_only_hit_i']:,}")

    result_df = pl.DataFrame(rows)
    out_parquet = report_dir / f"repeat_candidate_pool_audit_{args.split}.parquet"
    out_csv     = report_dir / f"repeat_candidate_pool_audit_{args.split}.csv"
    result_df.write_parquet(out_parquet)
    result_df.write_csv(out_csv)
    print(f"\nSaved: {out_parquet}")
    print(f"Saved: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
