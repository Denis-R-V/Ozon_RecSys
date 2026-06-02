"""
Evaluate the selection model (same as used for val NDCG=0.087751) on the test dataset.
This gives an apples-to-apples comparison with val metrics.

Val used: best_selection_record (trained on train_fit, 6M rows)
Test here: same model, evaluated on test_dataset_dir

Usage:
    python scripts/eval_test_selection.py
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_NAME     = "final_reranker_v12_repeat__merged_with_dl_all_repeat_history"
REPORTS_DIR  = PROJECT_ROOT / "artifacts" / "reports" / RUN_NAME
MODELS_DIR   = PROJECT_ROOT / "models" / "reranker" / RUN_NAME
TEST_DS_DIR  = PROJECT_ROOT / "data" / "processed" / "reranker" / RUN_NAME / "dataset" / "test"
LABELS_PATH  = PROJECT_ROOT / "data" / "processed" / "reranker" / RUN_NAME / "labels" / "labels_test_2025-07-02_2025-07-15.parquet"

# ── Load selection model ───────────────────────────────────────────────────────
model_path = MODELS_DIR / "lgbm_ranker_t02_selection.joblib"
print(f"Loading selection model: {model_path.name} ({model_path.stat().st_size//1024} KB)")
model = joblib.load(model_path)
print(f"  type: {type(model).__name__}")

# ── Load feature columns ───────────────────────────────────────────────────────
fcols_raw = json.loads((MODELS_DIR / "feature_cols.json").read_text(encoding="utf-8"))
if isinstance(fcols_raw, dict):
    feature_cols = fcols_raw.get("numeric_feature_cols", [])
else:
    feature_cols = list(fcols_raw)
print(f"  features: {len(feature_cols)}")

# ── Load test ground truth ────────────────────────────────────────────────────
print(f"\nLoading test labels...")
ldf = pd.read_parquet(LABELS_PATH)
test_gt = {}
for row in ldf[ldf["label"] == 1][["user_id", "item_id"]].itertuples(index=False):
    test_gt.setdefault(int(row.user_id), []).append(int(row.item_id))
print(f"  GT users: {len(test_gt):,}, total purchases: {ldf['label'].sum():,}")

# ── Score test dataset shard by shard ─────────────────────────────────────────
print(f"\nScoring test dataset ({TEST_DS_DIR})...")
shards = sorted(TEST_DS_DIR.iterdir())
print(f"  shards: {len(shards)}")

# Accumulate top-k per user across shards
K = 100
user_top: dict[int, list[tuple[float, int]]] = {}  # uid -> [(score, item_id)]

t0 = time.perf_counter()
for i, shard in enumerate(shards):
    pqs = list(shard.glob("*.parquet"))
    if not pqs:
        continue
    df = pd.read_parquet(pqs[0])

    # Fill missing features
    X = df.copy()
    for f in feature_cols:
        if f not in X.columns:
            X[f] = 0.0

    scores = model.predict(X[feature_cols].values.astype(np.float32))

    for uid, grp_idx in df.groupby("user_id").groups.items():
        uid = int(uid)
        items = df.loc[grp_idx, "item_id"].astype(int).values
        sc    = scores[grp_idx]
        # Merge with existing top-k
        combined = list(zip(sc.tolist(), items.tolist()))
        if uid in user_top:
            combined.extend(user_top[uid])
        combined.sort(key=lambda x: x[0], reverse=True)
        user_top[uid] = combined[:K]

    if (i + 1) % 64 == 0 or i == len(shards) - 1:
        elapsed = time.perf_counter() - t0
        print(f"  shard {i+1}/{len(shards)}  users={len(user_top):,}  {elapsed:.0f}s")

print(f"\nScoring complete. Scored users: {len(user_top):,}")

# ── Compute NDCG@k and recall@k ───────────────────────────────────────────────
print("\nComputing metrics...")

def dcg(hits, k):
    return sum(h / np.log2(i + 2) for i, h in enumerate(hits[:k]))

results = []
for k_eval in [10, 20, 50, 100]:
    ndcg_list, recall_list, hr_list = [], [], []
    for uid, top_items in user_top.items():
        gt_set = set(test_gt.get(uid, []))
        if not gt_set:
            continue
        pred = [iid for _, iid in top_items[:k_eval]]
        hits = [1.0 if iid in gt_set else 0.0 for iid in pred]
        ideal = [1.0] * min(len(gt_set), k_eval)
        idcg = dcg(ideal, k_eval) or 1.0
        ndcg_list.append(dcg(hits, k_eval) / idcg)
        recall_list.append(sum(hits) / len(gt_set))
        hr_list.append(1.0 if sum(hits) > 0 else 0.0)
    results.append({
        "model": "lgbm_ranker_t02_selection",
        "k": k_eval,
        "ndcg": np.mean(ndcg_list),
        "recall": np.mean(recall_list),
        "hit_rate": np.mean(hr_list),
        "n_users": len(ndcg_list),
    })

df_results = pd.DataFrame(results)
print("\n" + "=" * 65)
print("TEST METRICS — selection model (same as val evaluation)")
print("=" * 65)
print(df_results[["k", "ndcg", "recall", "hit_rate", "n_users"]].to_string(index=False))
print()
ndcg100 = df_results[df_results["k"] == 100]["ndcg"].values[0]
recall100 = df_results[df_results["k"] == 100]["recall"].values[0]
print(f"  Test NDCG@100   = {ndcg100:.6f}  (val: 0.087751)")
print(f"  Test recall@100 = {recall100:.6f}  (val: 0.145960)")
print("=" * 65)

# ── Save results ───────────────────────────────────────────────────────────────
out_path = REPORTS_DIR / "final_reranker_test_metrics_selection.parquet"
df_results.to_parquet(out_path, index=False)
print(f"\nSaved: {out_path.name}")
