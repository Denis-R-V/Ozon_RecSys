"""
Build model_comparison.csv + model_comparison.png from existing parquet artifacts,
log them + flat metrics into the existing PRD run (no new run created).

Usage:
    python scripts/log_comparison_artifacts.py
"""
import os, sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_NAME     = "final_reranker_v12_repeat__merged_with_dl_all_repeat_history"
REPORTS_DIR  = PROJECT_ROOT / "artifacts" / "reports" / RUN_NAME
# Both PRD runs: 6e08071b = original (has lgbm_ranker_t02 + inference output)
#                6cd846f3 = new metric names run (prd_run_id.txt points here)
_id_from_file = (REPORTS_DIR / "prd_run_id.txt").read_text(encoding="utf-8").strip()
PRD_RUN_IDS = list(dict.fromkeys([
    "6e08071b6021407a8ed6c9dd28318a8b",   # original PRD run
    _id_from_file,                          # current prd_run_id.txt
]))
MLFLOW_URI   = "http://localhost:5000"

# ── MinIO / boto3 creds ────────────────────────────────────────────────────────
_env = PROJECT_ROOT / ".env"
if _env.exists():
    for _l in _env.read_text(encoding="utf-8").splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, _, v = _l.partition("=")
            os.environ[k.strip()] = v.strip().strip('"').strip("'")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID",       "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY",   "minioadmin")
os.environ.setdefault("MLFLOW_S3_IGNORE_TLS",    "true")

try:
    import boto3
    boto3.setup_default_session(
        aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name           = "us-east-1",
    )
    _s3 = boto3.client("s3",
        endpoint_url          = os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name           = "us-east-1",
    )
    print(f"MinIO OK: buckets={[b['Name'] for b in _s3.list_buckets().get('Buckets',[])]}")
    del _s3
except Exception as e:
    print(f"[WARN] MinIO: {e}")

# ── Build comparison table ─────────────────────────────────────────────────────
import pandas as pd
import math

bdf = pd.read_parquet(REPORTS_DIR / "final_reranker_baseline_val_metrics.parquet")
mdf = pd.read_parquet(REPORTS_DIR / "final_reranker_best_val_metrics.parquet")

def val(df, model, metric, k=100):
    row = df[(df["model"] == model) & (df["k"] == k)]
    return float(row[metric].iloc[0]) if len(row) > 0 else float("nan")

# Models to show (label, model key, source df)
ROWS = [
    ("TopPopular",               "top_popular_rank",   bdf),
    ("ALS",                      "als_rank",           bdf),
    ("Candidate raw rank",       "candidate_rank",     bdf),
    ("2-stage reranker (PRD)",   mdf["model"].iloc[0], mdf),
]

reranker_ndcg   = val(mdf, mdf["model"].iloc[0], "ndcg",   100)
reranker_recall = val(mdf, mdf["model"].iloc[0], "recall", 100)

records = []
for label, key, df in ROWS:
    n = val(df, key, "ndcg",   100)
    r = val(df, key, "recall", 100)
    records.append({"method": label, "ndcg_at_100": n, "recall_at_100": r})

comp = pd.DataFrame(records)

# Lift vs TopPopular
tp_ndcg = val(bdf, "top_popular_rank", "ndcg", 100)
best_bl  = max(val(df, key, "ndcg", 100) for _, key, df in ROWS[:-1])  # best baseline

comp["lift_vs_top_popular"] = comp["ndcg_at_100"].apply(
    lambda x: round(x / tp_ndcg, 2) if tp_ndcg > 0 and not math.isnan(x) else float("nan")
)
comp["lift_vs_best_baseline"] = comp["ndcg_at_100"].apply(
    lambda x: round(x / best_bl, 2) if best_bl > 0 and not math.isnan(x) else float("nan")
)

CSV_PATH = REPORTS_DIR / "model_comparison.csv"
comp.to_csv(CSV_PATH, index=False, float_format="%.6f")
print(f"Saved: {CSV_PATH.name}")
print(comp.to_string(index=False))

# ── Build PNG ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PNG_PATH = REPORTS_DIR / "model_comparison.png"

methods = comp["method"].tolist()
ndcg    = comp["ndcg_at_100"].tolist()
recall  = comp["recall_at_100"].tolist()

colors = ["#aec6e8", "#aec6e8", "#aec6e8", "#2171b5"]   # last = PRD

x   = np.arange(len(methods))
w   = 0.38
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("2-Stage RecSys — Val Eval Comparison  (same candidate pool)",
             fontsize=13, fontweight="bold")

for ax, vals, title, ylab in [
    (axes[0], ndcg,   "NDCG@100",   "NDCG@100"),
    (axes[1], recall, "Recall@100", "Recall@100"),
]:
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_ylim(0, max(v for v in vals if not math.isnan(v)) * 1.22)
    for bar, v in zip(bars, vals):
        if not math.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

# Lift annotation on last bar
lift_x = round(reranker_ndcg / tp_ndcg, 1)
axes[0].annotate(f"×{lift_x} vs TopPop",
    xy=(x[-1], reranker_ndcg), xytext=(x[-1] - 0.8, reranker_ndcg * 0.7),
    arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9, color="#1a4f8a")

plt.tight_layout()
fig.savefig(PNG_PATH, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {PNG_PATH.name}")

# ── Flat metrics dict ──────────────────────────────────────────────────────────
flat_metrics = {
    "top_popular_ndcg_at_100":          val(bdf, "top_popular_rank", "ndcg",   100),
    "top_popular_recall_at_100":        val(bdf, "top_popular_rank", "recall", 100),
    "als_ndcg_at_100":                  val(bdf, "als_rank",         "ndcg",   100),
    "als_recall_at_100":                val(bdf, "als_rank",         "recall", 100),
    "candidate_rank_ndcg_at_100":       val(bdf, "candidate_rank",   "ndcg",   100),
    "candidate_rank_recall_at_100":     val(bdf, "candidate_rank",   "recall", 100),
    "reranker_ndcg_at_100":             reranker_ndcg,
    "reranker_recall_at_100":           reranker_recall,
    "lift_vs_top_popular_x_ndcg_at_100":  round(reranker_ndcg / tp_ndcg, 4),
    "lift_vs_best_baseline_x_ndcg_at_100": round(reranker_ndcg / best_bl, 4),
}

print("\nFlat metrics to log:")
for k, v in flat_metrics.items():
    print(f"  {k:<45} {v:.6f}")

# ── Log into existing PRD run (no new run) ─────────────────────────────────────
import mlflow
mlflow.set_tracking_uri(MLFLOW_URI)

for PRD_RUN_ID in PRD_RUN_IDS:
    run_info = mlflow.get_run(PRD_RUN_ID)
    print(f"\nLogging into run: {PRD_RUN_ID[:8]}…  ({run_info.info.run_name})")
    with mlflow.start_run(run_id=PRD_RUN_ID):
        mlflow.log_metrics(flat_metrics)
        print("  flat metrics logged ✓")
        for path, art_path in [(CSV_PATH, "reports"), (PNG_PATH, "reports")]:
            mlflow.log_artifact(str(path), artifact_path=art_path)
            print(f"  logged: {art_path}/{path.name}")

print(f"\nDone. MLflow UI: {MLFLOW_URI}")
