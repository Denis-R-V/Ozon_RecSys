"""
Create one lightweight eval run per model so MLflow Compare view
shows proper side-by-side bar charts.

Runs created (no training artifacts, just metrics):
  eval/top_popular
  eval/als
  eval/candidate_rank
  eval/2stage_reranker_prd  ← tagged PRD=true

Usage:
    python scripts/log_eval_runs.py
"""
import os, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_NAME     = "final_reranker_v12_repeat__merged_with_dl_all_repeat_history"
REPORTS_DIR  = PROJECT_ROOT / "artifacts" / "reports" / RUN_NAME
MLFLOW_URI   = "http://localhost:5000"
EXPERIMENT   = "recsys_reranker"

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
except Exception as e:
    print(f"[WARN] boto3: {e}")

# ── Load metrics from parquet ──────────────────────────────────────────────────
import pandas as pd

bdf = pd.read_parquet(REPORTS_DIR / "final_reranker_baseline_val_metrics.parquet")
mdf = pd.read_parquet(REPORTS_DIR / "final_reranker_best_val_metrics.parquet")

def row(df, model_key, k=100):
    r = df[(df["model"] == model_key) & (df["k"] == k)]
    return r.iloc[0] if len(r) else None

def metrics_for(df, model_key):
    """Return flat metrics dict for all k values."""
    out = {}
    for k in [10, 20, 50, 100]:
        r = row(df, model_key, k)
        if r is None:
            continue
        for col in ("ndcg", "recall", "hit_rate", "precision"):
            if col in r and not pd.isna(r[col]):
                out[f"{col}_at_{k}"] = float(r[col])
    return out

# Reranker: use all k from mdf
def reranker_metrics():
    out = {}
    for _, r in mdf.iterrows():
        k = int(r["k"])
        for col in ("ndcg", "recall", "hit_rate", "precision"):
            if col in r and not pd.isna(r[col]):
                out[f"{col}_at_{k}"] = float(r[col])
    return out

# ── Model definitions ──────────────────────────────────────────────────────────
MODELS = [
    dict(
        run_name    = "eval/top_popular",
        description = "TopPopular — most purchased items, no personalisation",
        metrics     = metrics_for(bdf, "top_popular_rank"),
        tags        = {"eval_group": "prd_checkpoint", "model_type": "heuristic"},
    ),
    dict(
        run_name    = "eval/als",
        description = "ALS — matrix factorisation retrieval score",
        metrics     = metrics_for(bdf, "als_rank"),
        tags        = {"eval_group": "prd_checkpoint", "model_type": "als"},
    ),
    dict(
        run_name    = "eval/candidate_rank",
        description = "Candidate raw rank — pool order without reranking",
        metrics     = metrics_for(bdf, "candidate_rank"),
        tags        = {"eval_group": "prd_checkpoint", "model_type": "pool_rank"},
    ),
    dict(
        run_name    = "eval/2stage_reranker_prd",
        description = "2-stage reranker PRD — retrieval + LightGBM LambdaRank",
        metrics     = reranker_metrics(),
        tags        = {"eval_group": "prd_checkpoint", "model_type": "lgbm_reranker",
                       "prd": "true", "model_stage": "PRD"},
    ),
]

# ── Create runs ────────────────────────────────────────────────────────────────
import mlflow
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

# Check for already-existing eval runs (idempotent re-run)
existing = mlflow.search_runs(
    experiment_names=[EXPERIMENT],
    filter_string="tags.eval_group = 'prd_checkpoint'",
    max_results=20,
)
existing_names = set(existing["tags.mlflow.runName"].tolist()) if len(existing) > 0 else set()

created = []
for m in MODELS:
    rname = m["run_name"]
    if rname in existing_names:
        print(f"  skip (exists): {rname}")
        continue
    with mlflow.start_run(run_name=rname) as run:
        mlflow.set_tags({
            "mlflow.note.content": m["description"],
            **m["tags"],
        })
        mlflow.log_metrics(m["metrics"])
        rid = run.info.run_id
    created.append((rname, rid))
    ndcg = m["metrics"].get("ndcg_at_100", float("nan"))
    recall = m["metrics"].get("recall_at_100", float("nan"))
    print(f"  created: {rname:<35}  ndcg@100={ndcg:.6f}  recall@100={recall:.6f}  run_id={rid[:8]}")

print()
if created:
    print(f"Created {len(created)} eval runs.")
else:
    print("All eval runs already exist.")

print()
print("=" * 65)
print("HOW TO VIEW COMPARISON IN MLFLOW UI:")
print("=" * 65)
print(f"  1. Open: {MLFLOW_URI}")
print(f"  2. Experiment: {EXPERIMENT}")
print("  3. Filter: tags.eval_group = 'prd_checkpoint'")
print("  4. Select all 4 eval/* runs  →  Compare")
print("  5. Choose metric: ndcg_at_100 or recall_at_100")
print("  6. View: Bar chart  →  one bar per model")
print("=" * 65)
