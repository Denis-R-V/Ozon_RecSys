"""
Create two groups of MLflow eval runs for visualization:

Group 1: retrieval/*  (eval_group=retrieval_sources)
  Shows recall@k for each individual candidate generator + combined pool + final model.
  Purpose: understand contribution of each retrieval source.

Group 2: pipeline/*  (eval_group=pipeline_comparison)
  Shows NDCG@k + recall@k for top_popular vs 2-stage reranker.
  Purpose: clear before/after comparison of the full system.

Usage:
    python scripts/log_analysis_runs.py
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

def get_metrics(df, model_key, metric_keys=("ndcg", "recall", "hit_rate", "precision")):
    """Return flat dict of all k values for given model."""
    out = {}
    for k in [10, 20, 50, 100]:
        r = df[(df["model"] == model_key) & (df["k"] == k)]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        for col in metric_keys:
            if col in r.index and not pd.isna(r[col]):
                out[f"{col}_at_{k}"] = float(r[col])
    return out

def reranker_metrics():
    out = {}
    for _, r in mdf.iterrows():
        k = int(r["k"])
        for col in ("ndcg", "recall", "hit_rate", "precision"):
            if col in r.index and not pd.isna(r[col]):
                out[f"{col}_at_{k}"] = float(r[col])
    return out

# ── Model definitions ──────────────────────────────────────────────────────────

# ── Group 1: Retrieval sources (recall is the primary metric here) ─────────────
RETRIEVAL_RUNS = [
    dict(
        run_name    = "retrieval/als",
        description = "ALS retrieval — collaborative filtering (matrix factorisation)",
        metrics     = get_metrics(bdf, "als_rank"),
        tags        = {"eval_group": "retrieval_sources", "source_type": "collaborative_filtering",
                       "stage": "retrieval"},
    ),
    dict(
        run_name    = "retrieval/top_popular",
        description = "TopPopular retrieval — most purchased items globally",
        metrics     = get_metrics(bdf, "top_popular_rank"),
        tags        = {"eval_group": "retrieval_sources", "source_type": "heuristic",
                       "stage": "retrieval"},
    ),
    dict(
        run_name    = "retrieval/dl_dssm",
        description = "DL DSSM retrieval — two-tower deep learning model",
        metrics     = get_metrics(bdf, "dl_dssm_final_rank"),
        tags        = {"eval_group": "retrieval_sources", "source_type": "deep_learning",
                       "stage": "retrieval"},
    ),
    dict(
        run_name    = "retrieval/dl_bert4rec",
        description = "DL BERT4Rec retrieval — transformer sequential model",
        metrics     = get_metrics(bdf, "dl_bert4rec_small_rank"),
        tags        = {"eval_group": "retrieval_sources", "source_type": "deep_learning",
                       "stage": "retrieval"},
    ),
    dict(
        run_name    = "retrieval/dl_sasrec",
        description = "DL SASRec retrieval — self-attention sequential model",
        metrics     = get_metrics(bdf, "dl_sasrec_v3_rank"),
        tags        = {"eval_group": "retrieval_sources", "source_type": "deep_learning",
                       "stage": "retrieval"},
    ),
    dict(
        run_name    = "retrieval/repeat_purchases",
        description = "Repeat-purchase retrieval — user's past bought items",
        metrics     = get_metrics(bdf, "repeat_cart_orders_rank"),
        tags        = {"eval_group": "retrieval_sources", "source_type": "repeat_history",
                       "stage": "retrieval"},
    ),
    dict(
        run_name    = "retrieval/combined_pool",
        description = "Combined candidate pool — all sources merged, ordered by pool rank",
        metrics     = get_metrics(bdf, "candidate_rank"),
        tags        = {"eval_group": "retrieval_sources", "source_type": "merged_pool",
                       "stage": "retrieval"},
    ),
    dict(
        run_name    = "retrieval/2stage_reranker",
        description = "2-stage reranker — same pool re-ranked by LightGBM (final model)",
        metrics     = reranker_metrics(),
        tags        = {"eval_group": "retrieval_sources", "source_type": "reranker",
                       "stage": "reranking", "prd": "true"},
    ),
]

# ── Group 2: Pipeline comparison (full system vs trivial baseline) ─────────────
PIPELINE_RUNS = [
    dict(
        run_name    = "pipeline/top_popular",
        description = "TopPopular — trivial baseline: 100 most purchased items for everyone",
        metrics     = get_metrics(bdf, "top_popular_rank"),
        tags        = {"eval_group": "pipeline_comparison", "model_type": "baseline"},
    ),
    dict(
        run_name    = "pipeline/als",
        description = "ALS — matrix factorisation retrieval, no reranking",
        metrics     = get_metrics(bdf, "als_rank"),
        tags        = {"eval_group": "pipeline_comparison", "model_type": "retrieval_only"},
    ),
    dict(
        run_name    = "pipeline/candidate_pool_rank",
        description = "Candidate pool rank — all retrieval sources merged, position-based rank",
        metrics     = get_metrics(bdf, "candidate_rank"),
        tags        = {"eval_group": "pipeline_comparison", "model_type": "retrieval_only"},
    ),
    dict(
        run_name    = "pipeline/2stage_reranker_prd",
        description = "2-stage reranker PRD — retrieval + LightGBM LambdaRank (production model)",
        metrics     = reranker_metrics(),
        tags        = {"eval_group": "pipeline_comparison", "model_type": "full_pipeline",
                       "prd": "true", "model_stage": "PRD"},
    ),
]

ALL_RUNS = RETRIEVAL_RUNS + PIPELINE_RUNS

# ── Create runs (idempotent) ───────────────────────────────────────────────────
import mlflow
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

# Collect existing run names for both groups (MLflow doesn't support IN — use two queries)
existing_names = set()
for _grp in ("retrieval_sources", "pipeline_comparison"):
    _ex = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string=f"tags.eval_group = '{_grp}'",
        max_results=50,
    )
    if len(_ex) > 0:
        existing_names.update(_ex["tags.mlflow.runName"].tolist())

created = []
skipped = []

for m in ALL_RUNS:
    rname = m["run_name"]
    if rname in existing_names:
        skipped.append(rname)
        continue
    with mlflow.start_run(run_name=rname) as run:
        mlflow.set_tags({
            "mlflow.note.content": m["description"],
            **m["tags"],
        })
        mlflow.log_metrics(m["metrics"])
        rid = run.info.run_id
    created.append((rname, rid, m["metrics"]))

# ── Report ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("GROUP 1: Retrieval sources  (filter: tags.eval_group = 'retrieval_sources')")
print(f"{'='*70}")
for m in RETRIEVAL_RUNS:
    r100  = m["metrics"].get("recall_at_100", float("nan"))
    ndcg  = m["metrics"].get("ndcg_at_100", float("nan"))
    status = "skip (exists)" if m["run_name"] in existing_names else "created"
    print(f"  [{status:<12}] {m['run_name']:<35}  recall@100={r100:.4f}  ndcg@100={ndcg:.4f}")

print(f"\n{'='*70}")
print("GROUP 2: Pipeline comparison  (filter: tags.eval_group = 'pipeline_comparison')")
print(f"{'='*70}")
for m in PIPELINE_RUNS:
    r100  = m["metrics"].get("recall_at_100", float("nan"))
    ndcg  = m["metrics"].get("ndcg_at_100", float("nan"))
    status = "skip (exists)" if m["run_name"] in existing_names else "created"
    print(f"  [{status:<12}] {m['run_name']:<38}  ndcg@100={ndcg:.4f}  recall@100={r100:.4f}")

print(f"\nCreated: {len(created)}  |  Skipped (already exist): {len(skipped)}")

print(f"""
{'='*70}
HOW TO VIEW IN MLFLOW UI
{'='*70}

1. Open: {MLFLOW_URI}  →  Experiment: {EXPERIMENT}

[Chart A] Candidate generator contributions — recall@100 bar chart:
  • Filter: tags.eval_group = 'retrieval_sources'
  • Select all retrieval/* runs  →  Compare
  • Metric: recall_at_100  |  Chart type: Bar chart

[Chart B] Full pipeline vs baseline — NDCG@100:
  • Filter: tags.eval_group = 'pipeline_comparison'
  • Select all pipeline/* runs  →  Compare
  • Metric: ndcg_at_100  |  Chart type: Bar chart
  • Also useful: recall_at_100 side-by-side

{'='*70}""")
