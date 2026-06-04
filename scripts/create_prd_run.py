"""
Воссоздаёт финальный PRD MLflow run из локальных артефактов.
Используется после docker compose down -v (потеря SQLite metadata).

Логгирует:
  - теги: prd=true, model_stage=PRD, архитектура
  - параметры: из run_config.json (seed, даты сплитов, гиперпараметры)
  - метрики: val NDCG/recall@k для всех моделей
  - артефакты: модель, feature_cols, отчёты, графики, ноутбук

Usage:
    python scripts/create_prd_run.py
"""
import os, sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_NAME     = "final_reranker_v12_repeat__merged_with_dl_all_repeat_history"
REPORTS_DIR  = PROJECT_ROOT / "artifacts" / "reports" / RUN_NAME
MODELS_DIR   = PROJECT_ROOT / "models" / "reranker" / RUN_NAME
NB_DIR       = PROJECT_ROOT / "notebooks"
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
    _s3 = boto3.client("s3",
        endpoint_url          = os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name           = "us-east-1",
    )
    _buckets = [b["Name"] for b in _s3.list_buckets().get("Buckets", [])]
    print(f"MinIO OK: buckets={_buckets}")
    del _s3
except Exception as e:
    print(f"[WARN] MinIO: {e}")

# ── Load source data ───────────────────────────────────────────────────────────
import pandas as pd

cfg     = json.loads((REPORTS_DIR / "run_config.json").read_text(encoding="utf-8"))
bdf     = pd.read_parquet(REPORTS_DIR / "final_reranker_baseline_val_metrics.parquet")
mdf     = pd.read_parquet(REPORTS_DIR / "final_reranker_best_val_metrics.parquet")

def val(df, model, metric, k=100):
    r = df[(df["model"] == model) & (df["k"] == k)]
    return float(r[metric].iloc[0]) if len(r) > 0 else float("nan")

# ── Parameters (from run_config) ───────────────────────────────────────────────
PARAM_KEYS = [
    "random_state", "train_context_start", "train_context_end",
    "train_target_start", "train_target_end",
    "val_context_start", "val_context_end",
    "val_target_start", "val_target_end",
    "test_target_start", "candidate_input_name",
    "optuna_trials", "early_stopping_rounds",
    "run_lightgbm", "run_catboost",
    "refit_production_on_train_val", "enable_test",
    "model_max_numeric_features", "model_train_max_rows",
    "final_k", "positive_weight", "feature_store_version",
    "notes",
]
params = {k: str(cfg[k]) for k in PARAM_KEYS if k in cfg}

# ── Metrics ────────────────────────────────────────────────────────────────────
metrics = {}

# Final reranker @ all k
reranker_model = mdf["model"].iloc[0]
for _, r in mdf.iterrows():
    k = int(r["k"])
    for col in ("ndcg", "recall", "hit_rate", "precision"):
        if col in r.index and not pd.isna(r[col]):
            metrics[f"reranker_{col}_at_{k}"] = float(r[col])

# Baselines @ 100
for bname, bkey in [
    ("top_popular", "top_popular_rank"),
    ("als",         "als_rank"),
    ("candidate_rank", "candidate_rank"),
]:
    for col in ("ndcg", "recall"):
        v = val(bdf, bkey, col, 100)
        if not pd.isna(v):
            metrics[f"{bname}_{col}_at_100"] = v

# Lift metrics
tp_ndcg = metrics.get("top_popular_ndcg_at_100", float("nan"))
cr_ndcg = metrics.get("candidate_rank_ndcg_at_100", float("nan"))
rr_ndcg = metrics.get("reranker_ndcg_at_100", float("nan"))
import math
if not math.isnan(tp_ndcg) and tp_ndcg > 0:
    metrics["lift_vs_top_popular_ndcg_at_100"] = round(rr_ndcg / tp_ndcg, 4)
if not math.isnan(cr_ndcg) and cr_ndcg > 0:
    metrics["lift_vs_best_baseline_ndcg_at_100"] = round(rr_ndcg / cr_ndcg, 4)

# Candidate pool recall
cand_recall = val(bdf, "candidate_rank", "recall", 100)
if not math.isnan(cand_recall):
    metrics["candidate_pool_recall_at_100"] = cand_recall

# ── Artifact manifest ──────────────────────────────────────────────────────────
# (local_path, artifact_path_in_mlflow)
ARTIFACTS = [
    # Model
    (MODELS_DIR / "best_ranker.joblib",             "model"),
    (MODELS_DIR / "feature_cols.json",              "model"),
    # Reports — metrics
    (REPORTS_DIR / "final_reranker_best_val_metrics.parquet",      "reports"),
    (REPORTS_DIR / "final_reranker_baseline_val_metrics.parquet",  "reports"),
    (REPORTS_DIR / "baseline_comparison.parquet",                  "reports"),
    # Reports — comparison
    (REPORTS_DIR / "model_comparison.png",          "reports"),
    (REPORTS_DIR / "model_comparison.csv",          "reports"),
    # Reports — feature importance
    (REPORTS_DIR / "feature_importance.png",        "reports"),
    (REPORTS_DIR / "v12_feature_importance.csv",    "reports"),
    # Reports — error analysis
    (REPORTS_DIR / "error_analysis_samples.parquet",  "reports"),
    (REPORTS_DIR / "error_analysis_summary.parquet",  "reports"),
    # Reproducibility
    (REPORTS_DIR / "run_config.json",               "reports"),
    (REPORTS_DIR / "versions.json",                 "reports"),
    (REPORTS_DIR / "requirements_snapshot.txt",     "reports"),
    # Notebook
    (NB_DIR / "6_prd_inference.ipynb",              "notebooks"),
]

# ── Create PRD run ─────────────────────────────────────────────────────────────
import mlflow
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

print(f"\nCreating PRD run...")
with mlflow.start_run(run_name="2stage_reranker_prd") as run:
    prd_run_id = run.info.run_id

    # Tags
    mlflow.set_tags({
        "mlflow.note.content": (
            "2-stage recsys: retrieval (ALS + TopPop + DL + repeat-history) → "
            "LightGBM LambdaRank reranker. "
            f"val_eval NDCG@100={rr_ndcg:.6f}, recall@100={metrics.get('reranker_recall_at_100', float('nan')):.6f}. "
            "Candidate pool: merged_with_dl_all_repeat_history."
        ),
        "prd":               "true",
        "model_stage":       "PRD",
        "architecture":      "2stage_retrieval_rerank",
        "stage1_retrieval":  "als_toppop_dl_repeat_history",
        "stage2_reranker":   "lightgbm_lambdarank",
        "eval_type":         "val_eval",
        "candidate_pool":    "merged_with_dl_all_repeat_history",
    })

    # Params
    mlflow.log_params(params)
    print(f"  params logged: {len(params)}")

    # Metrics
    mlflow.log_metrics(metrics)
    print(f"  metrics logged: {len(metrics)}")

    # Artifacts
    logged, missing = [], []
    for local_path, art_path in ARTIFACTS:
        p = Path(local_path)
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path=art_path)
            logged.append(f"{art_path}/{p.name}")
        else:
            missing.append(str(p.name))

    print(f"  artifacts logged: {len(logged)}")
    if missing:
        print(f"  missing (skipped): {missing}")

# ── Save new run_id to files ───────────────────────────────────────────────────
prd_model_uri = f"runs:/{prd_run_id}/model"

(REPORTS_DIR / "prd_run_id.txt").write_text(prd_run_id, encoding="utf-8")
(REPORTS_DIR / "prd_model_uri.txt").write_text(prd_model_uri, encoding="utf-8")
(REPORTS_DIR / "prd_artifact_uri.txt").write_text(
    f"mlflow-artifacts:/1/{prd_run_id}/artifacts", encoding="utf-8"
)

print(f"""
{'='*65}
PRD run created successfully
{'='*65}
  run_id        : {prd_run_id}
  run_name      : 2stage_reranker_prd
  model_uri     : {prd_model_uri}

  reranker NDCG@100  : {rr_ndcg:.6f}
  lift vs TopPopular : ×{metrics.get('lift_vs_top_popular_ndcg_at_100', float('nan')):.1f}
  lift vs best_bl    : ×{metrics.get('lift_vs_best_baseline_ndcg_at_100', float('nan')):.1f}

  MLflow UI : {MLFLOW_URI}/#/experiments/1/runs/{prd_run_id}
  prd_run_id.txt updated ✓
  prd_model_uri.txt  updated ✓
{'='*65}

Next: re-run notebooks/6_prd_inference.ipynb (Run All)
""")
