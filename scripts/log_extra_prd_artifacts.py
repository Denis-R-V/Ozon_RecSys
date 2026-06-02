"""
Log extra artifacts to the existing PRD MLflow run.
Does NOT create a new run — uses mlflow.start_run(run_id=...).

Artifacts logged:
  notebooks/  ← 6_prd_inference.ipynb
  reports/    ← error_analysis_samples, error_analysis_summary,
                feature_importance.png, robustness_report (if exists)

Usage:
    python scripts/log_extra_prd_artifacts.py
"""
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_NAME     = "final_reranker_v12_repeat__merged_with_dl_all_repeat_history"
REPORTS_DIR  = PROJECT_ROOT / "artifacts" / "reports" / RUN_NAME
NB_DIR       = PROJECT_ROOT / "notebooks"

# ── MLflow / MinIO config ──────────────────────────────────────────────────────
PRD_RUN_ID          = "6e08071b6021407a8ed6c9dd28318a8b"   # existing PRD run
MLFLOW_TRACKING_URI = "http://localhost:5000"

# Force-load .env for MinIO credentials
_env = PROJECT_ROOT / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, _, v = _line.partition("=")
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID",       "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY",   "minioadmin")
os.environ.setdefault("MLFLOW_S3_IGNORE_TLS",    "true")

# Reset boto3 default session
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
    print(f"MinIO OK: {os.environ['MLFLOW_S3_ENDPOINT_URL']}  buckets={_buckets}")
    del _s3
except ImportError:
    print("[WARN] boto3 not installed — artifact upload may fail")
except Exception as e:
    print(f"[WARN] MinIO check failed: {e}")

# ── Connect MLflow ─────────────────────────────────────────────────────────────
import mlflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Verify run exists
_run = mlflow.get_run(PRD_RUN_ID)
print(f"\nPRD run found : {PRD_RUN_ID}")
print(f"  status      : {_run.info.status}")
print(f"  run_name    : {_run.info.run_name}")
print(f"  prd tag     : {_run.data.tags.get('prd', '?')}")
print()

# ── Generate feature_importance.png ───────────────────────────────────────────
FI_CSV = REPORTS_DIR / "v12_feature_importance.csv"
FI_PNG = REPORTS_DIR / "feature_importance.png"

if FI_CSV.exists() and not FI_PNG.exists():
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")          # headless
        import matplotlib.pyplot as plt

        fi = pd.read_csv(FI_CSV)

        # Detect columns
        feat_col  = next((c for c in fi.columns if "feature" in c.lower()), fi.columns[0])
        score_col = next((c for c in fi.columns if c.lower() in ("importance", "gain", "split", "score")
                          and c != feat_col), None)
        if score_col is None:
            score_col = [c for c in fi.columns if c != feat_col][0]

        top30 = fi.nlargest(30, score_col)[[feat_col, score_col]].sort_values(score_col)

        fig, ax = plt.subplots(figsize=(10, 9))
        ax.barh(top30[feat_col], top30[score_col], color="#4e79a7")
        ax.set_xlabel(score_col)
        ax.set_title("Feature Importance — Top 30  (LightGBM Reranker, PRD)")
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        fig.savefig(FI_PNG, dpi=120)
        plt.close(fig)
        print(f"Created: {FI_PNG.name}")
    except Exception as e:
        print(f"[WARN] feature_importance.png generation failed: {e}")
        FI_PNG = None
elif FI_PNG.exists():
    print(f"feature_importance.png already exists, reusing")

# ── Artifact manifest ──────────────────────────────────────────────────────────
#  (local_path, artifact_path_in_mlflow)
ARTIFACTS = [
    (NB_DIR  / "6_prd_inference.ipynb",          "notebooks"),
    (REPORTS_DIR / "error_analysis_samples.parquet",  "reports"),
    (REPORTS_DIR / "error_analysis_summary.parquet",  "reports"),
    (REPORTS_DIR / "robustness_report.parquet",        "reports"),
    (FI_PNG  if FI_PNG and FI_PNG.exists() else None,  "reports"),
    (FI_CSV,                                           "reports"),  # already there but idempotent
]

# ── Log into existing PRD run ──────────────────────────────────────────────────
logged  = []
missing = []
skipped = []

with mlflow.start_run(run_id=PRD_RUN_ID):
    for local_path, artifact_path in ARTIFACTS:
        if local_path is None:
            continue
        p = Path(local_path)
        if not p.exists():
            missing.append(str(p.name))
            continue
        try:
            mlflow.log_artifact(str(p), artifact_path=artifact_path)
            logged.append(f"{artifact_path}/{p.name}")
        except Exception as e:
            skipped.append(f"{p.name}: {e}")

# ── Report ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PRD run  : {PRD_RUN_ID}")
print(f"Tracking : {MLFLOW_TRACKING_URI}")
print()
print(f"Logged ({len(logged)}):")
for x in logged:
    print(f"  logged  : {x}")
print()
print(f"Missing ({len(missing)}):")
for x in missing:
    print(f"  missing : {x}")
if skipped:
    print(f"\nErrors ({len(skipped)}):")
    for x in skipped:
        print(f"  error   : {x}")
print(f"{'='*60}")
print("Done. Check MLflow UI: http://localhost:5000")
