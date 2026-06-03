#!/usr/bin/env python3
"""
CLI inference script: load PRD model from MLflow and generate recommendations.

Finds the latest run tagged model_stage=PRD, downloads the model and feature
columns from MinIO via MLflow artifact store, scores a sample of candidates,
and prints top-N recommendations per user.

Usage:
    # Basic: top-10 recs for 5 demo users
    python scripts/predict.py

    # Specify user IDs:
    python scripts/predict.py --user-ids 12800,1758720,3423970

    # Top-N:
    python scripts/predict.py --top-n 20

    # Use specific run_id instead of latest PRD:
    python scripts/predict.py --run-id ba67e7e4efd842778a8b5d6dca78d1fe

    # Use a specific model file (skip MLflow):
    python scripts/predict.py --local-model models/reranker/final_reranker_v12.../best_ranker.joblib
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def find_project_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "data").exists() and (p / "models").exists():
            return p
    raise FileNotFoundError("Cannot find project root")


def load_env(root: Path) -> None:
    env_path = root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip().strip('"').strip("'")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("AWS_ACCESS_KEY_ID",       "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY",   "minioadmin")
    os.environ.setdefault("MLFLOW_S3_IGNORE_TLS",    "true")


def setup_boto3() -> None:
    try:
        import boto3
        boto3.setup_default_session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name="us-east-1",
        )
    except Exception as e:
        print(f"[WARN] boto3: {e}")


def find_prd_run(mlflow_uri: str, experiment: str) -> str:
    import mlflow
    mlflow.set_tracking_uri(mlflow_uri)
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string="tags.model_stage = 'PRD'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if len(runs) == 0:
        raise RuntimeError(
            f"No PRD run found in experiment '{experiment}'.\n"
            "Run: python scripts/create_prd_run.py"
        )
    run_id = runs.iloc[0]["run_id"]
    run_name = runs.iloc[0]["tags.mlflow.runName"]
    print(f"PRD run found: {run_id[:8]}... ({run_name})")
    return run_id


def load_model_from_mlflow(run_id: str):
    import mlflow
    model = None
    model_source = None

    # (a) MLflow model format
    try:
        model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
        model_source = f"mlflow_model:{run_id[:8]}"
        print(f"[a] loaded via mlflow.lightgbm.load_model ✓")
        return model, model_source
    except Exception as e:
        print(f"[a] ✗ {e}")

    # (b) download_artifacts + joblib
    import joblib
    for artifact_path in ["model/best_ranker.joblib", "model/lgbm_ranker_t02_selection.joblib"]:
        try:
            local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
            model = joblib.load(local)
            model_source = f"mlflow_artifact:{artifact_path}"
            print(f"[b] loaded from {artifact_path} ✓")
            return model, model_source
        except Exception as e:
            print(f"[b] ✗ {artifact_path}: {e}")

    raise RuntimeError(f"Could not load model from run {run_id}")


def load_feature_cols(run_id: str, local_fallback: Path | None = None) -> list[str]:
    import mlflow
    try:
        local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model/feature_cols.json")
        raw = json.loads(Path(local).read_text(encoding="utf-8"))
        cols = raw.get("numeric_feature_cols", raw) if isinstance(raw, dict) else list(raw)
        print(f"Feature cols from MLflow: {len(cols)}")
        return cols
    except Exception as e:
        print(f"[WARN] MLflow feature_cols: {e}")

    if local_fallback and local_fallback.exists():
        raw = json.loads(local_fallback.read_text(encoding="utf-8"))
        cols = raw.get("numeric_feature_cols", raw) if isinstance(raw, dict) else list(raw)
        print(f"Feature cols from local fallback: {len(cols)}")
        return cols

    raise RuntimeError("Cannot load feature_cols.json")


def load_sample_data(root: Path, user_ids: list[int] | None, feature_cols: list[str]) -> "pd.DataFrame":
    """Load a small sample from val_eval dataset for demo predictions."""
    import polars as pl
    import pandas as pd
    import numpy as np

    RUN_NAME = "final_reranker_v12_repeat__merged_with_dl_all_repeat_history"
    val_eval_dir = root / "data" / "processed" / "reranker" / RUN_NAME / "dataset" / "val_eval"

    if not val_eval_dir.exists():
        raise FileNotFoundError(f"val_eval dataset not found: {val_eval_dir}")

    shards = sorted(val_eval_dir.iterdir())[:3]
    dfs = []
    for shard in shards:
        pqs = list(shard.glob("*.parquet"))
        if pqs:
            dfs.append(pl.read_parquet(str(pqs[0])).to_pandas())

    df = pd.concat(dfs, ignore_index=True)

    if user_ids:
        df = df[df["user_id"].isin(user_ids)]
        if len(df) == 0:
            raise ValueError(f"None of the requested users found in val_eval sample. Try without --user-ids.")
    else:
        # Pick 5 random users
        rng = np.random.default_rng(42)
        users = df["user_id"].unique()
        sample_users = rng.choice(users, size=min(5, len(users)), replace=False)
        df = df[df["user_id"].isin(sample_users)]

    # Fill missing features
    for f in feature_cols:
        if f not in df.columns:
            df[f] = 0.0

    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Load PRD model from MLflow and generate top-N recommendations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--run-id",       default=None, help="MLflow run_id (default: latest PRD run)")
    parser.add_argument("--user-ids",     default=None, help="Comma-separated user IDs to score")
    parser.add_argument("--top-n",        type=int, default=10, help="Top-N items per user (default: 10)")
    parser.add_argument("--mlflow-uri",   default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--experiment",   default="recsys_reranker", help="MLflow experiment name")
    parser.add_argument("--local-model",  default=None, help="Path to local joblib model (skip MLflow)")
    args = parser.parse_args()

    import mlflow
    import joblib
    import numpy as np
    import pandas as pd

    root = find_project_root()
    load_env(root)
    setup_boto3()

    mlflow.set_tracking_uri(args.mlflow_uri)

    print(f"\n{'='*60}")
    print("PRD MODEL INFERENCE")
    print(f"{'='*60}")

    # ── Load model ────────────────────────────────────────────────────────────
    if args.local_model:
        model = joblib.load(args.local_model)
        model_source = f"local:{args.local_model}"
        run_id = args.run_id
        print(f"Model loaded from local: {args.local_model}")
    else:
        run_id = args.run_id or find_prd_run(args.mlflow_uri, args.experiment)
        model, model_source = load_model_from_mlflow(run_id)

    print(f"model_source : {model_source}")
    print(f"model type   : {type(model).__name__}")

    # ── Load feature columns ──────────────────────────────────────────────────
    RUN_NAME = "final_reranker_v12_repeat__merged_with_dl_all_repeat_history"
    local_feat = root / "models" / "reranker" / RUN_NAME / "feature_cols.json"
    feature_cols = load_feature_cols(run_id, local_feat) if run_id else json.loads(local_feat.read_text())["numeric_feature_cols"]

    assert not hasattr(model, "booster_") or len(feature_cols) > 0, "feature_cols empty"
    print(f"feature cols : {len(feature_cols)}")

    # ── Load data ─────────────────────────────────────────────────────────────
    user_ids = [int(x.strip()) for x in args.user_ids.split(",")] if args.user_ids else None
    df = load_sample_data(root, user_ids, feature_cols)
    print(f"sample shape : {df.shape}  ({df['user_id'].nunique()} users)")

    # ── Score ─────────────────────────────────────────────────────────────────
    X = df[feature_cols].values.astype(np.float32)
    scores = model.predict(X)
    df["_score"] = scores

    # ── Top-N per user ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Top-{args.top_n} recommendations per user")
    print(f"{'='*60}")

    results = {}
    for uid, grp in df.groupby("user_id"):
        top = grp.nlargest(args.top_n, "_score")[["item_id", "_score"]]
        item_ids = top["item_id"].astype(int).tolist()
        sc = top["_score"].round(4).tolist()
        results[int(uid)] = item_ids
        print(f"\nuser {uid}:")
        print(f"  items  : {item_ids}")
        print(f"  scores : {sc}")
        if "label" in grp.columns:
            hits = sum(1 for iid in item_ids if grp[grp["item_id"] == iid]["label"].sum() > 0)
            print(f"  gt hits: {hits}/{args.top_n}")

    print(f"\n{'='*60}")
    print(f"PRD model loaded from MLflow/MinIO  ✓")
    print(f"Inference working  ✓")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    main()
