#!/usr/bin/env python3
"""
CLI training script for 2-stage reranker (LightGBM LambdaRank).
Trains on pre-built datasets, logs to MLflow.

Usage:
    # Default config (configs/train.yaml):
    python scripts/train.py

    # Override hyperparameters:
    python scripts/train.py model.n_estimators=500 model.learning_rate=0.05

    # Custom run name:
    python scripts/train.py mlflow.run_name=experiment_v2

    # Quick smoke test (fewer rows):
    python scripts/train.py data.max_train_rows=100000 data.max_val_rows=50000

Prerequisites:
    pip install hydra-core lightgbm joblib pandas polars pyarrow mlflow boto3
    docker compose up -d   # MLflow + MinIO must be running
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


# ── Paths ──────────────────────────────────────────────────────────────────────

def find_project_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "data").exists() and (p / "models").exists():
            return p
    raise FileNotFoundError("Cannot find project root (needs data/ and models/)")


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_split(dataset_dir: Path, split: str, max_rows: int | None, feature_cols: list[str], seed: int = 42):
    """Load a dataset split as pandas DataFrame, optionally downsampled."""
    import numpy as np
    import polars as pl

    split_dir = dataset_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    parquets = sorted(split_dir.rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquets in {split_dir}")

    log.info(f"Loading {split}: {len(parquets)} shards...")
    lf = pl.scan_parquet([str(p) for p in parquets], hive_partitioning=True)

    need_cols = ["user_id", "item_id", "label"] + feature_cols
    existing = set(lf.collect_schema().names())
    select_cols = [c for c in need_cols if c in existing]

    df = lf.select(select_cols).collect().to_pandas()

    # Fill missing feature cols with 0
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    if max_rows and len(df) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=max_rows, replace=False)
        df = df.iloc[sorted(idx)].reset_index(drop=True)
        log.info(f"  downsampled to {len(df):,} rows")
    else:
        log.info(f"  loaded {len(df):,} rows")

    return df


def make_group_array(df, key: str = "user_id"):
    """LightGBM requires group sizes (number of items per query)."""
    return df.groupby(key, sort=False)[key].count().values


def evaluate_scored(df_eval: "pd.DataFrame", scores, gt: dict, ks=(10, 20, 50, 100)):
    """Compute NDCG@k and recall@k from model scores on eval split."""
    import numpy as np

    df = df_eval.copy()
    df["_score"] = scores

    results = {}
    for k in ks:
        ndcg_list, recall_list = [], []
        for uid, grp in df.groupby("user_id"):
            gt_items = set(gt.get(int(uid), []))
            if not gt_items:
                continue
            top_k = grp.nlargest(k, "_score")["item_id"].astype(int).values
            hits = [1.0 if iid in gt_items else 0.0 for iid in top_k]
            dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_items), k))) or 1.0
            ndcg_list.append(dcg / idcg)
            recall_list.append(sum(hits) / len(gt_items))
        results[k] = {
            "ndcg": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
            "recall": float(np.mean(recall_list)) if recall_list else 0.0,
            "n_users": len(ndcg_list),
        }
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    import joblib
    import mlflow
    import numpy as np
    import pandas as pd
    from lightgbm import LGBMRanker

    sys.stdout.reconfigure(encoding="utf-8")

    root = find_project_root()
    log.info(f"Project root: {root}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ── Paths ──────────────────────────────────────────────────────────────────
    run_name   = cfg.data.run_name
    dataset_dir = root / "data" / "processed" / "reranker" / run_name / "dataset"
    models_dir  = root / "models" / "reranker" / run_name
    reports_dir = root / "artifacts" / "reports" / run_name
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Feature columns ────────────────────────────────────────────────────────
    feat_path = models_dir / "feature_cols.json"
    if not feat_path.exists():
        raise FileNotFoundError(f"feature_cols.json not found: {feat_path}")
    raw = json.loads(feat_path.read_text(encoding="utf-8"))
    feature_cols: list[str] = raw.get("numeric_feature_cols", raw) if isinstance(raw, dict) else list(raw)
    feature_cols = feature_cols[: cfg.model.get("max_features", len(feature_cols))]
    log.info(f"Features: {len(feature_cols)}")

    # ── Load val_eval ground truth for evaluation ──────────────────────────────
    labels_dir = root / "data" / "processed" / "reranker" / run_name / "labels"
    val_gt: dict[int, list[int]] = {}
    if labels_dir.exists():
        import polars as pl
        for lf_path in sorted(labels_dir.glob("labels_val*.parquet")):
            ldf = pl.read_parquet(str(lf_path)).filter(pl.col("label") == 1)
            for row in ldf.iter_rows(named=True):
                val_gt.setdefault(int(row["user_id"]), []).append(int(row["item_id"]))
            log.info(f"val GT: {len(val_gt):,} users from {lf_path.name}")
            break

    # ── Load datasets ──────────────────────────────────────────────────────────
    log.info("Loading train_fit...")
    df_train = load_split(dataset_dir, cfg.data.train_split, cfg.data.max_train_rows, feature_cols, cfg.data.random_state)
    log.info("Loading val_fit (early stopping)...")
    df_val   = load_split(dataset_dir, cfg.data.val_split, cfg.data.max_val_rows, feature_cols, cfg.data.random_state)

    X_train = df_train[feature_cols].values.astype("float32")
    y_train = df_train["label"].fillna(0).astype(int).values
    g_train = make_group_array(df_train)

    X_val = df_val[feature_cols].values.astype("float32")
    y_val = df_val["label"].fillna(0).astype(int).values
    g_val = make_group_array(df_val)

    sample_weight_train = np.where(y_train > 0, cfg.model.positive_weight, 1.0)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    for _l in (root / ".env").read_text(encoding="utf-8").splitlines():
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
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name="us-east-1",
        )
    except Exception as e:
        log.warning(f"boto3: {e}")

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # ── Train ─────────────────────────────────────────────────────────────────
    model_params = {
        "n_estimators":              cfg.model.n_estimators,
        "learning_rate":             cfg.model.learning_rate,
        "max_depth":                 cfg.model.max_depth,
        "num_leaves":                cfg.model.num_leaves,
        "min_child_samples":         cfg.model.min_child_samples,
        "colsample_bytree":          cfg.model.colsample_bytree,
        "subsample":                 cfg.model.subsample,
        "reg_alpha":                 cfg.model.reg_alpha,
        "reg_lambda":                cfg.model.reg_lambda,
        "lambdarank_truncation_level": cfg.model.lambdarank_truncation_level,
        "objective":                 "lambdarank",
        "metric":                    "ndcg",
        "random_state":              cfg.data.random_state,
        "n_jobs":                    -1,
        "verbose":                   -1,
    }

    log.info("Training LightGBM LambdaRank...")
    t0 = time.perf_counter()
    model = LGBMRanker(**model_params)
    model.fit(
        X_train, y_train,
        group=g_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_val, y_val)],
        eval_group=[g_val],
        eval_metric="ndcg",
        callbacks=[
            __import__("lightgbm").early_stopping(cfg.model.early_stopping_rounds, verbose=False),
            __import__("lightgbm").log_evaluation(50),
        ],
    )
    train_time = time.perf_counter() - t0
    log.info(f"Training complete in {train_time:.1f}s. Best iteration: {model.best_iteration_}")

    # ── Evaluate on val_eval ──────────────────────────────────────────────────
    eval_metrics: dict = {}
    if val_gt:
        log.info("Evaluating on val_eval...")
        df_eval = load_split(dataset_dir, cfg.data.val_eval_split, None, feature_cols, cfg.data.random_state)
        scores  = model.predict(df_eval[feature_cols].values.astype("float32"))
        results = evaluate_scored(df_eval, scores, val_gt)
        for k, m in results.items():
            eval_metrics[f"val_eval/ndcg_at_{k}"]   = m["ndcg"]
            eval_metrics[f"val_eval/recall_at_{k}"] = m["recall"]
        log.info(f"val_eval NDCG@100 = {results[100]['ndcg']:.6f}")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = models_dir / "lgbm_cli_ranker.joblib"
    joblib.dump(model, model_path)
    log.info(f"Model saved: {model_path}")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    tags = dict(cfg.mlflow.tags) if cfg.mlflow.get("tags") else {}
    tags["cli"] = "hydra"

    with mlflow.start_run(run_name=cfg.mlflow.run_name, tags=tags) as run:
        # Params
        mlflow.log_params({
            **{f"model/{k}": v for k, v in model_params.items()},
            "data/train_split":    cfg.data.train_split,
            "data/val_split":      cfg.data.val_split,
            "data/max_train_rows": cfg.data.max_train_rows,
            "data/max_val_rows":   cfg.data.max_val_rows,
            "data/random_state":   cfg.data.random_state,
            "n_features":          len(feature_cols),
            "best_iteration":      model.best_iteration_,
            "train_time_sec":      round(train_time, 1),
        })

        # Metrics
        mlflow.log_metrics(eval_metrics)

        # Config artifact
        cfg_path = Path("hydra_config.yaml")
        cfg_path.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
        mlflow.log_artifact(str(cfg_path), artifact_path="config")

        # Model artifact
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(feat_path), artifact_path="model")

        run_id = run.info.run_id
        log.info(f"MLflow run: {run_id}")
        if eval_metrics:
            log.info(f"val_eval NDCG@100 = {eval_metrics.get('val_eval/ndcg_at_100', 0):.6f}")

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  run_id: {run_id}")
    print(f"  model:  {model_path}")
    if eval_metrics:
        print(f"  val_eval NDCG@100 = {eval_metrics.get('val_eval/ndcg_at_100', 0):.6f}")
    print(f"  MLflow: {cfg.mlflow.tracking_uri}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
