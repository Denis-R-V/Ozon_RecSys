"""
RankerStore: PRD LightGBM inference backend for the FastAPI demo service.

Responsibilities:
- Load best_ranker.joblib and feature_cols at startup.
- Read candidates/features per user_id via DuckDB (read-only analytical layer).
- Score candidates with model.predict(), sort by score desc.
- Join item metadata, user history, future events.
- Provide popular-items fallback for cold users.
- Expose health_checks() for /health/ready.

DuckDB is used as a local read-only query layer over Parquet files.
All tables are small (<20 MB total) so metadata/history/events are
kept as pandas DataFrames after initial load.
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Source-flag columns in candidates parquet -> short label for API response
_SOURCE_FLAG_LABELS: dict[str, str] = {
    "in_als": "als",
    "in_clip_faiss": "clip_faiss",
    "in_clip_hnsw": "clip_hnsw",
    "in_dl_bert4rec_small": "bert4rec",
    "in_dl_dssm_final": "dssm",
    "in_dl_sasrec_v3": "sasrec",
    "in_i2i_bm25": "i2i_bm25",
    "in_top_popular": "top_popular",
    "in_user_category_popular": "user_cat_popular",
    "is_repeat_history_candidate": "repeat_history",
}

_PURCHASE_ACTION_TYPES = frozenset({"to_cart", "order", "purchase", "buy"})


class RankerStore:
    """
    Stateful inference store. Instantiate once at startup via RankerStore.load().
    Thread-safe for reads (DuckDB connection is per-query, pandas is read-only).
    """

    def __init__(
        self,
        model: Any,
        feature_cols: list[str],
        candidates_path: pathlib.Path,
        item_meta: pd.DataFrame,
        user_history: pd.DataFrame,
        future_events: pd.DataFrame,
        popular_items: pd.DataFrame,
        model_info: dict,
        duckdb_threads: int = 2,
    ) -> None:
        self._model = model
        self._feature_cols = feature_cols
        self._candidates_path = candidates_path
        self._item_meta = item_meta          # indexed on item_id after load
        self._user_history = user_history
        self._future_events = future_events
        self._popular_items = popular_items  # sorted by global_pop_rank
        self._model_info = model_info
        self._duckdb_threads = duckdb_threads

        # Pre-build sets/indices for O(1) lookup
        self._known_user_ids: frozenset[int] = self._load_known_users()
        self._item_meta_idx = item_meta.set_index("item_id") if not item_meta.empty else item_meta

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, cfg: Any) -> "RankerStore":
        """Load all artifacts and return a ready RankerStore instance."""
        errors: list[str] = []

        # Model
        model_path = pathlib.Path(cfg.RANKER_MODEL_PATH)
        if not model_path.exists():
            raise RuntimeError(f"Ranker model not found: {model_path}")
        log.info("Loading ranker model: %s", model_path)
        model = joblib.load(model_path)
        log.info("Model loaded: %s  features=%d", type(model).__name__, model.n_features_in_)

        # Feature cols
        fc_path = pathlib.Path(cfg.FEATURE_COLS_PATH)
        if not fc_path.exists():
            raise RuntimeError(f"feature_cols.json not found: {fc_path}")
        fc_data = json.loads(fc_path.read_text(encoding="utf-8"))
        feature_cols: list[str] = fc_data["numeric_feature_cols"]
        if model.n_features_in_ != len(feature_cols):
            raise RuntimeError(
                f"Feature count mismatch: model={model.n_features_in_}, "
                f"feature_cols.json={len(feature_cols)}"
            )
        log.info("Feature cols loaded: %d columns", len(feature_cols))

        # Candidates path (DuckDB reads on demand — only check file exists here)
        candidates_path = pathlib.Path(cfg.CANDIDATE_FEATURES_PATH)
        if not candidates_path.exists():
            raise RuntimeError(f"Candidate features not found: {candidates_path}")

        # Item metadata
        meta_path = pathlib.Path(cfg.ITEM_METADATA_PATH)
        item_meta = pd.read_parquet(meta_path) if meta_path.exists() else pd.DataFrame()
        if item_meta.empty:
            log.warning("Item metadata not found or empty: %s", meta_path)

        # User history
        hist_path = pathlib.Path(cfg.USER_HISTORY_PATH)
        user_history = pd.read_parquet(hist_path) if hist_path.exists() else pd.DataFrame()
        if user_history.empty:
            log.warning("User history not found or empty: %s", hist_path)

        # Future events
        fut_path = pathlib.Path(cfg.FUTURE_EVENTS_PATH)
        future_events = pd.read_parquet(fut_path) if fut_path.exists() else pd.DataFrame()
        if future_events.empty:
            log.warning("Future events not found or empty: %s", fut_path)

        # Popular items (cold fallback)
        pop_path = pathlib.Path(cfg.RANKER_POPULAR_ITEMS_PATH)
        popular_items = pd.read_parquet(pop_path) if pop_path.exists() else pd.DataFrame()
        if popular_items.empty:
            log.warning("Popular items not found: %s", pop_path)

        # Model info JSON
        info_path = pathlib.Path(cfg.MODEL_INFO_PATH)
        model_info: dict = {}
        if info_path.exists():
            model_info = json.loads(info_path.read_text(encoding="utf-8"))

        store = cls(
            model=model,
            feature_cols=feature_cols,
            candidates_path=candidates_path,
            item_meta=item_meta,
            user_history=user_history,
            future_events=future_events,
            popular_items=popular_items,
            model_info=model_info,
            duckdb_threads=getattr(cfg, "DUCKDB_THREADS", 2),
        )
        log.info(
            "RankerStore ready: %d known users, %d items in metadata",
            len(store._known_user_ids),
            len(item_meta),
        )
        return store

    # ------------------------------------------------------------------ #
    # Internal: DuckDB query                                               #
    # ------------------------------------------------------------------ #

    def _duckdb_conn(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(":memory:")
        con.execute(f"SET threads TO {self._duckdb_threads}")
        return con

    def _query_candidates(self, user_id: int) -> pd.DataFrame:
        """Read candidate rows for user_id from Parquet via DuckDB."""
        con = self._duckdb_conn()
        try:
            df = con.execute(
                "SELECT * FROM read_parquet(?) WHERE user_id = ?",
                [str(self._candidates_path), user_id],
            ).df()
        finally:
            con.close()
        return df

    def _load_known_users(self) -> frozenset[int]:
        """Read distinct user_ids from candidates parquet via DuckDB."""
        con = self._duckdb_conn()
        try:
            rows = con.execute(
                "SELECT DISTINCT user_id FROM read_parquet(?)",
                [str(self._candidates_path)],
            ).fetchall()
        finally:
            con.close()
        return frozenset(int(r[0]) for r in rows)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def score_user(self, user_id: int, k: int) -> tuple[bool, list[dict]]:
        """
        Score candidates for user_id with the PRD LightGBM model.

        Returns:
            (is_cold, recommendations)
            is_cold=True means user was not in candidate store; popular fallback used.
        """
        if user_id not in self._known_user_ids:
            return True, self._popular_fallback(k)

        user_df = self._query_candidates(user_id)
        if user_df.empty:
            return True, self._popular_fallback(k)

        # Feature matrix — strict column order from feature_cols.json
        missing = [c for c in self._feature_cols if c not in user_df.columns]
        if missing:
            raise RuntimeError(
                f"Candidate store is missing {len(missing)} model feature columns: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
                "Rebuild api/demo_store with: python scripts/build_api_demo_store.py"
            )

        X = user_df[self._feature_cols].fillna(0)
        scores = self._model.predict(X)

        user_df = user_df.copy()
        user_df["_score"] = scores
        user_df = user_df.sort_values("_score", ascending=False).head(k).reset_index(drop=True)

        # Future hit flags
        future_item_ids: frozenset[int] = frozenset()
        if not self._future_events.empty and "user_id" in self._future_events.columns:
            fut_user = self._future_events[self._future_events["user_id"] == user_id]
            future_item_ids = frozenset(fut_user["item_id"].astype(int).tolist())

        # Build response
        recs: list[dict] = []
        for rank, (_, row) in enumerate(user_df.iterrows(), 1):
            iid = int(row["item_id"])
            meta = self._item_meta_idx.loc[iid] if iid in self._item_meta_idx.index else None

            sources = [
                label
                for flag, label in _SOURCE_FLAG_LABELS.items()
                if bool(row.get(flag, False))
            ]

            recs.append({
                "rank": rank,
                "item_id": iid,
                "score": round(float(row["_score"]), 6),
                "title": _safe_str(meta, "title") if meta is not None else None,
                "brand": _safe_str(meta, "brand") if meta is not None else None,
                "item_type": _safe_str(meta, "item_type") if meta is not None else None,
                "category_name": _safe_str(meta, "category_name") if meta is not None else None,
                "category_path": _safe_str(meta, "category_path") if meta is not None else None,
                "sources": sources,
                "future_hit": iid in future_item_ids,
                "is_repeat": bool(row.get("is_repeat_history_candidate", False)),
            })

        return False, recs

    def get_user_history(self, user_id: int, limit: int = 20) -> list[dict]:
        if self._user_history.empty:
            return []
        h = self._user_history[self._user_history["user_id"] == user_id]
        if "timestamp" in h.columns:
            h = h.sort_values("timestamp", ascending=False)
        h = h.head(limit)
        return [
            {
                "item_id": int(r["item_id"]),
                "timestamp": str(r.get("timestamp", "")),
                "action_type": str(r.get("action_type", "")),
                "title": str(r["title"]) if "title" in r and pd.notna(r["title"]) else None,
                "category_name": (
                    str(r["category_name"])
                    if "category_name" in r and pd.notna(r["category_name"])
                    else None
                ),
            }
            for _, r in h.iterrows()
        ]

    def get_future_events(self, user_id: int) -> list[dict]:
        if self._future_events.empty:
            return []
        f = self._future_events[self._future_events["user_id"] == user_id]
        if "timestamp" in f.columns:
            f = f.sort_values("timestamp")
        return [
            {
                "item_id": int(r["item_id"]),
                "timestamp": str(r.get("timestamp", "")),
                "action_type": str(r.get("action_type", "")),
                "title": str(r["title"]) if "title" in r and pd.notna(r["title"]) else None,
                "category_name": (
                    str(r["category_name"])
                    if "category_name" in r and pd.notna(r["category_name"])
                    else None
                ),
            }
            for _, r in f.iterrows()
        ]

    def get_demo_users(
        self, limit: int = 20, min_hits: int | None = 1
    ) -> list[dict]:
        """Return demo users sorted by label_recall_at_20 desc."""
        con = self._duckdb_conn()
        try:
            users_df = con.execute(
                "SELECT DISTINCT user_id FROM read_parquet(?)",
                [str(self._candidates_path)],
            ).df()
        finally:
            con.close()

        user_ids = users_df["user_id"].astype(int).tolist()

        rows: list[dict] = []
        for uid in user_ids:
            _, recs = self.score_user(uid, k=20)
            rec_item_ids = {r["item_id"] for r in recs}

            # Candidate store: label-based metrics
            cand_df = self._query_candidates(uid)
            cand_count = len(cand_df)
            label_items: set[int] = set()
            if "label" in cand_df.columns:
                label_items = set(
                    cand_df[cand_df["label"] > 0]["item_id"].astype(int).tolist()
                )
            pos_count = len(label_items)
            label_hits_20 = len(rec_item_ids & label_items)
            label_recall_20 = label_hits_20 / max(pos_count, 1)

            # Future-events-based metrics (hit = item appeared in any future event)
            future_hits_20 = sum(1 for r in recs if r["future_hit"])
            future_hit_rate_20 = future_hits_20 / max(len(recs), 1)

            # History / future event counts
            hist_count = 0
            if not self._user_history.empty:
                hist_count = int((self._user_history["user_id"] == uid).sum())
            fut_count = 0
            if not self._future_events.empty:
                fut_count = int((self._future_events["user_id"] == uid).sum())

            # Filter: min_hits applies to label_hits_at_20
            if min_hits is not None and label_hits_20 < min_hits:
                continue

            rows.append({
                "user_id": uid,
                "candidate_count": cand_count,
                "positive_count": pos_count,
                "future_event_count": fut_count,
                "history_count": hist_count,
                "label_hits_at_20": label_hits_20,
                "label_recall_at_20": round(label_recall_20, 4),
                "future_hits_at_20": future_hits_20,
                "future_hit_rate_at_20": round(future_hit_rate_20, 4),
            })

        rows.sort(key=lambda x: x["label_recall_at_20"], reverse=True)
        return rows[:limit]

    def get_demo_user_full(self, user_id: int, k: int = 20, history_limit: int = 20) -> dict:
        """
        Rich per-user demo: history + recommendations + bought analysis.
        """
        is_cold, recs = self.score_user(user_id, k=k)
        history = self.get_user_history(user_id, limit=history_limit)
        future = self.get_future_events(user_id)

        rec_item_ids = {r["item_id"] for r in recs}
        future_purchase_ids = {
            e["item_id"]
            for e in future
            if e.get("action_type", "").lower() in _PURCHASE_ACTION_TYPES
        }

        bought_from_recs = sorted(rec_item_ids & future_purchase_ids)
        bought_not_rec = sorted(future_purchase_ids - rec_item_ids)

        hits_at_k = len(bought_from_recs)
        recall_at_k = hits_at_k / max(len(future_purchase_ids), 1)

        # Top categories from history
        top_categories: list[str] = []
        if history:
            from collections import Counter
            cat_counts = Counter(
                e["category_name"] for e in history if e.get("category_name")
            )
            top_categories = [cat for cat, _ in cat_counts.most_common(5)]

        explanation = (
            f"User has {len(history)} tracked interactions "
            f"and {len(future)} future events "
            f"(including {len(future_purchase_ids)} purchase-like actions). "
            f"Model is {'cold-start popular-fallback' if is_cold else 'PRD LightGBM ranker'}. "
            f"{hits_at_k} of top-{k} recommendations were later purchased."
        )

        return {
            "user_id": user_id,
            "history": history,
            "top_categories": top_categories,
            "recommendations": recs,
            "bought_from_recommendations": bought_from_recs,
            "bought_not_recommended": bought_not_rec,
            "hits_at_k": hits_at_k,
            "recall_at_k": round(recall_at_k, 4),
            "explanation": explanation,
        }

    def get_model_info(self) -> dict:
        info = dict(self._model_info)
        info.setdefault("model_name", "2stage_reranker_v12_repeat")
        info.setdefault("feature_count", len(self._feature_cols))
        info.setdefault("candidate_store", str(self._candidates_path))
        info["artifact_status"] = self.health_checks()
        return info

    def health_checks(self) -> dict[str, str]:
        checks: dict[str, str] = {}

        # Ranker model
        checks["ranker_model"] = "ok" if self._model is not None else "missing"

        # Feature cols
        checks["feature_cols"] = "ok" if self._feature_cols else "missing"

        # Candidate store
        if self._candidates_path.exists():
            checks["candidate_store"] = "ok"
        else:
            checks["candidate_store"] = f"missing: {self._candidates_path}"

        # Item metadata
        checks["item_metadata"] = "ok" if not self._item_meta.empty else "missing"

        # User history
        checks["user_history"] = "ok" if not self._user_history.empty else "missing"

        # Future events
        checks["future_events"] = "ok" if not self._future_events.empty else "missing"

        return checks

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _popular_fallback(self, k: int) -> list[dict]:
        if self._popular_items.empty:
            return []
        top = self._popular_items.head(k)
        return [
            {
                "rank": i + 1,
                "item_id": int(row["item_id"]),
                "score": float(row.get("score", 0.0)),
                "title": None,
                "brand": None,
                "item_type": None,
                "category_name": None,
                "category_path": None,
                "sources": ["top_popular"],
                "future_hit": False,
                "is_repeat": False,
            }
            for i, (_, row) in enumerate(top.iterrows())
        ]


def _safe_str(series_or_row: Any, key: str) -> str | None:
    try:
        v = series_or_row[key]
        return str(v) if pd.notna(v) else None
    except (KeyError, TypeError):
        return None
