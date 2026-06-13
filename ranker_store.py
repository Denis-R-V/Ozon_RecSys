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
from collections import Counter
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

_PURCHASE_ACTION_TYPES = frozenset({"order", "purchase", "buy"})
_NEGATIVE_ACTION_TYPES = frozenset({"unfavorite", "remove"})

# Higher = stronger user intent signal (used for future-event annotation)
_ACTION_STRENGTH: dict[str, int] = {
    "order":            7,
    "purchase":         7,
    "buy":              7,
    "to_cart":          6,
    "favorite":         5,
    "review_view":      3,
    "view_description": 2,
    "page_view":        1,
    "remove":           0,
}

# Priority for selecting strongest_history_action from pre-cutoff history.
# Negative signals (unfavorite/remove) get low but non-zero priority so they
# still appear as strongest_history_action when no positive action exists.
# The UI suppresses unfavorite/remove from display — but is_repeat stays True.
_HISTORY_ACTION_PRIORITY: dict[str, int] = {
    "order":            100,
    "purchase":         100,
    "buy":              100,
    "to_cart":           80,
    "favorite":          70,
    "review_view":       60,
    "view_description":  50,
    "page_view":         40,
    "unfavorite":        10,
    "remove":            10,
}


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
        split_info: dict | None = None,
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
        self._split_info: dict = split_info or {}
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

        # Split info JSON (optional — describes data sources and temporal windows)
        split_info: dict = {}
        split_info_path = candidates_path.parent / "split_info.json"
        if split_info_path.exists():
            split_info = json.loads(split_info_path.read_text(encoding="utf-8"))
            log.info("split_info.json loaded")
        else:
            # Fallback: hard-coded defaults based on known build_api_demo_store.py sources
            split_info = {
                "candidates_source": "val_eval",
                "history_source": "train_tracker",
                "future_events_source": "test_tracker_08",
                "label_definition": "val_target_window_positive",
                "ground_truth_note": (
                    "MISMATCH: label (val period) != future_events (test period 08). "
                    "Use label_hits_at_k for metrics consistent with /demo/users sidebar."
                ),
            }
            log.warning("split_info.json not found; using inferred defaults")

        store = cls(
            model=model,
            feature_cols=feature_cols,
            candidates_path=candidates_path,
            item_meta=item_meta,
            user_history=user_history,
            future_events=future_events,
            popular_items=popular_items,
            model_info=model_info,
            split_info=split_info,
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

        has_label_col = "label" in user_df.columns

        # Future hit flags — any interaction vs real purchase only
        future_item_ids: frozenset[int] = frozenset()
        purchase_item_ids: frozenset[int] = frozenset()
        if not self._future_events.empty and "user_id" in self._future_events.columns:
            fut_user = self._future_events[self._future_events["user_id"] == user_id]
            future_item_ids = frozenset(fut_user["item_id"].astype(int).tolist())
            if "action_type" in fut_user.columns:
                _pur = fut_user[
                    fut_user["action_type"].str.lower().isin(_PURCHASE_ACTION_TYPES)
                ]
                purchase_item_ids = frozenset(_pur["item_id"].astype(int).tolist())

        # Single source of truth for all history fields: event count, action counts,
        # strongest action, and is_repeat.  Never derives is_repeat from candidate
        # source flags (repeat_history / is_repeat_history_candidate).
        _history_by_item = self._build_history_by_item(user_id)

        # Build response
        recs: list[dict] = []
        for rank, (_, row) in enumerate(user_df.iterrows(), 1):
            iid = int(row["item_id"])
            meta = self._item_meta_idx.loc[iid] if iid in self._item_meta_idx.index else None

            sources = [
                lbl
                for flag, lbl in _SOURCE_FLAG_LABELS.items()
                if bool(row.get(flag, False))
            ]

            # Val-period GT: read label value directly from this candidate row.
            # This is the single source of truth — consistent with /demo/users sidebar.
            label_value = 0
            if has_label_col:
                try:
                    raw = row["label"]
                    label_value = int(raw) if pd.notna(raw) else 0
                except Exception:
                    label_value = 0
            val_gt_hit = label_value > 0

            _item_hist       = _history_by_item.get(iid, {})
            _hist_evt_count  = _item_hist.get("history_event_count", 0)
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
                "label": label_value,
                "val_gt_hit": val_gt_hit,
                "label_hit": val_gt_hit,
                "future_hit": iid in future_item_ids,
                "future_interaction_hit": iid in future_item_ids,
                "bought_after": iid in purchase_item_ids,
                # History fields — single source of truth from _build_history_by_item.
                "is_repeat":               _hist_evt_count > 0,
                "history_event_count":     _hist_evt_count,
                "history_action_counts":   _item_hist.get("history_action_counts", {}),
                "strongest_history_action": _item_hist.get("strongest_history_action"),
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
        """Return demo users sorted by test-period purchases.

        Batch approach: ONE DuckDB query + ONE LightGBM predict for all users
        (~2 s for 500 users vs. 30+ s with the old per-user loop).
        """
        # ── 1. Load only needed columns (parquet projection = 14x faster) ────
        try:
            import pyarrow.parquet as pq
            _pq_cols = set(pq.read_schema(str(self._candidates_path)).names)
        except Exception:
            _pq_cols = None  # fall back to loading all cols

        _base = ["user_id", "item_id"]
        if _pq_cols is not None:
            _label = ["label"] if "label" in _pq_cols else []
            _feats = [c for c in self._feature_cols if c in _pq_cols]
            all_cands = pd.read_parquet(
                self._candidates_path, columns=_base + _label + _feats
            )
        else:
            # Fallback: full read via DuckDB (also correct, just slower)
            con = self._duckdb_conn()
            try:
                all_cands = con.execute(
                    "SELECT * FROM read_parquet(?)", [str(self._candidates_path)]
                ).df()
            finally:
                con.close()

        if all_cands.empty:
            return []

        # ── 2. Score ALL candidates in ONE LightGBM call ──────────────────
        missing = [c for c in self._feature_cols if c not in all_cands.columns]
        if missing:
            log.warning("get_demo_users: %d feature cols missing, zeroing", len(missing))
            for c in missing:
                all_cands[c] = 0.0

        X = all_cands[self._feature_cols].fillna(0)  # keep DataFrame — model needs feature names
        all_cands = all_cands.copy()
        all_cands["_score"] = self._model.predict(X)

        # ── 3. Pre-aggregate future events per user (vectorised) ──────────
        if not self._future_events.empty and "user_id" in self._future_events.columns:
            fut_all = (
                self._future_events.groupby("user_id")["item_id"]
                .apply(frozenset)
                .to_dict()
            )
            if "action_type" in self._future_events.columns:
                pur_df = self._future_events[
                    self._future_events["action_type"].str.lower().isin(_PURCHASE_ACTION_TYPES)
                ]
                pur_all    = pur_df.groupby("user_id")["item_id"].apply(frozenset).to_dict()
                pur_counts = pur_df.groupby("user_id").size().to_dict()
            else:
                pur_all = {}
                pur_counts = {}
            fut_counts = self._future_events.groupby("user_id").size().to_dict()
        else:
            fut_all = {}
            pur_all = {}
            fut_counts = {}
            pur_counts = {}

        # ── 4. Pre-aggregate history counts ───────────────────────────────
        if not self._user_history.empty and "user_id" in self._user_history.columns:
            hist_counts = self._user_history.groupby("user_id").size().to_dict()
        else:
            hist_counts = {}

        # ── 5. Per-user metrics via pandas groupby (no DuckDB per user) ───
        rows: list[dict] = []
        for uid, user_df in all_cands.groupby("user_id"):
            uid = int(uid)
            user_df = user_df.sort_values("_score", ascending=False).reset_index(drop=True)

            top100_items = frozenset(user_df.head(100)["item_id"].astype(int))
            top20_items  = frozenset(user_df.head(20)["item_id"].astype(int))

            fut_u = fut_all.get(uid, frozenset())
            pur_u = pur_all.get(uid, frozenset())

            if "label" in user_df.columns:
                label_items = frozenset(
                    user_df.loc[user_df["label"] > 0, "item_id"].astype(int)
                )
            else:
                label_items = frozenset()
            pos_count = len(label_items)

            purchase_from_recs_20  = len(top20_items  & pur_u)
            purchase_from_recs_100 = len(top100_items & pur_u)
            future_hits_20         = len(top20_items  & fut_u)
            label_hits_20  = len(top20_items  & label_items)
            label_hits_100 = len(top100_items & label_items)

            if min_hits is not None and future_hits_20 < min_hits:
                continue

            rows.append({
                "user_id":               uid,
                "candidate_count":       len(user_df),
                "positive_count":        pos_count,
                "future_event_count":    fut_counts.get(uid, 0),
                "purchase_count":        pur_counts.get(uid, 0),
                "purchase_from_recs_20":  purchase_from_recs_20,
                "purchase_from_recs_100": purchase_from_recs_100,
                "history_count":         hist_counts.get(uid, 0),
                "label_hits_at_20":      label_hits_20,
                "label_hits_at_100":     label_hits_100,
                "label_recall_at_20":    round(label_hits_20 / max(pos_count, 1), 4),
                "future_hits_at_20":     future_hits_20,
                "future_hit_rate_at_20": round(future_hits_20 / max(20, 1), 4),
            })

        rows.sort(
            key=lambda x: (
                x["purchase_from_recs_20"], x["purchase_from_recs_100"],
                x["future_hits_at_20"],     x["purchase_count"]
            ),
            reverse=True,
        )
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
            f"(including {len(future_purchase_ids)} real purchases). "
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

    # ------------------------------------------------------------------ #
    # Validation metrics helper (single source of truth for both          #
    # /demo/users and /demo/story)                                        #
    # ------------------------------------------------------------------ #

    def get_validation_metrics_for_user(
        self, user_id: int, recommendations: list[dict]
    ) -> dict:
        """
        Compute val-period GT metrics by intersecting top-k rec items with
        label=1 items from the candidate store.

        This is the ONLY place label metrics are computed.  Both get_demo_users()
        and get_demo_story() must call this method so sidebar and story always agree.

        Returns:
            positive_count        - total label=1 items in full candidate set
            label_hit_item_ids    - sorted list[int] of intersection items
            label_hits_at_k       - len(intersection)
            label_recall_at_k     - hits / positive_count  (0.0 if positive_count==0)
            candidate_count       - total candidate rows for this user
            has_label_column      - bool, False if parquet has no label column
        """
        rec_item_ids: set[int] = {r["item_id"] for r in recommendations}

        label_items: set[int] = set()
        cand_count = 0
        has_label_col = False
        try:
            cand_df = self._query_candidates(user_id)
            cand_count = len(cand_df)
            if not cand_df.empty and "label" in cand_df.columns:
                has_label_col = True
                label_items = set(
                    cand_df[cand_df["label"] > 0]["item_id"].astype(int).tolist()
                )
        except Exception as exc:
            log.warning("get_validation_metrics_for_user: _query_candidates failed: %s", exc)

        hit_set = label_items & rec_item_ids
        label_hit_item_ids: list[int] = sorted(hit_set)
        positive_count = len(label_items)
        label_hits_at_k = len(label_hit_item_ids)
        label_recall_at_k = label_hits_at_k / positive_count if positive_count > 0 else 0.0

        return {
            "positive_count": positive_count,
            "label_hit_item_ids": label_hit_item_ids,
            "label_hits_at_k": label_hits_at_k,
            "label_recall_at_k": round(label_recall_at_k, 4),
            "candidate_count": cand_count,
            "has_label_column": has_label_col,
        }

    # ------------------------------------------------------------------ #
    # Demo story                                                           #
    # ------------------------------------------------------------------ #

    def get_demo_story(
        self,
        user_id: int,
        k: int = 20,
        history_limit: int = 20,
        analysis_k: int = 100,
    ) -> dict:
        """
        Aggregated view-model for Streamlit demo.
        Returns one JSON with everything needed to render the full user story:
        profile, history, recommendations, what-happened-later, charts.

        k           — number of cards to display in the UI.
        analysis_k  — candidate pool size for bought/interaction analysis.
                      Always >= k. Buying/interaction rates computed over this pool.
        """
        _score_k = max(k, analysis_k)
        is_cold, recs_all = self.score_user(user_id, k=_score_k)
        # Display slice (UI table)
        history = self.get_user_history(user_id, limit=history_limit)
        # Full history for badge computation — history_limit may be only last 20 events,
        # but strongest_history_action must reflect the entire known history
        _hist_full = self.get_user_history(user_id, limit=5000)
        future_events = self.get_future_events(user_id)

        # Single source of truth: history fields for all recommended items.
        # Reads self._user_history (pre-cutoff only); never touches future_events.
        _history_by_item = self._build_history_by_item(user_id)
        for _r in recs_all:
            _item_hist  = _history_by_item.get(_r["item_id"], {})
            _evt_count  = _item_hist.get("history_event_count", 0)
            _r["history_event_count"]     = _evt_count
            _r["history_action_counts"]   = _item_hist.get("history_action_counts", {})
            _r["strongest_history_action"] = _item_hist.get("strongest_history_action")
            _r["is_repeat"]               = _evt_count > 0

        # Invariant: strongest_history_action must not exist without history events
        for _r in recs_all:
            if _r.get("strongest_history_action") and _r.get("history_event_count", 0) == 0:
                log.warning(
                    "BAD_HISTORY_INVARIANT user=%s item=%s sha=%s sources=%s",
                    user_id, _r["item_id"],
                    _r["strongest_history_action"], _r.get("sources"),
                )

        # Display set (top-k cards); analysis set (top-analysis_k) keeps full recs_all
        recs = recs_all[:k]

        rec_item_ids = {r["item_id"] for r in recs_all}   # analysis pool
        rec_by_id    = {r["item_id"]: r for r in recs_all}

        future_purchase_events = [
            e for e in future_events
            if e.get("action_type", "").lower() in _PURCHASE_ACTION_TYPES
        ]
        future_purchase_ids = {e["item_id"] for e in future_purchase_events}

        # Group future events by item_id for O(1) per-rec lookup
        fut_by_item: dict[int, list[dict]] = {}
        for _fe in future_events:
            fut_by_item.setdefault(_fe["item_id"], []).append(_fe)

        # Annotate recs with later interaction info (applied to full analysis pool)
        for _r in recs_all:
            _evts     = fut_by_item.get(_r["item_id"], [])
            # Exclude negative/removal signals from positive interaction logic
            _pos_evts = [
                e for e in _evts
                if (e.get("action_type") or "").lower() not in _NEGATIVE_ACTION_TYPES
            ]
            if _pos_evts:
                _best = max(
                    _pos_evts,
                    key=lambda e: _ACTION_STRENGTH.get((e.get("action_type") or "").lower(), 0),
                )
                _r["strongest_later_action"]    = _best.get("action_type")
                _r["strongest_later_action_ts"] = _best.get("timestamp")
            else:
                _r["strongest_later_action"]    = None
                _r["strongest_later_action_ts"] = None
            _r["bought_after"]      = _r["item_id"] in future_purchase_ids
            _r["later_interaction"] = bool(_pos_evts)

        # Enriched bought_from_recommendations (sorted by rank)
        bought_from_recs = []
        for iid in sorted(rec_item_ids & future_purchase_ids, key=lambda x: rec_by_id[x]["rank"]):
            rec = rec_by_id[iid]
            bought_from_recs.append({
                "item_id": iid,
                "title": rec.get("title"),
                "brand": rec.get("brand"),
                "category_name": rec.get("category_name"),
                "rank_in_recommendations": rec["rank"],
                "score": rec["score"],
                "action_type": rec.get("strongest_later_action"),
                "timestamp": rec.get("strongest_later_action_ts"),
            })

        # Enriched bought_not_recommended (deduplicated by item_id)
        seen_not_rec: set[int] = set()
        bought_not_rec = []
        for e in future_purchase_events:
            iid = e["item_id"]
            if iid in rec_item_ids or iid in seen_not_rec:
                continue
            seen_not_rec.add(iid)
            meta = self._item_meta_idx.loc[iid] if iid in self._item_meta_idx.index else None
            bought_not_rec.append({
                "item_id": iid,
                "title": e.get("title") or (_safe_str(meta, "title") if meta is not None else None),
                "brand": _safe_str(meta, "brand") if meta is not None else None,
                "category_name": (
                    e.get("category_name")
                    or (_safe_str(meta, "category_name") if meta is not None else None)
                ),
                "action_type": e.get("action_type"),
                "timestamp": e.get("timestamp"),
            })

        # Enrich not-recommended purchases with context flags (top-k display pool + full history)
        _topk_categories = {r.get("category_name") for r in recs if r.get("category_name")}
        _topk_brands     = {r.get("brand") for r in recs if r.get("brand")}
        _hist_categories = {e.get("category_name") for e in _hist_full if e.get("category_name")}
        _hist_brands     = {e.get("brand") for e in _hist_full if e.get("brand")}
        for _nr in bought_not_rec:
            _nr["category_was_in_topk"]    = _nr.get("category_name") in _topk_categories
            _nr["brand_was_in_topk"]       = _nr.get("brand") in _topk_brands
            _nr["category_was_in_history"] = _nr.get("category_name") in _hist_categories
            _nr["brand_was_in_history"]    = _nr.get("brand") in _hist_brands

        bought_from_recs_count = len(bought_from_recs)
        purchase_recall_at_k = bought_from_recs_count / max(len(future_purchase_ids), 1)
        future_interaction_hits_at_k = sum(1 for r in recs_all if r.get("future_interaction_hit"))
        future_interaction_rate_at_k = future_interaction_hits_at_k / max(len(recs_all), 1)

        # Val-period GT — single shared helper, same logic as /demo/users sidebar.
        # Patches val_gt_hit/label_hit on the display set only.
        vm = self.get_validation_metrics_for_user(user_id, recs_all)
        hit_set: set[int] = set(vm["label_hit_item_ids"])
        for r in recs_all:
            r["val_gt_hit"] = r["item_id"] in hit_set
            r["label_hit"] = r["val_gt_hit"]

        # Non-purchase interactions from recommendations (positive actions only, no unfavorite/remove)
        later_int_from_recs = [
            {
                "item_id": r["item_id"],
                "title": r.get("title"),
                "brand": r.get("brand"),
                "category_name": r.get("category_name"),
                "rank_in_recommendations": r["rank"],
                "score": r["score"],
                "action_type": r.get("strongest_later_action"),
                "timestamp": r.get("strongest_later_action_ts"),
            }
            for r in recs_all
            if r.get("later_interaction") and not r.get("bought_after")
        ]

        # Enrich history items with brand/item_type from metadata
        for _h in history:
            _hid   = _h["item_id"]
            _hmeta = self._item_meta_idx.loc[_hid] if _hid in self._item_meta_idx.index else None
            if _hmeta is not None:
                if not _h.get("brand"):
                    _h["brand"]     = _safe_str(_hmeta, "brand")
                if not _h.get("item_type"):
                    _h["item_type"] = _safe_str(_hmeta, "item_type")

        top_cat_hist: Counter = Counter(
            e["category_name"] for e in history if e.get("category_name")
        )
        top_categories = [cat for cat, _ in top_cat_hist.most_common(5)]

        # Recommended items with any later interaction — from full analysis pool
        recommended_with_later = [
            {
                "rank":             r["rank"],
                "item_id":          r["item_id"],
                "title":            r.get("title"),
                "brand":            r.get("brand"),
                "category_name":    r.get("category_name"),
                "score":            r["score"],
                "strongest_action": r.get("strongest_later_action"),
                "timestamp":        r.get("strongest_later_action_ts"),
                "is_purchase":      r.get("bought_after", False),
            }
            for r in recs_all
            if r.get("later_interaction")
        ]

        # Summary block — counts over full analysis pool
        _all_strongest = [r["strongest_later_action"] for r in recs_all if r.get("strongest_later_action")]
        _best_overall  = (
            max(_all_strongest, key=lambda a: _ACTION_STRENGTH.get(a.lower(), 0))
            if _all_strongest else None
        )
        summary = {
            "strongest_recommended_action":         _best_overall,
            "recommended_later_interactions_count": sum(1 for r in recs_all if r.get("later_interaction")),
            "recommended_bought_count":      sum(1 for r in recs_all if r.get("bought_after")),
            "not_recommended_bought_count":  len(bought_not_rec),
        }

        # Analysis block — over full analysis pool
        _hist_cat_cnt   = Counter(e.get("category_name") for e in history if e.get("category_name"))
        _rec_cat_cnt    = Counter(r.get("category_name") for r in recs_all if r.get("category_name"))
        _hist_brand_cnt = Counter(e["brand"] for e in history if e.get("brand"))
        _n_seen         = sum(1 for r in recs_all if r.get("is_repeat"))
        analysis_block = {
            "history_top_categories":    [{"category": c, "count": n} for c, n in _hist_cat_cnt.most_common(10)],
            "recommendation_categories": [{"category": c, "count": n} for c, n in _rec_cat_cnt.most_common(10)],
            "history_top_brands":        [{"brand": b, "count": n} for b, n in _hist_brand_cnt.most_common(5)],
            "category_overlap":          sorted(set(_hist_cat_cnt) & set(_rec_cat_cnt)),
            "category_miss":             sorted(set(_hist_cat_cnt) - set(_rec_cat_cnt)),
            "source_distribution":       self._source_distribution(recs_all),
            "seen_before_count":         _n_seen,
            "new_items_count":           len(recs_all) - _n_seen,
        }

        # Lightweight model section (no artifact_status, no health_checks)
        mi = self._model_info
        model_section = {
            "model_name": mi.get("model_name", "prd_reranker"),
            "prd_run_id": mi.get("prd_run_id"),
            "feature_count": mi.get("feature_count", len(self._feature_cols)),
            "metrics": mi.get("metrics", {}),
        }

        explanation = (
            f"User has {len(history)} tracked interactions "
            f"and {len(future_events)} future events "
            f"({len(future_purchase_ids)} real purchases, "
            f"{future_interaction_hits_at_k} any interactions in top-{k}). "
            f"Model is {'cold-start popular-fallback' if is_cold else 'PRD LightGBM ranker'}. "
            f"{bought_from_recs_count} of top-{k} recommendations were later purchased."
        )

        return {
            "user_id": user_id,
            "model": model_section,
            "profile": {
                "is_cold": is_cold,
                "history_count": len(history),
                "future_event_count": len(future_events),
                "future_purchase_count": len(future_purchase_ids),
                "top_categories": top_categories,
            },
            "history": history,
            "display_k": k,
            "analysis_k": _score_k,
            "recommendations": recs,          # top-k display set (cards)
            "recommendations_analysis": recs_all,  # full analysis pool
            "future": {
                "later_interactions_from_recommendations": later_int_from_recs,
                "bought_from_recommendations": bought_from_recs,
                "bought_not_recommended": bought_not_rec,
            },
            # "validation" is the authoritative val-period GT block.
            # Streamlit sidebar and story must read from here — not from "metrics".
            "validation": {
                "positive_count": vm["positive_count"],
                "label_hit_item_ids": vm["label_hit_item_ids"],
                "label_hits_at_k": vm["label_hits_at_k"],
                "label_recall_at_k": vm["label_recall_at_k"],
            },
            # "metrics" holds test-period event counts only.
            "metrics": {
                "bought_from_recs_count": bought_from_recs_count,
                "purchase_recall_at_k": round(purchase_recall_at_k, 4),
                "future_interaction_hits_at_k": future_interaction_hits_at_k,
                "future_interaction_rate_at_k": round(future_interaction_rate_at_k, 4),
                "strongest_recommended_action": _best_overall,
            },
            "summary": summary,
            "after_recommendation": {
                "recommended_with_later_actions": recommended_with_later,
                "recommended_bought":      bought_from_recs,
                "not_recommended_bought":  bought_not_rec,
            },
            "analysis": analysis_block,
            "split_info": self._split_info,
            "charts": {
                "source_distribution": self._source_distribution(recs_all),
                "history_category_distribution": self._category_distribution(history),
                "recommendation_category_distribution": self._category_distribution(recs_all),
                "cumulative_hits_by_rank": self._cumulative_hits_by_rank(recs),
            },
            "explanation": explanation,
        }

    def _source_distribution(self, recommendations: list[dict]) -> list[dict]:
        counter: Counter = Counter()
        for r in recommendations:
            for src in r.get("sources", []):
                counter[src] += 1
        return [{"source": s, "count": c} for s, c in counter.most_common()]

    def _category_distribution(
        self, items: list[dict], field: str = "category_name"
    ) -> list[dict]:
        cats = [item.get(field) for item in items if item.get(field)]
        counter: Counter = Counter(cats)
        return [{"category": c, "count": n} for c, n in counter.most_common()]

    def _cumulative_hits_by_rank(self, recommendations: list[dict]) -> list[dict]:
        """Counts val-period GT hits (val_gt_hit) cumulatively by rank."""
        cumulative = 0
        result = []
        for r in recommendations:
            if r.get("val_gt_hit", False):
                cumulative += 1
            result.append({"rank": r["rank"], "cumulative_hits": cumulative})
        return result

    def _enrich_item_ids(self, item_ids: list[int]) -> list[dict]:
        result = []
        for iid in item_ids:
            meta = self._item_meta_idx.loc[iid] if iid in self._item_meta_idx.index else None
            result.append({
                "item_id": iid,
                "title": _safe_str(meta, "title") if meta is not None else None,
                "brand": _safe_str(meta, "brand") if meta is not None else None,
                "category_name": _safe_str(meta, "category_name") if meta is not None else None,
            })
        return result

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

    def _build_history_by_item(self, user_id: int) -> dict[int, dict]:
        """
        Single source of truth for pre-cutoff history per item.

        Returns {item_id: {history_event_count, history_action_counts, strongest_history_action}}.
        Only reads self._user_history (pre-test-period events).  Never touches future_events.
        """
        if self._user_history.empty or "user_id" not in self._user_history.columns:
            return {}
        hist_u = self._user_history[self._user_history["user_id"] == user_id]
        result: dict[int, dict] = {}
        for _, row in hist_u.iterrows():
            iid   = int(row["item_id"])
            atype = str(row.get("action_type") or "").lower()
            if iid not in result:
                result[iid] = {"history_event_count": 0, "history_action_counts": {}}
            result[iid]["history_event_count"] += 1
            if atype:
                counts = result[iid]["history_action_counts"]
                counts[atype] = counts.get(atype, 0) + 1
        for info in result.values():
            counts = info["history_action_counts"]
            info["strongest_history_action"] = (
                max(counts, key=lambda a: _HISTORY_ACTION_PRIORITY.get(a, 0))
                if counts else None
            )
        return result

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
                "label": 0,
                "val_gt_hit": False,
                "label_hit": False,
                "future_hit": False,
                "future_interaction_hit": False,
                "bought_after": False,
                "is_repeat": False,
                "history_event_count": 0,
                "history_action_counts": {},
                "strongest_history_action": None,
            }
            for i, (_, row) in enumerate(top.iterrows())
        ]


def _safe_str(series_or_row: Any, key: str) -> str | None:
    try:
        v = series_or_row[key]
        return str(v) if pd.notna(v) else None
    except (KeyError, TypeError):
        return None
