"""
Ozon Apparel RecSys — Streamlit Demo
User story: history → PRD recommendations → later interactions
"""
from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

try:
    import plotly.express as px
    _PLOTLY = True
except ImportError:
    _PLOTLY = False


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ozon Apparel RecSys Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────

_SOURCE_COLORS: dict[str, str] = {
    "als":              "#1565c0",
    "clip_hnsw":        "#6a1b9a",
    "clip_faiss":       "#6a1b9a",
    "dssm":             "#2e7d32",
    "sasrec":           "#e65100",
    "bert4rec":         "#c62828",
    "i2i_bm25":         "#4527a0",
    "top_popular":      "#37474f",
    "user_cat_popular": "#455a64",
    "repeat_history":   "#ff6b00",
}

_CAT_PALETTE = [
    "#1565c0", "#6a1b9a", "#2e7d32", "#e65100", "#c62828",
    "#4527a0", "#37474f", "#455a64", "#bf360c", "#006064",
]

_SRC_DESC: dict[str, str] = {
    "als":              "Коллаборативная фильтрация (ALS)",
    "clip_hnsw":        "Визуальное сходство (CLIP+HNSW)",
    "clip_faiss":       "Визуальное сходство (CLIP+FAISS)",
    "dssm":             "Глубокое семантическое сходство (DSSM)",
    "sasrec":           "Последовательная модель (SASRec)",
    "bert4rec":         "Последовательная модель (BERT4Rec)",
    "i2i_bm25":         "Сходство товаров (BM25)",
    "top_popular":      "Глобальная популярность",
    "user_cat_popular": "Популярность по категории пользователя",
    "repeat_history":   "Ранее просмотренные товары",
}

_COL_RU: dict[str, str] = {
    "rank":                    "Ранг",
    "rank_in_recommendations": "Ранг",
    "item_id":                 "ID",
    "title":                   "Товар",
    "brand":                   "Бренд",
    "item_type":               "Тип",
    "category_name":           "Категория",
    "category":                "Категория",
    "action_type":             "Действие",
    "timestamp":               "Время",
    "score":                   "score",
    "seen_before":             "В истории",
    "category_in_history":     "Кат. в истории",
    "brand_in_history":        "Бренд в истории",
    "sources":                 "Источники",
    "interacted_in_test":      "Взаимодействовал (тест)",
    "bought_in_test":          "Куплено (тест)",
    "category_was_in_topk":    "Кат. в топ-k",
    "brand_was_in_topk":       "Бренд в топ-k",
    "category_was_in_history": "Кат. в истории",
    "brand_was_in_history":    "Бренд в истории",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_df(df: "pd.DataFrame") -> "pd.DataFrame":
    df = df.rename(columns=_COL_RU)
    if "Время" in df.columns:
        df["Время"] = df["Время"].astype(str).str.slice(0, 19).replace("nan", "")
    return df




def _api_get(
    base: str, path: str, params: dict | None = None, timeout: int = 30
) -> tuple[dict | list | None, str | None]:
    try:
        r = requests.get(f"{base}{path}", params=params or {}, timeout=timeout)
        try:
            data = r.json()
        except Exception:
            data = None
        return (data, None) if r.ok else (None, f"HTTP {r.status_code}")
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API"
    except Exception as exc:
        return None, str(exc)


def _api_post(
    base: str, path: str, params: dict | None = None, timeout: int = 30
) -> tuple[dict | None, str | None]:
    try:
        r = requests.post(f"{base}{path}", params=params or {}, timeout=timeout)
        try:
            data = r.json()
        except Exception:
            data = None
        return (data, None) if r.ok else (None, f"HTTP {r.status_code}: {r.text[:200]}")
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API"
    except Exception as exc:
        return None, str(exc)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_story(api_url: str, uid: int):
    # Always fetch full analysis depth; k-slider and history filter are applied client-side.
    # Fixed params mean cache is keyed only by (api_url, uid) → k/history sliders never cause misses.
    return _api_get(
        api_url,
        f"/demo/story/{uid}",
        {"k": 100, "history_limit": 500, "analysis_k": 100},
    )


@st.cache_data(ttl=60, show_spinner=False)
def _cached_users(api_url: str):
    return _api_get(api_url, "/demo/users", {"limit": 500, "min_hits": 0})


@st.cache_data(ttl=30, show_spinner=False)
def _cached_health(api_url: str):
    return _api_get(api_url, "/health")


@st.cache_data(ttl=30, show_spinner=False)
def _cached_health_ready(api_url: str):
    return _api_get(api_url, "/health/ready")


@st.cache_data(ttl=3600, show_spinner=False)
def _catalog_categories() -> frozenset[str]:
    """All unique category_names from the full item metadata (256 categories)."""
    import pathlib
    _path = pathlib.Path(__file__).parent / "api" / "demo_store" / "api_item_metadata.parquet"
    if not _path.exists():
        return frozenset()
    _meta = pd.read_parquet(_path, columns=["category_name"])
    return frozenset(_meta["category_name"].dropna().unique().tolist())


@st.cache_data(ttl=60, show_spinner=False)
def _cached_model_info(api_url: str):
    return _api_get(api_url, "/model/info")


def _source_badge(src: str) -> str:
    color = _SOURCE_COLORS.get(src, "#666")
    return (
        f'<span style="background:{color};color:white;padding:1px 4px;'
        f'border-radius:3px;font-size:9px;margin-right:2px;'
        f'white-space:nowrap;display:inline-block">{src}</span>'
    )


def _cat_color(cat: str) -> str:
    return _CAT_PALETTE[abs(hash(cat or "")) % len(_CAT_PALETTE)]


def _render_rec_card(
    rec: dict,
    show_all_sources: bool = False,
    show_hist_action: bool = True,
    show_test_action: bool = True,
) -> None:
    title       = (rec.get("title") or "Unknown item")[:60]
    brand       = rec.get("brand") or ""
    cat         = rec.get("category_name") or ""
    sources     = rec.get("sources", [])
    score       = rec.get("score", 0.0)
    rank        = rec.get("rank", 0)
    pur_after   = rec.get("bought_after", False)
    later_int   = rec.get("later_interaction", False)

    _iid = rec.get("item_id", 0)
    # Backend is the single source of truth: history fields come from _build_history_by_item.
    # Never derive is_repeat from sources[], repeat_history flag, or slider filtering.
    repeat      = rec.get("history_event_count", 0) > 0
    hist_action = rec.get("strongest_history_action") or ""

    test_action = rec.get("strongest_later_action") or ""     # test period

    # Green border for items the user later purchased
    if pur_after:
        border_color = "#00b050"
        border_width = "2px"
        bg           = "#f4fef6"
        title_color  = "#004d1a"
    elif later_int:
        border_color = "#1565c0"
        border_width = "2px"
        bg           = "#f0f4ff"
        title_color  = "#0d3a7a"
    else:
        border_color = "#dde"
        border_width = "1px"
        bg           = "#fafafa"
        title_color  = "#111111"

    _NEGATIVE_HIST = {"unfavorite", "remove"}

    badges = ""
    # Rendering invariants (enforced by backend _build_history_by_item):
    #   1. history_event_count > 0  →  show ↩ в истории
    #   2. strongest_history_action →  show action (unless suppressed by checkbox or negative)
    #   3. action shown             →  ↩ в истории always co-present (guaranteed: SHA implies evt_count>0)
    #   4. repeat_history source    →  does NOT affect ↩ в истории (source provenance only)
    if show_hist_action and hist_action and hist_action not in _NEGATIVE_HIST:
        badges += (
            f'<span title="Действие с товаром до тест-периода" '
            f'style="background:#546e7a;color:white;padding:2px 5px;'
            f'border-radius:3px;font-size:9px;margin-right:3px;cursor:help">'
            f'{hist_action}</span>'
        )
    if repeat:  # history_event_count > 0
        badges += (
            '<span title="Товар есть в истории пользователя (до тест-периода)" '
            'style="background:#e65100;color:white;padding:2px 5px;'
            'border-radius:3px;font-size:9px;margin-right:3px;cursor:help">↩ в истории</span>'
        )
    if show_test_action:
        if pur_after:
            _t_label = test_action if test_action else "order"
            badges += (
                f'<span title="Купил после рекомендации" '
                f'style="background:#00b050;color:white;padding:2px 6px;'
                f'border-radius:3px;font-size:10px;font-weight:600;margin-right:3px;cursor:help">'
                f'→ {_t_label}</span>'
            )
        elif later_int and test_action:
            badges += (
                f'<span title="Взаимодействовал в тест-периоде" '
                f'style="background:#1565c0;color:white;padding:2px 6px;'
                f'border-radius:3px;font-size:10px;margin-right:3px;cursor:help">'
                f'→ {test_action}</span>'
            )

    if show_all_sources:
        src_html = "".join(_source_badge(s) for s in sources)
    else:
        src_html = "".join(_source_badge(s) for s in sources[:2])
        if len(sources) > 2:
            src_html += (
                f'<span style="background:#bbb;color:white;padding:1px 4px;'
                f'border-radius:3px;font-size:9px">+{len(sources)-2}</span>'
            )

    subtitle = f"{brand}  ·  {cat}" if (brand and cat) else (brand or cat)

    st.markdown(
        f"""<div style="border:{border_width} solid {border_color};border-radius:8px;
                        padding:10px 12px 12px 12px;background:{bg};margin-bottom:4px;
                        min-height:155px;box-sizing:border-box;">
  <div style="font-size:10px;color:#888;margin-bottom:2px">#{rank} &nbsp; {score:.3f}</div>
  <div style="font-weight:700;font-size:13px;color:{title_color};line-height:1.35;
              overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;
              -webkit-box-orient:vertical;margin-bottom:2px">{title}</div>
  <div style="font-size:11px;color:#666;white-space:nowrap;overflow:hidden;
              text-overflow:ellipsis;margin-bottom:5px">{subtitle}</div>
  <div style="min-height:26px;margin-bottom:4px">{badges}</div>
  <div style="line-height:1.8">{src_html}</div>
</div>""",
        unsafe_allow_html=True,
    )


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Настройки")
    api_url: str = st.text_input("API URL", value="http://localhost:8000")

    st.subheader("Статус сервиса")
    sc1, sc2 = st.columns(2)
    with sc1:
        live_data, _ = _cached_health(api_url)
        if live_data and live_data.get("status") == "ok":
            st.success("API: ok")
        else:
            st.error("API: fail")
    with sc2:
        ready_data, _ = _cached_health_ready(api_url)
        if ready_data and ready_data.get("status") in ("ok", "degraded"):
            st.success("Модель: ok")
        else:
            st.warning("Модель: fail")

    st.divider()
    st.subheader("Пользователь")

    users_data, users_err = _cached_users(api_url)
    demo_users: list[dict] = users_data or []  # type: ignore[assignment]

    manual_id: str = st.text_input("Ввести user_id вручную", placeholder="например 3737600")

    selected_uid: int | None = None
    if manual_id.strip().isdigit():
        selected_uid = int(manual_id.strip())
    elif demo_users:
        fmt_map = {
            u["user_id"]: (
                f"User {u['user_id']}"
                f"  ·  топ20: {u.get('purchase_from_recs_20', 0)}/{u.get('purchase_count', 0)}"
                f"  ·  топ100: {u.get('purchase_from_recs_100', 0)}/{u.get('purchase_count', 0)}"
            )
            for u in demo_users
        }
        selected_uid = st.selectbox(
            "Демо-пользователи",
            options=[u["user_id"] for u in demo_users],
            format_func=lambda uid: fmt_map.get(uid, str(uid)),
        )
    else:
        st.warning(users_err or "Demo store недоступен")

    st.divider()
    k: int = st.select_slider("Рекомендаций (топ-k)", options=[5, 10, 20, 50, 100], value=20)
    cols_per_row: int = st.select_slider("Карточек в строке", options=[2, 3, 4, 5, 6, 8], value=4)

    _TW_OPTS = ["1 нед.", "2 нед.", "3 нед.", "4 нед.", "1 мес.", "2 мес.", "3 мес."]
    _TW_DAYS = {"1 нед.": 7, "2 нед.": 14, "3 нед.": 21, "4 нед.": 28,
                "1 мес.": 30, "2 мес.": 60, "3 мес.": 90}
    hist_window_label: str = st.select_slider(
        "Горизонт истории",
        options=_TW_OPTS,
        value="2 мес.",
        help="Горизонт отображения панели истории пользователя. Бейджи на карточках (↩ в истории, до: action) отражают полную pre-test историю из backend и не зависят от этого ползунка.",
    )
    hist_badge_days: int = _TW_DAYS[hist_window_label]
    analysis_k = 100  # fixed analysis pool depth, not user-configurable
    history_limit = 500  # always fetch enough; client-side time filter applied

    st.divider()
    show_hist_action: bool = st.checkbox("Показывать действие до рекомендаций", value=True)
    show_test_action: bool = st.checkbox("Показывать действия в тестовом периоде", value=True)
    show_all_sources: bool = st.checkbox("Показывать источники кандидата", value=False)
    with st.expander("Легенда источников", expanded=False):
        for src, color in _SOURCE_COLORS.items():
            st.markdown(
                f'<div style="margin:3px 0">'
                f'<span style="background:{color};color:white;padding:2px 6px;'
                f'border-radius:3px;font-size:10px;margin-right:6px">{src}</span>'
                f'<span style="font-size:11px;color:#aaa">{_SRC_DESC.get(src,"")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if st.button("Обновить", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── load story ────────────────────────────────────────────────────────────────

story: dict | None = None
story_err: str | None = None
if selected_uid is not None:
    story, story_err = _cached_story(api_url, selected_uid)


# ── page header ───────────────────────────────────────────────────────────────

st.title("Ozon Apparel Recommendation Demo")
st.caption(f"Данные из FastAPI-сервиса: {api_url}")
st.divider()

tabs = st.tabs(["Демо", "Анализ", "Система"])


# =============================================================================
# Tab 1: Demo  (test-period user story)
# =============================================================================

with tabs[0]:
    if selected_uid is None:
        st.info("Выберите пользователя в боковой панели")
    elif story is None:
        st.error(f"Ошибка загрузки: {story_err}")
    else:
        profile      = story.get("profile", {})
        summ         = story.get("summary", {})
        _recs_full   = story.get("recommendations", [])   # full 100-item pool from API
        recs         = _recs_full[:k]                      # visible cards (client-side slice)
        _hist_full_r = story.get("history", [])            # up to 500 events loaded

        # Filter history to the selected time window
        _si_ui     = story.get("split_info", {})
        _ts_max    = _si_ui.get("history_ts_max")
        try:
            if _ts_max:
                _cutoff_dt = pd.to_datetime(_ts_max) - pd.Timedelta(days=hist_badge_days)
                history = [e for e in _hist_full_r
                           if pd.to_datetime(e.get("timestamp", "1970-01-01")) >= _cutoff_dt]
            else:
                history = _hist_full_r
        except Exception:
            history = _hist_full_r

        after    = story.get("after_recommendation", {})

        # Map story response fields (future/metrics) → local after/summ dicts
        if not after:
            _fut = story.get("future", {})
            after = {
                "recommended_with_later_actions": _fut.get("later_interactions_from_recommendations", []),
                "recommended_bought":  _fut.get("bought_from_recommendations", []),
                "not_recommended_bought": _fut.get("bought_not_recommended", []),
            }
        if not summ:
            _tm = story.get("metrics", {})
            summ = {
                "strongest_recommended_action":         _tm.get("strongest_recommended_action"),
                "recommended_later_interactions_count": len(after.get("recommended_with_later_actions", [])),
                "recommended_bought_count":             _tm.get("bought_from_recs_count", 0),
                "not_recommended_bought_count":         len(after.get("not_recommended_bought", [])),
            }

        top_cats         = profile.get("top_categories", [])
        is_cold          = profile.get("is_cold", False)
        _actual_analysis_k = story.get("analysis_k", analysis_k)

        # Top-k counts — consistent with what user can visually count on cards
        _bought_topk        = sum(1 for r in recs if r.get("bought_after", False))
        _interactions_topk  = sum(1 for r in recs if r.get("later_interaction", False) and not r.get("bought_after", False))
        # Analysis-pool counts (for one-line summary below header)
        _bought_analysis    = summ.get("recommended_bought_count", 0)
        _bought_outside     = summ.get("not_recommended_bought_count", 0)

        # ── Header ────────────────────────────────────────────────────────────
        st.subheader(f"Пользователь {story['user_id']}")
        if is_cold:
            st.warning("Холодный пользователь — нет кандидатов, показывается популярное.")

        # ── 5 metrics (all computed from visible top-k) ────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        _total_purch = profile.get("future_purchase_count", 0)
        with m1:
            st.metric(
                "Покупок (тест-период)",
                _total_purch,
                help="Реальные покупки пользователя в тест-периоде (order/buy/purchase)",
            )
        with m2:
            st.metric(
                "Рекомендаций",
                len(recs),
                help=f"Карточек показано в топ-{k}",
            )
        with m3:
            st.metric(
                f"Куплено из топ-{k}",
                _bought_topk,
                help=f"Зелёные карточки в топ-{k}: реальные покупки (order/buy/purchase)",
            )
        with m4:
            st.metric(
                f"Взаимодействий (топ-{k})",
                _interactions_topk,
                help=f"Синие карточки в топ-{k}: просмотр/избранное без покупки (unfavorite/remove не учитывается)",
            )
        with m5:
            st.metric(
                f"Вне топ-{_actual_analysis_k}",
                _bought_outside,
                help=f"Покупок товаров, которых НЕ было в топ-{_actual_analysis_k} (модель пропустила)",
            )

        st.caption(
            f"В тестовом периоде пользователь купил {_total_purch} товар(ов). "
            f"Видимый топ-{k} покрывает {_bought_topk} из {_total_purch}, "
            f"расширенный топ-{_actual_analysis_k} покрывает {_bought_analysis} из {_total_purch}. "
            f"Вне топ-{_actual_analysis_k} осталось {_bought_outside} покупок."
        )
        st.divider()

        # ── User history ──────────────────────────────────────────────────────
        st.subheader("История пользователя")
        if history:
            # Category chips
            if top_cats:
                chips = "".join(
                    f'<span style="background:{_cat_color(c)};color:white;padding:3px 9px;'
                    f'border-radius:12px;font-size:11px;margin:2px 3px;'
                    f'display:inline-block">{c}</span>'
                    for c in top_cats[:8]
                )
                st.markdown(chips, unsafe_allow_html=True)

            # Top brands
            _analysis = story.get("analysis", {})
            _top_brands = [b["brand"] for b in _analysis.get("history_top_brands", [])[:5] if b.get("brand")]
            if _top_brands:
                st.caption("Топ бренды: " + "  ·  ".join(_top_brands))

            # Action counts (over all loaded history, not just display slice)
            action_counts: dict[str, int] = {}
            for ev in history:
                at = (ev.get("action_type") or "unknown").lower()
                action_counts[at] = action_counts.get(at, 0) + 1
            if action_counts:
                st.caption(
                    "Действия: "
                    + "  ·  ".join(
                        f"{at}: {cnt}"
                        for at, cnt in sorted(action_counts.items(), key=lambda x: -x[1])
                    )
                )

            h_df = pd.DataFrame(history)
            for _col, _fallback in [
                ("title",         "—"),
                ("category_name", "—"),
                ("brand",         "—"),
                ("item_type",     ""),
            ]:
                if _col in h_df.columns:
                    h_df[_col] = h_df[_col].fillna(_fallback)

            show_h = [
                c for c in
                ["timestamp", "action_type", "title", "brand",
                 "category_name", "item_type", "item_id"]
                if c in h_df.columns
            ]
            st.dataframe(_fmt_df(h_df[show_h]), use_container_width=True, hide_index=True, height=350)
        else:
            st.info("История не найдена")

        st.divider()

        # ── Recommendation cards ───────────────────────────────────────────────
        st.subheader(f"Рекомендации (топ-{k})")
        st.caption(
            "Тест-период: июль 2025. "
            "**Зелёный** — куплен. "
            "**Синий** — действия в тестовом периоде без покупки. "
            "**«action»** — действие до рекомендаций. "
            "**↩ в истории** — тренировочная история до тест-периода. "
            "**источники кандидата** — показаны внизу каждой карточки."
        )

        if recs:
            n_repeat = sum(1 for r in recs if r.get("is_repeat"))
            n_new    = len(recs) - n_repeat
            rec_cats_t1  = {r.get("category_name") for r in recs if r.get("category_name")}
            hist_cats_t1 = {ev.get("category_name") for ev in history if ev.get("category_name")}
            all_srcs     = {s for r in recs for s in r.get("sources", [])}

            ri1, ri2, ri3, ri4 = st.columns(4)
            with ri1:
                st.metric("Повторные", f"{n_repeat} / {len(recs)}",
                          help="Товары, которые уже были в истории пользователя")
            with ri2:
                st.metric("Новые товары", n_new)
            with ri3:
                st.metric(
                    "Категорий в топ",
                    len(rec_cats_t1),
                    help=f"Уникальных категорий в топ-{k}",
                )
            with ri4:
                st.metric("Источников", len(all_srcs),
                          help=f"Уникальных кандидатных генераторов в топ-{k}")

            for row_start in range(0, len(recs), cols_per_row):
                row_slice = recs[row_start: row_start + cols_per_row]
                card_cols = st.columns(cols_per_row)
                for col, rec in zip(card_cols, row_slice):
                    with col:
                        _render_rec_card(
                            rec,
                            show_all_sources=show_all_sources,
                            show_hist_action=show_hist_action,
                            show_test_action=show_test_action,
                        )
        else:
            st.info("Рекомендации отсутствуют")

        st.divider()

        # ── Debug: history invariant table ────────────────────────────────────
        with st.expander("🔍 Debug: история по карточкам (history invariant)", expanded=False):
            _debug_rows = []
            for _dr in recs:
                _debug_rows.append({
                    "rank":                    _dr.get("rank"),
                    "item_id":                 _dr.get("item_id"),
                    "title":                   (_dr.get("title") or "")[:40],
                    "is_repeat":               _dr.get("history_event_count", 0) > 0,
                    "history_event_count":     _dr.get("history_event_count", 0),
                    "strongest_history_action": _dr.get("strongest_history_action"),
                    "history_action_counts":   str(_dr.get("history_action_counts") or {}),
                    "sources":                 ", ".join(_dr.get("sources") or []),
                })
            if _debug_rows:
                st.dataframe(pd.DataFrame(_debug_rows), use_container_width=True, hide_index=True)
            # Invariant check
            _bad = [r for r in _debug_rows if r["strongest_history_action"] and not r["is_repeat"]]
            if _bad:
                st.error(f"BAD_HISTORY_INVARIANT: {len(_bad)} карточек с action но history_event_count=0")
            else:
                st.success("Инвариант OK: все strongest_history_action согласованы с history_event_count")

        # ── What happened in the test period ──────────────────────────────────
        st.subheader("Взаимодействия пользователя в тестовом периоде")

        wa_later   = after.get("recommended_with_later_actions", [])
        wa_pur     = after.get("recommended_bought", [])
        wa_outside = after.get("not_recommended_bought", [])

        # Split bought_from_recs into top-k slice and extra top-100 slice
        wa_pur_topk = [p for p in wa_pur if p.get("rank_in_recommendations", 0) <= k]
        wa_pur_ext  = [p for p in wa_pur if p.get("rank_in_recommendations", 0) > k]

        wc1, wc2, wc3 = st.columns(3)

        with wc1:
            st.markdown(f"**Куплено из показанного топ-{k}** ({len(wa_pur_topk)})")
            if wa_pur_topk:
                df_pur_topk = pd.DataFrame(wa_pur_topk)
                show_pur = [
                    c for c in
                    ["rank_in_recommendations", "item_id", "title",
                     "brand", "category_name", "action_type", "timestamp"]
                    if c in df_pur_topk.columns
                ]
                st.dataframe(_fmt_df(df_pur_topk[show_pur]), use_container_width=True, hide_index=True)
            else:
                st.info(f"В топ-{k} покупок нет")

        with wc2:
            st.markdown(f"**Дополнительно в топ-{_actual_analysis_k}** ({len(wa_pur_ext)})")
            if wa_pur_ext:
                df_pur_ext = pd.DataFrame(wa_pur_ext)
                show_pur_ext = [
                    c for c in
                    ["rank_in_recommendations", "item_id", "title",
                     "brand", "category_name", "action_type", "timestamp"]
                    if c in df_pur_ext.columns
                ]
                st.dataframe(_fmt_df(df_pur_ext[show_pur_ext]), use_container_width=True, hide_index=True)
            else:
                st.info(f"В топ-{_actual_analysis_k} (вне топ-{k}) покупок нет")

        with wc3:
            st.markdown(f"**Куплено вне топ-{_actual_analysis_k}** ({len(wa_outside)})")
            if wa_outside:
                df_out = pd.DataFrame(wa_outside)
                show_out = [
                    c for c in
                    ["item_id", "title", "brand", "category_name",
                     "action_type", "timestamp",
                     "category_was_in_topk", "brand_was_in_topk",
                     "category_was_in_history", "brand_was_in_history"]
                    if c in df_out.columns
                ]
                st.dataframe(_fmt_df(df_out[show_out]), use_container_width=True, hide_index=True)
            else:
                st.info("Все покупки были из рекомендаций")

        if wa_later:
            st.markdown(f"**Взаимодействовал без покупки ({len(wa_later)})**")
            df_later = pd.DataFrame(wa_later)
            show_later = [
                c for c in
                ["rank_in_recommendations", "item_id", "title", "brand",
                 "category_name", "action_type", "score"]
                if c in df_later.columns
            ]
            st.dataframe(_fmt_df(df_later[show_later]), use_container_width=True, hide_index=True)
        else:
            st.info("Взаимодействий с рекомендованными товарами (без покупок) нет")


# =============================================================================
# Tab 2: Analysis
# =============================================================================

with tabs[1]:
    if selected_uid is None:
        st.info("Выберите пользователя в боковой панели")
    elif story is None:
        st.warning("Нет данных для отображения")
    else:
        _analysis = story.get("analysis", {})
        _recs_an  = story.get("recommendations", [])
        _hist_an  = story.get("history", [])

        # Fallback to charts block if analysis not yet in story
        if not _analysis:
            _charts = story.get("charts", {})
            _analysis = {
                "history_top_categories":    _charts.get("history_category_distribution", []),
                "recommendation_categories": _charts.get("recommendation_category_distribution", []),
                "source_distribution":       _charts.get("source_distribution", []),
                "seen_before_count":         sum(1 for r in _recs_an if r.get("is_repeat")),
                "new_items_count":           sum(1 for r in _recs_an if not r.get("is_repeat")),
                "category_overlap":          [],
                "category_miss":             [],
                "history_top_brands":        [],
            }

        hcat = _analysis.get("history_top_categories") or []
        rcat = _analysis.get("recommendation_categories") or []
        hist_cat_set_an = {item.get("category") for item in hcat if item.get("category")}

        # ── Section 1: Category coverage ─────────────────────────────────────
        st.subheader("Категории: история vs рекомендации")

        ca1, ca2 = st.columns(2)
        with ca1:
            if hcat and _PLOTLY:
                df_hc = pd.DataFrame(hcat).head(15).sort_values("count", ascending=True)
                fig = px.bar(df_hc, x="count", y="category", orientation="h",
                             title="История пользователя",
                             labels={"count": "Кол-во", "category": "Категория"})
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)
            elif hcat:
                st.dataframe(pd.DataFrame(hcat), use_container_width=True, hide_index=True)

        with ca2:
            if rcat and _PLOTLY:
                df_rc = pd.DataFrame(rcat).head(15).sort_values("count", ascending=True)
                df_rc["in_history"] = df_rc["category"].isin(hist_cat_set_an)
                fig = px.bar(
                    df_rc, x="count", y="category", orientation="h",
                    title=f"Рекомендованные категории (зелёный — из истории)",
                    labels={"count": "Кол-во", "category": "Категория"},
                    color="in_history",
                    color_discrete_map={True: "#2e7d32", False: "#90a4ae"},
                )
                fig.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            elif rcat:
                st.dataframe(pd.DataFrame(rcat), use_container_width=True, hide_index=True)

        # Category overlap / miss table
        # Use ALL catalog categories as rows — fixed base regardless of user
        _all_cat_set = _catalog_categories()
        all_cats_an = _all_cat_set if _all_cat_set else {item.get("category") for item in hcat + rcat if item.get("category")}
        if all_cats_an:
            _hcc = {item["category"]: item["count"] for item in hcat}
            _rcc = {item["category"]: item["count"] for item in rcat}

            # Purchases per category — ALL test-period purchases (from recs + outside recs).
            _future_an = story.get("future", {})
            _all_pur_an = (
                _future_an.get("bought_from_recommendations", [])
                + _future_an.get("bought_not_recommended", [])
            )
            _all_int_an  = _future_an.get("later_interactions_from_recommendations", [])
            _pur_by_cat: dict[str, int] = {}
            for _p in _all_pur_an:
                _pc = _p.get("category_name")
                if _pc:
                    _pur_by_cat[_pc] = _pur_by_cat.get(_pc, 0) + 1
            _int_by_cat: dict[str, int] = {}
            for _i in _all_int_an:
                _ic = _i.get("category_name")
                if _ic:
                    _int_by_cat[_ic] = _int_by_cat.get(_ic, 0) + 1

            overlap_rows = [
                {"Категория": c,
                 "В истории":             _hcc.get(c, 0),
                 "В топ-k":               _rcc.get(c, 0),
                 "Куплено (тест)":        _pur_by_cat.get(c, 0),
                 "Взаимодействовал (тест)": _int_by_cat.get(c, 0)}
                for c in all_cats_an
                if _hcc.get(c, 0) + _rcc.get(c, 0) + _pur_by_cat.get(c, 0) + _int_by_cat.get(c, 0) > 0
            ]
            overlap_rows.sort(key=lambda x: (-x["Куплено (тест)"], -x["Взаимодействовал (тест)"], -x["В истории"]))
            n_match = sum(1 for r in overlap_rows if r["В истории"] > 0 and r["В топ-k"] > 0)
            n_miss  = sum(1 for r in overlap_rows if r["В истории"] > 0 and r["В топ-k"] == 0)
            st.caption(
                f"всего в каталоге: {len(all_cats_an)}  |  "
                f"непустых строк: {len(overlap_rows)}  |  "
                f"история ∩ топ-k: {n_match}  |  "
                f"пропущено: {n_miss}"
            )
            st.dataframe(pd.DataFrame(overlap_rows),
                         use_container_width=True, hide_index=True)

        st.divider()

        # ── Section 2: Candidate sources ─────────────────────────────────────
        st.subheader("Источники кандидатов")
        st.caption(
            "Откуда пришёл каждый товар — какой генератор его выдвинул. "
            "Это не SHAP/feature importance LightGBM-реранкера, а источник кандидата на этапе ретривала."
        )

        src_data = _analysis.get("source_distribution") or []
        if not src_data:
            src_data = story.get("charts", {}).get("source_distribution", [])

        if src_data and _PLOTLY:
            # Top-20 source distribution computed client-side
            _src_20: dict[str, int] = {}
            for _r in _recs_an[:20]:
                for _s in _r.get("sources", []):
                    _src_20[_s] = _src_20.get(_s, 0) + 1
            df_src_20 = (
                pd.DataFrame([{"source": s, "count": c} for s, c in _src_20.items()])
                .sort_values("count", ascending=True)
            ) if _src_20 else pd.DataFrame()

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                if not df_src_20.empty:
                    fig20 = px.bar(df_src_20, x="count", y="source", orientation="h",
                                   title="Источники в топ-20",
                                   labels={"count": "Кол-во", "source": "Источник"},
                                   color="count", color_continuous_scale="Greens")
                    fig20.update_layout(coloraxis_showscale=False, height=320)
                    st.plotly_chart(fig20, use_container_width=True)

            with sc2:
                df_src = pd.DataFrame(src_data).sort_values("count", ascending=True)
                fig = px.bar(df_src, x="count", y="source", orientation="h",
                             title="Источники в топ-100",
                             labels={"count": "Кол-во", "source": "Источник"},
                             color="count", color_continuous_scale="Blues")
                fig.update_layout(coloraxis_showscale=False, height=320)
                st.plotly_chart(fig, use_container_width=True)

            with sc3:
                n_recs_an = len(_recs_an)
                if n_recs_an >= 51:
                    bands = [(1, 5), (6, 20), (21, 50), (51, n_recs_an)]
                elif n_recs_an >= 21:
                    bands = [(1, 5), (6, 20), (21, n_recs_an)]
                elif n_recs_an >= 8:
                    _mid = n_recs_an // 2
                    bands = [(1, _mid), (_mid + 1, n_recs_an)]
                else:
                    bands = [(1, n_recs_an)]

                band_rows: list[dict] = []
                for lo, hi in bands:
                    _band_recs = [r for r in _recs_an if lo <= r.get("rank", 0) <= hi]
                    _sc: dict[str, int] = {}
                    for r in _band_recs:
                        for s in r.get("sources", []):
                            _sc[s] = _sc.get(s, 0) + 1
                    for s, cnt in _sc.items():
                        band_rows.append({"Rank band": f"#{lo}–{hi}", "source": s, "count": cnt})

                if band_rows:
                    df_band = pd.DataFrame(band_rows)
                    fig_b = px.bar(
                        df_band, x="Rank band", y="count", color="source",
                        title="Источники по позиции в выдаче",
                        labels={"count": "Кол-во", "source": "Источник", "Rank band": "Позиция"},
                        color_discrete_map=_SOURCE_COLORS, barmode="stack",
                    )
                    fig_b.update_layout(height=420, legend_title="Источник")
                    st.plotly_chart(fig_b, use_container_width=True)
                    st.caption("Ось Y — число источников в полосе позиций (один товар может иметь несколько источников).")
        elif src_data:
            st.dataframe(pd.DataFrame(src_data), use_container_width=True, hide_index=True)

        st.divider()

        # ── Section 3: Seen-before / new split ───────────────────────────────
        st.subheader("Повторные vs новые товары")

        _recs_an_20  = _recs_an[:20]
        _recs_an_100 = _recs_an[:100]
        _n20  = len(_recs_an_20)  or 1
        _n100 = len(_recs_an_100) or 1

        _seen_20  = sum(1 for r in _recs_an_20  if r.get("is_repeat"))
        _seen_100 = sum(1 for r in _recs_an_100 if r.get("is_repeat"))

        sb1, sb2, sb3, sb4 = st.columns(4)
        with sb1:
            st.metric("Повторные топ-20",  f"{_seen_20} / {len(_recs_an_20)}",
                      help="Товаров из топ-20, которые уже были в истории (is_repeat)")
        with sb2:
            st.metric("Повторные топ-100", f"{_seen_100} / {len(_recs_an_100)}",
                      help="Товаров из топ-100, которые уже были в истории (is_repeat)")
        with sb3:
            st.metric("Новые топ-20",  len(_recs_an_20)  - _seen_20)
        with sb4:
            st.metric("Новые топ-100", len(_recs_an_100) - _seen_100)

        st.divider()

        # ── Section 4: Prediction evidence table ──────────────────────────────
        st.subheader("Детализация по товарам (топ-k)")
        st.caption(
            "По каждому товару из показанного топ-k: что модель знала при скоринге "
            "и что пользователь сделал в тест-периоде."
        )

        if _recs_an:
            _hist_cats_ev  = {ev.get("category_name") for ev in _hist_an if ev.get("category_name")}
            _hist_brands_ev = {b["brand"] for b in _analysis.get("history_top_brands", []) if b.get("brand")}

            evidence = []
            for r in _recs_an:
                _srcs = r.get("sources", [])
                evidence.append({
                    "rank":                 r["rank"],
                    "title":                (r.get("title") or "?")[:50],
                    "brand":                r.get("brand") or "",
                    "category":             r.get("category_name") or "",
                    "score":                round(r["score"], 4),
                    "seen_before":          r.get("is_repeat", False),
                    "category_in_history":  r.get("category_name") in _hist_cats_ev,
                    "brand_in_history":     r.get("brand") in _hist_brands_ev if r.get("brand") else False,
                    "sources":              ", ".join(_srcs[:3]) + ("…" if len(_srcs) > 3 else ""),
                    "interacted_in_test":   r.get("later_interaction", False),
                    "bought_in_test":       r.get("bought_after", False),
                })

            st.dataframe(_fmt_df(pd.DataFrame(evidence)), use_container_width=True, hide_index=True)


# =============================================================================
# Tab 3: System
# =============================================================================

with tabs[2]:
    model_info, mi_err = _cached_model_info(api_url)
    status_label = (ready_data or {}).get("status", "unknown")

    # ── Model identity ─────────────────────────────────────────────────────
    st.subheader("Модель")
    if model_info:
        _model_name = model_info.get("model_name") or "—"
        _prd_run    = model_info.get("prd_run_id") or "—"
        _feat_count = model_info.get("feature_count", "—")
        _model_uri  = model_info.get("model_uri") or "—"

        sys_c1, sys_c2, sys_c3 = st.columns([2, 2, 1])
        with sys_c1:
            st.markdown("**Модель**")
            st.code(_model_name, language=None)
        with sys_c2:
            st.markdown("**MLflow run_id (PRD)**")
            st.code(_prd_run, language=None)
        with sys_c3:
            st.metric("Признаков", _feat_count)
            st.metric("API", status_label)

        if _model_uri != "—":
            st.caption(f"URI: `{_model_uri}`")

    else:
        st.error(f"Не удалось загрузить /model/info: {mi_err}")

    st.divider()

    # ── Offline metrics ────────────────────────────────────────────────────
    st.subheader("Офлайн-метрики (тест)")
    if model_info:
        metrics_dict: dict = model_info.get("metrics", {})
        baseline_dict: dict = model_info.get("baseline", {})

        def _mv(d: dict, *keys) -> str:
            for k in keys:
                v = d.get(k)
                if v is not None:
                    return f"{v:.4f}"
            return "—"

        # NDCG — only columns with actual values
        _ndcg_items = [
            (_n, _mv(metrics_dict, f"test_ndcg_at_{_n}", f"ndcg_at_{_n}", f"reranker_ndcg_at_{_n}"),
                  _mv(baseline_dict, f"ndcg_at_{_n}", f"baseline_ndcg_at_{_n}"))
            for _n in [10, 20, 50, 100]
        ]
        _ndcg_items = [x for x in _ndcg_items if x[1] != "—"]
        if _ndcg_items:
            for _col, (_n, _v, _b) in zip(st.columns(len(_ndcg_items)), _ndcg_items):
                with _col:
                    _delta = f"{float(_v) - float(_b):+.4f} vs baseline" if _b != "—" else None
                    st.metric(f"NDCG@{_n}", _v, delta=_delta)

        st.divider()

        # Recall — only columns with actual values; candidate recall appended if present
        _recall_items = [
            (_n, _mv(metrics_dict, f"test_recall_at_{_n}", f"recall_at_{_n}", f"reranker_recall_at_{_n}"),
                  _mv(baseline_dict, f"recall_at_{_n}", f"baseline_recall_at_{_n}"))
            for _n in [20, 50, 100]
        ]
        _recall_items = [x for x in _recall_items if x[1] != "—"]
        _cand_v = _mv(metrics_dict, "candidate_recall_at_100", "retrieval_recall_at_100", "recall_candidates")
        if _cand_v != "—":
            _recall_items.append(("cand", _cand_v, "—"))
        if _recall_items:
            for _col, (_n, _v, _b) in zip(st.columns(len(_recall_items)), _recall_items):
                with _col:
                    if _n == "cand":
                        st.metric("Candidate recall@100", _v)
                    else:
                        _delta = f"{float(_v) - float(_b):+.4f} vs baseline" if _b != "—" else None
                        st.metric(f"Recall@{_n}", _v, delta=_delta)

        # All remaining metrics in a collapsible table
        _shown = {k for k in metrics_dict if any(
            k.endswith(f"_{n}") for n in ["10", "20", "50", "100"]
        )}
        _extra = {k: v for k, v in metrics_dict.items() if k not in _shown}
        if _extra:
            with st.expander("Все метрики MLflow", expanded=False):
                st.dataframe(
                    pd.DataFrame([{"Метрика": k, "Значение": v} for k, v in sorted(_extra.items())]),
                    use_container_width=True, hide_index=True,
                )

        art = model_info.get("artifact_status", {})
        if art:
            st.divider()
            st.subheader("Артефакты")
            st.dataframe(
                pd.DataFrame([{"Артефакт": k_a, "Статус": v} for k_a, v in art.items()]),
                use_container_width=True, hide_index=True,
            )

    st.divider()

    # ── Pipeline ───────────────────────────────────────────────────────────
    st.subheader("Пайплайн")
    st.markdown("""
**Датасет:** Ozon Apparel — одежда и аксессуары, июнь–июль 2025.
Обучение/валидация/тест: `< Jun 18` / `Jun 18 – Jul 1` / `Jul 1 – Jul 15`.

**Стадия 1 — Ретривал (10 генераторов кандидатов):**

| Генератор | Тип | Описание |
|-----------|-----|----------|
| ALS | Коллаборативная фильтрация | Matrix factorization (implicit) |
| DSSM | Двухбашенная сеть | User/item эмбеддинги, dot-product |
| SASRec | Трансформер | Self-attention на последовательности действий |
| BERT4Rec | Трансформер | BERT-style masked LM на сессиях |
| CLIP+HNSW | Визуальный поиск | Эмбеддинги изображений, граф HNSW |
| CLIP+FAISS | Визуальный поиск | Эмбеддинги изображений, FAISS индекс |
| I2I BM25 | Item-to-item | BM25 по текстовым атрибутам товара |
| TopPopular | Популярность | Глобально популярные товары |
| UserCatPopular | Популярность | Популярные в категориях из истории пользователя |
| RepeatHistory | Детерминированный | Товары из истории пользователя (~2 мес.) — прямой повтор |

**Стадия 2 — Реранкинг:**
- **Модель:** LightGBM LambdaRank
- **Цель:** NDCG@100
- **Признаки:** ~240 числовых (user / item / category / candidate-level)
- **Группы признаков:** статистика взаимодействий пользователя (7/14/30/60d), статистика товара, категориальное сходство, флаги повторного просмотра, позиция в кандидатном пуле

**Сервинг:** FastAPI → Streamlit.
Demo store: 100 pre-scored пользователей, артефакты ~18 MB (`api/demo_store/`).
""")


