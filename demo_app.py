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


# ── helpers ───────────────────────────────────────────────────────────────────

_ACTION_STRENGTH_UI: dict[str, int] = {
    "order": 7, "purchase": 7, "buy": 7,
    "to_cart": 6, "favorite": 5,
    "review_view": 3, "view_description": 2, "page_view": 1,
}


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
def _cached_story(api_url: str, uid: int, k: int, history_limit: int, analysis_k: int):
    return _api_get(
        api_url,
        f"/demo/story/{uid}",
        {"k": k, "history_limit": history_limit, "analysis_k": analysis_k},
    )


@st.cache_data(ttl=60, show_spinner=False)
def _cached_users(api_url: str):
    return _api_get(api_url, "/demo/users", {"limit": 100, "min_hits": 0})


def _source_badge(src: str) -> str:
    color = _SOURCE_COLORS.get(src, "#666")
    return (
        f'<span style="background:{color};color:white;padding:1px 4px;'
        f'border-radius:3px;font-size:9px;margin-right:2px">{src}</span>'
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
    repeat      = rec.get("is_repeat", False)
    pur_after   = rec.get("bought_after", False)
    later_int   = rec.get("later_interaction", False)
    hist_action = rec.get("strongest_history_action") or ""   # pre-test
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

    # unfavorite/remove are negative/zero-strength signals — showing them as "до:" is misleading
    _SUPPRESS_HIST = {"unfavorite", "remove"}

    badges = ""
    if show_hist_action and hist_action and hist_action not in _SUPPRESS_HIST:
        badges += (
            f'<span title="Strongest action before test period" '
            f'style="background:#546e7a;color:white;padding:2px 5px;'
            f'border-radius:3px;font-size:9px;margin-right:3px;cursor:help">'
            f'до: {hist_action}</span>'
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
    if repeat:
        badges += (
            '<span title="Товар из истории пользователя — ранее просмотрен/куплен" '
            'style="background:#e65100;color:white;padding:2px 5px;'
            'border-radius:3px;font-size:9px;cursor:help">↩ повтор</span>'
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
        live_data, _ = _api_get(api_url, "/health")
        if live_data and live_data.get("status") == "ok":
            st.success("Live: ok")
        else:
            st.error("Live: fail")
    with sc2:
        ready_data, _ = _api_get(api_url, "/health/ready")
        if ready_data and ready_data.get("status") in ("ok", "degraded"):
            st.success("Ready: ok")
        else:
            st.warning("Ready: fail")

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
                f"  {u.get('purchase_count', 0)} покуп."
                f"  hist={u.get('history_count', 0)}"
            )
            for u in demo_users
        }
        selected_uid = st.selectbox(
            "Demo users",
            options=[u["user_id"] for u in demo_users],
            format_func=lambda uid: fmt_map.get(uid, str(uid)),
        )
    else:
        st.warning(users_err or "Demo store недоступен")

    st.divider()
    k: int = st.select_slider("Карточек (топ-k)", options=[5, 10, 20, 50, 100, 200], value=20)
    history_limit: int = st.slider(
        "История (строк)", min_value=5, max_value=200, value=20, step=5
    )
    analysis_k = 100  # fixed analysis pool depth, not user-configurable

    st.divider()
    cols_per_row: int = st.select_slider("Карточек в строке", options=[2, 3, 4, 5, 6, 8], value=4)
    show_hist_action: bool = st.checkbox("Показывать действие ДО рек. на карточке", value=True)
    show_test_action: bool = st.checkbox("Показывать действие тест. периода на карточке", value=True)
    show_all_sources: bool = st.checkbox("Показать все источники в карточках", value=False)
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
        st.rerun()


# ── load story ────────────────────────────────────────────────────────────────

story: dict | None = None
story_err: str | None = None
if selected_uid is not None:
    story, story_err = _cached_story(api_url, selected_uid, k, history_limit, analysis_k)


# ── page header ───────────────────────────────────────────────────────────────

st.title("Ozon Apparel Recommendation Demo")
st.caption(
    "9 ретриверов (ALS, CLIP, DSSM, SASRec…)  →  LightGBM LambdaRank реранкер  |  FastAPI + Streamlit"
)
st.divider()

tabs = st.tabs(["Demo", "Analysis", "System", "Debug"])


# =============================================================================
# Tab 1: Demo  (test-period user story)
# =============================================================================

with tabs[0]:
    if selected_uid is None:
        st.info("Выберите пользователя в боковой панели")
    elif story is None:
        st.error(f"Ошибка загрузки: {story_err}")
    else:
        profile  = story.get("profile", {})
        summ     = story.get("summary", {})
        recs     = story.get("recommendations", [])
        history  = story.get("history", [])
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
        st.subheader(f"User {story['user_id']}")
        if is_cold:
            st.warning("Cold user — no candidates, showing popular-items fallback.")

        st.info(
            "**Демо тест-периода.** "
            "FastAPI scoring в реальном времени — PRD LightGBM reranker.  "
            "**Зелёный** = пользователь купил товар после рекомендации.  "
            "**Синий** = взаимодействовал, но не купил."
        )

        # ── 5 metrics ─────────────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric(
                "Покупок (тест-период)",
                profile.get("future_purchase_count", 0),
                help="Реальные покупки пользователя в тест-периоде (order/buy)",
            )
        with m2:
            st.metric(
                f"Показано (топ-{k})",
                len(recs),
                help="Карточки рекомендаций на экране",
            )
        with m3:
            st.metric(
                "Куплено из рек.",
                _bought_topk,
                help=f"Зелёные карточки в топ-{k}: реальные покупки после рекомендации",
            )
        with m4:
            st.metric(
                "Взаимодействий",
                _interactions_topk,
                help=f"Синие карточки в топ-{k}: просмотр/избранное без покупки (unfavorite/remove не считается)",
            )
        with m5:
            st.metric(
                "Куплено вне рек.",
                _bought_outside,
                help=f"Покупки товаров, которых НЕ было в топ-{_actual_analysis_k} рекомендациях (модель пропустила)",
            )

        st.caption(
            f"Покупки: топ-{k}: {_bought_topk}"
            f"  |  топ-{_actual_analysis_k}: {_bought_analysis}"
            f"  |  вне рек. (мимо модели): {_bought_outside}"
        )
        st.divider()

        # ── User history ──────────────────────────────────────────────────────
        st.subheader("User history")
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
                st.caption("Top brands: " + "  ·  ".join(_top_brands))

            # Action counts
            action_counts: dict[str, int] = {}
            for ev in history:
                at = (ev.get("action_type") or "unknown").lower()
                action_counts[at] = action_counts.get(at, 0) + 1
            if action_counts:
                st.caption(
                    "Actions: "
                    + "  ·  ".join(
                        f"{at}: {cnt}"
                        for at, cnt in sorted(action_counts.items(), key=lambda x: -x[1])
                    )
                )

            h_df = pd.DataFrame(history)
            for _col, _fallback in [
                ("title",         "Unknown item"),
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
            st.dataframe(h_df[show_h], use_container_width=True, hide_index=True, height=350)
        else:
            st.info("История не найдена")

        st.divider()

        # ── Recommendation cards ───────────────────────────────────────────────
        st.subheader(f"Recommendations (top-{k})")

        if recs:
            n_repeat = sum(1 for r in recs if r.get("is_repeat"))
            n_new    = len(recs) - n_repeat
            rec_cats_t1  = {r.get("category_name") for r in recs if r.get("category_name")}
            hist_cats_t1 = {ev.get("category_name") for ev in history if ev.get("category_name")}
            all_srcs     = {s for r in recs for s in r.get("sources", [])}

            ri1, ri2, ri3, ri4 = st.columns(4)
            with ri1:
                st.metric("Seen before", f"{n_repeat} / {len(recs)}",
                          help="Items already in user history")
            with ri2:
                st.metric("New items", n_new)
            with ri3:
                st.metric(
                    "Category match",
                    f"{len(hist_cats_t1 & rec_cats_t1)} / {len(hist_cats_t1) or 1}",
                    help="History categories represented in top-k",
                )
            with ri4:
                st.metric("Unique sources", len(all_srcs),
                          help="Distinct candidate generators in top-k")

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

        # ── What happened after ────────────────────────────────────────────────
        st.subheader("Что купил и что посмотрел после рекомендаций")
        st.caption(
            f"Тест-период. Пул анализа: топ-{_actual_analysis_k} рекомендаций "
            f"(карточки выше показывают топ-{k})."
        )

        wa_later   = after.get("recommended_with_later_actions", [])
        wa_pur     = after.get("recommended_bought", [])
        wa_outside = after.get("not_recommended_bought", [])

        if wa_later:
            _later_act_types = [r.get("action_type") for r in wa_later if r.get("action_type")]
            _strongest_later_int = (
                max(_later_act_types, key=lambda a: _ACTION_STRENGTH_UI.get(a.lower(), 0))
                if _later_act_types else None
            )
            _later_hdr = f"**Взаимодействовал (без покупки) ({len(wa_later)})"
            if _strongest_later_int:
                _later_hdr += f"  ·  max: {_strongest_later_int}"
            _later_hdr += "**"
            st.markdown(_later_hdr)
            df_later = pd.DataFrame(wa_later)
            show_later = [
                c for c in
                ["rank_in_recommendations", "item_id", "title", "brand",
                 "category_name", "action_type", "score"]
                if c in df_later.columns
            ]
            st.dataframe(df_later[show_later], use_container_width=True, hide_index=True)
        else:
            st.info("Взаимодействий с рекомендованными товарами (без покупок) нет")

        wc1, wc2 = st.columns(2)

        with wc1:
            st.markdown(f"**Caught** — куплено из рек. (топ-{_actual_analysis_k})")
            if wa_pur:
                df_pur = pd.DataFrame(wa_pur)
                show_pur = [
                    c for c in
                    ["rank_in_recommendations", "item_id", "title",
                     "brand", "category_name", "action_type", "timestamp", "score"]
                    if c in df_pur.columns
                ]
                st.dataframe(df_pur[show_pur], use_container_width=True, hide_index=True)
            else:
                st.info("Покупок из рекомендаций нет")

        with wc2:
            st.markdown(f"**Missed** — куплено вне рек. (мимо топ-{_actual_analysis_k})")
            if wa_outside:
                df_out = pd.DataFrame(wa_outside)
                show_out = [
                    c for c in
                    ["item_id", "title", "brand", "category_name",
                     "action_type", "category_was_recommended",
                     "brand_was_recommended", "timestamp"]
                    if c in df_out.columns
                ]
                st.dataframe(df_out[show_out], use_container_width=True, hide_index=True)
            else:
                st.info("Все покупки были из рекомендаций")


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
        st.subheader("Category coverage: history vs recommendations")

        ca1, ca2 = st.columns(2)
        with ca1:
            if hcat and _PLOTLY:
                df_hc = pd.DataFrame(hcat).head(10).sort_values("count", ascending=True)
                fig = px.bar(df_hc, x="count", y="category", orientation="h",
                             title="History categories")
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)
            elif hcat:
                st.dataframe(pd.DataFrame(hcat), use_container_width=True, hide_index=True)

        with ca2:
            if rcat and _PLOTLY:
                df_rc = pd.DataFrame(rcat).head(10).sort_values("count", ascending=True)
                df_rc["in_history"] = df_rc["category"].isin(hist_cat_set_an)
                fig = px.bar(
                    df_rc, x="count", y="category", orientation="h",
                    title="Recommended categories (green = in history)",
                    color="in_history",
                    color_discrete_map={True: "#2e7d32", False: "#90a4ae"},
                )
                fig.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            elif rcat:
                st.dataframe(pd.DataFrame(rcat), use_container_width=True, hide_index=True)

        # Category overlap / miss table
        all_cats_an = {item.get("category") for item in hcat + rcat if item.get("category")}
        if all_cats_an:
            _hcc = {item["category"]: item["count"] for item in hcat}
            _rcc = {item["category"]: item["count"] for item in rcat}

            def _cstatus(c: str) -> str:
                return ("match" if (_hcc.get(c, 0) > 0 and _rcc.get(c, 0) > 0)
                        else ("miss" if _hcc.get(c, 0) > 0 else ""))

            overlap_rows = [
                {"Category": c, "In history": _hcc.get(c, 0),
                 "In top-k": _rcc.get(c, 0), "Status": _cstatus(c)}
                for c in all_cats_an
            ]
            _go = {"match": 0, "miss": 1, "": 2}
            overlap_rows.sort(key=lambda x: (_go[x["Status"]], -x["In history"]))
            n_match = sum(1 for r in overlap_rows if r["Status"] == "match")
            n_miss  = sum(1 for r in overlap_rows if r["Status"] == "miss")
            st.caption(f"match: {n_match}  |  miss (in history, not in top-k): {n_miss}")
            st.dataframe(pd.DataFrame(overlap_rows[:20]),
                         use_container_width=True, hide_index=True)

        st.divider()

        # ── Section 2: Candidate sources ─────────────────────────────────────
        st.subheader("Candidate sources")
        st.caption(
            "Source provenance of candidates — which generator nominated each item. "
            "Not a direct SHAP/feature importance explanation of the LightGBM ranking."
        )

        src_data = _analysis.get("source_distribution") or []
        if not src_data:
            src_data = story.get("charts", {}).get("source_distribution", [])

        if src_data and _PLOTLY:
            sc1, sc2 = st.columns(2)
            with sc1:
                df_src = pd.DataFrame(src_data).sort_values("count", ascending=True)
                fig = px.bar(df_src, x="count", y="source", orientation="h",
                             title="Sources in top-k",
                             color="count", color_continuous_scale="Blues")
                fig.update_layout(coloraxis_showscale=False, height=300)
                st.plotly_chart(fig, use_container_width=True)

            with sc2:
                n_recs_an = len(_recs_an)
                if n_recs_an >= 15:
                    bands = [(1, 5), (6, 10), (11, n_recs_an)]
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
                        title="Sources by rank position",
                        color_discrete_map=_SOURCE_COLORS, barmode="stack",
                    )
                    fig_b.update_layout(height=300, legend_title="Source")
                    st.plotly_chart(fig_b, use_container_width=True)
                    st.caption("Y-axis = source occurrences per rank band (one item can have multiple sources).")
        elif src_data:
            st.dataframe(pd.DataFrame(src_data), use_container_width=True, hide_index=True)

        st.divider()

        # ── Section 3: Seen-before / new split ───────────────────────────────
        st.subheader("Seen-before vs new items")
        n_seen = _analysis.get("seen_before_count", sum(1 for r in _recs_an if r.get("is_repeat")))
        n_new  = _analysis.get("new_items_count",  len(_recs_an) - n_seen)

        sb1, sb2, sb3 = st.columns(3)
        with sb1:
            st.metric("Seen before", n_seen, help="Items the user previously interacted with")
        with sb2:
            st.metric("New items", n_new)
        with sb3:
            pct = round(100 * n_seen / max(len(_recs_an), 1))
            st.metric("Repeat rate", f"{pct}%")

        st.divider()

        # ── Section 4: Prediction evidence table ──────────────────────────────
        st.subheader("Prediction evidence")
        st.caption(
            "Per-item view: what the ranker knew at score time "
            "and what user did after."
        )

        if _recs_an:
            _hist_cats_ev  = {ev.get("category_name") for ev in _hist_an if ev.get("category_name")}
            _hist_brands_ev = {b["brand"] for b in _analysis.get("history_top_brands", []) if b.get("brand")}

            evidence = []
            for r in _recs_an:
                _srcs = r.get("sources", [])
                evidence.append({
                    "rank":           r["rank"],
                    "title":          (r.get("title") or "?")[:50],
                    "brand":          r.get("brand") or "",
                    "category":       r.get("category_name") or "",
                    "score":          round(r["score"], 4),
                    "seen_before":    r.get("is_repeat", False),
                    "cat_in_hist":    r.get("category_name") in _hist_cats_ev,
                    "brand_in_hist":  r.get("brand") in _hist_brands_ev if r.get("brand") else False,
                    "sources":        ", ".join(_srcs[:3]) + ("…" if len(_srcs) > 3 else ""),
                    "later_int":  r.get("later_interaction", False),
                    "bought":     r.get("bought_after", False),
                })

            st.dataframe(pd.DataFrame(evidence), use_container_width=True, hide_index=True)


# =============================================================================
# Tab 3: System
# =============================================================================

with tabs[2]:
    model_info, mi_err = _api_get(api_url, "/model/info")

    if model_info:
        st.subheader("Model and offline metrics")
        metrics_dict: dict = model_info.get("metrics", {})

        # Test metrics only — no val fallback
        ndcg_val = (
            metrics_dict.get("test_ndcg_at_100")
            or metrics_dict.get("ndcg_at_100")
            or metrics_dict.get("reranker_ndcg_at_100")
        )
        recall_val = (
            metrics_dict.get("test_recall_at_100")
            or metrics_dict.get("recall_at_100")
            or metrics_dict.get("reranker_recall_at_100")
        )
        status_label = (ready_data or {}).get("status", "unknown")

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1:
            st.metric("Model", model_info.get("model_name", "—"))
        with mc2:
            st.metric("Features", model_info.get("feature_count", "—"))
        with mc3:
            st.metric(
                "NDCG@100 (test, offline)",
                f"{ndcg_val:.4f}" if isinstance(ndcg_val, float) else "—",
                help="Offline metric from test split (reranker_ndcg_at_100 or test_ndcg_at_100)",
            )
        with mc4:
            st.metric(
                "Recall@100 (offline)",
                f"{recall_val:.4f}" if isinstance(recall_val, float) else "—",
            )
        with mc5:
            st.metric("API status", status_label)

        art = model_info.get("artifact_status", {})
        if art:
            st.subheader("Artifact health")
            st.dataframe(
                pd.DataFrame([{"Artifact": k_a, "Status": v} for k_a, v in art.items()]),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.error(f"Failed to load /model/info: {mi_err}")

    st.divider()
    st.subheader("Architecture")
    st.markdown(
        """
```
Offline store            Online scoring             Serving
─────────────────────    ──────────────────────     ─────────────────
Parquet / DuckDB   ──►   LightGBM LambdaRank  ──►  FastAPI  ──►  Streamlit
(api/demo_store/)         (real-time predict)        (REST)       (this UI)
```

| Stage | Description |
|-------|-------------|
| **Retrieval** | 9 generators: ALS, DSSM, SASRec, BERT4Rec, CLIP+HNSW, CLIP+FAISS, I2I BM25, TopPopular, UserCatPopular, RepeatHistory |
| **Features** | 240 numeric features (user / item / category / candidate-level) |
| **Reranking** | LightGBM LambdaRank, optimized for NDCG@100 |
| **Demo store** | 100 pre-scored users, pre-built artifacts ~18 MB (`api/demo_store/`) |
| **Labeled replay** | Val-period labels used in training — see Debug tab for sanity-check |
"""
    )


# =============================================================================
# Tab 4: Debug
# =============================================================================

with tabs[3]:
    st.subheader("Raw API responses")

    if story:
        with st.expander(f"GET /demo/story/{selected_uid}?k={k}", expanded=False):
            st.json(story)

    mi_raw, _ = _api_get(api_url, "/model/info")
    if mi_raw:
        with st.expander("GET /model/info", expanded=False):
            st.json(mi_raw)

    hr_raw, _ = _api_get(api_url, "/health/ready")
    if hr_raw:
        with st.expander("GET /health/ready", expanded=False):
            st.json(hr_raw)

    st.subheader("Curl examples")
    if selected_uid is not None:
        st.code(
            f"# Story endpoint\n"
            f'curl "{api_url}/demo/story/{selected_uid}?k={k}'
            f'&history_limit={history_limit}"\n'
            f"\n# Rich recommendations\n"
            f'curl "{api_url}/recommendations/{selected_uid}?k={k}"\n'
            f"\n# Forward (compatible)\n"
            f'curl -X POST "{api_url}/forward" \\\n'
            f'  -H "Content-Type: application/json" \\\n'
            f'  -d \'{{"user_id": {selected_uid}, "k": {k}}}\'\n'
            f"\n# Demo users list\n"
            f'curl "{api_url}/demo/users?limit=20&min_hits=0"\n'
            f"\n# Health\n"
            f'curl "{api_url}/health/ready"',
            language="bash",
        )
    else:
        st.info("Select a user to see curl examples")

    st.divider()
    with st.expander("Пересборка демо данных", expanded=False):
        st.caption(
            "Пересобирает api/demo_store/ из train/test данных (~1-2 мин). "
            "После завершения нужен рестарт uvicorn."
        )
        _use_test_flag = st.checkbox("--use-test (test_eval кандидаты)", value=True)
        if st.button("Запустить пересборку", type="secondary"):
            _rb, _rb_err = _api_post(
                api_url,
                "/admin/rebuild-demo-store",
                {"use_test": str(_use_test_flag).lower()},
            )
            if _rb:
                st.success(f"Запущено (PID {_rb.get('pid')}). {_rb.get('note', '')}")
                st.code(_rb.get("command", ""))
            else:
                st.error(f"Ошибка: {_rb_err}")

    with st.expander("Labeled replay sanity-check (val-period labels, not primary metric)", expanded=False):
        st.caption("Val-period labels are training data — not the demo quality metric.")
        if story:
            val_block = story.get("validation", {})
            if val_block:
                st.json(val_block)
            else:
                st.info("No validation block in story response")
