from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    rank: int
    item_id: int
    score: float
    title: str | None = None
    brand: str | None = None
    item_type: str | None = None
    category_name: str | None = None
    category_path: str | None = None
    sources: list[str] = Field(default_factory=list)
    label: int = 0                          # raw label value from candidate store (0/1)
    val_gt_hit: bool = False                # label > 0 — val-period GT, matches /demo/users sidebar
    label_hit: bool = False                 # backward alias for val_gt_hit
    future_hit: bool = False                # legacy alias = future_interaction_hit
    future_interaction_hit: bool = False    # item appeared in any future event (test period)
    bought_after: bool = False              # item was purchased in test period (order/purchase/buy only)
    later_interaction: bool = False         # item had any future event (same as future_interaction_hit, set in story context)
    strongest_history_action: str | None = None   # strongest action before test period
    strongest_later_action: str | None = None     # strongest action in test period
    strongest_later_action_ts: str | None = None  # timestamp of strongest later action
    is_repeat: bool = False
    history_event_count: int = 0                  # total pre-cutoff history events for this item
    history_action_counts: dict[str, int] = Field(default_factory=dict)  # action_type -> count


class RichRecommendationResponse(BaseModel):
    user_id: int
    model: str
    is_cold: bool
    recommendations: list[RecommendationItem]


class DemoUserSummary(BaseModel):
    user_id: int
    candidate_count: int
    positive_count: int
    future_event_count: int
    purchase_count: int = 0           # real purchases (order/buy) in test period
    purchase_from_recs_20: int = 0    # real purchases in top-20 recommendations
    purchase_from_recs_100: int = 0   # real purchases in top-100 recommendations
    history_count: int
    # label-based (candidates with label=1) — val-period GT, matches /demo/users sidebar
    label_hits_at_20: int
    label_hits_at_100: int = 0
    label_recall_at_20: float   # label_hits_at_20 / positive_count
    # future-events-based
    future_hits_at_20: int
    future_hit_rate_at_20: float  # future_hits_at_20 / k  (= hit rate among recommendations)


class HistoryEvent(BaseModel):
    item_id: int
    timestamp: str | None = None
    action_type: str | None = None
    title: str | None = None
    category_name: str | None = None


class DemoUserResponse(BaseModel):
    user_id: int
    history: list[HistoryEvent]
    top_categories: list[str]
    recommendations: list[RecommendationItem]
    bought_from_recommendations: list[int]
    bought_not_recommended: list[int]
    hits_at_k: int
    recall_at_k: float
    explanation: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_uri: str | None = None
    prd_run_id: str | None = None
    feature_count: int
    candidate_store: str
    metrics: dict[str, float] = Field(default_factory=dict)
    baseline: dict[str, float] = Field(default_factory=dict)
    description: str | None = None
    artifact_status: dict[str, str] = Field(default_factory=dict)


# ── DemoStory view-model (for /demo/story/{user_id}) ─────────────────────────

class DemoStoryModelInfo(BaseModel):
    model_name: str
    prd_run_id: str | None = None
    feature_count: int
    metrics: dict[str, float] = Field(default_factory=dict)


class DemoStoryProfile(BaseModel):
    is_cold: bool
    history_count: int
    future_event_count: int
    future_purchase_count: int
    top_categories: list[str]


class BoughtFromRecommendation(BaseModel):
    item_id: int
    title: str | None = None
    brand: str | None = None
    category_name: str | None = None
    rank_in_recommendations: int
    score: float
    action_type: str | None = None
    timestamp: str | None = None


class BoughtNotRecommended(BaseModel):
    item_id: int
    title: str | None = None
    brand: str | None = None
    category_name: str | None = None
    action_type: str | None = None
    timestamp: str | None = None
    category_was_in_topk: bool = False
    brand_was_in_topk: bool = False
    category_was_in_history: bool = False
    brand_was_in_history: bool = False


class DemoStoryFuture(BaseModel):
    later_interactions_from_recommendations: list[BoughtFromRecommendation]
    bought_from_recommendations: list[BoughtFromRecommendation]
    bought_not_recommended: list[BoughtNotRecommended]


class ValidationMetrics(BaseModel):
    """Val-period ground truth — single source of truth, matches /demo/users sidebar."""
    positive_count: int
    label_hit_item_ids: list[int] = Field(default_factory=list)
    label_hits_at_k: int
    label_recall_at_k: float


class DemoStoryMetrics(BaseModel):
    """Test-period event counts (different time window from val labels)."""
    bought_from_recs_count: int
    purchase_recall_at_k: float
    future_interaction_hits_at_k: int
    future_interaction_rate_at_k: float
    strongest_recommended_action: str | None = None


class SourceDistributionPoint(BaseModel):
    source: str
    count: int


class CategoryDistributionPoint(BaseModel):
    category: str
    count: int


class CumulativeHitsPoint(BaseModel):
    rank: int
    cumulative_hits: int


class DemoStoryCharts(BaseModel):
    source_distribution: list[SourceDistributionPoint]
    history_category_distribution: list[CategoryDistributionPoint]
    recommendation_category_distribution: list[CategoryDistributionPoint]
    cumulative_hits_by_rank: list[CumulativeHitsPoint]


class DemoStoryResponse(BaseModel):
    user_id: int
    model: DemoStoryModelInfo
    profile: DemoStoryProfile
    history: list[HistoryEvent]
    recommendations: list[RecommendationItem]
    future: DemoStoryFuture
    validation: ValidationMetrics          # val-period GT — matches /demo/users sidebar
    metrics: DemoStoryMetrics              # test-period event counts
    charts: DemoStoryCharts
    explanation: str
    split_info: dict = Field(default_factory=dict)
