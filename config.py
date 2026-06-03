import os

from dotenv import load_dotenv

# Загрузка переменных из .env файла
load_dotenv()

# DATABASE
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./history.db")

# MODEL
MODEL_DIR = os.getenv("MODEL_DIR", "models")
ALS_MODEL_PATH = os.getenv(
    "ALS_MODEL_PATH",
    f"{MODEL_DIR}/als_model.npz",
)

ALS_USER_MAPPING_PATH = os.getenv(
    "ALS_USER_MAPPING_PATH",
    f"{MODEL_DIR}/als_user_mapping.parquet",
)

ALS_ITEM_MAPPING_PATH = os.getenv(
    "ALS_ITEM_MAPPING_PATH",
    f"{MODEL_DIR}/als_item_mapping.parquet",
)

ALS_PURCHASED_PATH = os.getenv(
    "ALS_PURCHASED_PATH",
    f"{MODEL_DIR}/als_purchased_by_user_idx.npy",
)

POPULAR_ITEMS_PATH = os.getenv(
    "POPULAR_ITEMS_PATH",
    f"{MODEL_DIR}/popular_items.parquet",
)

DEFAULT_K = int(os.getenv("DEFAULT_K", "100"))
CANDIDATE_MULTIPLIER = int(os.getenv("CANDIDATE_MULTIPLIER", "10"))

# PRD RANKER
_DEMO_STORE = "api/demo_store"
_RANKER_DIR = (
    "models/reranker"
    "/final_reranker_v12_repeat__merged_with_dl_all_repeat_history"
)
RANKER_MODEL_PATH = os.getenv(
    "RANKER_MODEL_PATH", f"{_RANKER_DIR}/best_ranker.joblib"
)
FEATURE_COLS_PATH = os.getenv(
    "FEATURE_COLS_PATH", f"{_RANKER_DIR}/feature_cols.json"
)
CANDIDATE_FEATURES_PATH = os.getenv(
    "CANDIDATE_FEATURES_PATH", f"{_DEMO_STORE}/api_candidates_features.parquet"
)
ITEM_METADATA_PATH = os.getenv(
    "ITEM_METADATA_PATH", f"{_DEMO_STORE}/api_item_metadata.parquet"
)
USER_HISTORY_PATH = os.getenv(
    "USER_HISTORY_PATH", f"{_DEMO_STORE}/api_user_history.parquet"
)
FUTURE_EVENTS_PATH = os.getenv(
    "FUTURE_EVENTS_PATH", f"{_DEMO_STORE}/api_future_events.parquet"
)
MODEL_INFO_PATH = os.getenv(
    "MODEL_INFO_PATH", f"{_DEMO_STORE}/api_model_info.json"
)
RANKER_POPULAR_ITEMS_PATH = os.getenv(
    "RANKER_POPULAR_ITEMS_PATH", f"{_DEMO_STORE}/popular_items.parquet"
)
ENABLE_LEGACY_ALS_FALLBACK = (
    os.getenv("ENABLE_LEGACY_ALS_FALLBACK", "false").strip().lower() == "true"
)
DUCKDB_THREADS = int(os.getenv("DUCKDB_THREADS", "2"))

# AUTH
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGO = os.getenv("JWT_ALGO", "HS256")
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN", "60"))

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# HISTORY
HISTORY_LIMIT_DEFAULT = 100
HISTORY_LIMIT_MAX = 1000

# VALIDATION
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET is not set")

if not ADMIN_PASSWORD:
    raise RuntimeError("ADMIN_PASSWORD is not set")
