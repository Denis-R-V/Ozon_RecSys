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

# AUTH
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGO = os.getenv("JWT_ALGO", "HS256")
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN", "60"))

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# VALIDATION
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET is not set")

if not ADMIN_PASSWORD:
    raise RuntimeError("ADMIN_PASSWORD is not set")












