import json
import os
import requests
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm

import dask.dataframe as dd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

OWNER_SLUG = "thunderpede"
DATASET_SLUG = "testststs"


def dataset_ok() -> bool:
    root_parquet = "ml_ozon_recsys_test.snappy.parquet"
    expected_counts = {
        "archive_extra_orders_data": 28,
        "final_apparel_tracker_data_08_action_widget": 2800,
        "ml_ozon_recsys_train_final_apparel_orders_data": 5,
        "ml_ozon_recsys_train_final_apparel_tracker_data": 200,
        "ml_ozon_recsys_train_final_apparel_items_data": 100,
        "ml_ozon_recsys_train_final_categories_tree": 1,
    }

    # корневой файл parquet
    f = DATA_DIR / root_parquet
    if not f.is_file() or f.stat().st_size == 0:
        return False

    # структура каталогов и количество parquet
    for name, exp in expected_counts.items():
        d = DATA_DIR / name
        if not d.is_dir():
            return False
        if len(list(d.rglob("*.parquet"))) != exp:
            return False

    return True


def download_from_kaggle(owner_slug: str, dataset_slug: str):
    """
    Загрузка датасета с kaggle (библиотека kaggle падает по памяти)
    """
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        key = json.load(f)["key"]

    url = f"https://www.kaggle.com/api/v1/datasets/download/{owner_slug}/{dataset_slug}"
    headers = {"Authorization": f"Bearer {key}"}

    output_path = DATA_DIR / f"{dataset_slug}.zip"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(output_path, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as bar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def unzip_with_progress(zip_path: str, extract_to: str):
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.infolist()
        total_files = len(members)

        with tqdm(total=total_files, desc="Unzipping", unit="file") as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)

    zip_path.unlink(missing_ok=True)


def unwrap_data_dirs():
    """Перенос вложенных каталогов на уровень выше (ошибка архивирования)"""
    outer_names = [
        "extra_ml_ozon_recsys_train%202",
        "extra_ml_ozon_recsys_train",
        "ml_ozon_recsys_train_final_apparel_orders_data",
        "ml_ozon_recsys_train_final_apparel_tracker_data",
        "ml_ozon_recsys_train_final_apparel_items_data",
        "ml_ozon_recsys_train_final_categories_tree",
    ]

    for outer_name in outer_names:
        outer = DATA_DIR / outer_name
        if not outer.is_dir():
            continue

        subs = [p for p in outer.iterdir() if p.is_dir()]
        if not subs:
            continue

        for inner in subs:
            # случай дублированного имени
            if inner.name == outer.name:
                for item in inner.iterdir():
                    shutil.move(str(item), str(outer / item.name))
                inner.rmdir()
            else:
                # переносим в DATA_DIR
                dst = DATA_DIR / inner.name
                dst.mkdir(exist_ok=True)
                for item in inner.iterdir():
                    shutil.move(str(item), str(dst / item.name))
                inner.rmdir()

        # если внешний каталог пуст после переноса - удалить
        if not any(outer.iterdir()):
            outer.rmdir()


def download_dataset(force: bool = False):
    """Загрузка датасета с Kaggle, если он не загружен ранее"""

    if not force and dataset_ok():
        print("Датасет загружен ранее")
        return

    download_from_kaggle(OWNER_SLUG, DATASET_SLUG)
    unzip_with_progress(DATA_DIR / f"{DATASET_SLUG}.zip", DATA_DIR)
    unwrap_data_dirs()
    print("Датасет загружен")


def load_orders(train: bool = True):
    if train:
        path = DATA_DIR / "ml_ozon_recsys_train_final_apparel_orders_data"
    else:
        path = DATA_DIR / "archive_extra_orders_data" / "final_apparel_orders_data_07"
    return dd.read_parquet(path)


def load_interactions(train: bool = True):
    if train:
        path = DATA_DIR / "ml_ozon_recsys_train_final_apparel_tracker_data"
    else:
        path = DATA_DIR / "final_apparel_tracker_data_08_action_widget"
    return dd.read_parquet(path)


def load_items():
    path = DATA_DIR / "ml_ozon_recsys_train_final_apparel_items_data"
    return dd.read_parquet(path)


def load_categories():
    path = DATA_DIR / "ml_ozon_recsys_train_final_categories_tree"
    return dd.read_parquet(path)
