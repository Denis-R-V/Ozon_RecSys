import shutil
import subprocess
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def ensure_data_dir():
    """Создание каталога data, если не создан ранее"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


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


def download_dataset():
    """Загрузка датасета с Kaggle, если он не загружен ранее"""
    ensure_data_dir()

    if any(DATA_DIR.glob("**/*.parquet")):
        print("Датасет загружен ранее")
        return
    
    dataset = "thunderpede/testststs"
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(DATA_DIR), "--unzip"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for zip_file in DATA_DIR.glob("*.zip"):
        zip_file.unlink(missing_ok=True)

    unwrap_data_dirs()
    print("Датасет загружен")