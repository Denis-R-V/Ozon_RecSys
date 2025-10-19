import subprocess
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def ensure_data_dir():
    """Создание каталога data, если не создан ранее"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


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

    print("Датасет загружен")