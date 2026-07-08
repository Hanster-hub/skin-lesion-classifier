"""Shared paths and hyperparameters for the skin lesion pipeline."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get("SKIN_LESION_DATA_DIR", os.path.join(BASE_DIR, "data"))
RAW_DATA_DIR = os.path.join(DATA_DIR, "skin-cancer-mnist-ham10000")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODEL_DIR = os.environ.get("SKIN_LESION_MODEL_DIR", os.path.join(BASE_DIR, "models"))
LOG_DIR = os.path.join(BASE_DIR, "logs")

IMAGE_DIRECTORIES = [
    os.path.join(RAW_DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(RAW_DATA_DIR, "HAM10000_images_part_2"),
]

METADATA_CSV = os.path.join(RAW_DATA_DIR, "HAM10000_metadata.csv")
METADATA_PARQUET = os.path.join(PROCESSED_DATA_DIR, "df_metadata.parquet")

KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000"

CATEGORICAL_COLS = ["sex", "dx_type", "localization", "dx"]

SEED = 420
NUM_FOLDS = 5
IMAGE_SIZE = 256
BATCH_SIZE = 64
NUM_WORKERS = 2

for _dir in (DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR):
    os.makedirs(_dir, exist_ok=True)
