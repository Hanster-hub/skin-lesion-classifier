"""Step 1: download the HAM10000 skin cancer dataset from Kaggle.

Requires a Kaggle API token: create one at kaggle.com -> Account -> Create New
API Token, then either place kaggle.json in ~/.kaggle/ or answer the
interactive username/key prompt the first time this runs.
"""
import opendatasets as od

from config import DATA_DIR, KAGGLE_DATASET_URL


def download():
    od.download(KAGGLE_DATASET_URL, data_dir=DATA_DIR, force=False)


if __name__ == "__main__":
    download()
