# Skin Lesion Classifier

A PyTorch pipeline for classifying skin lesions from the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000),
fine-tuning MobileNetV2 under three training strategies, and a Gradio UI for
exploring predictions against the held-out test set.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `download_data.py` | Downloads the HAM10000 dataset from Kaggle |
| 2 | `preprocess_data.py` | Cleans metadata: imputes missing age, label-encodes categorical columns |
| 3 | `split_data.py` | Holds out a stratified 20% test set, assigns the rest to 5 CV folds (grouped by lesion) |
| 4 | `dataset.py` | `CustomImageDataset` + augmentation/standard transform pipelines |
| 5 | `dataloaders.py` | Builds train (augmented/unaugmented)/val/test `DataLoader`s for a given fold |
| 6 | `model.py` | MobileNetV2 builder (`scratch`, `full_finetune`, `last_layer` strategies) |
| 7 | `engine.py` | Training and evaluation loops |
| 8 | `train.py` | Runs all 6 experiments (2 data variants x 3 strategies), saves models + summary CSV |
| 9 | `app.py` | Gradio UI to pick a model + test image and inspect predictions/metrics |

## Setup

```bash
pip install -r requirements.txt
```

You'll need a Kaggle API token: create one at kaggle.com -> Account -> Create
New API Token, then place `kaggle.json` in `~/.kaggle/` (or answer the
interactive prompt the first time `download_data.py` runs).

## Usage

Run the pipeline in order:

```bash
python download_data.py
python preprocess_data.py
python split_data.py
python train.py --fold 0 --epochs 10 --lr 1e-4
python app.py
```

`train.py` saves one `.pth` checkpoint per experiment plus
`mobilenet_experiment_summary.csv` to `models/`. `app.py` picks up every
`.pth` file in that folder automatically.

## Configuration

All paths and hyperparameters live in `config.py`. Override the data/model
locations without editing code via environment variables:

```bash
export SKIN_LESION_DATA_DIR=/path/to/data
export SKIN_LESION_MODEL_DIR=/path/to/models
```

## Project layout

```
config.py             # paths + hyperparameters
download_data.py       # step 1: pull data
preprocess_data.py     # step 2: clean + encode metadata
split_data.py          # step 3: stratified group k-fold split
dataset.py             # step 4: Dataset + transforms
dataloaders.py         # step 5: DataLoader construction
model.py               # step 6: MobileNetV2 builder
engine.py              # step 7: train/eval loops
train.py               # step 8: experiment runner
app.py                 # step 9: Gradio UI
utils.py               # seeding helper
```
