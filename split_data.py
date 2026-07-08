"""Step 3: split lesions into a held-out test set plus stratified group K-folds.

Splitting is grouped by `lesion_id` (a lesion can have multiple images) and
stratified by `dx_numeric` so class balance is preserved across folds. A
20% slice is held out as a true test set, marked with fold == -2; the
remaining lesions are assigned to folds 0..NUM_FOLDS-1.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import METADATA_PARQUET, NUM_FOLDS, SEED


def add_fold_column(df_metadata):
    df_metadata = df_metadata.copy()
    df_metadata["fold"] = -2  # -2 marks the held-out test set

    lesion_ids = df_metadata["lesion_id"].values
    target_labels = df_metadata["dx_numeric"].values
    num_samples = len(df_metadata)

    # Isolate ~20% of lesions as a true holdout test set
    holdout_splitter = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    cv_train_indices, _ = next(holdout_splitter.split(np.zeros(num_samples), target_labels, groups=lesion_ids))

    # Assign folds 0..NUM_FOLDS-1 to the remaining ~80%
    cv_splitter = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=False)
    y_train = target_labels[cv_train_indices]
    train_lesion_ids = lesion_ids[cv_train_indices]

    fold_splits = cv_splitter.split(np.zeros(len(cv_train_indices)), y_train, groups=train_lesion_ids)
    for fold_id, (_, fold_val_idx) in enumerate(fold_splits):
        master_val_indices = cv_train_indices[fold_val_idx]
        df_metadata.iloc[master_val_indices, df_metadata.columns.get_loc("fold")] = fold_id

    return df_metadata


def verify_folds(df_metadata):
    print("=== UNIQUE LESIONS PER FOLD ===")
    lesions_per_fold = df_metadata.groupby("fold")["lesion_id"].nunique()
    print(lesions_per_fold)

    print("\n=== CLASS DISTRIBUTION PER FOLD (%) ===")
    class_fold_pct = pd.crosstab(df_metadata["fold"], df_metadata["dx_numeric"], normalize="index") * 100
    print(class_fold_pct.round(1))


def split():
    df_metadata = pd.read_parquet(METADATA_PARQUET)
    df_metadata = add_fold_column(df_metadata)
    verify_folds(df_metadata)

    df_metadata.to_parquet(METADATA_PARQUET, index=False)
    print("\nSaved fold assignments to:", METADATA_PARQUET)

    return df_metadata


if __name__ == "__main__":
    split()
