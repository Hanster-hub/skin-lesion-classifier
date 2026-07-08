"""Step 2: clean the HAM10000 metadata and label-encode categorical columns.

- Imputes missing `age` with the median.
- Label-encodes sex / dx_type / localization / dx into `<col>_numeric` columns.
- Saves the cleaned metadata to a parquet file for the rest of the pipeline.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import CATEGORICAL_COLS, METADATA_CSV, METADATA_PARQUET


def load_and_clean_metadata():
    df_metadata = pd.read_csv(METADATA_CSV)

    median_age = df_metadata["age"].median()
    df_metadata["age"] = df_metadata["age"].fillna(median_age)

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_metadata[f"{col}_numeric"] = le.fit_transform(df_metadata[col].astype(str))
        encoders[col] = le

    return df_metadata, encoders


def preprocess():
    df_metadata, encoders = load_and_clean_metadata()

    df_unique_lesions = df_metadata.drop_duplicates(subset="lesion_id", keep="first")

    print("Original image-level samples:", len(df_metadata))
    print("Unique lesion-level samples:", len(df_unique_lesions))
    print("\nClass counts (unique lesions):\n", df_unique_lesions["dx"].value_counts())

    df_metadata.to_parquet(METADATA_PARQUET, index=False)
    print("\nSaved cleaned metadata to:", METADATA_PARQUET)

    return df_metadata, encoders


if __name__ == "__main__":
    preprocess()
