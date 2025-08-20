import pandas as pd
import numpy as np
import os, sys
import pandera.pandas as pa
from pandera.pandas import Column, Check
REPO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPO_PATH)
from settings import TARGET, DATA_DIR_PATH

# Since our EDA revealed no missing columns we just have to convert target column to binary (instead of ">50k" and "<=50K")

def load_clean(path):
    df = pd.read_csv(path, na_values=["?"]) # replace "?" with NaN

    # Drop rows with NaN and duplicates as they're only a small ratio of data
    df = df.dropna().drop_duplicates() 

    # Make binary 0/1 target
    df[TARGET] = (df[TARGET] == ">50K").astype(int)
    return df

# Expected numerical features ranges to validate
AdultSchema = pa.DataFrameSchema({
    "age": Column(int, Check.in_range(17, 90)),
    "fnlwgt": Column(int, Check.ge(1)),
    "educational-num": Column(int, Check.in_range(1, 16)),
    "capital-gain": Column(int, Check.ge(0)),
    "capital-loss": Column(int, Check.ge(0)),
    "hours-per-week": Column(int, Check.in_range(1, 99)),
}, coerce=True, strict=False)

if __name__ == "__main__":
    dataset_path = os.path.join(DATA_DIR_PATH, "raw/adult.csv")
    df = load_clean(dataset_path)
    AdultSchema.validate(df)
    print("Data loaded and validated successfully.")
    cleaned_dataset_path = os.path.join(DATA_DIR_PATH, "processed/adult.csv")
    df.to_csv(cleaned_dataset_path, index=False)