import os, sys
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
REPO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPO_PATH)
from sklearn.pipeline import Pipeline

from settings import CAT_COLS, NUM_COLS, DATA_DIR_PATH

def make_preprocess_pipeline(sparse_ohe=False): # sparse=False to avoid implicit conversion to dense in the ColumnTransformer as StandardScaler outputs dense arrays
    ohe = OneHotEncoder(sparse_output=sparse_ohe)
    scaler = StandardScaler(with_mean=True, with_std=True)
    ct = ColumnTransformer(
        transformers=[
            ("num", scaler, NUM_COLS),
            ("cat", ohe, CAT_COLS),
        ],
        remainder="drop" # don't transform the target column
    )
    return ct

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR_PATH, "raw/adult.csv"))
    preprocess_pipeline = make_preprocess_pipeline()
    df_processed = preprocess_pipeline.fit_transform(df)
    print("Processed DataFrame shape:", df_processed.shape)