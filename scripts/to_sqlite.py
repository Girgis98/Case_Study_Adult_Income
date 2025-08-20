import os, sys
import sqlite3

REPO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPO_PATH)

from settings import DATA_DIR_PATH
from src.data_utils import load_clean, AdultSchema

def get_df():
    dataset_path = os.path.join(DATA_DIR_PATH, "raw/adult.csv")
    print(f"[INFO] Loading dataset from {dataset_path} ...")
    df = load_clean(dataset_path)
    print(f"[INFO] Loaded dataset with shape: {df.shape}")
    AdultSchema.validate(df)
    print("[INFO] Schema validation passed.")
    return df

def save_to_sqlite(df, db_path):
    print(f"[INFO] Saving DataFrame to SQLite database at {db_path} ...")
    conn = sqlite3.connect(db_path)
    df.to_sql("adult", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[INFO] Data saved to {db_path}")

def main():
    df = get_df()
    db_path = os.path.join(DATA_DIR_PATH, "adult.sqlite")
    save_to_sqlite(df, db_path)

if __name__ == "__main__":
    main()