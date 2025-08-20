import os, sys
import sqlite3
import pandas as pd

REPO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPO_PATH)

from settings import DATA_DIR_PATH

DB_PATH   = os.path.join(DATA_DIR_PATH, "adult.sqlite")
OUT_DIR = os.path.join(DATA_DIR_PATH, "sql_queries_results")
os.makedirs(OUT_DIR, exist_ok=True)

QUERIES = [
    ("basic_aggs",      os.path.join(REPO_PATH, "sql/basic_aggs.sql")),
    ("top5",            os.path.join(REPO_PATH, "sql/top5.sql")),
    ("window",          os.path.join(REPO_PATH, "sql/window.sql")),
    ("join_and_indexes",os.path.join(REPO_PATH, "sql/join_and_indexes.sql")),
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_income ON adult(income);",
    "CREATE INDEX IF NOT EXISTS idx_workclass ON adult(workclass);",
    "CREATE INDEX IF NOT EXISTS idx_occupation ON adult(occupation);",
    'CREATE INDEX IF NOT EXISTS idx_native_country ON adult("native-country");',
]

def read_query(path):
    with open(path, "r") as f:
        return f.read()

def save_df(df, out_name):
    csv_path = os.path.join(OUT_DIR, f"{out_name}.csv")
    df.to_csv(csv_path, index=False)
    # markdown is nice but optional
    try:
        md_path = os.path.join(OUT_DIR, f"{out_name}.md")
        with open(md_path, "w") as md:
            md.write(df.to_markdown(index=False))
    except Exception:
        pass
    print(f"[OK] saved → {csv_path}")

def run_one(con, name, path, tag):
    sql = read_query(path)
    df = pd.read_sql_query(sql, con)
    save_df(df, f"{name}_{tag}")
    dfp = pd.read_sql_query("EXPLAIN QUERY PLAN " + sql, con)
    save_df(dfp, f"{name}_{tag}_explain")

if __name__ == "__main__":
    print(f"[INFO] DB: {DB_PATH}")
    with sqlite3.connect(DB_PATH) as con:
        # create the lookup table
        with open(os.path.join(REPO_PATH, "sql/create_country_table.sql")) as f:
            con.executescript(f.read())
        print("[INFO] country_region created/loaded")

        # run all queries without indexes
        for name, path in QUERIES:
            print(f"[INFO] running {name} before indexes")
            run_one(con, name, path, "before")

        # add indexes
        for idx in INDEXES:
            con.execute(idx)
        print("[INFO] indexes created")

        # run all queries after indexes
        for name, path in QUERIES:
            print(f"[INFO] running {name} after indexes")
            run_one(con, name, path, "after")

    print(f"[DONE] outputs in {OUT_DIR}")
