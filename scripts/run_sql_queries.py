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
    ("basic_aggs",      "sql/basic_aggs.sql"),
    ("top5",            "sql/top5.sql"),
    # ("window",          "sql/window.sql"),
    # ("join_and_indexes","sql/join_and_indexes.sql"),
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

def run_one(con, name, path):
    sql = read_query(path)
    # results
    df = pd.read_sql_query(sql, con)
    save_df(df, name)
    # explain
    explain_sql = "EXPLAIN QUERY PLAN " + sql
    dfp = pd.read_sql_query(explain_sql, con)
    save_df(dfp, f"{name}_explain")

if __name__ == "__main__":
    print(f"[INFO] DB: {DB_PATH}")
    with sqlite3.connect(DB_PATH) as con:
        for name, path in QUERIES:
            print(f"[INFO] running {name} ({path})")
            run_one(con, name, path)
    print(f"[DONE] outputs in {OUT_DIR}")
