import os

USER_NAME = "mladmin"

# Current dir path
REPO_PATH = os.path.join(os.path.dirname(__file__))

SCRIPTS_DIR_PATH = os.path.join(REPO_PATH, "scripts")
NOTEBOOKS_DIR_PATH = os.path.join(REPO_PATH, "notebooks")
DATA_DIR_PATH = os.path.join(REPO_PATH, "data")
SQL_DIR_PATH = os.path.join(REPO_PATH, "sql")
SRC_DIR_PATH = os.path.join(REPO_PATH, "src")

# Constants
CAT_COLS = ["workclass","education","marital-status","occupation",
            "relationship","race","gender","native-country"]
NUM_COLS = ["age","fnlwgt","educational-num","capital-gain","capital-loss","hours-per-week"]
TARGET   = "income"