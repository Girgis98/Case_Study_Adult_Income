import os
import sys
import shutil
import kagglehub
from fireducks import pandas as pd

REPO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPO_PATH)

from settings import USER_NAME, DATA_DIR_PATH

# Move kaggle.json to /home/mladmin/.kaggle/
os.makedirs(f"/home/{USER_NAME}/.kaggle/", exist_ok=True)
kaggle_json_path = os.path.join(REPO_PATH,"kaggle.json")
shutil.copyfile(kaggle_json_path, f"/home/{USER_NAME}/.kaggle/kaggle.json")

# Download latest version
file_path = kagglehub.dataset_download("wenruliu/adult-income-dataset")
downloaded_data_path = os.path.join(file_path, "adult.csv")
data_path = os.path.abspath(os.path.join(DATA_DIR_PATH, "raw/adult.csv"))
shutil.copyfile(downloaded_data_path, data_path)

print("Path to dataset files:", data_path)

# Load the latest version
df = pd.read_csv(data_path)

print("First 5 records:", df.head())
print(df.shape, df.isna().sum().to_dict())