import os, sys, time, contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

REPO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPO_PATH)

from settings import NUM_COLS, CAT_COLS, TARGET, DATA_DIR_PATH
from src.data_utils import load_clean, AdultSchema

OUT_DIR = os.path.join(DATA_DIR_PATH, "supervised_results")
os.makedirs(OUT_DIR, exist_ok=True)


def get_df():
    path = os.path.join(DATA_DIR_PATH, "raw", "adult.csv")
    print(f"[INFO] loading {path}")
    df = load_clean(path)
    print(f"[INFO] df shape: {df.shape}")
    AdultSchema.validate(df)
    print("[INFO] schema ok")
    return df

def split(df):
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET].astype(int)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_tst, y_val, y_tst = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )
    print(f"[INFO] splits -> train:{X_tr.shape[0]} valid:{X_val.shape[0]} test:{X_tst.shape[0]}")
    return X_tr, X_val, X_tst, y_tr, y_val, y_tst


def fit_logreg(X, y):
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT_COLS),
    ])
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    t0 = time.perf_counter()
    pipe.fit(X, y)
    t = time.perf_counter() - t0
    print(f"[INFO] LogReg fit in {t:.3f}s")
    return pipe

def fit_hgb(X, y):
    # ordinal cats + std numerics; tiny grid
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
    ])
    base = Pipeline([("prep", prep), ("clf", HistGradientBoostingClassifier(random_state=42))])
    grid = {
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_leaf_nodes": [31, 63],
        "clf__min_samples_leaf": [20, 50],
    }
    gs = GridSearchCV(base, grid, cv=3, scoring="roc_auc", n_jobs=-1)
    t0 = time.perf_counter()
    gs.fit(X, y)
    t = time.perf_counter() - t0
    print(f"[INFO] HGB fit (grid) in {t:.3f}s | best params={gs.best_params_}")
    return gs.best_estimator_

def evaluate(model, name, X_val, y_val, X_tst, y_tst):
    pv = model.predict_proba(X_val)[:,1]
    pt = model.predict_proba(X_tst)[:,1]

    # metrics
    res = {
        "model": name,
        "valid_auc": roc_auc_score(y_val, pv),
        "valid_pr_auc": average_precision_score(y_val, pv),
        "test_auc": roc_auc_score(y_tst, pt),
        "test_pr_auc": average_precision_score(y_tst, pt),
    }
    print(f"[{name}] valid AUC={res['valid_auc']:.4f} PR-AUC={res['valid_pr_auc']:.4f} | "
          f"test AUC={res['test_auc']:.4f} PR-AUC={res['test_pr_auc']:.4f}")

    # choose threshold to maximize F1 on valid
    ts = np.linspace(0.1, 0.9, 33)
    f1s = [f1_score(y_val, (pv>t).astype(int)) for t in ts]
    t_star = float(ts[int(np.argmax(f1s))])
    yv_hat = (pv>t_star).astype(int)
    yt_hat = (pt>t_star).astype(int)
    cm_v = confusion_matrix(y_val, yv_hat)
    cm_t = confusion_matrix(y_tst, yt_hat)
    print(f"[{name}] threshold={t_star:.2f}\nvalid CM:\n{cm_v}\ntest CM:\n{cm_t}")

    # save metrics
    mpath = os.path.join(OUT_DIR, f"metrics_{name}.csv")
    pd.DataFrame([{
        **res,
        "threshold": t_star,
        "valid_cm_tn": cm_v[0,0], "valid_cm_fp": cm_v[0,1],
        "valid_cm_fn": cm_v[1,0], "valid_cm_tp": cm_v[1,1],
        "test_cm_tn": cm_t[0,0],  "test_cm_fp": cm_t[0,1],
        "test_cm_fn": cm_t[1,0],  "test_cm_tp": cm_t[1,1],
    }]).to_csv(mpath, index=False)
    print(f"[INFO] saved metrics -> {mpath}")

    # plots (valid only)
    RocCurveDisplay.from_predictions(y_val, pv)
    plt.title(f"ROC (valid) - {name}")
    plt.savefig(os.path.join(OUT_DIR, f"roc_{name}.png")); plt.close()

    PrecisionRecallDisplay.from_predictions(y_val, pv)
    plt.title(f"PR (valid) - {name}")
    plt.savefig(os.path.join(OUT_DIR, f"pr_{name}.png")); plt.close()

    return res, t_star

def dump_importances_logreg(model):
    # OHE names
    ohe = model.named_steps["prep"].named_transformers_["cat"]
    feat_names = NUM_COLS + list(ohe.get_feature_names_out(CAT_COLS))
    coef = model.named_steps["clf"].coef_.ravel()
    df = pd.DataFrame({"feature": feat_names, "coef": coef, "abs_coef": np.abs(coef)})
    df.sort_values("abs_coef", ascending=False, inplace=True)
    path = os.path.join(OUT_DIR, "feature_importance_logreg.csv")
    df.to_csv(path, index=False)
    print(f"[INFO] saved logreg coeffs -> {path}")

def dump_importances_hgb(model, X_val, y_val):
    r = permutation_importance(
        model, X_val, y_val,
        scoring="roc_auc", n_repeats=5, n_jobs=-1, random_state=42
    )
    # with OrdinalEncoder we have one column per original feature
    names = NUM_COLS + CAT_COLS
    imp = pd.DataFrame({
        "feature": names,
        "perm_importance_mean": r.importances_mean,
        "perm_importance_std":  r.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)
    path = os.path.join(OUT_DIR, "feature_importance_hgb.csv")
    imp.to_csv(path, index=False)
    print(f"[INFO] saved HGB permutation importance -> {path}")

if __name__ == "__main__":
    log_path = os.path.join(OUT_DIR, "supervised_run.log")
    with open(log_path, "w") as f, contextlib.redirect_stdout(f):
        df = get_df()
        X_tr, X_val, X_tst, y_tr, y_val, y_tst = split(df)

        logreg = fit_logreg(X_tr, y_tr)
        hgb    = fit_hgb(X_tr, y_tr)

        m1, thr1 = evaluate(logreg, "logreg", X_val, y_val, X_tst, y_tst)
        m2, thr2 = evaluate(hgb,    "hgb",    X_val, y_val, X_tst, y_tst)

        dump_importances_logreg(logreg)
        dump_importances_hgb(hgb, X_val, y_val)

        # save predictions on test for audit
        preds = pd.DataFrame({
            "y_test": y_tst.values,
            "p_logreg": logreg.predict_proba(X_tst)[:,1],
            "p_hgb":    hgb.predict_proba(X_tst)[:,1],
        })
        preds.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)
        print("[INFO] wrote test_predictions.csv")

    print(f"[INFO] all prints saved to {log_path}")
