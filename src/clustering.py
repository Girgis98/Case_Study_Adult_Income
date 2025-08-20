import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import contextlib
import hdbscan

REPO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPO_PATH)

from src.preprocessing import make_preprocess_pipeline
from settings import NUM_COLS, CAT_COLS, TARGET, DATA_DIR_PATH
from src.data_utils import load_clean, AdultSchema


OUT_DIR = os.path.join(DATA_DIR_PATH, "results_stats_and_weights")
os.makedirs(OUT_DIR, exist_ok=True)


def get_df():
    dataset_path = os.path.join(DATA_DIR_PATH, "raw/adult.csv")
    print(f"[INFO] Loading dataset from {dataset_path} ...")
    df = load_clean(dataset_path)
    print(f"[INFO] Loaded dataset with shape: {df.shape}")
    AdultSchema.validate(df)
    print("[INFO] Schema validation passed.")
    return df

def run_hdbscan(X_pca):
    print("[INFO] Running HDBSCAN on PCA space...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(X_pca)  # -1 = noise
    probs = getattr(clusterer, "probabilities_", None)

    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[INFO] HDBSCAN clusters={n_clusters}, noise={n_noise}")

    # silhouette only if we have >=2 clusters and at least some non-noise
    sil = None
    valid_idx = labels != -1
    if n_clusters >= 2 and valid_idx.any():
        try:
            sil = float(silhouette_score(X_pca[valid_idx], labels[valid_idx]))
            print(f"[INFO] HDBSCAN silhouette (non-noise) = {sil:.4f}")
        except Exception as e:
            print(f"[WARN] Silhouette failed: {e}")

    # save label/prob tables
    df_hdb = pd.DataFrame({
        "cluster_hdb": labels,
        "probability": probs if probs is not None else np.ones_like(labels, dtype=float),
    })
    path = os.path.join(OUT_DIR, "hdbscan_labels.csv")
    df_hdb.to_csv(path, index=False)
    print(f"[INFO] Saved HDBSCAN labels → {path}")
    return labels, probs, sil

def run_clustering_pipeline_kmeans(df, dry_run=False):
    X = df[NUM_COLS + CAT_COLS].copy()
    print(f"[INFO] Running preprocessing + PCA on features: {X.shape}")

    pipe_base = Pipeline([
        ("prep", make_preprocess_pipeline(sparse_ohe=False)),
        ("pca", PCA(n_components=20, random_state=42)),
    ])

    X_pca = pipe_base.fit_transform(X)
    print(f"[INFO] PCA transformed shape: {X_pca.shape}")

    # Elbow & silhouette for KMeans
    scores = []
    max_k = 3 if dry_run else 10
    print(f"[INFO] Evaluating KMeans for k=2..{max_k-1}")
    for k in range(2, max_k):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels)
        print(f"   k={k:2d} | inertia={km.inertia_:.2f} | silhouette={sil:.4f}")
        scores.append((k, km.inertia_, sil))

    # save scores
    df_scores = pd.DataFrame(scores, columns=["k", "inertia", "silhouette"])
    out_path = os.path.join(OUT_DIR, "clustering_scores.csv")
    df_scores.to_csv(out_path, index=False)
    print(f"[INFO] Saved scores → {out_path}")

    return scores, X_pca


def plot_elb_sil(scores):
    print(f"[INFO] Plotting elbow and silhouette for {len(scores)} k-values...")
    ks, inertias, sils = zip(*scores)
    plt.plot(ks, inertias, marker="o")
    plt.title("Elbow")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.savefig(os.path.join(OUT_DIR, "elbow.png"))
    plt.close()

    plt.plot(ks, sils, marker="o")
    plt.title("Silhouette vs k")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.savefig(os.path.join(OUT_DIR, "silhouette.png"))
    plt.close()
    print(f"[INFO] Saved elbow.png and silhouette.png to {OUT_DIR}")

def plot_2d(X_pca, labels, name):
    # 2D PCA for plotting
    p2 = PCA(n_components=2, random_state=42).fit_transform(X_pca)
    plt.scatter(p2[:,0], p2[:,1], c=labels, s=8, alpha=0.6)
    plt.title(f"{name} Clustering (PCA-2D)")
    out_path = os.path.join(OUT_DIR, f"{name}_pca2.png")
    plt.savefig(out_path); plt.close()
    print(f"[INFO] Saved {name} 2D plot to {out_path}")

def final_clustering(df, X_pca, scores):
    # pick k with max silhouette
    best_k = max(scores, key=lambda t: t[2])[0]
    print(f"[INFO] Best k chosen by silhouette: {best_k}")
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    clusters_km = km.fit_predict(X_pca)

    df = df.copy()
    df["clusters_km"] = clusters_km
    print(f"[INFO] Cluster KMeans counts:\n{df['clusters_km'].value_counts()}")

    # Profile distributions per cluster
    summary_num_km = df.groupby("clusters_km")[NUM_COLS].median()
    summary_cat_km = df.groupby("clusters_km")[CAT_COLS].agg(lambda x: x.value_counts().index[0])
    summary_km = pd.concat([summary_num_km, summary_cat_km], axis=1)
    print("[INFO] Cluster KMeans summary stats:")
    print(summary_km)

    # save clustered df
    clustered_km_path = os.path.join(OUT_DIR, "adult_with_clusters_km.csv")
    df.to_csv(clustered_km_path, index=False)
    print(f"[INFO] Saved dataframe with clusters to: {clustered_km_path}")

    # save summary both csv + markdown
    summary_km_path_csv = os.path.join(OUT_DIR, "cluster_km_summary.csv")
    summary_km_path_md = os.path.join(OUT_DIR, "cluster_km_summary.md")
    summary_km.to_csv(summary_km_path_csv)
    summary_km.to_markdown(summary_km_path_md)
    print(f"[INFO] Saved cluster km summary → {summary_km_path_csv}, {summary_km_path_md}")

    # save cluster centers (PCA space)
    centers_pca = pd.DataFrame(
        km.cluster_centers_,
        columns=[f"PC{i+1}" for i in range(km.cluster_centers_.shape[1])],
    )
    centers_pca["cluster_km"] = np.arange(best_k)
    centers_path = os.path.join(OUT_DIR, "cluster_centers_pca.csv")
    centers_pca.to_csv(centers_path, index=False)
    print(f"[INFO] Saved KMeans cluster centers (PCA) → {centers_path}")


    # HDBSCAN
    labels_hdb, probs_hdb, sil_hdb = run_hdbscan(X_pca)
    df["cluster_hdb"] = labels_hdb
    df["hdb_prob"] = probs_hdb if probs_hdb is not None else 1.0

    # HDBSCAN summary, exclude noise
    if (labels_hdb != -1).any():
        summary_num_hdb = df[df["cluster_hdb"] != -1].groupby("cluster_hdb")[NUM_COLS].median()
        summary_cat_hdb = df[df["cluster_hdb"] != -1].groupby("cluster_hdb")[CAT_COLS].agg(lambda x: x.value_counts().index[0])
        summary_hdb = pd.concat([summary_num_hdb, summary_cat_hdb], axis=1)
        
        # save summary both csv + markdown
        summary_hdb_path_csv = os.path.join(OUT_DIR, "cluster_hdb_summary.csv")
        summary_hdb_path_md = os.path.join(OUT_DIR, "cluster_hdb_summary.md")
        summary_hdb.to_csv(summary_hdb_path_csv)
        summary_hdb.to_markdown(summary_hdb_path_md)
        print(f"[INFO] Saved cluster hdb summary → {summary_hdb_path_csv}, {summary_hdb_path_md}")
    else:
        print("[INFO] HDBSCAN produced only noise; no summary written.")
        summary_hdb = None

    # save full df with both labels
    out_df_path = os.path.join(OUT_DIR, "adult_with_clusters_km_hdb.csv")
    df.to_csv(out_df_path, index=False)
    print(f"[INFO] Saved dataframe with KMeans+HDBSCAN labels → {out_df_path}")

    return df, summary_km, summary_hdb, sil_hdb, centers_pca

if __name__ == "__main__":
    log_path = os.path.join(OUT_DIR, "clustering_run.log")
    with open(log_path, "w") as f, contextlib.redirect_stdout(f):
        df = get_df()
        scores, X_pca = run_clustering_pipeline_kmeans(df, dry_run=False)
        plot_elb_sil(scores)
        df_final, summary_km, summary_hdb, sil_hdb, centers = final_clustering(df, X_pca, scores)
        plot_2d(X_pca, df_final["cluster_hdb"].values, "HDBSCAN")
        plot_2d(X_pca, df_final["clusters_km"].values, "KMeans")
        print(f"[INFO] All prints also saved to {log_path}")
