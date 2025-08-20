# Case Study — Adult Income

## How to run
```bash
# clustering
python src/clustering.py
# supervised
python src/supervised.py
# sql (before/after explain)
python scripts/run_sql.py
```

Outputs land under `data/`:
- `clustering_results/` — scores, plots, labels, summaries, centroids, logs
- `supervised_results/` — metrics, ROC/PR plots, feature importances, log
- `sql_queries_results/` — query results and EXPLAIN plans (before/after)

## Approach
- **Cleaning:** parse `?` as NaN, trim strings, drop rows with NaN/duplicates, map `income` to 0/1.
- **Unsupervised:** Standardize numerics + OHE categoricals → PCA(20). KMeans (k via silhouette+elbow) and HDBSCAN. Profile clusters by numeric medians and top categorical modes. Visualize on PCA‑2D only for plotting.
- **Supervised:** 60/20/20 stratified split. Baseline Logistic Regression (OHE). Advanced HistGradientBoosting (OrdinalEncoder) with a tiny grid. Metrics on valid and test (AUC, PR‑AUC, CM with F1‑tuned threshold). Interpret via coefficients and permutation importance.
- **SQL:** SQLite with a small `country_region` lookup. Four queries: basic aggs, top‑5, window, join. Indexes added on common filter/group/join keys; compare EXPLAIN before/after.

## Notes
- Random seeds set to 42.
- No leakage: all transforms live inside pipelines.
- Plots saved as PNGs for the README.
