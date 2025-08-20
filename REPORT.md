
# Analysis Report

## Dataset
- **Adult Income (UCI)**, ~48k rows, mixed numeric/categorical.
- `?` parsed as NaN, rows with NaN dropped, duplicates removed, `income` mapped to 0/1.

## Unsupervised (KMeans + HDBSCAN)
- PCA(20) on scaled/encoded features to stabilize distances.
- KMeans silhouette peaks at **k=9** (silhouette=0.146).
- HDBSCAN found **2 clusters**, capturing a dense majority group and a high-capital-gain niche.

**Notable KMeans clusters (income rate and size):**
| index | p_income_gt_50k | count |
|-------|-----------------|-------|
| 5     | 1.000000        |   229 |
| 7     | 0.682440        |  5558 |
| 8     | 0.524653        |  2089 |
| 2     | 0.301714        |  9045 |
| 4     | 0.301239        |  2825 |
| 0     | 0.201063        |  5456 |
| 3     | 0.169636        |  4097 |
| 1     | 0.086014        |  6964 |

**Interpretation (≤300 words):**
We embedded the cleaned tabular data into a 20‑D PCA space after scaling numerics and one‑hot encoding categoricals, then clustered in that space. KMeans achieved its best separation at k=9 (silhouette 0.146). One small segment (cluster 5; ~229 records) concentrates extreme `capital-gain≈99,999`, long hours, professional occupations, and a 100% `income>50K` rate—an obvious high‑earner niche. Larger segments show gradations: clusters with higher `educational-num`, professional/managerial occupations, and longer `hours-per-week` trend toward higher income rates (e.g., clusters 7 and 8 with ~0.68 and ~0.52 high‑income). Conversely, clusters tied to lower `educational-num`, clerical/service roles, or fewer weekly hours show markedly lower positive rates (clusters 1 and 6). HDBSCAN, operating without a preset k, recovered a dominant core cluster and a compact high‑gain cluster, aligning with the KMeans picture and lending stability to the segmentation. These segments can guide differentiated thresholds (e.g., risk or targeting), and the concordance across algorithms increases confidence that separation is driven by education/occupation/hours and capital flows rather than noise.

## Supervised (Classification)
Two models:
- **Logistic Regression (baseline)** — Valid AUC=0.902, Test AUC=0.904; Valid PR‑AUC=0.763.
- **HistGradientBoosting (tuned)** — Valid AUC=0.924, Test AUC=0.925; Valid PR‑AUC=0.826.

On the test set, HistGB improves both AUC and PR‑AUC over the baseline. Thresholds were chosen by maximizing F1 on the validation set, and we report confusion matrices in the metrics CSVs.

**Feature signals**
- LogReg (top |coef|): see `feature_importance_logreg.csv` (capital‑gain, marital‑status variants, select native‑country signals).
- HistGB (permutation importance): `feature_importance_hgb.csv` highlights age, capital‑gain, educational‑num, marital‑status, and relationship as key drivers.

## SQL
We loaded `adult` into SQLite and added a small `country_region` lookup. For each query we saved **before/after** `EXPLAIN QUERY PLAN`. A representative improvement: GROUP BY on `workclass` changed from `SCAN adult + TEMP B‑TREE` to `SCAN adult USING INDEX idx_workclass`, eliminating the grouping temp structure (ORDER BY on aggregates still uses a temp B‑tree, as expected). See `data/sql_queries_results/*_before*.csv` vs `*_after*.csv`.

## How to Run (summary)
- `python src/clustering.py` → writes to `data/clustering_results/` (scores, plots, labels, summaries, centers, logs).
- `python src/supervised.py` → writes to `data/supervised_results/` (metrics, plots, importances, log).
- `python scripts/run_sql.py` → writes to `data/sql_queries_results/` (results + explain, before/after).

