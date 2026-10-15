 # Customer Segmentation Using RFM and K-Means

Beginner-friendly Data Science internship project using transaction data, Pandas, Matplotlib/Seaborn, and scikit-learn.

## Dataset (not bundled)
**Recommended source:** UCI Machine Learning Repository, *Online Retail* dataset: https://archive.ics.uci.edu/dataset/352/online+retail  
Direct file commonly listed by UCI: https://archive.ics.uci.edu/static/public/352/online+retail.xlsx

The dataset contains 541,909 transaction-line rows and 8 fields in the original release. Columns: `InvoiceNo` (invoice/transaction identifier; invoices starting with C indicate cancellations), `StockCode` (product code), `Description` (product description), `Quantity` (units), `InvoiceDate` (timestamp), `UnitPrice` (price per unit in GBP), `CustomerID` (customer identifier; some missing), `Country` (customer country). It covers transactions for a UK-based online retailer from 2010-12-01 through 2011-12-09. The source page should be checked at download time for current access and licensing terms; UCI pages have historically listed a Creative Commons Attribution 4.0 International (CC BY 4.0) license for this dataset, but verify the current source record before redistribution. Raw data is deliberately not included in this repository.

### Obtain the dataset
1. Open the UCI source page above and download `Online Retail.xlsx`.
2. Place it at `data/online_retail.xlsx`.
3. Install dependencies and run the commands below.

A compatible CSV is accepted with the same column names. No dataset download is silently attempted by the pipeline; this makes data provenance and failures transparent.

## Run it

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_analysis.py
```

Optional explicit reference date and CSV path:

```bash
python run_analysis.py --data data/online_retail.csv --reference-date 2011-12-10 --max-k 8
```

Tests:

```bash
pytest -q
```

## Cleaning decisions
- Exact duplicate rows are removed and counted in `outputs/reports/cleaning_audit.json`.
- Missing customer IDs are excluded from customer-level RFM; they are not invented as a customer.
- Invalid dates/numbers, non-positive quantity or unit price, and cancellation invoices (invoice number begins with `C`) are excluded from positive-purchase RFM.
- `LineValue = Quantity × UnitPrice`. Since cancellations/returns are excluded, Monetary is **gross qualifying purchase value**, not net revenue after returns. A production net-sales analysis should match returns to original invoices where possible and calculate net value separately.
- Invoice lines are aggregated to customer level. Frequency counts distinct invoice numbers, not rows.
- Reference date defaults to midnight on the day after the latest valid purchase date. Supply `--reference-date` for comparisons across runs.

## RFM
- **Recency:** number of whole days between reference date and the customer's most recent qualifying purchase date (smaller is more recent).
- **Frequency:** number of distinct qualifying invoices per customer.
- **Monetary:** sum of qualifying line values per customer (GBP in the original dataset).

## Clustering method
RFM columns are non-negative and typically right-skewed. The pipeline applies `log1p`, then `StandardScaler`, and fits K-Means with `random_state=42` and `n_init=10`. It evaluates k from 2 through `min(--max-k, customer_count - 1)` using inertia (elbow curve) and silhouette score. The default selection is the tested k with the highest silhouette score; this is a transparent heuristic, not a guarantee of business-optimal segments. Cluster IDs have no natural order. Segment names and proposed actions are based on computed medians and are hypotheses to validate, not proven outcomes. Inspect cluster sizes and the figures before presenting the results.

## Outputs
- `outputs/tables/clean_transactions.csv`: cleaned qualifying transaction lines.
- `outputs/tables/customer_rfm.csv`: one row per identified customer.
- `outputs/tables/customer_segments.csv`: RFM values plus assigned cluster.
- `outputs/tables/segment_summary.csv`: counts, medians, monetary contribution, profile and proposed action.
- `outputs/tables/cluster_evaluation.csv`: inertia and silhouette for tested k.
- `outputs/figures/`: transaction-value distribution, customer spending distribution, RFM correlation, elbow curve, silhouette scores, and cluster scatter plot.
- `outputs/reports/cleaning_audit.json`: counts of excluded/removed records.
- `outputs/reports/evaluator_report.md`: concise report generated from the actual run.
- `outputs/models/kmeans_pipeline.joblib`: fitted model, scaler, feature names, transform description, and reference date. Load only trusted joblib files.

## Limitations and extensions
- Cancellation/return lines are excluded rather than reconciled to original sales; monetary value is not net of returns.
- K-Means favors roughly spherical clusters and may be sensitive to outliers despite log transformation.
- Silhouette alone cannot decide whether segments are commercially useful; compare with the elbow curve, segment sizes, stability, and business context.
- No causal claim is made about the proposed marketing actions.
- Potential extensions: match returns to original purchases, compare robust scaling, test cluster stability, and validate campaigns with controlled experiments.

## Source citation
Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. Dataset page: https://archive.ics.uci.edu/dataset/352/online+retail. License listed by UCI: CC BY 4.0.

## AI assistance disclosure
If the internship rules require it, disclose that AI assistance was used to scaffold and review parts of the code/documentation, and that the submitter ran the tests, inspected outputs, and can explain the methodology. Adapt this statement to match the actual assistance permitted by QSkill rules.
