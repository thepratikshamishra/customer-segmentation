from pathlib import Path
import argparse, json, sys
import pandas as pd
import joblib
from src.load_data import load_data
from src.clean import clean_transactions
from src.rfm import calculate_rfm
from src.clustering import prepare_features, evaluate_cluster_counts, fit_clusters
from src.visualize import make_figures

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Run customer RFM analysis and K-Means segmentation.")
    parser.add_argument("--data", default=str(ROOT/"data"/"online_retail.xlsx"), help="Path to UCI Online Retail .xlsx or compatible CSV")
    parser.add_argument("--reference-date", default=None, help="Optional reference date, e.g. 2011-12-10")
    parser.add_argument("--max-k", type=int, default=8)
    args = parser.parse_args()
    out = ROOT/"outputs"
    for folder in ["figures","tables","reports","models"]:
        (out/folder).mkdir(parents=True, exist_ok=True)
    raw = load_data(args.data)
    clean, audit = clean_transactions(raw)
    rfm, reference_date = calculate_rfm(clean, args.reference_date)
    scaled, scaler, _ = prepare_features(rfm)
    evaluation = evaluate_cluster_counts(scaled, max_k=args.max_k)
    # Select highest silhouette score among evaluated values; metrics and caveat documented in report.
    best = evaluation.sort_values(["silhouette","k"], ascending=[False,True]).iloc[0]
    k = int(best["k"])
    clustered, model, scaler, transformed = fit_clusters(rfm, k)
    # Descriptive segment table: medians are robust to skew; sums give actual monetary contribution.
    summary = clustered.groupby("Cluster").agg(
        Customers=("CustomerID","nunique"),
        MedianRecency=("Recency","median"),
        MedianFrequency=("Frequency","median"),
        MedianMonetary=("Monetary","median"),
        MeanMonetary=("Monetary","mean"),
        TotalMonetary=("Monetary","sum"),
    ).reset_index()
    summary["CustomerSharePct"] = summary["Customers"] / len(clustered) * 100
    total = summary["TotalMonetary"].sum()
    summary["MonetarySharePct"] = summary["TotalMonetary"] / total * 100 if total else 0
    # Generate neutral, data-driven interpretations from segment medians.
    overall = clustered[["Recency","Frequency","Monetary"]].median()
    def describe(row):
        labels = []
        if row["MedianRecency"] <= overall["Recency"]: labels.append("more recent")
        else: labels.append("less recent")
        if row["MedianFrequency"] >= overall["Frequency"]: labels.append("more frequent")
        else: labels.append("less frequent")
        if row["MedianMonetary"] >= overall["Monetary"]: labels.append("higher spending")
        else: labels.append("lower spending")
        return ", ".join(labels)
    summary["Profile"] = summary.apply(describe, axis=1)
    summary["ProposedAction"] = summary.apply(lambda r:
        "Reward loyalty and test relevant cross-sell offers" if r["MedianRecency"] <= overall["Recency"] and r["MedianFrequency"] >= overall["Frequency"] and r["MedianMonetary"] >= overall["Monetary"]
        else "Use a targeted win-back message and measure response" if r["MedianRecency"] > overall["Recency"]
        else "Test low-cost onboarding, bundles, or repeat-purchase incentives", axis=1)
    min_cluster_size = int(summary["Customers"].min())
    small_cluster_warning = min_cluster_size < max(5, int(0.02 * len(clustered)))
    clean.to_csv(out/"tables"/"clean_transactions.csv", index=False)
    rfm.to_csv(out/"tables"/"customer_rfm.csv", index=False)
    clustered.to_csv(out/"tables"/"customer_segments.csv", index=False)
    summary.to_csv(out/"tables"/"segment_summary.csv", index=False)
    evaluation.to_csv(out/"tables"/"cluster_evaluation.csv", index=False)
    (out/"reports"/"cleaning_audit.json").write_text(json.dumps(audit, indent=2))
    make_figures(clean, rfm, evaluation, clustered, out/"figures")
    joblib.dump({"model":model, "scaler":scaler, "features":["Recency","Frequency","Monetary"], "transform":"log1p then StandardScaler", "reference_date":str(reference_date)}, out/"models"/"kmeans_pipeline.joblib")
    report = f"""# Customer Segmentation — Results Report

## Run facts
- Source file: `{args.data}`
- Raw rows: {audit['input_rows']:,}
- Qualifying purchase line items after cleaning: {audit['clean_purchase_rows']:,}
- Customers with usable IDs: {audit['clean_customers']:,}
- Distinct invoices: {audit['clean_invoices']:,}
- RFM reference date: {reference_date.date()}
- Selected k: {k} (highest silhouette among tested k values; this is a heuristic, not a universal truth)
- Selected silhouette score: {best['silhouette']:.4f}
- Selected inertia: {best['inertia']:.2f}
- Smallest segment size: {min_cluster_size} customers
- Segment-size check: {"WARNING — at least one segment is very small; interpret cautiously" if small_cluster_warning else "No segment below the simple size-warning threshold"}

## Cleaning audit
```json
{json.dumps(audit, indent=2)}
```

## Segment summary
{summary.to_markdown(index=False, floatfmt=".2f")}

## Interpretation and proposed actions
Cluster IDs are arbitrary. Profiles are generated by comparing each segment's median RFM to the whole-customer median. Actions are hypotheses to test, not proven outcomes. See `tables/segment_summary.csv` and `figures/` for the detailed outputs.

## Caveats
Cancellation invoices (InvoiceNo beginning with C), non-positive quantities/prices, invalid dates/numbers, missing customer IDs, and exact duplicate rows are excluded from positive-purchase RFM. This keeps Monetary as gross qualifying purchase value rather than net revenue after returns. Returns are not subtracted because the dataset's cancellation rows are excluded; a separate net-sales analysis would need explicit invoice-line matching and return allocation. Reference date defaults to one day after the latest valid purchase date unless specified.
"""
    (out/"reports"/"evaluator_report.md").write_text(report)
    print("Analysis completed.")
    print(f"Customers: {len(rfm)} | selected k: {k} | silhouette: {best['silhouette']:.4f}")
    print(f"Report: {out/'reports'/'evaluator_report.md'}")

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
