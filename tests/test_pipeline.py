import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.load_data import load_data
from src.clean import clean_transactions
from src.rfm import calculate_rfm
from src.clustering import prepare_features, evaluate_cluster_counts, fit_clusters

def sample():
    return pd.DataFrame([
        ["1001","A","item",2,"2024-01-01",10,"1","UK"],
        ["1001","B","item2",1,"2024-01-01",5,"1","UK"],
        ["1002","A","item",3,"2024-01-10",10,"1","UK"],
        ["1003","A","item",1,"2024-01-03",20,"2","UK"],
        ["C1004","A","return",-1,"2024-01-04",20,"2","UK"],
        ["1005","A","bad",0,"2024-01-05",10,"3","UK"],
        ["1006","A","missing",1,"2024-01-05",10,None,"UK"],
    ], columns=["InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","CustomerID","Country"])

def test_cleaning_audit_and_rules():
    raw = pd.concat([sample(), sample().iloc[[0]]], ignore_index=True)
    clean, audit = clean_transactions(raw)
    assert audit["duplicate_rows_removed"] == 1
    assert audit["cancellation_rows_excluded"] == 1
    assert audit["missing_customer_rows_excluded"] == 1
    assert audit["invalid_or_nonpositive_rows_excluded"] == 1
    assert len(clean) == 4

def test_rfm_customer_aggregation_and_invoice_frequency():
    clean, _ = clean_transactions(sample())
    rfm, ref = calculate_rfm(clean, "2024-01-11")
    one = rfm.set_index("CustomerID").loc["1"]
    assert one["Frequency"] == 2
    assert one["Monetary"] == 55
    assert one["Recency"] == 1
    assert ref == pd.Timestamp("2024-01-11")

def test_empty_rfm_fails_cleanly():
    with pytest.raises(ValueError, match="No qualifying"):
        calculate_rfm(pd.DataFrame())

def test_loader_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data(ROOT/"does_not_exist.xlsx")

def test_cluster_evaluation_and_reproducibility():
    rfm = pd.DataFrame({
        "CustomerID":[str(i) for i in range(12)],
        "Recency":[1,2,3,4,20,22,24,25,40,42,44,45],
        "Frequency":[20,18,17,16,8,7,6,5,2,2,1,1],
        "Monetary":[1000,900,850,800,300,250,220,200,50,40,30,20]
    })
    scaled, _, _ = prepare_features(rfm)
    ev = evaluate_cluster_counts(scaled, max_k=4)
    assert set(["k","inertia","silhouette"]).issubset(ev.columns)
    a, _, _, _ = fit_clusters(rfm, 3)
    b, _, _, _ = fit_clusters(rfm, 3)
    assert a["Cluster"].tolist() == b["Cluster"].tolist()
    assert a["Cluster"].nunique() == 3

def test_too_few_customers():
    rfm = pd.DataFrame({"CustomerID":["1","2"],"Recency":[1,2],"Frequency":[1,2],"Monetary":[10,20]})
    scaled, _, _ = prepare_features(rfm)
    with pytest.raises(ValueError, match="At least 3"):
        evaluate_cluster_counts(scaled)


def test_loader_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"InvoiceNo":["1"], "Quantity":[1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="schema mismatch"):
        load_data(path)

def test_cleaning_rejects_bad_dates_and_prices():
    raw = sample().iloc[[0]].copy()
    raw.loc[raw.index[0], "InvoiceDate"] = "not-a-date"
    clean, audit = clean_transactions(raw)
    assert clean.empty
    assert audit["missing_date_rows"] == 1

def test_prepare_features_rejects_nonfinite_values():
    rfm = pd.DataFrame({"Recency":[1, np.inf], "Frequency":[1,2], "Monetary":[10,20]})
    with pytest.raises(ValueError, match="non-finite"):
        prepare_features(rfm)
