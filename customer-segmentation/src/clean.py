import pandas as pd
import numpy as np

def clean_transactions(df):
    """Return (clean purchase lines, audit). Excludes cancellations/returns from positive-purchase RFM."""
    data = df.copy()
    audit = {"input_rows": int(len(data))}
    data["InvoiceNo"] = data["InvoiceNo"].astype("string").str.strip()
    data["CustomerID"] = data["CustomerID"].astype("string").str.strip()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["UnitPrice"] = pd.to_numeric(data["UnitPrice"], errors="coerce")
    audit["missing_customer_id_rows"] = int(data["CustomerID"].isna().sum() + data["CustomerID"].eq("").sum())
    audit["missing_date_rows"] = int(data["InvoiceDate"].isna().sum())
    audit["duplicate_rows_removed"] = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()
    # A leading C in InvoiceNo is the documented UCI cancellation convention.
    cancellation = data["InvoiceNo"].str.upper().str.startswith("C", na=False)
    audit["cancellation_rows_excluded"] = int(cancellation.sum())
    missing_customer = data["CustomerID"].isna() | data["CustomerID"].eq("")
    audit["missing_customer_rows_excluded"] = int(missing_customer.sum())
    invalid = data["InvoiceDate"].isna() | data["Quantity"].isna() | data["UnitPrice"].isna()
    invalid |= data["Quantity"].le(0) | data["UnitPrice"].le(0)
    audit["invalid_or_nonpositive_rows_excluded"] = int((invalid & ~cancellation & ~missing_customer).sum())
    keep = ~(cancellation | missing_customer | invalid)
    data = data.loc[keep].copy()
    data["CustomerID"] = data["CustomerID"].str.replace(r"\.0$", "", regex=True)
    data["LineValue"] = data["Quantity"] * data["UnitPrice"]
    data = data[data["LineValue"].gt(0)].copy()
    audit["clean_purchase_rows"] = int(len(data))
    audit["clean_customers"] = int(data["CustomerID"].nunique())
    audit["clean_invoices"] = int(data["InvoiceNo"].nunique())
    return data, audit
