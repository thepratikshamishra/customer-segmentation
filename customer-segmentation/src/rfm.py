import pandas as pd

def calculate_rfm(clean_df, reference_date=None):
    if clean_df.empty:
        raise ValueError("No qualifying purchase rows remain; cannot calculate RFM.")
    data = clean_df.copy()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    if data["InvoiceDate"].isna().any():
        raise ValueError("Clean input contains invalid InvoiceDate values.")
    if reference_date is None:
        reference_date = data["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    reference_date = pd.Timestamp(reference_date)
    # Frequency counts distinct invoices, not individual line items.
    grouped = data.groupby("CustomerID", dropna=True).agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("LineValue", "sum"),
    )
    grouped["Recency"] = (reference_date.normalize() - grouped["LastPurchase"].dt.normalize()).dt.days
    result = grouped[["Recency", "Frequency", "Monetary"]].reset_index()
    if (result["Recency"] < 0).any():
        raise ValueError("Reference date precedes at least one purchase.")
    return result, reference_date
