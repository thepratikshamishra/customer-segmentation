from pathlib import Path
import pandas as pd

REQUIRED = {"InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"}

def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. See README.md for download instructions.")
    # Excel is used by the UCI Online Retail dataset; CSV is also supported.
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Supported dataset formats: .xlsx, .xls, .csv")
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Dataset schema mismatch. Missing columns: {sorted(missing)}")
    return df
