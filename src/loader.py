from __future__ import annotations
import pandas as pd, re

CASH_PATTERN = re.compile(r"^\$\$CASH_TX")
REQUIRED_COLUMNS = ["Symbol", "Current Price", "Purchase Price", "Quantity"]

def load_yahoo_csv(path: str):
    # sep=None + engine=python: auto-detect (anche CSV con ';')
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]

    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Colonna mancante nel CSV: {c}")

    # Cassa
    cash_df = df[df["Symbol"].astype(str).str.match(CASH_PATTERN)]
    cash_total = float(cash_df["Quantity"].fillna(0).sum()) if not cash_df.empty else 0.0

    # Posizioni
    pos_df = df[~df["Symbol"].astype(str).str.match(CASH_PATTERN)].copy()
    keep = ["Symbol","Current Price","Purchase Price","Quantity","Comment"]
    for k in keep:
        if k not in pos_df.columns:
            pos_df[k] = None
    pos_df = pos_df[keep]

    for col in ["Current Price","Purchase Price","Quantity"]:
        pos_df[col] = pd.to_numeric(pos_df[col], errors="coerce")
    pos_df.dropna(subset=["Quantity"], inplace=True)
    pos_df = pos_df[pos_df["Quantity"] > 0].reset_index(drop=True)

    return pos_df, cash_total
