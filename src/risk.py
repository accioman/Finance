from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

def daily_returns_from_history(df: pd.DataFrame) -> pd.Series:
    close = df["Close"].dropna()
    return close.pct_change().dropna()

def risk_metrics_from_history(df: pd.DataFrame):
    r = daily_returns_from_history(df)
    if r.empty:
        return {"vol_ann": None, "dd_max": None, "ret_ann": None, "sharpe_naive": None}
    vol_ann = r.std() * (252 ** 0.5)
    ret_ann = (1 + r.mean()) ** 252 - 1
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    dd_max = dd.min()
    sharpe = ret_ann / vol_ann if vol_ann else None
    return {
        "vol_ann": float(vol_ann),
        "ret_ann": float(ret_ann),
        "dd_max": float(dd_max),
        "sharpe_naive": float(sharpe) if sharpe is not None else None,
    }

def correlation_matrix(symbols, period="3mo", interval="1d"):
    """Costruisce matrice di correlazione solo con serie di rendimenti valide (>=2 punti)."""
    series_list = []
    for s in symbols:
        try:
            df = yf.download(s, period=period, interval=interval, progress=False, auto_adjust=False)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            r = df["Close"].pct_change().dropna()
            if r.empty or len(r) < 2:
                continue
            series_list.append(r.rename(s))
        except Exception:
            continue
    if not series_list:
        return pd.DataFrame()
    R = pd.concat(series_list, axis=1)
    R = R.dropna(how="all")
    if R.empty:
        return pd.DataFrame()
    return R.corr()
