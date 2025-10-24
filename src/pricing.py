from __future__ import annotations
import pandas as pd
import yfinance as yf
from typing import Optional

def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def get_last_price(ticker: str) -> Optional[float]:
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if not hist.empty:
            return _safe_float(hist["Close"].iloc[-1])
        info = getattr(t, "fast_info", None)
        if info:
            p = getattr(info, "last_price", None)
            if p is not None:
                return _safe_float(p)
    except Exception:
        pass
    return None

def get_currency(ticker: str) -> Optional[str]:
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        cur = None
        if fi and hasattr(fi, "currency"):
            cur = fi.currency
        if not cur:
            info = getattr(t, "info", {}) or {}
            cur = info.get("currency")
        if isinstance(cur, str):
            return cur.upper()
    except Exception:
        pass
    return None

def get_fx_to_eur(currency: Optional[str]) -> Optional[float]:
    if not currency or currency.upper() == "EUR":
        return 1.0
    cur = currency.upper()
    pair = f"EUR{cur}=X"  # {cur} per 1 EUR
    try:
        q = yf.Ticker(pair)
        hist = q.history(period="1d")
        if not hist.empty:
            eur_cur = _safe_float(hist["Close"].iloc[-1])
        else:
            fi = getattr(q, "fast_info", None)
            eur_cur = _safe_float(getattr(fi, "last_price", None)) if fi else None
        if eur_cur and eur_cur > 0:
            return 1.0 / eur_cur   # prezzo_cur -> EUR
    except Exception:
        pass
    return None

def enrich_with_prices(df: pd.DataFrame, use_live: bool = False) -> pd.DataFrame:
    """
    Restituisce colonne: Currency, Live Price, Price Used, FX_to_EUR, Price EUR, Purchase EUR.
    Tutte le colonne nuove sono pandas Series (no ndarray) e allineate a df.index.
    """
    d = df.copy()

    # --- Prezzi live e valute (liste -> Series) ---
    live_prices = []
    currencies = []
    for tk in d["Symbol"]:
        live = get_last_price(tk) if use_live else None
        cur  = get_currency(tk) or "EUR"
        live_prices.append(live)
        currencies.append(cur)

    d["Live Price"]   = pd.Series(pd.to_numeric(live_prices, errors="coerce"), index=d.index, dtype="float64")
    d["Currency"]     = pd.Series(currencies, index=d.index, dtype="object")

    # --- Price Used: usa numerici + combine_first tra Series ---
    curr_series = pd.to_numeric(d["Current Price"], errors="coerce")
    d["Price Used"]  = d["Live Price"].combine_first(curr_series)

    # --- FX -> EUR come Series ---
    fx_list = []
    for cur in d["Currency"]:
        fx = get_fx_to_eur(cur)
        fx_list.append(1.0 if fx is None else fx)
    d["FX_to_EUR"] = pd.Series(pd.to_numeric(fx_list, errors="coerce"), index=d.index).fillna(1.0)

    # --- Prezzi in EUR (tutto come Series) ---
    price_used_num   = pd.to_numeric(d["Price Used"], errors="coerce")
    purchase_num     = pd.to_numeric(d["Purchase Price"], errors="coerce")
    d["Price EUR"]    = pd.Series(price_used_num * d["FX_to_EUR"], index=d.index)
    d["Purchase EUR"] = pd.Series(purchase_num * d["FX_to_EUR"], index=d.index)

    return d

def get_history(symbol: str, period="3mo", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()