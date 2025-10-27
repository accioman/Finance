from __future__ import annotations
import pandas as pd
import yfinance as yf
from typing import Optional

def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None
    
def _get_prev_close(ticker: str) -> Optional[float]:
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi and getattr(fi, "previous_close", None) is not None:
            return _safe_float(fi.previous_close)
        hist = t.history(period="5d")
        if hist is not None and not hist.empty:
            if len(hist["Close"]) >= 2:
                return _safe_float(hist["Close"].iloc[-2])
    except Exception:
        pass
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
    Restituisce colonne: Currency, Live Price, Price Used, FX_to_EUR,
    Price EUR, Purchase EUR, Prev Close (native), Prev Close EUR,
    Day_Delta (native), Day_Delta_% (pct), Day_Delta_EUR (per azione).
    """
    d = df.copy()

    # --- Prezzi live/valute/prev close ---
    live_prices, currencies, prev_closes = [], [], []
    for tk in d["Symbol"]:
        live = get_last_price(tk) if use_live else None
        cur  = get_currency(tk) or "EUR"
        pc   = _get_prev_close(tk)
        live_prices.append(live)
        currencies.append(cur)
        prev_closes.append(pc)

    d["Live Price"] = pd.Series(pd.to_numeric(live_prices, errors="coerce"), index=d.index, dtype="float64")
    d["Currency"]   = pd.Series(currencies, index=d.index, dtype="object")
    d["Prev Close"] = pd.Series(pd.to_numeric(prev_closes, errors="coerce"), index=d.index, dtype="float64")

    # --- Price Used ---
    curr_series = pd.to_numeric(d["Current Price"], errors="coerce")
    d["Price Used"] = d["Live Price"].combine_first(curr_series)

    # --- FX -> EUR ---
    fx_list = []
    for cur in d["Currency"]:
        fx = get_fx_to_eur(cur)
        fx_list.append(1.0 if fx is None else fx)
    d["FX_to_EUR"] = pd.Series(pd.to_numeric(fx_list, errors="coerce"), index=d.index).fillna(1.0)

    # --- EUR ---
    price_used_num = pd.to_numeric(d["Price Used"], errors="coerce")
    purchase_num   = pd.to_numeric(d["Purchase Price"], errors="coerce")
    prev_close_num = pd.to_numeric(d["Prev Close"], errors="coerce")

    d["Price EUR"]     = pd.Series(price_used_num * d["FX_to_EUR"], index=d.index)
    d["Purchase EUR"]  = pd.Series(purchase_num * d["FX_to_EUR"], index=d.index)
    d["Prev Close EUR"]= pd.Series(prev_close_num * d["FX_to_EUR"], index=d.index)

    # --- Delta giornaliero (per azione) in nativo + EUR ---
    d["Day_Delta"]      = price_used_num - prev_close_num
    d["Day_Delta_%"]    = (d["Day_Delta"] / prev_close_num * 100.0).replace([pd.NA, pd.NaT], 0.0)
    d["Day_Delta_EUR"]  = d["Day_Delta"] * d["FX_to_EUR"]

    return d

def get_history(symbol: str, period="3mo", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()