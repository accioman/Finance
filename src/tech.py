from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

def get_history(symbol: str, period="1y", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.title)
        return df
    except Exception:
        return pd.DataFrame()

def add_indicators(df: pd.DataFrame, sma=50, ema=21, rsi_len=14, macd_fast=12, macd_slow=26, macd_signal=9):
    d = df.copy()
    close = d["Close"]

    d["SMA"] = close.rolling(window=sma, min_periods=sma).mean()
    d["EMA"] = close.ewm(span=ema, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=close.index).rolling(rsi_len).mean()
    roll_down = pd.Series(loss, index=close.index).rolling(rsi_len).mean()
    rs = roll_up / roll_down
    d["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=macd_signal, adjust=False).mean()
    d["MACD"] = macd
    d["MACD_signal"] = macd_signal
    d["MACD_hist"] = macd - macd_signal

    return d

def signals_from_indicators(d: pd.DataFrame):
    msgs = []
    if d.empty or len(d) < 3:
        return msgs
    last = d.iloc[-1]
    prev = d.iloc[-2]

    # Segnale trend
    if last["Close"] > last["SMA"] > last["EMA"]:
        msgs.append("📈 Trend rialzista (Close > SMA > EMA).")
    elif last["Close"] < last["SMA"] < last["EMA"]:
        msgs.append("📉 Trend ribassista (Close < SMA < EMA).")

    # RSI
    if last["RSI"] >= 70:
        msgs.append("⚠️ RSI in ipercomprato (≥70). Possibile ritracciamento.")
    elif last["RSI"] <= 30:
        msgs.append("💡 RSI in ipervenduto (≤30). Possibile rimbalzo.")

    # MACD cross
    if prev["MACD"] < prev["MACD_signal"] and last["MACD"] > last["MACD_signal"]:
        msgs.append("✅ MACD: incrocio rialzista (bullish crossover).")
    if prev["MACD"] > prev["MACD_signal"] and last["MACD"] < last["MACD_signal"]:
        msgs.append("❌ MACD: incrocio ribassista (bearish crossover).")

    # Momentum istogramma
    if last["MACD_hist"] > 0 and prev["MACD_hist"] <= 0:
        msgs.append("➕ MACD istogramma passa positivo: momentum in miglioramento.")
    if last["MACD_hist"] < 0 and prev["MACD_hist"] >= 0:
        msgs.append("➖ MACD istogramma passa negativo: momentum in peggioramento.")

    return msgs
