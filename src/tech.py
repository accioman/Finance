from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple, Dict

def get_history(symbol: str, period="1y", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.title)
        # rimuovi righe completamente NaN
        df = df.dropna(how="all")
        return df
    except Exception:
        return pd.DataFrame()

def _parse_list(inp, default: List[int]) -> List[int]:
    if inp is None:
        return default
    if isinstance(inp, (list, tuple)):
        return [int(x) for x in inp if str(x).strip().isdigit()]
    # stringa "20,50,200"
    parts = [p.strip() for p in str(inp).replace(";", ",").split(",")]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            pass
    return out or default

def add_indicators(
    df: pd.DataFrame,
    sma=(20, 50, 200),
    ema=(21, 50),
    rsi_len=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    bb_window=20,
    bb_std=2.0,
    atr_window=14,
) -> pd.DataFrame:
    d = df.copy()
    close = d["Close"].astype(float)
    high  = d["High"].astype(float)
    low   = d["Low"].astype(float)
    vol   = d["Volume"].astype(float) if "Volume" in d.columns else pd.Series(index=d.index, dtype=float)

    # --- SMA / EMA multipli ---
    for n in _parse_list(sma, [20, 50, 200]):
        d[f"SMA_{n}"] = close.rolling(window=n, min_periods=n).mean()
    for n in _parse_list(ema, [21, 50]):
        d[f"EMA_{n}"] = close.ewm(span=n, adjust=False).mean()

    # --- RSI ---
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=close.index).rolling(rsi_len).mean()
    roll_down = pd.Series(loss, index=close.index).rolling(rsi_len).mean()
    rs = roll_up / roll_down
    d["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    # --- MACD ---
    ema_fast_s = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow_s = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast_s - ema_slow_s
    macd_sig = macd.ewm(span=macd_signal, adjust=False).mean()
    d["MACD"] = macd
    d["MACD_signal"] = macd_sig
    d["MACD_hist"] = macd - macd_sig

    # --- Bollinger Bands ---
    ma = close.rolling(bb_window, min_periods=bb_window).mean()
    std = close.rolling(bb_window, min_periods=bb_window).std()
    d["BB_mid"] = ma
    d["BB_up"] = ma + bb_std * std
    d["BB_low"] = ma - bb_std * std

    # --- ATR (True Range) ---
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(atr_window, min_periods=atr_window).mean()

    # --- Volumi medi ---
    if "Volume" in d.columns:
        d["Vol_MA20"] = vol.rolling(20, min_periods=1).mean()
        d["Vol_MA50"] = vol.rolling(50, min_periods=1).mean()

    # --- 52w stats ---
    win = min(len(d), 252)
    d["52w_high"] = close.rolling(win, min_periods=1).max()
    d["52w_low"]  = close.rolling(win, min_periods=1).min()
    d["dist_from_52w_high_%"] = (close / d["52w_high"] - 1.0) * 100.0
    d["dist_from_52w_low_%"]  = (close / d["52w_low"]  - 1.0) * 100.0

    # --- Gap del giorno (vs prev close) ---
    if "Open" in d.columns:
        d["Gap_%"] = (d["Open"] / prev_close - 1.0) * 100.0

    return d

def _last_cross(series_fast: pd.Series, series_slow: pd.Series) -> Tuple[str, pd.Timestamp]:
    """Ritorna ('bullish'|'bearish'|'none', ts) per l'ultimo incrocio rilevato."""
    s1 = series_fast - series_slow
    sign = np.sign(s1)
    cs = sign.diff()
    idx = cs[cs != 0].index
    if len(idx) == 0:
        return "none", pd.NaT
    last_idx = idx[-1]
    # determinare tipo
    if s1.loc[last_idx] > 0:
        return "bullish", last_idx
    else:
        return "bearish", last_idx

def summarize_signals(
    d: pd.DataFrame,
    sma_list: List[int],
    ema_list: List[int],
) -> List[str]:
    """Restituisce messaggi comprensibili e compatti."""
    msgs: List[str] = []
    if d.empty or len(d) < 5:
        return msgs

    last = d.iloc[-1]
    close = float(last["Close"])

    # Trend per medie (maggiore consenso = più forte)
    above_sma = [n for n in sma_list if not pd.isna(last.get(f"SMA_{n}", np.nan)) and close > last[f"SMA_{n}"]]
    below_sma = [n for n in sma_list if not pd.isna(last.get(f"SMA_{n}", np.nan)) and close < last[f"SMA_{n}"]]
    if above_sma and not below_sma:
        msgs.append(f"📈 Trend rialzista: prezzo sopra tutte le SMA {above_sma}.")
    elif below_sma and not above_sma:
        msgs.append(f"📉 Trend ribassista: prezzo sotto tutte le SMA {below_sma}.")
    else:
        if above_sma:
            msgs.append(f"🙂 Prezzo sopra SMA {above_sma}, ma non tutte.")
        if below_sma:
            msgs.append(f"🙃 Prezzo sotto SMA {below_sma}, ma non tutte.")

    # RSI
    rsi = float(last.get("RSI", np.nan))
    if not np.isnan(rsi):
        if rsi >= 70:
            msgs.append("⚠️ RSI in ipercomprato (≥70): rischio ritracciamento.")
        elif rsi <= 30:
            msgs.append("💡 RSI in ipervenduto (≤30): possibile rimbalzo.")
        else:
            msgs.append(f"ℹ️ RSI neutro: {rsi:.1f}")

    # MACD
    macd = float(last.get("MACD", np.nan))
    macd_sig = float(last.get("MACD_signal", np.nan))
    macd_hist = float(last.get("MACD_hist", np.nan))
    if not any(np.isnan([macd, macd_sig, macd_hist])):
        if macd > macd_sig:
            msgs.append("✅ MACD sopra il segnale (momentum positivo).")
        else:
            msgs.append("❌ MACD sotto il segnale (momentum debole).")
        if macd_hist > 0:
            msgs.append("➕ Istogramma MACD positivo (forza crescente).")
        elif macd_hist < 0:
            msgs.append("➖ Istogramma MACD negativo (forza calante).")

    # Bollinger squeeze / breakout
    bb_up, bb_low = last.get("BB_up", np.nan), last.get("BB_low", np.nan)
    if not np.isnan(bb_up) and not np.isnan(bb_low):
        width = (bb_up - bb_low) / close * 100.0 if close else np.nan
        if not np.isnan(width):
            if width < 5:
                msgs.append("🫙 Bollinger squeeze (<5%): volatilità compressa, possibili breakout.")
            if close > bb_up:
                msgs.append("🚀 Close oltre BB superiore: breakout rialzista.")
            elif close < bb_low:
                msgs.append("🧊 Close sotto BB inferiore: breakout ribassista.")

    # Distanza 52w high/low
    dfh = float(last.get("dist_from_52w_high_%", np.nan))
    dfl = float(last.get("dist_from_52w_low_%", np.nan))
    if not np.isnan(dfh) and not np.isnan(dfl):
        msgs.append(f"📏 Distanza dai 52w: High {dfh:.1f}% | Low {dfl:.1f}%.")

    # ATR / volatilità
    atr = float(last.get("ATR", np.nan))
    if not np.isnan(atr) and close:
        msgs.append(f"🌪️ ATR: {atr:.2f} ({atr/close*100:.1f}% del prezzo).")

    # Ultimi cross (SMA e EMA, dal più corto al più lungo)
    sma_list = sorted([n for n in sma_list if f"SMA_{n}" in d.columns])
    ema_list = sorted([n for n in ema_list if f"EMA_{n}" in d.columns])
    if len(sma_list) >= 2:
        fast, slow = sma_list[0], sma_list[-1]
        typ, when = _last_cross(d[f"SMA_{fast}"], d[f"SMA_{slow}"])
        if typ != "none":
            emoji = "🟢" if typ == "bullish" else "🔴"
            msgs.append(f"{emoji} Ultimo cross SMA {fast}/{slow}: {typ} in data {when.date()}.")
    if len(ema_list) >= 2:
        fast, slow = ema_list[0], ema_list[-1]
        typ, when = _last_cross(d[f"EMA_{fast}"], d[f"EMA_{slow}"])
        if typ != "none":
            emoji = "🟢" if typ == "bullish" else "🔴"
            msgs.append(f"{emoji} Ultimo cross EMA {fast}/{slow}: {typ} in data {when.date()}.")

    # Gap
    gap = float(last.get("Gap_%", np.nan))
    if not np.isnan(gap):
        if gap >= 2:
            msgs.append(f"⬆️ Gap up del {gap:.2f}%.")
        elif gap <= -2:
            msgs.append(f"⬇️ Gap down del {gap:.2f}%.")

    return msgs

def latest_table(d: pd.DataFrame, sma_list: List[int], ema_list: List[int]) -> pd.DataFrame:
    """Tabella compatta di valori correnti (close, MA/EMA, distanze %)."""
    if d.empty:
        return pd.DataFrame()
    last = d.iloc[[-1]].copy()
    close = float(last["Close"])
    rows: Dict[str, Dict[str, float]] = {"Close": {"Value": close, "Δ% vs Close": 0.0}}

    for n in sma_list:
        col = f"SMA_{n}"
        if col in d.columns and not pd.isna(last[col].iloc[0]) and close:
            v = float(last[col].iloc[0])
            rows[f"SMA {n}"] = {"Value": v, "Δ% vs Close": (close / v - 1) * 100.0}
    for n in ema_list:
        col = f"EMA_{n}"
        if col in d.columns and not pd.isna(last[col].iloc[0]) and close:
            v = float(last[col].iloc[0])
            rows[f"EMA {n}"] = {"Value": v, "Δ% vs Close": (close / v - 1) * 100.0}

    # 52w
    for label in ["52w_high", "52w_low"]:
        if label in d.columns and not pd.isna(last[label].iloc[0]) and close:
            v = float(last[label].iloc[0])
            rows[label] = {"Value": v, "Δ% vs Close": (close / v - 1) * 100.0}

    # ATR
    if "ATR" in d.columns and not pd.isna(last["ATR"].iloc[0]) and close:
        v = float(last["ATR"].iloc[0])
        rows["ATR"] = {"Value": v, "Δ% vs Close": v / close * 100.0}

    # RSI
    if "RSI" in d.columns and not pd.isna(last["RSI"].iloc[0]):
        v = float(last["RSI"].iloc[0])
        rows["RSI"] = {"Value": v, "Δ% vs Close": np.nan}

    # Volumi
    for label in ["Vol_MA20", "Vol_MA50"]:
        if label in d.columns and not pd.isna(last[label].iloc[0]):
            rows[label] = {"Value": float(last[label].iloc[0]), "Δ% vs Close": np.nan}

    out = pd.DataFrame(rows).T
    return out.reset_index(names=["Metric"])
