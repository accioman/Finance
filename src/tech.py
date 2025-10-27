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
    rsi_fast_len=10,            # NEW
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    bb_window=20,
    bb_std=2.0,
    kc_window=20,               # NEW
    kc_mult=1.5,                # NEW
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

    # --- RSI standard + RSI fast (trigger) ---
    def _rsi(series: pd.Series, length: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        roll_up = gain.rolling(length).mean()
        roll_down = loss.rolling(length).mean()
        rs = roll_up / roll_down
        return 100.0 - (100.0 / (1.0 + rs))
    d["RSI"] = _rsi(close, rsi_len)
    d["RSI_fast"] = _rsi(close, rsi_fast_len)

    # --- MACD ---
    ema_fast_s = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow_s = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast_s - ema_slow_s
    macd_sig = macd.ewm(span=macd_signal, adjust=False).mean()
    d["MACD"] = macd
    d["MACD_signal"] = macd_sig
    d["MACD_hist"] = macd - macd_sig

    # --- Bollinger Bands + feature per squeeze ---
    ma = close.rolling(bb_window, min_periods=bb_window).mean()
    std = close.rolling(bb_window, min_periods=bb_window).std()
    d["BB_mid"] = ma
    d["BB_up"] = ma + bb_std * std
    d["BB_low"] = ma - bb_std * std
    # %B e Bandwidth
    d["BB_pctB"] = (close - d["BB_low"]) / (d["BB_up"] - d["BB_low"])
    d["BB_bandwidth_%"] = (d["BB_up"] - d["BB_low"]) / ma * 100.0

    # --- ATR (True Range) + normalizzato ---
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=atr_window).mean()
    d["ATR"] = atr
    d["ATR_norm_%"] = (atr / close) * 100.0

    # --- Keltner Channels (EMA o SMA + ATR) ---
    # base = EMA(kc_window) più comune per KC
    ema_kc = close.ewm(span=kc_window, adjust=False).mean()
    d["KC_mid"] = ema_kc
    d["KC_up"] = ema_kc + kc_mult * atr
    d["KC_low"] = ema_kc - kc_mult * atr

    # --- TTM Squeeze flag: Bollinger dentro Keltner ---
    d["SQUEEZE_ON"] = (d["BB_up"] <= d["KC_up"]) & (d["BB_low"] >= d["KC_low"])
    d["SQUEEZE_OFF_UP"] = (~d["SQUEEZE_ON"]) & (d["BB_up"] > d["KC_up"]) & (d["BB_low"] > d["KC_low"])
    d["SQUEEZE_OFF_DOWN"] = (~d["SQUEEZE_ON"]) & (d["BB_up"] < d["KC_up"]) & (d["BB_low"] < d["KC_low"])

    # --- Volumi medi ---
    if "Volume" in d.columns:
        d["Vol_MA20"] = vol.rolling(20, min_periods=1).mean()
        d["Vol_MA50"] = vol.rolling(50, min_periods=1).mean()

    # --- 52w stats / Gap (come già avevi) ---
    win = min(len(d), 252)
    d["52w_high"] = close.rolling(win, min_periods=1).max()
    d["52w_low"]  = close.rolling(win, min_periods=1).min()
    d["dist_from_52w_high_%"] = (close / d["52w_high"] - 1.0) * 100.0
    d["dist_from_52w_low_%"]  = (close / d["52w_low"]  - 1.0) * 100.0
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
    
    # Squeeze (TTM-style) + trigger
    sq_on = bool(last.get("SQUEEZE_ON", False))
    sq_up = bool(last.get("SQUEEZE_OFF_UP", False))
    sq_dn = bool(last.get("SQUEEZE_OFF_DOWN", False))

    rsi_fast = float(last.get("RSI_fast", np.nan))
    vol_ma20 = float(last.get("Vol_MA20", np.nan)) if "Vol_MA20" in d.columns else np.nan
    vol = float(last.get("Volume", np.nan)) if "Volume" in d.columns else np.nan
    vol_ok = (not np.isnan(vol)) and (not np.isnan(vol_ma20)) and (vol > 1.5 * vol_ma20)
    
    if sq_on:
        msgs.append("🫙 Squeeze ON: Bollinger dentro Keltner (volatilità compressa).")
    elif sq_up or sq_dn:
        direction = "rialzista" if sq_up else "ribassista"
        extra = " + volume >1.5×MA20" if vol_ok else ""
        rsi_note = f" (RSI_fast {rsi_fast:.0f})" if not np.isnan(rsi_fast) else ""
        msgs.append(f"🚀 Uscita dallo squeeze {direction}{extra}{rsi_note}.")

    atr_n = float(last.get("ATR_norm_%", np.nan))
    if not np.isnan(atr_n):
        msgs.append(f"🌡️ ATR normalizzato: {atr_n:.1f}% (basso = compressione).")

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
