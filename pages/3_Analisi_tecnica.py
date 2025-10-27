# pages/3_Analisi_tecnica.py — robust/failsafe + compat width + safe mode + diagnostica
from __future__ import annotations
import time
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from typing import Optional, Dict, List
from src.tech import get_history, add_indicators, summarize_signals, latest_table
from src.utils import ensure_state, require_data, reload_portfolio_from_state, glossary_md
from src import ui_compat as ui

# Altair: disattiva max_rows, renderer snello
alt.data_transformers.disable_max_rows()
try:
    alt.renderers.set_embed_options(actions=False)
except Exception:
    pass

st.set_page_config(page_title="Analisi tecnica", page_icon="📈", layout="wide")

# ------------- Stato / dati base -------------
ensure_state()
reload_portfolio_from_state()
require_data()

st.title("📈 Analisi tecnica (avanzata)")

# Toggle Safe Mode (evita Altair se crea problemi)
safe_mode = st.sidebar.toggle(
    "Safe Mode (fallback grafici semplici)",
    value=False,
    help="Se i grafici non compaiono o vedi schermo nero, attiva questo."
)

# ----------------- Controlli -----------------
symbols = st.session_state.df["Symbol"].unique().tolist()
with st.container():
    c1, c2, c3, c4, c5 = st.columns([2,1,1,1,1])
    symbol = c1.selectbox("Simbolo", symbols)
    period = c2.selectbox("Periodo", ["5d","1mo","3mo","6mo","1y","2y","5y"], index=4)
    interval = c3.selectbox("Intervallo", ["1d","1h","30m","15m"], index=0)
    show_volume = c4.checkbox("Mostra volumi", value=True)
    use_candles = c5.checkbox("Usa candele", value=True, help="Candlestick invece della sola linea Close")

with st.expander("Parametri indicatori", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    sma_str = c1.text_input("SMA (comma-sep)", value="20,50,200", help="Es: 10,20,50,200")
    ema_str = c2.text_input("EMA (comma-sep)", value="21,50")
    rsi_len = c3.number_input("RSI (periodi)", value=14, min_value=2)
    macd_fast = c4.number_input("MACD fast", value=12, min_value=2)

    c5, c6, c7, c8 = st.columns(4)
    macd_slow = c5.number_input("MACD slow", value=26, min_value=2)
    macd_signal = c6.number_input("MACD signal", value=9, min_value=2)
    bb_window = c7.number_input("Bollinger window", value=20, min_value=5)
    bb_std = c8.number_input("Bollinger σ", value=2.0, step=0.1)

    c9, c10 = st.columns(2)
    atr_window = c9.number_input("ATR window", value=14, min_value=2)
    show_bb = c10.checkbox("Mostra Bollinger", value=True)

    c11, c12, c13, c14 = st.columns(4)
    kc_window = c11.number_input("Keltner window", value=20, min_value=5, help="Di solito 20")
    kc_mult = c12.number_input("Keltner ATR×", value=1.5, step=0.1, help="1.5 classico per TTM Squeeze")
    show_kc = c13.checkbox("Mostra Keltner", value=False)
    rsi_fast_len = c14.number_input("RSI fast (trigger)", value=10, min_value=2, help="Per segnali più reattivi")

    add_ema8 = st.checkbox("Aggiungi EMA 8", value=True)

# Downsample compatibile con timeframe
def _rules_for_interval(iv: str) -> List[str]:
    iv = iv.lower()
    if iv == "1d":
        return ["1D", "W", "2W", "M", "Q"]
    if iv == "1h":
        return ["1H", "4H", "1D", "W"]
    if iv in ("30m", "15m"):
        return ["15min", "30min", "1H", "4H", "1D"]
    return ["1D", "W", "M"]

with st.expander("Prestazioni e downsample", expanded=True):
    d1, d2, d3 = st.columns([1,1,2])
    enable_down = d1.checkbox("Attiva downsample", value=True, help="Riduce punti sul grafico per fluidità")
    mode = d2.selectbox("Modalità", ["Auto", "Manuale"], index=0)
    rule = d3.selectbox("Raggruppa per", _rules_for_interval(interval), index=0, help="Usato se Modalità = Manuale")

# ----------------- Dati & indicatori -----------------
try:
    hist = get_history(symbol, period=period, interval=interval)
except Exception as e:
    st.error(f"Errore nel download storico: {type(e).__name__}: {e}")
    st.stop()

if hist is None or hist.empty:
    st.warning("Dati storici non disponibili per questo simbolo/intervallo (ticker delistato o borsa chiusa senza dati intraday). Prova a cambiare periodo/intervallo.")
    st.stop()

sma_list = [int(x) for x in sma_str.replace(";", ",").split(",") if x.strip().isdigit()]
ema_list = [int(x) for x in ema_str.replace(";", ",").split(",") if x.strip().isdigit()]

try:
    df_ind = add_indicators(
        hist,
        sma=tuple(sma_list or [20, 50, 200]),
        ema=tuple(([8] if add_ema8 else []) + (ema_list or [21, 50])),
        rsi_len=int(rsi_len),
        rsi_fast_len=int(rsi_fast_len),
        macd_fast=int(macd_fast),
        macd_slow=int(macd_slow),
        macd_signal=int(macd_signal),
        bb_window=int(bb_window),
        bb_std=float(bb_std),
        kc_window=int(kc_window),
        kc_mult=float(kc_mult),
        atr_window=int(atr_window),
    )
except Exception as e:
    st.error(f"Errore nel calcolo indicatori: {type(e).__name__}: {e}")
    st.stop()

# Normalizza la colonna tempo
try:
    df_plot = (
        df_ind.reset_index().rename(columns={"Date": "dt"})
        if "Date" in df_ind.columns
        else df_ind.reset_index().rename(columns={df_ind.index.name or "index": "dt"})
    )
    df_plot["dt"] = pd.to_datetime(df_plot["dt"], errors="coerce").dt.tz_localize(None)
    df_plot = df_plot.dropna(subset=["dt"]).sort_values("dt")
    # Dedup timestamp (tieni l'ultimo)
    df_plot = df_plot.drop_duplicates(subset=["dt"], keep="last")
except Exception as e:
    st.error(f"Errore nella preparazione temporale: {type(e).__name__}: {e}")
    st.stop()

if df_plot.empty:
    st.error("Problema sul campo data: impossibile costruire la serie temporale.")
    st.stop()

# ----------------- Downsample robusto -----------------
def _choose_auto_rule(curr_interval: str, n_points: int) -> Optional[str]:
    if n_points <= 12000:
        return None
    ci = curr_interval.lower()
    if ci in ("15m", "30m"):
        if n_points > 80000: return "4H"
        if n_points > 40000: return "1H"
        if n_points > 20000: return "30min"
        return "15min"
    if ci == "1h":
        if n_points > 80000: return "1D"
        if n_points > 40000: return "4H"
        return "1H"
    if n_points > 5000:  # daily
        return "W"
    return None

def _downsample_for_plot(dfin: pd.DataFrame, rule: str) -> pd.DataFrame:
    if "dt" not in dfin.columns:
        raise ValueError("Colonna dt mancante")
    dfi = dfin.copy()
    if not isinstance(dfi.index, pd.DatetimeIndex):
        dfi = dfi.set_index(pd.to_datetime(dfi["dt"]))
    agg: Dict[str, str] = {}
    for c, how in (("Open","first"),("High","max"),("Low","min"),("Close","last")):
        if c in dfi.columns: agg[c] = how
    if "Volume" in dfi.columns:
        agg["Volume"] = "sum"
    for c in [c for c in dfi.columns if c not in agg and c != "dt"]:
        agg[c] = "last"
    dfo = dfi.resample(rule).agg(agg)
    dfo = dfo.dropna(subset=["Close"])
    dfo["dt"] = dfo.index
    return dfo.reset_index(drop=True)

n_raw = len(df_plot)
chosen_rule: Optional[str] = None
if enable_down:
    chosen_rule = _choose_auto_rule(interval, n_raw) if mode == "Auto" else rule

if chosen_rule:
    try:
        dfp = _downsample_for_plot(df_plot, chosen_rule)
        if dfp.empty:
            st.warning(f"Nessun dato con la regola '{chosen_rule}'. Uso dati originali.")
            dfp = df_plot.copy()
            chosen_rule = None
    except Exception as e:
        st.warning(f"Downsample fallito ({type(e).__name__}: {e}). Uso dati originali.")
        dfp = df_plot.copy()
        chosen_rule = None
else:
    dfp = df_plot.copy()

n_plot = len(dfp)

# ----------------- KPI -----------------
try:
    last = dfp.iloc[-1]
    close = float(last["Close"])
    prev = float(dfp["Close"].iloc[-2]) if len(dfp) >= 2 else close
    day_pl = (close - prev) / prev * 100 if prev else 0
    dist_high = float(last.get("dist_from_52w_high_%", float("nan")))
    dist_low  = float(last.get("dist_from_52w_low_%", float("nan")))
    squeeze_on = bool(last.get("SQUEEZE_ON", False))
    squeeze_up = bool(last.get("SQUEEZE_OFF_UP", False))
    squeeze_dn = bool(last.get("SQUEEZE_OFF_DOWN", False))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Prezzo", f"{close:.2f}")
    k2.metric("Day Δ%", f"{day_pl:.2f}%")
    k3.metric("Dist. 52W High", f"{dist_high:.1f}%" if pd.notna(dist_high) else "—")
    k4.metric("Dist. 52W Low", f"{dist_low:.1f}%" if pd.notna(dist_low) else "—")
    k5.metric("Squeeze", "ON" if squeeze_on else ("UP" if squeeze_up else ("DOWN" if squeeze_dn else "—")))
    st.caption(f"📉 Punti raw: {n_raw:,} → grafico: {n_plot:,}" + (f" (rule: {chosen_rule})" if chosen_rule else ""))
except Exception as e:
    st.warning(f"KPI non disponibili: {type(e).__name__}: {e}")

# ----------------- Grafici -----------------
axis_x = alt.Axis(title="", labelAngle=-30, labelFlush=False, labelOverlap=True)
legend_bottom = alt.Legend(title=None, orient="bottom", direction="horizontal",
                           columns=6, labelLimit=1000, symbolSize=120)

# ---- helpers serie/layer
def _prepare_price_long(dfin: pd.DataFrame, sma_list: List[int], ema_list: List[int]) -> pd.DataFrame:
    frames = []
    def _sf(df: pd.DataFrame, col: str, label: str):
        if col in df.columns:
            sr = df[["dt", col]].copy().rename(columns={col: "value"}).assign(series=label)
            if sr["value"].notna().any():
                frames.append(sr)

    _sf(dfin, "Close", "Close")
    for n in sma_list: _sf(dfin, f"SMA_{n}", f"SMA {n}")
    for n in ema_list: _sf(dfin, f"EMA_{n}", f"EMA {n}")

    if not frames:
        frames = [dfin[["dt","Close"]].rename(columns={"Close":"value"}).assign(series="Close")]

    out = pd.concat(frames, ignore_index=True)

    # pulizia forte
    out = out.dropna(subset=["dt", "value"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    out = out[np.isfinite(out["value"])]
    out["dt"] = pd.to_datetime(out["dt"], errors="coerce")
    out = out.dropna(subset=["dt"]).sort_values("dt")
    return out

def _price_chart_full(dfin: pd.DataFrame, price_long: pd.DataFrame):
    price_lines = alt.Chart(price_long).mark_line().encode(
        x=alt.X("dt:T", axis=axis_x),
        y=alt.Y("value:Q", title="Prezzo"),
        color=alt.Color("series:N", legend=legend_bottom),
        tooltip=[alt.Tooltip("dt:T", title="Data"),
                 alt.Tooltip("series:N", title="Serie"),
                 alt.Tooltip("value:Q", title="Valore", format=".2f")]
    )

    layers = []
    if show_kc and {"KC_low","KC_up"}.issubset(dfin.columns):
        layers.append(alt.Chart(dfin).mark_area(opacity=0.12).encode(x="dt:T", y="KC_low:Q", y2="KC_up:Q"))
    if show_bb and {"BB_low","BB_up"}.issubset(dfin.columns):
        layers.append(alt.Chart(dfin).mark_area(opacity=0.15).encode(x="dt:T", y="BB_low:Q", y2="BB_up:Q"))

    if use_candles and {"Open","High","Low","Close"}.issubset(dfin.columns):
        up = alt.value("#16a34a"); dn = alt.value("#ef4444")
        wick = alt.Chart(dfin).mark_rule().encode(
            x="dt:T", y="Low:Q", y2="High:Q",
            color=alt.condition("datum.Open <= datum.Close", up, dn),
            tooltip=[alt.Tooltip("dt:T", title="Data"),
                     alt.Tooltip("Open:Q", format=".2f"),
                     alt.Tooltip("High:Q", format=".2f"),
                     alt.Tooltip("Low:Q", format=".2f"),
                     alt.Tooltip("Close:Q", format=".2f")]
        )
        body = alt.Chart(dfin).mark_bar(size=5).encode(
            x="dt:T", y="Open:Q", y2="Close:Q",
            color=alt.condition("datum.Open <= datum.Close", up, dn)
        )
        chart = alt.layer(*(layers + [wick, body, price_lines]))
    else:
        chart = alt.layer(*(layers + [price_lines]))

    return (chart
            .configure(padding={"top":8,"left":8,"right":8,"bottom":90})
            .configure_view(clip=False)
            .configure_axis(labelLimit=0)
            .configure_legend(orient="bottom")
            .interactive())

def _price_chart_lines_only(price_long: pd.DataFrame):
    chart = (alt.Chart(price_long)
        .mark_line()
        .encode(x=alt.X("dt:T", axis=axis_x),
                y=alt.Y("value:Q", title="Prezzo"),
                color=alt.Color("series:N", legend=legend_bottom))
        .configure_view(clip=False)
        .interactive())
    return chart

# Limita punti per serie (no groupby.apply → niente FutureWarning)
def _cap_points(df: pd.DataFrame, per_series_max: int = 4000) -> pd.DataFrame:
    if "series" not in df.columns or df.empty:
        return df
    chunks = []
    for _, g in df.groupby("series", sort=False):
        n = len(g)
        step = max(1, n // per_series_max)
        chunks.append(g.iloc[::step] if step > 1 else g)
    return pd.concat(chunks, ignore_index=True) if chunks else df

# Diagnostica dati
def _diagnose_price_data(dfin: pd.DataFrame, price_long: Optional[pd.DataFrame]) -> Dict[str, any]:
    di: Dict[str, any] = {}
    try:
        di["rows_dfp"] = len(dfin)
        di["unique_dt"] = int(dfin["dt"].nunique())
        di["dup_dt"] = int(len(dfin) - dfin["dt"].nunique())
        if "Close" in dfin.columns:
            close_num = pd.to_numeric(dfin["Close"], errors="coerce")
            di["close_nan"] = int(close_num.isna().sum())
            di["close_inf"] = int(np.isinf(close_num).sum())
            v = close_num.dropna()
            if not v.empty:
                di["close_min"] = float(v.min())
                di["close_max"] = float(v.max())
        if price_long is not None and not price_long.empty:
            di["rows_price_long"] = len(price_long)
            di["series_counts"] = price_long["series"].value_counts().to_dict()
    except Exception as e:
        di["diag_error"] = f"{type(e).__name__}: {e}"
    return di

# ————— Render a scalini + diagnostica
try:
    price_long = _prepare_price_long(dfp, sma_list, ema_list)
    price_long = _cap_points(price_long, per_series_max=4000)

    if safe_mode:
        raise RuntimeError("Safe mode attivo")

    render_ok = False
    render_error_msgs: List[str] = []

    # FULL
    try:
        ui.altair_chart(_price_chart_full(dfp, price_long), height=360)
        render_ok = True
    except Exception as e1:
        render_error_msgs.append(f"FULL: {type(e1).__name__}: {e1}")

    # SOLO LINEE
    if not render_ok:
        try:
            ui.altair_chart(_price_chart_lines_only(price_long), height=360)
            render_ok = True
        except Exception as e2:
            render_error_msgs.append(f"LINEE: {type(e2).__name__}: {e2}")

    # SOLO CLOSE (Altair)
    if not render_ok:
        try:
            close_only = price_long.loc[price_long["series"] == "Close", ["dt","value"]].copy()
            if close_only.empty:
                raise RuntimeError("Serie Close vuota dopo pulizia")
            basic = alt.Chart(close_only).mark_line().encode(
                x="dt:T", y=alt.Y("value:Q", title="Prezzo")
            ).interactive()
            ui.altair_chart(basic, height=360)
            render_ok = True
        except Exception as e3:
            render_error_msgs.append(f"CLOSE-ALT: {type(e3).__name__}: {e3}")

    # Fallback Streamlit
    if not render_ok:
        try:
            ui.line_chart(dfp.set_index("dt")["Close"], height=360)
            render_ok = True
        except Exception as e4:
            render_error_msgs.append(f"FALLBACK-ST: {type(e4).__name__}: {e4}")

    if not render_ok:
        st.error("Impossibile renderizzare il grafico prezzo dopo tutti i fallback.")
        if render_error_msgs:
            with st.expander("Dettagli errori render"):
                for msg in render_error_msgs:
                    st.code(msg)

except Exception as e:
    st.warning(f"Prezzo fallback (Safe Mode / errore: {type(e).__name__}: {e}).")
    try:
        ui.line_chart(dfp.set_index("dt")["Close"], height=360)
    except Exception as e2:
        st.error(f"Fallback line_chart fallito: {type(e2).__name__}: {e2}")

with st.expander("Diagnostica grafico prezzo"):
    info = _diagnose_price_data(dfp, price_long if 'price_long' in locals() else None)
    st.json(info)
    try:
        st.caption("Sample dfp (head/tail)")
        st.dataframe(dfp.head(3))
        st.dataframe(dfp.tail(3))
        if 'price_long' in locals() and price_long is not None and not price_long.empty:
            st.caption("Sample price_long (head/tail)")
            st.dataframe(price_long.head(5))
            st.dataframe(price_long.tail(5))
    except Exception:
        pass

# ---- RSI
st.subheader("RSI")
if safe_mode:
    try:
        ui.line_chart(dfp.set_index("dt")["RSI"].dropna(), height=190)
    except Exception as e:
        st.warning(f"RSI semplice non disponibile: {type(e).__name__}: {e}")
else:
    try:
        axis_x2 = alt.Axis(title="", labelAngle=-30, labelFlush=False, labelOverlap=True)
        rsi_line = alt.Chart(dfp).mark_line().encode(
            x=alt.X("dt:T", axis=axis_x2),
            y=alt.Y("RSI:Q", title="RSI")
        )
        rsi_thresh = pd.DataFrame({"y": [30, 70], "soglia": ["RSI 30", "RSI 70"]})
        rsi_rules = alt.Chart(rsi_thresh).mark_rule().encode(y="y:Q", color=alt.Color("soglia:N", title="Soglie RSI"))
        ui.altair_chart((rsi_line + rsi_rules).configure(padding={"bottom": 40}).configure_view(clip=False).interactive(),
                        height=190)
    except Exception as e:
        st.warning(f"RSI Altair non renderizzato ({type(e).__name__}: {e}). Fallback semplice:")
        try:
            ui.line_chart(dfp.set_index("dt")["RSI"].dropna(), height=190)
        except Exception as e2:
            st.error(f"Fallback RSI fallito: {type(e2).__name__}: {e2}")

# ---- MACD
st.subheader("MACD")
if safe_mode:
    try:
        macd_simple = dfp.set_index("dt")[["MACD","MACD_signal"]].dropna()
        ui.line_chart(macd_simple, height=190)
    except Exception as e:
        st.warning(f"MACD semplice non disponibile: {type(e).__name__}: {e}")
else:
    try:
        axis_x2 = alt.Axis(title="", labelAngle=-30, labelFlush=False, labelOverlap=True)
        macd_long = pd.concat([
            dfp[["dt","MACD"]].rename(columns={"MACD":"value"}).assign(series="MACD"),
            dfp[["dt","MACD_signal"]].rename(columns={"MACD_signal":"value"}).assign(series="MACD signal"),
        ], ignore_index=True)
        macd_lines = alt.Chart(macd_long).mark_line().encode(
            x=alt.X("dt:T", axis=axis_x2),
            y=alt.Y("value:Q", title="MACD"),
            color=alt.Color("series:N", title="Linee MACD"),
        )
        macd_hist = alt.Chart(dfp).transform_calculate(
            sign="datum.MACD_hist >= 0 ? '≥ 0' : '< 0'"
        ).mark_bar().encode(
            x=alt.X("dt:T", axis=axis_x2),
            y="MACD_hist:Q",
            color=alt.Color("sign:N", title="Istogramma MACD", scale=alt.Scale(domain=["≥ 0","< 0"]))
        ).properties(height=170)
        ui.altair_chart((macd_hist + macd_lines).configure(padding={"bottom": 50}).configure_view(clip=False).interactive())
    except Exception as e:
        st.warning(f"MACD Altair non renderizzato ({type(e).__name__}: {e}). Fallback semplice:")
        try:
            macd_simple = dfp.set_index("dt")[["MACD","MACD_signal"]].dropna()
            ui.line_chart(macd_simple, height=190)
        except Exception as e2:
            st.error(f"Fallback MACD fallito: {type(e2).__name__}: {e2}")

# ---- Volumi
if show_volume and "Volume" in dfp.columns:
    st.subheader("Volumi")
    if safe_mode:
        try:
            ui.line_chart(dfp.set_index("dt")[["Volume"]], height=190)
        except Exception as e:
            st.warning(f"Volumi semplici non disponibili: {type(e).__name__}: {e}")
    else:
        try:
            axis_x2 = alt.Axis(title="", labelAngle=-30, labelFlush=False, labelOverlap=True)
            vol_bar = alt.Chart(dfp).mark_bar().encode(
                x=alt.X("dt:T", axis=axis_x2),
                y=alt.Y("Volume:Q", title="Volume")
            )
            layers = [vol_bar]
            vser = []
            if "Vol_MA20" in dfp.columns:
                vser.append(dfp[["dt","Vol_MA20"]].rename(columns={"Vol_MA20":"value"}).assign(series="Vol MA20"))
            if "Vol_MA50" in dfp.columns:
                vser.append(dfp[["dt","Vol_MA50"]].rename(columns={"Vol_MA50":"value"}).assign(series="Vol MA50"))
            if vser:
                vol_long = pd.concat(vser, ignore_index=True)
                vol_lines = alt.Chart(vol_long).mark_line().encode(
                    x=alt.X("dt:T", axis=axis_x2),
                    y="value:Q",
                    color=alt.Color("series:N", title="Medie Volume")
                )
                layers.append(vol_lines)
            ui.altair_chart(alt.layer(*layers).configure(padding={"bottom": 40}).configure_view(clip=False).interactive(),
                            height=190)
        except Exception as e:
            st.warning(f"Volumi Altair non renderizzati ({type(e).__name__}: {e}). Fallback semplice:")
            try:
                ui.line_chart(dfp.set_index("dt")[["Volume"]], height=190)
            except Exception as e2:
                st.error(f"Fallback Volumi fallito: {type(e2).__name__}: {e2}")

# ---- Segnali + tabella
st.subheader("Segnali sintetici")
try:
    for txt in summarize_signals(df_ind, sma_list, ema_list):
        st.info(txt)
except Exception as e:
    st.warning(f"Impossibile calcolare i segnali: {type(e).__name__}: {e}")

st.subheader("Valori e distanze attuali")
try:
    tab = latest_table(df_ind, sma_list, ema_list)
    if not tab.empty:
        def _styler(df):
            def colorize(v):
                try:
                    if pd.isna(v): return ""
                    v = float(v)
                    return "color: green;" if v > 0 else ("color: red;" if v < 0 else "")
                except Exception:
                    return ""
            sty = df.style.map(colorize, subset=pd.IndexSlice[:, ["Δ% vs Close"]])
            sty = sty.format({"Value": "{:.4f}", "Δ% vs Close": "{:.2f}%"})
            return sty
        ui.dataframe(_styler(tab), height=280)
    else:
        st.caption("Nessun dato disponibile per la tabella.")
except Exception as e:
    st.warning(f"Tabella non disponibile: {type(e).__name__}: {e}")

with st.expander("Glossario acronimi"):
    st.markdown(glossary_md(), unsafe_allow_html=True)

# ---- Auto refresh (con guardia)
try:
    if st.session_state.get("auto_refresh", False):
        secs = int(st.session_state.get("refresh_secs", 5) or 5)
        time.sleep(max(1, secs))
        st.rerun()
except Exception as e:
    st.warning(f"Auto refresh disattivato: {type(e).__name__}: {e}")
