# pages/3_Analisi_tecnica.py — Analisi avanzata + autorefresh locale
from __future__ import annotations
import time
import pandas as pd
import altair as alt
import streamlit as st
from src.tech import get_history, add_indicators, summarize_signals, latest_table
from src.utils import ensure_state, require_data, reload_portfolio_from_state, glossary_md

st.set_page_config(page_title="Analisi tecnica", page_icon="📈", layout="wide")

ensure_state()
reload_portfolio_from_state()
require_data()

st.title("📈 Analisi tecnica (avanzata)")

symbols = st.session_state.df["Symbol"].unique().tolist()
col1, col2, col3, col4 = st.columns([2,1,1,1])
symbol = col1.selectbox("Simbolo", symbols)
period = col2.selectbox("Periodo", ["1mo","3mo","6mo","1y","2y","5y"], index=3)
interval = col3.selectbox("Intervallo", ["1d","1h","30m","15m"], index=0)
show_volume = col4.checkbox("Mostra volumi", value=True)

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

hist = get_history(symbol, period=period, interval=interval)
if hist.empty:
    st.warning("Dati storici non disponibili per questo simbolo.")
    st.stop()

# Calcoli
sma_list = [int(x) for x in sma_str.replace(";", ",").split(",") if x.strip().isdigit()]
ema_list = [int(x) for x in ema_str.replace(";", ",").split(",") if x.strip().isdigit()]

df_ind = add_indicators(
    hist,
    sma=tuple(sma_list or [20,50,200]),
    ema=tuple(ema_list or [21,50]),
    rsi_len=int(rsi_len),
    macd_fast=int(macd_fast),
    macd_slow=int(macd_slow),
    macd_signal=int(macd_signal),
    bb_window=int(bb_window),
    bb_std=float(bb_std),
    atr_window=int(atr_window),
)

df_plot = df_ind.reset_index().rename(columns={"Date": "dt"}) if "Date" in df_ind.columns else df_ind.reset_index().rename(columns={df_ind.index.name or "index": "dt"})

# =========================
#   CHART: Prezzo + MA/EMA
# =========================
base = alt.Chart(df_plot).encode(x=alt.X("dt:T", title=""))

price_line = alt.Chart(df_plot).mark_line().encode(
    x=alt.X("dt:T", title=""),
    y=alt.Y("Close:Q", title="Prezzo")
)
layers = [price_line]

# SMA selezionate
for n in sma_list:
    col = f"SMA_{n}"
    if col in df_plot.columns:
        layers.append(
            alt.Chart(df_plot).mark_line().encode(
                x="dt:T",
                y=alt.Y(f"{col}:Q"),
                color=alt.value("#6b7280")  # grigio
            )
        )

# EMA selezionate
for n in ema_list:
    col = f"EMA_{n}"
    if col in df_plot.columns:
        layers.append(
            alt.Chart(df_plot).mark_line().encode(
                x="dt:T",
                y=alt.Y(f"{col}:Q"),
                color=alt.value("#a855f7")  # viola
            )
        )

# Bollinger opzionali
if show_bb and all(c in df_plot.columns for c in ["BB_low", "BB_up"]):
    bb_band = alt.Chart(df_plot).mark_area(opacity=0.15).encode(
        x="dt:T",
        y="BB_low:Q",
        y2="BB_up:Q"
    )
    layers.insert(0, bb_band)  # banda sotto le linee

price_chart = alt.layer(*layers).properties(height=320).interactive()
st.altair_chart(price_chart, use_container_width=True)

st.subheader("Prezzo con SMA/EMA" + (" + Bollinger" if show_bb else ""))
st.altair_chart(price_chart, use_container_width=True)

# =========
#   RSI
# =========
rsi_chart = alt.Chart(df_plot).mark_line().encode(
    x="dt:T", y=alt.Y("RSI:Q", title="RSI")
).properties(height=180)

# Soglie 30/70
rsi_band_30 = alt.Chart(pd.DataFrame({"y": [30]})).mark_rule().encode(y="y:Q", color=alt.value("#ef4444"))
rsi_band_70 = alt.Chart(pd.DataFrame({"y": [70]})).mark_rule().encode(y="y:Q", color=alt.value("#22c55e"))

st.subheader("RSI")
st.altair_chart((rsi_chart + rsi_band_30 + rsi_band_70).interactive(), use_container_width=True)

# =========
#   MACD
# =========
macd_line = alt.Chart(df_plot).mark_line().encode(x="dt:T", y=alt.Y("MACD:Q", title="MACD"))
macd_sig_line = alt.Chart(df_plot).mark_line().encode(x="dt:T", y="MACD_signal:Q", color=alt.value("#6b7280"))
macd_hist = alt.Chart(df_plot).mark_bar().encode(
    x="dt:T",
    y="MACD_hist:Q",
    color=alt.condition("datum.MACD_hist >= 0", alt.value("#22c55e"), alt.value("#ef4444"))
).properties(height=160)

st.subheader("MACD")
st.altair_chart((macd_hist + macd_line + macd_sig_line).interactive(), use_container_width=True)

# =========
#   Volumi
# =========
if show_volume and "Volume" in df_plot.columns:
    vol_bar = alt.Chart(df_plot).mark_bar().encode(x="dt:T", y=alt.Y("Volume:Q", title="Volume"))
    overlays = [vol_bar]
    if "Vol_MA20" in df_plot.columns:
        overlays.append(alt.Chart(df_plot).mark_line().encode(x="dt:T", y="Vol_MA20:Q", color=alt.value("#f59e0b")))
    if "Vol_MA50" in df_plot.columns:
        overlays.append(alt.Chart(df_plot).mark_line().encode(x="dt:T", y="Vol_MA50:Q", color=alt.value("#9333ea")))
    st.subheader("Volumi")
    st.altair_chart(alt.layer(*overlays).properties(height=180).interactive(), use_container_width=True)

# =========
#   Segnali + Tabella
# =========
st.subheader("Segnali sintetici")
for txt in summarize_signals(df_ind, sma_list, ema_list):
    st.info(txt)

st.subheader("Valori e distanze attuali")
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
        sty = df.style.applymap(colorize, subset=["Δ% vs Close"])
        sty = sty.format({"Value": "{:.4f}", "Δ% vs Close": "{:.2f}%"})
        return sty
    st.dataframe(_styler(tab), use_container_width=True, height=280)
else:
    st.caption("Nessun dato disponibile per la tabella.")

with st.expander("Glossario acronimi"):
    st.markdown(glossary_md(), unsafe_allow_html=True)

# Auto refresh locale
if st.session_state.auto_refresh:
    time.sleep(int(st.session_state.refresh_secs))
    st.rerun()
