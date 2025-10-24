# pages/3_Analisi_tecnica.py — Price chart + SMA/EMA/RSI/MACD + segnali
from __future__ import annotations
import streamlit as st
from src.tech import get_history, add_indicators, signals_from_indicators
from src.utils import require_data, glossary_md

st.set_page_config(page_title="Analisi tecnica", page_icon="📈", layout="wide")
require_data()

st.title("📈 Analisi tecnica")

symbols = st.session_state.df["Symbol"].unique().tolist()
col1, col2, col3, col4 = st.columns([2,1,1,1])
symbol = col1.selectbox("Simbolo", symbols)
period = col2.selectbox("Periodo", ["1mo","3mo","6mo","1y","2y","5y"], index=3)
interval = col3.selectbox("Intervallo", ["1d","1h","30m","15m"], index=0)
show_volume = col4.checkbox("Mostra volumi", value=True)

with st.expander("Parametri indicatori", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    sma = c1.number_input("SMA (gg)", value=50, min_value=2)
    ema = c2.number_input("EMA (gg)", value=21, min_value=2)
    rsi_len = c3.number_input("RSI (periodi)", value=14, min_value=2)
    macd_fast = c4.number_input("MACD fast", value=12, min_value=2)
    c5, c6, c7 = st.columns(3)
    macd_slow = c5.number_input("MACD slow", value=26, min_value=2)
    macd_signal = c6.number_input("MACD signal", value=9, min_value=2)

hist = get_history(symbol, period=period, interval=interval)
if hist.empty:
    st.warning("Dati storici non disponibili per questo simbolo.")
    st.stop()

df_ind = add_indicators(hist, sma=int(sma), ema=int(ema), rsi_len=int(rsi_len),
                        macd_fast=int(macd_fast), macd_slow=int(macd_slow), macd_signal=int(macd_signal))

st.subheader("Prezzo (Close, SMA, EMA)")
st.line_chart(df_ind[["Close","SMA","EMA"]].dropna(), height=320, width="stretch")

st.subheader("RSI")
st.line_chart(df_ind[["RSI"]].dropna(), height=200, width="stretch")

st.subheader("MACD")
st.line_chart(df_ind[["MACD","MACD_signal","MACD_hist"]].dropna(), height=200, width="stretch")

if show_volume and "Volume" in df_ind.columns:
    st.subheader("Volumi")
    st.bar_chart(df_ind[["Volume"]], height=180, width="stretch")

st.subheader("Segnali")
for txt in signals_from_indicators(df_ind):
    st.info(txt)

with st.expander("Glossario acronimi"):
    st.markdown(glossary_md(), unsafe_allow_html=True)
