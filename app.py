# app.py — Home multipagina, refresh rapido, setup EUR
from __future__ import annotations
import time
import streamlit as st
from pathlib import Path
from src.utils import ensure_state, glossary_md
from src.loader import load_yahoo_csv
from src.pricing import enrich_with_prices

st.set_page_config(page_title="Portafoglio • Dashboard EUR", page_icon="💶", layout="wide")
ensure_state()

st.title("💶 Dashboard Portafoglio (EUR) — yfinance")
st.caption("Aggiornamento frequente; i dati possono essere in differita a seconda dell'exchange.")

with st.sidebar:
    st.header("⚙️ Impostazioni")
    uploaded = st.file_uploader("Carica CSV del portafoglio", type=["csv"])
    st.session_state.csv_path = st.text_input("...oppure percorso locale", value=st.session_state.csv_path or "sample.csv")
    st.session_state.use_live = st.checkbox("Usa prezzi live (yfinance)", value=st.session_state.use_live)
    st.session_state.upper = st.number_input("Soglia UPPER (%)", value=float(st.session_state.upper), step=1.0)
    st.session_state.lower = st.number_input("Soglia LOWER (%)", value=float(st.session_state.lower), step=1.0)
    st.session_state.auto_refresh = st.checkbox("Auto-refresh globale", value=st.session_state.auto_refresh)
    st.session_state.refresh_secs = st.number_input("Intervallo refresh (sec)", value=int(st.session_state.refresh_secs), step=1, min_value=2)
    st.divider()
    st.markdown("### ℹ️ Glossario")
    st.markdown(glossary_md(), unsafe_allow_html=True)

def read_portfolio_df():
    if uploaded is not None:
        tmp = Path("uploaded_tmp.csv")
        tmp.write_bytes(uploaded.getbuffer())
        st.session_state.csv_path = str(tmp)
    df, cash = load_yahoo_csv(st.session_state.csv_path)
    df = enrich_with_prices(df, use_live=st.session_state.use_live)
    st.session_state.df = df
    st.session_state.cash_total = cash
    st.session_state.last_loaded_ok = True

try:
    read_portfolio_df()
    st.success(f"Dati caricati: `{st.session_state.csv_path}`")
except Exception as e:
    st.session_state.last_loaded_ok = False
    st.error(f"Errore nel caricamento: {e}")

st.info("Usa il menu **Pagine** a sinistra: Panoramica, Posizioni, Analisi tecnica, Report/Export.")

if st.session_state.auto_refresh:
    time.sleep(int(st.session_state.refresh_secs))
    st.rerun()

# Note: other pages (panoramica.py, posizioni.py, analisi_tecnica.py, report.py) will call
# reload_portfolio_from_state() from utils.py to refresh data on each page load.