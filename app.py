# app.py — Home multipagina, refresh rapido, setup EUR
from __future__ import annotations
import io
import time
import tempfile
from pathlib import Path

import streamlit as st
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
    if uploaded is not None:
        st.session_state.uploaded_bytes = uploaded.getvalue()
        st.session_state.uploaded_name = uploaded.name
        st.session_state.csv_source = "uploaded_bytes"
        st.session_state.csv_path = None
        st.info(f"Caricato da upload: {uploaded.name}")
    else:
        st.session_state.csv_path = st.text_input("...oppure percorso locale",
                                                  value=st.session_state.csv_path or "portfolio.csv")
        st.session_state.csv_source = "path"

    st.session_state.use_live = st.checkbox("Usa prezzi live (yfinance)", value=st.session_state.use_live)
    st.session_state.upper = st.number_input("Soglia UPPER (%)", value=float(st.session_state.upper), step=1.0)
    st.session_state.lower = st.number_input("Soglia LOWER (%)", value=float(st.session_state.lower), step=1.0)
    st.session_state.auto_refresh = st.checkbox("Auto-refresh globale", value=st.session_state.auto_refresh)
    st.session_state.refresh_secs = st.number_input("Intervallo refresh (sec)",
                                                    value=int(st.session_state.refresh_secs), step=1, min_value=2)
    st.divider()
    st.markdown("### ℹ️ Glossario")
    st.markdown(glossary_md(), unsafe_allow_html=True)

def _load_df_from_session():
    source = st.session_state.get("csv_source")

    if source == "uploaded_bytes" and st.session_state.get("uploaded_bytes"):
        data = st.session_state.uploaded_bytes
        # 1) prova file-like
        try:
            buf = io.BytesIO(data)
            df, cash = load_yahoo_csv(buf)
        except TypeError:
            # 2) fallback temp file
            suffix = Path(st.session_state.get("uploaded_name", "uploaded.csv")).suffix or ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempfile.gettempdir()) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            df, cash = load_yahoo_csv(tmp_path)
            st.session_state._last_tmp_csv = tmp_path
        return df, cash, st.session_state.get("uploaded_name") or "upload.csv"

    # path locale
    csv_path = st.session_state.get("csv_path")
    if not csv_path:
        raise FileNotFoundError("Nessuna sorgente CSV selezionata (upload o path).")
    df, cash = load_yahoo_csv(csv_path)
    return df, cash, csv_path

def read_portfolio_df():
    df, cash, label = _load_df_from_session()
    df = enrich_with_prices(df, use_live=st.session_state.use_live)
    st.session_state.df = df
    st.session_state.cash_total = cash
    st.session_state.last_loaded_ok = True
    return label

try:
    label = read_portfolio_df()
    st.success(f"Dati caricati da: `{label}`")
except Exception as e:
    st.session_state.last_loaded_ok = False
    st.error(f"Errore nel caricamento: {e}")

st.info("Usa il menu **Pagine** a sinistra: Panoramica, Posizioni, Analisi tecnica, Report/Export.")

if st.session_state.auto_refresh:
    time.sleep(int(st.session_state.refresh_secs))
    st.rerun()
