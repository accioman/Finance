from __future__ import annotations
import streamlit as st
from pathlib import Path
from src.loader import load_yahoo_csv
from src.pricing import enrich_with_prices

def ensure_state():
    st.session_state.setdefault("csv_path", None)
    st.session_state.setdefault("use_live", True)
    st.session_state.setdefault("upper", 12.0)
    st.session_state.setdefault("lower", 8.0)
    st.session_state.setdefault("auto_refresh", True)
    st.session_state.setdefault("refresh_secs", 5)
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("cash_total", 0.0)
    st.session_state.setdefault("last_loaded_ok", False)
    st.session_state.setdefault("load_error", "")

def require_data():
    """Se non sono mai stati caricati dati, mostra errore e interrompi."""
    if not st.session_state.get("last_loaded_ok", False):
        st.error("Dati non caricati. Torna alla Home (app.py) e seleziona un CSV.")
        st.stop()

def reload_portfolio_from_state():
    """
    Ricarica SEMPRE dati e prezzi in EUR usando csv_path/use_live dalla sessione.
    Da chiamare all'inizio di OGNI pagina, così anche le pagine interne aggiornano i dati.
    """
    path = st.session_state.csv_path
    if not path:
        st.session_state.last_loaded_ok = False
        st.session_state.load_error = "Percorso CSV non impostato."
        return

    try:
        df, cash = load_yahoo_csv(path)
        df = enrich_with_prices(df, use_live=st.session_state.use_live)  # calcolo EUR
        st.session_state.df = df
        st.session_state.cash_total = cash
        st.session_state.last_loaded_ok = True
        st.session_state.load_error = ""
    except Exception as e:
        st.session_state.last_loaded_ok = False
        st.session_state.load_error = str(e)

def glossary_md():
    return """
**Acronimi e termini**  
- **SMA** (Simple Moving Average): media mobile semplice del prezzo.  
- **EMA** (Exponential Moving Average): media mobile esponenziale (più reattiva).  
- **RSI** (Relative Strength Index): oscillatore 0-100; >70 ipercomprato, <30 ipervenduto.  
- **MACD** (Moving Average Convergence Divergence): differenza tra due EMA (trend/momentum).  
- **P/L** (Profit/Loss): profitto o perdita.  
- **Drawdown**: calo dal massimo storico del capitale.  
- **Volatilità annualizzata**: dev. std. dei rendimenti giornalieri * √252.  
- **Sharpe (naive)**: rendimento annuo / volatilità annua (tasso risk-free = 0).  
- **Weight %**: peso percentuale della posizione sul valore totale (posizioni + cassa).  
    """
