from __future__ import annotations
import streamlit as st

def ensure_state():
    st.session_state.setdefault("csv_path", None)
    st.session_state.setdefault("use_live", True)
    st.session_state.setdefault("upper", 12.0)
    st.session_state.setdefault("lower", 8.0)
    st.session_state.setdefault("auto_refresh", False)
    st.session_state.setdefault("refresh_secs", 60)
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("cash_total", 0.0)
    st.session_state.setdefault("last_loaded_ok", False)
    st.session_state.setdefault("load_error", "")

def require_data():
    if not st.session_state.get("last_loaded_ok", False):
        st.error("Dati non caricati dalla Home. Torna a **app.py** e carica un CSV.")
        st.stop()

def glossary_md():
    return """
**Acronimi e termini**  
- **SMA** (Simple Moving Average): media mobile semplice del prezzo.  
- **EMA** (Exponential Moving Average): media mobile esponenziale (più reattiva).  
- **RSI** (Relative Strength Index): oscillatore 0-100; >70 ipercomprato, <30 ipervenduto.  
- **MACD** (Moving Average Convergence Divergence): differenza tra due EMA (trend/momentum).  
- **P/L** (Profit/Loss): profitto o perdita.  
- **Drawdown**: calo dal massimo storico del capitale.  
- **Volatilità annualizzata**: deviazione standard dei rendimenti giornalieri * √252.  
- **Sharpe (naive)**: rendimento annuo / volatilità annua (tasso privo di rischio = 0).  
- **Weight %**: peso percentuale della posizione sul valore totale (posizioni + cassa).  
    """
