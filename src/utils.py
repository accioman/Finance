from __future__ import annotations
import io
import tempfile
from pathlib import Path
import streamlit as st
from src.loader import load_yahoo_csv
from src.pricing import enrich_with_prices

def ensure_state():
    # Sorgente dati
    st.session_state.setdefault("csv_source", "path")   # "uploaded_bytes" | "path"
    st.session_state.setdefault("csv_path", None)       # usato se csv_source == "path"
    st.session_state.setdefault("uploaded_bytes", None) # bytes dell'upload
    st.session_state.setdefault("uploaded_name", "")    # nome originale
    st.session_state.setdefault("_last_tmp_csv", None)  # debug: ultimo temp

    # Impostazioni app
    st.session_state.setdefault("use_live", True)
    st.session_state.setdefault("upper", 12.0)
    st.session_state.setdefault("lower", 8.0)
    st.session_state.setdefault("auto_refresh", True)
    st.session_state.setdefault("refresh_secs", 5)

    # Dati calcolati
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("cash_total", 0.0)

    # Stato caricamento
    st.session_state.setdefault("last_loaded_ok", False)
    st.session_state.setdefault("load_error", "")

def require_data():
    """Se non sono mai stati caricati dati, mostra errore e interrompi."""
    if not st.session_state.get("last_loaded_ok", False):
        st.error("Dati non caricati. Torna alla Home (app.py) e seleziona un CSV.")
        st.stop()

def _load_from_uploaded_bytes():
    """
    Prova a caricare il CSV direttamente dai byte in memoria.
    Se load_yahoo_csv richiede un path, fa fallback a un file temporaneo sicuro.
    Ritorna (df, cash, label_per_ui)
    """
    data: bytes | None = st.session_state.get("uploaded_bytes")
    if not data:
        raise FileNotFoundError("Nessun upload disponibile in sessione.")

    # 1) Tentativo: file-like in memoria
    try:
        buf = io.BytesIO(data)
        df, cash = load_yahoo_csv(buf)  # funziona se accetta file-like
        label = st.session_state.get("uploaded_name") or "upload.csv"
        return df, cash, label
    except TypeError:
        # 2) Fallback: salvataggio in temp (permessi garantiti in prod)
        suffix = Path(st.session_state.get("uploaded_name") or "upload.csv").suffix or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        st.session_state._last_tmp_csv = tmp_path
        df, cash = load_yahoo_csv(tmp_path)
        return df, cash, st.session_state.get("uploaded_name") or tmp_path

def reload_portfolio_from_state():
    """
    Ricarica SEMPRE dati e prezzi in EUR usando csv_source/use_live dalla sessione.
    Da chiamare all'inizio di OGNI pagina (Panoramica, Posizioni, ecc.).
    """
    try:
        source = st.session_state.get("csv_source", "path")

        if source == "uploaded_bytes":
            df, cash, _ = _load_from_uploaded_bytes()
        else:
            path = st.session_state.get("csv_path")
            if not path:
                raise FileNotFoundError("Percorso CSV non impostato.")
            df, cash = load_yahoo_csv(path)

        # Arricchimento prezzi (EUR / live o cache)
        df = enrich_with_prices(df, use_live=st.session_state.get("use_live", True))

        # Stato condiviso
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
- **BB** (Bollinger Bands): bande di volatilità attorno a una SMA.
- **ATR** (Average True Range): misura della volatilità basata sui range di prezzo.
- **Keltner Channels**: bande di volatilità basate su EMA e ATR.
- **PL % vs Cost**: profitto/perdita % rispetto al costo medio di acquisto.  
- **P/L % vs Prev**: profitto/perdita % rispetto alla chiusura precedente.  
- **Day P/L**: profitto/perdita giornaliero (prezzo attuale vs chiusura precedente).
- **P/L** (Profit/Loss): profitto o perdita.  
- **Drawdown**: calo dal massimo storico del capitale.  
- **Volatilità annualizzata**: dev. std. dei rendimenti giornalieri * √252.  
- **Sharpe (naive)**: rendimento annuo / volatilità annua (tasso risk-free = 0).  
- **Weight %**: peso percentuale della posizione sul valore totale (posizioni + cassa).  
    """
