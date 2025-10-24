# pages/4_Report_Export.py — Report/Export con autorefresh locale
from __future__ import annotations
import io
import time
import pandas as pd
import streamlit as st

from src.risk import risk_metrics_from_history, correlation_matrix
from src.tech import get_history
from src.stats import compute_stats_tables
from src.utils import ensure_state, require_data, reload_portfolio_from_state

st.set_page_config(page_title="Report / Export", page_icon="📤", layout="wide")

ensure_state()
reload_portfolio_from_state()
require_data()

st.title("📤 Report & Export")

df = st.session_state.df
cash_total = st.session_state.cash_total
tables = compute_stats_tables(df, cash_total)

st.subheader("📎 Riepilogo totale")
st.json(tables["totals"])

st.subheader("🔗 Correlazioni (90 giorni, Close)")
syms = df["Symbol"].unique().tolist()
corr = correlation_matrix(syms, period="3mo", interval="1d")
if corr.empty:
    st.info("Correlazioni non disponibili (storico insufficiente).")
else:
    st.dataframe(corr, use_container_width=True, height=360)

st.subheader("⚠️ Metriche di rischio per simbolo (90 giorni)")
rows = []
for s in syms:
    h = get_history(s, period="3mo", interval="1d")
    if not h.empty:
        rows.append({"Symbol": s, **risk_metrics_from_history(h)})
risk_df = pd.DataFrame(rows)
if not risk_df.empty:
    st.dataframe(risk_df, use_container_width=True)
else:
    st.info("Storico insufficiente per calcolare le metriche.")

# Export (xlsx o zip fallback)
try:
    import xlsxwriter
    engine = "xlsxwriter"
except Exception:
    try:
        import openpyxl
        engine = "openpyxl"
    except Exception:
        engine = None

st.subheader("⬇️ Esporta report")
if engine:
    with io.BytesIO() as buf:
        with pd.ExcelWriter(buf, engine=engine) as xw:
            tables["detail"].to_excel(xw, index=False, sheet_name="Dettaglio")
            tables["by_weight"].to_excel(xw, index=False, sheet_name="Allocazione")
            tables["by_pl"].to_excel(xw, index=False, sheet_name="PL_Ord")
            if not corr.empty:
                corr.to_excel(xw, sheet_name="Correlazioni")
            if not risk_df.empty:
                risk_df.to_excel(xw, index=False, sheet_name="Rischio")
        st.download_button("Scarica report.xlsx", buf.getvalue(),
                           file_name="report_portafoglio.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    import zipfile
    with io.BytesIO() as zbuf:
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("Dettaglio.csv", tables["detail"].to_csv(index=False))
            z.writestr("Allocazione.csv", tables["by_weight"].to_csv(index=False))
            z.writestr("PL_Ord.csv", tables["by_pl"].to_csv(index=False))
            if not corr.empty:
                z.writestr("Correlazioni.csv", corr.to_csv())
            if not risk_df.empty:
                z.writestr("Rischio.csv", risk_df.to_csv(index=False))
        st.download_button("Scarica report.zip", zbuf.getvalue(),
                           file_name="report_portafoglio.zip",
                           mime="application/zip")

if st.session_state.auto_refresh:
    time.sleep(int(st.session_state.refresh_secs))
    st.rerun()
