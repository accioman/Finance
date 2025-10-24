# pages/2_Posizioni.py — Tabella posizioni in EUR con autorefresh locale
from __future__ import annotations
import time
import streamlit as st
import pandas as pd
from src.utils import ensure_state, require_data, reload_portfolio_from_state

st.set_page_config(page_title="Posizioni (EUR)", page_icon="📑", layout="wide")

ensure_state()
reload_portfolio_from_state()
require_data()

st.title("📑 Posizioni (EUR)")

df = st.session_state.df.copy()
df["Valore €"] = df["Price EUR"] * df["Quantity"]
df["P/L %"] = (df["Price EUR"] - df["Purchase EUR"]) / df["Purchase EUR"] * 100.0
df["P/L €"] = (df["Price EUR"] - df["Purchase EUR"]) * df["Quantity"]

col1, col2, col3, col4 = st.columns(4)
f_sym = col1.text_input("Filtra simbolo contiene", "")
f_pl_min = col2.number_input("P/L % minimo", value=float(-100), step=1.0)
f_pl_max = col3.number_input("P/L % massimo", value=float(100), step=1.0)
sort_by_val = col4.checkbox("Ordina per valore €", value=True)

mask = df["Symbol"].str.contains(f_sym, case=False, na=False)
mask &= df["P/L %"].between(f_pl_min, f_pl_max, inclusive="both")
df_filt = df[mask].copy()
if sort_by_val:
    df_filt = df_filt.sort_values("Valore €", ascending=False)

cols = ["Symbol","Quantity","Currency","Price EUR","Purchase EUR","Valore €","P/L €","P/L %","Comment"]
st.dataframe(df_filt[cols], use_container_width=True, height=520)

csv_bytes = df_filt[cols].to_csv(index=False).encode()
st.download_button("⬇️ Scarica CSV filtrato (EUR)", data=csv_bytes,
                   file_name="posizioni_eur_filtrate.csv", mime="text/csv")

if st.session_state.auto_refresh:
    time.sleep(int(st.session_state.refresh_secs))
    st.rerun()
