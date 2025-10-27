# Streamlit Portfolio Dashboard — Stats, Alerts & Hints
import os
import time
import pandas as pd
import streamlit as st
from pathlib import Path

from src.loader import load_yahoo_csv
from src.pricing import enrich_with_prices
from src.stats import compute_stats_tables
from src.alerts import find_alerts

st.set_page_config(page_title="Portfolio Dashboard", page_icon="📊", layout="wide")

st.title("📊 Portfolio Dashboard — yfinance")
st.caption("CSV stile Yahoo → statistiche, alert e suggerimenti basati su regole semplici.")

# Sidebar inputs
with st.sidebar:
    st.header("⚙️ Impostazioni")
    csv_file = st.file_uploader("Carica CSV del portafoglio", type=["csv"])
    csv_path = st.text_input("...oppure percorso locale", value="portfolio.csv")
    use_live = st.checkbox("Usa prezzi live (yfinance)", value=True)
    upper = st.number_input("Soglia UPPER (%)", value=12.0, step=1.0)
    lower = st.number_input("Soglia LOWER (%)", value=8.0, step=1.0)
    auto_refresh = st.checkbox("Auto-refresh", value=False, help="Ricarica automaticamente ogni N secondi")
    refresh_secs = st.number_input("Intervallo (sec)", value=60, step=10, min_value=30)
    run_btn = st.button("Aggiorna adesso")

def _read_csv_to_tmp(uploaded):
    tmp = Path("uploaded_tmp.csv")
    tmp.write_bytes(uploaded.getbuffer())
    return str(tmp)

def _load_df():
    if csv_file is not None:
        path = _read_csv_to_tmp(csv_file)
    else:
        path = csv_path
    df, cash = load_yahoo_csv(path)
    df = enrich_with_prices(df, use_live=use_live)
    return df, cash

def derive_suggestions(df: pd.DataFrame, upper: float, lower: float):
    """Semplici suggerimenti rule-based: rebalance, taglia perdite, prendi profitto, diversifica."""
    hints = []
    if df.empty: 
        return hints

    # Ricalcola colonne base come in stats
    d = df.copy()
    d["Value"] = d["Price Used"] * d["Quantity"]
    d["PL_pct"] = (d["Price Used"] - d["Purchase Price"]) / d["Purchase Price"] * 100.0
    tot_val = d["Value"].sum()
    if tot_val:
        d["Weight_%"] = d["Value"] / tot_val * 100.0
    else:
        d["Weight_%"] = 0.0

    # 1) Take profit sopra upper*1.5 per %
    tp = d[d["PL_pct"] >= upper * 1.5]
    if not tp.empty:
        syms = ", ".join(tp.sort_values("PL_pct", ascending=False)["Symbol"].tolist())
        hints.append(f"🏁 Valuta presa di profitto/parziale su: {syms} (P/L% molto elevato).")

    # 2) Stop loss sotto -(lower*1.5)
    sl = d[d["PL_pct"] <= -abs(lower) * 1.5]
    if not sl.empty:
        syms = ", ".join(sl.sort_values("PL_pct")["Symbol"].tolist())
        hints.append(f"🛑 Valuta riduzione/uscita su: {syms} (perdita prolungata).")

    # 3) Rebalance: peso > 15% su singolo titolo
    heavy = d[d["Weight_%"] > 15]
    if not heavy.empty:
        syms = ", ".join(heavy.sort_values("Weight_%", ascending=False)["Symbol"].tolist())
        hints.append(f"⚖️ Ribilanciamento: questi pesano >15%: {syms}.")

    # 4) Concentrazione top 3 > 50%
    top3 = d.sort_values("Value", ascending=False).head(3)
    if top3["Value"].sum() / max(tot_val, 1) > 0.5:
        hints.append("🧩 Portafoglio concentrato: top 3 strumenti >50% del valore totale.")

    # 5) Aggiungi mediazione: losers tra -lower e -lower*1.5 con peso < 5%
    zone = d[(d["PL_pct"] <= -abs(lower)) & (d["PL_pct"] > -abs(lower)*1.5) & (d["Weight_%"] < 5)]
    if not zone.empty:
        syms = ", ".join(zone.sort_values("PL_pct")["Symbol"].tolist())
        hints.append(f"➕ Possibile DCA su: {syms} (drawdown moderato e peso contenuto).")

    return hints

placeholder = st.empty()

def render_once():
    df, cash_total = _load_df()
    tables = compute_stats_tables(df, cash_total)
    alerts = find_alerts(df, upper=upper, lower=lower)

    # KPIs
    t = tables["totals"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valore posizioni", f"{t['positions_value']:.2f}")
    c2.metric("Cassa", f"{t['cash_total']:.2f}")
    c3.metric("Valore portafoglio", f"{t['portfolio_total']:.2f}")
    c4.metric("P/L totale", f"{t['pl_total']:.2f}", f"{t['pl_pct_vs_cost']:.2f}%")

    # Layout tabelle
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Allocazione (Top)")
        st.dataframe(tables["by_weight"], use_container_width=True)

        st.subheader("P/L peggiore → migliore")
        st.dataframe(tables["by_pl"], use_container_width=True)

    with right:
        st.subheader("🔔 Alerts")
        up = alerts["upper"]; lo = alerts["lower"]
        if up.empty and lo.empty:
            st.success("Nessuna soglia raggiunta.")
        else:
            if not up.empty:
                st.write("**Soglie UPPER superate:**")
                st.dataframe(up[["Symbol","PL_pct","Price Used","Purchase Price","Quantity"]])
            if not lo.empty:
                st.write("**Soglie LOWER superate:**")
                st.dataframe(lo[["Symbol","PL_pct","Price Used","Purchase Price","Quantity"]])

        st.subheader("💡 Suggerimenti")
        for hint in derive_suggestions(df, upper, lower):
            st.info(hint)

    # Grafici semplici con st.bar_chart / st.pyplot
    st.subheader("Grafici")
    det = tables["detail"].copy()
    det = det.sort_values("Value", ascending=False)
    chart_alloc = det.set_index("Symbol")["Value"]
    st.bar_chart(chart_alloc)

    chart_pl = det.set_index("Symbol")["PL_pct"].sort_values()
    st.bar_chart(chart_pl)

with placeholder.container():
    render_once()

if auto_refresh and not run_btn:
    # Auto-refresh loop (soft): Streamlit reruns the script, so we just sleep a bit
    time.sleep(int(refresh_secs))
    st.rerun()
