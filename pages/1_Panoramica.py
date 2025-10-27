# pages/1_Panoramica.py — KPI, Alert, Suggerimenti (EUR) con autorefresh locale
from __future__ import annotations
import time
import streamlit as st
from src.stats import compute_stats_tables
from src.alerts import find_alerts
from src.suggestions import derive_suggestions
from src.utils import ensure_state, require_data, reload_portfolio_from_state

st.set_page_config(page_title="Panoramica (EUR)", page_icon="💶", layout="wide")

# Ricarica sempre dati prima di disegnare
ensure_state()
reload_portfolio_from_state()
require_data()

st.title("💶 Panoramica (valori in EUR)")
st.caption("Aggiornamento frequente (polling).")

df = st.session_state.df
cash_total = st.session_state.cash_total
tables = compute_stats_tables(df, cash_total)
alerts = find_alerts(df, upper=st.session_state.upper, lower=st.session_state.lower)

# KPI
t = tables["totals"]
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Valore posizioni (€)", f"{t['positions_value']:.2f}")
c2.metric("Cassa (€)", f"{t['cash_total']:.2f}")
c3.metric("Valore portafoglio (€)", f"{t['portfolio_total']:.2f}")
c4.metric("P/L totale (€)", f"{t['pl_total']:.2f}", f"{t['pl_pct_vs_cost']:.2f}%")
# Nuovi KPI giornalieri: Streamlit colora verde/rosso automaticamente sul delta
c5.metric("P/L giornaliero (€)", f"{t['day_pl_total']:.2f}", f"{t['day_pl_pct_vs_prev']:.2f}%")
c6.metric("P/L % giornaliero", f"{t['day_pl_pct_vs_prev']:.2f}%",  # valore principale
          f"{t['day_pl_total']:.2f} €")                            # delta in €

left, right = st.columns([2, 1])

def _style_pos_neg(df, cols):
    def colorize(v):
        if pd.isna(v): return ""
        return "color: green;" if v > 0 else ("color: red;" if v < 0 else "")
    return df.style.applymap(colorize, subset=cols) \
                   .format({c: "{:.2f}" for c in cols if c.endswith("_abs") or c.endswith("EUR")}) \
                   .format({c: "{:.2f}%" for c in cols if c.endswith("_pct") or c.endswith("%")})

with left:
    st.subheader("Allocazione (Top) — EUR")
    st.dataframe(
        _style_pos_neg(tables["by_weight"], ["PL_abs","PL_pct","Day_abs","Day_pct"]),
        use_container_width=True, height=360
    )

    st.subheader("Movers giornalieri — % (peggiore → migliore)")
    st.dataframe(
        _style_pos_neg(tables["by_day"], ["Day_pct","Day_abs"]),
        use_container_width=True, height=360
    )

    st.subheader("P/L peggiore → migliore — %")
    st.dataframe(
        _style_pos_neg(tables["by_pl"], ["PL_pct","PL_abs"]),
        use_container_width=True, height=360
    )

with right:
    st.subheader("🔔 Alerts (calcolati in EUR)")
    up = alerts["upper"]; lo = alerts["lower"]
    if up.empty and lo.empty:
        st.success("Nessuna soglia raggiunta.")
    else:
        if not up.empty:
            st.write("**Soglie UPPER superate**")
            st.dataframe(up[["Symbol","PL_pct","Price EUR","Purchase EUR","Quantity"]],
                         use_container_width=True)
        if not lo.empty:
            st.write("**Soglie LOWER superate**")
            st.dataframe(lo[["Symbol","PL_pct","Price EUR","Purchase EUR","Quantity"]],
                         use_container_width=True)

    st.subheader("💡 Suggerimenti")
    for hint in derive_suggestions(df, st.session_state.upper, st.session_state.lower):
        st.info(hint)

st.subheader("Grafici rapidi (EUR)")
det = tables["detail"].copy().sort_values("Value", ascending=False)
st.bar_chart(det.set_index("Symbol")["Value"], height=240, use_container_width=True)
st.bar_chart(det.set_index("Symbol")["PL_pct"].sort_values(), height=240, use_container_width=True)

# (opzionale) grafico delta giornaliero
st.bar_chart(det.set_index("Symbol")["Day_pct"].sort_values(), height=240, use_container_width=True)

# AUTORERUN locale
if st.session_state.auto_refresh:
    time.sleep(int(st.session_state.refresh_secs))
    st.rerun()