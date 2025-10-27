# pages/3_Analisi_tecnica.py — Analisi avanzata + autorefresh locale
from __future__ import annotations
import time
import pandas as pd
import altair as alt
import streamlit as st
from src.tech import get_history, add_indicators, summarize_signals, latest_table
from src.utils import ensure_state, require_data, reload_portfolio_from_state, glossary_md

# -----------------------
# Config + Stato applicazione
# -----------------------
st.set_page_config(page_title="Analisi tecnica", page_icon="📈", layout="wide")
ensure_state()
reload_portfolio_from_state()
require_data()

st.title("📈 Analisi tecnica (avanzata)")

# -----------------------
# Selettori principali
# -----------------------
symbols = st.session_state.df["Symbol"].unique().tolist()
top = st.container()
with top:
    c1, c2, c3, c4, c5 = st.columns([2,1,1,1,1])
    symbol = c1.selectbox("Simbolo", symbols)
    period = c2.selectbox("Periodo", ["1mo","3mo","6mo","1y","2y","5y"], index=3)
    interval = c3.selectbox("Intervallo", ["1d","1h","30m","15m"], index=0)
    show_volume = c4.checkbox("Mostra volumi", value=True)
    use_candles = c5.checkbox("Usa candele", value=True, help="Candlestick invece della sola linea Close")

with st.expander("Parametri indicatori", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    sma_str = c1.text_input("SMA (comma-sep)", value="20,50,200", help="Es: 10,20,50,200")
    ema_str = c2.text_input("EMA (comma-sep)", value="21,50")
    rsi_len = c3.number_input("RSI (periodi)", value=14, min_value=2)
    macd_fast = c4.number_input("MACD fast", value=12, min_value=2)

    c5, c6, c7, c8 = st.columns(4)
    macd_slow = c5.number_input("MACD slow", value=26, min_value=2)
    macd_signal = c6.number_input("MACD signal", value=9, min_value=2)
    bb_window = c7.number_input("Bollinger window", value=20, min_value=5)
    bb_std = c8.number_input("Bollinger σ", value=2.0, step=0.1)

    c9, c10 = st.columns(2)
    atr_window = c9.number_input("ATR window", value=14, min_value=2)
    show_bb = c10.checkbox("Mostra Bollinger", value=True)

    c11, c12, c13, c14 = st.columns(4)
    kc_window = c11.number_input("Keltner window", value=20, min_value=5, help="Di solito 20")
    kc_mult = c12.number_input("Keltner ATR×", value=1.5, step=0.1, help="1.5 classico per TTM Squeeze")
    show_kc = c13.checkbox("Mostra Keltner", value=False)
    rsi_fast_len = c14.number_input("RSI fast (trigger)", value=10, min_value=2, help="Per segnali più reattivi")

    add_ema8 = st.checkbox("Aggiungi EMA 8", value=True)

# -----------------------
# Dati + Indicatori (get_history è cache-ato in src.tech)
# -----------------------
hist = get_history(symbol, period=period, interval=interval)
if hist.empty:
    st.warning("Dati storici non disponibili per questo simbolo.")
    st.stop()

sma_list = [int(x) for x in sma_str.replace(";", ",").split(",") if x.strip().isdigit()]
ema_list = [int(x) for x in ema_str.replace(";", ",").split(",") if x.strip().isdigit()]

df_ind = add_indicators(
    hist,
    sma=tuple(sma_list or [20, 50, 200]),
    ema=tuple(([8] if add_ema8 else []) + (ema_list or [21, 50])),
    rsi_len=int(rsi_len),
    rsi_fast_len=int(rsi_fast_len),
    macd_fast=int(macd_fast),
    macd_slow=int(macd_slow),
    macd_signal=int(macd_signal),
    bb_window=int(bb_window),
    bb_std=float(bb_std),
    kc_window=int(kc_window),
    kc_mult=float(kc_mult),
    atr_window=int(atr_window),
)

df_plot = (
    df_ind.reset_index().rename(columns={"Date": "dt"})
    if "Date" in df_ind.columns
    else df_ind.reset_index().rename(columns={df_ind.index.name or "index": "dt"})
)

# -----------------------
# CRUSCOTTO INVESTITORE
# -----------------------
last = df_plot.iloc[-1]
close = float(last["Close"])
prev = float(df_plot["Close"].iloc[-2]) if len(df_plot) >= 2 else close
day_pl = (close - prev) / prev * 100 if prev else 0
dist_high = float(last.get("dist_from_52w_high_%", float("nan")))
dist_low  = float(last.get("dist_from_52w_low_%", float("nan")))
squeeze_on = bool(last.get("SQUEEZE_ON", False))
squeeze_up = bool(last.get("SQUEEZE_OFF_UP", False))
squeeze_dn = bool(last.get("SQUEEZE_OFF_DOWN", False))

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Prezzo", f"{close:.2f}")
k2.metric("Day Δ%", f"{day_pl:.2f}%")
k3.metric("Dist. 52W High", f"{dist_high:.1f}%" if pd.notna(dist_high) else "—")
k4.metric("Dist. 52W Low", f"{dist_low:.1f}%" if pd.notna(dist_low) else "—")
k5.metric("Squeeze", "ON" if squeeze_on else ("UP" if squeeze_up else ("DOWN" if squeeze_dn else "—")))

# -----------------------
# Impostazioni comuni Altair
# -----------------------
axis_x = alt.Axis(
    title="",
    labelAngle=-30,
    labelFlush=False,
    labelOverlap=True
)
legend_bottom = alt.Legend(
    title=None,
    orient="bottom",
    direction="horizontal",
    columns=6,
    labelLimit=1000,
    symbolSize=120
)

# -----------------------
# GRAFICO PREZZO (candlestick + MA/EMA + bande)
# -----------------------
# Serie per la legenda
series_frames = [df_plot[["dt", "Close"]].rename(columns={"Close": "value"}).assign(series="Close")]
for n in sma_list:
    col = f"SMA_{n}"
    if col in df_plot.columns:
        series_frames.append(df_plot[["dt", col]].rename(columns={col: "value"}).assign(series=f"SMA {n}"))
for n in ema_list:
    col = f"EMA_{n}"
    if col in df_plot.columns:
        series_frames.append(df_plot[["dt", col]].rename(columns={col: "value"}).assign(series=f"EMA {n}"))
for col, label in [("KC_low", "KC low"), ("KC_up", "KC up"), ("KC_mid", "KC mid")]:
    if show_kc and col in df_plot.columns:
        series_frames.append(df_plot[["dt", col]].rename(columns={col: "value"}).assign(series=label))
price_long = pd.concat(series_frames, ignore_index=True)

# Linee (con legenda)
price_lines = alt.Chart(price_long).mark_line().encode(
    x=alt.X("dt:T", axis=axis_x),
    y=alt.Y("value:Q", title="Prezzo"),
    color=alt.Color("series:N", legend=legend_bottom),
    tooltip=[alt.Tooltip("dt:T", title="Data"),
             alt.Tooltip("series:N", title="Serie"),
             alt.Tooltip("value:Q", title="Valore", format=".2f")]
)

# Bande visive (no legenda)
bands = []
if show_kc and {"KC_low","KC_up"}.issubset(df_plot.columns):
    bands.append(
        alt.Chart(df_plot).mark_area(opacity=0.12).encode(
            x="dt:T", y="KC_low:Q", y2="KC_up:Q",
            tooltip=[alt.Tooltip("dt:T", title="Data"),
                     alt.Tooltip("KC_low:Q", title="KC low", format=".2f"),
                     alt.Tooltip("KC_up:Q", title="KC up", format=".2f")]
        )
    )
if show_bb and {"BB_low","BB_up"}.issubset(df_plot.columns):
    bands.append(
        alt.Chart(df_plot).mark_area(opacity=0.15).encode(
            x="dt:T", y="BB_low:Q", y2="BB_up:Q",
            tooltip=[alt.Tooltip("dt:T", title="Data"),
                     alt.Tooltip("BB_low:Q", title="BB low", format=".2f"),
                     alt.Tooltip("BB_up:Q", title="BB up", format=".2f")]
        )
    )

# Candlestick (opzionale)
if use_candles and {"Open","High","Low","Close"}.issubset(df_plot.columns):
    up = alt.value("#16a34a")   # verde
    dn = alt.value("#ef4444")   # rosso
    rule = alt.Chart(df_plot).mark_rule().encode(
        x="dt:T",
        y="Low:Q",
        y2="High:Q",
        color=alt.condition("datum.Open <= datum.Close", up, dn),
        tooltip=[
            alt.Tooltip("dt:T", title="Data"),
            alt.Tooltip("Open:Q", format=".2f"),
            alt.Tooltip("High:Q", format=".2f"),
            alt.Tooltip("Low:Q", format=".2f"),
            alt.Tooltip("Close:Q", format=".2f"),
        ]
    )
    bar = alt.Chart(df_plot).mark_bar(size=5).encode(
        x="dt:T",
        y="Open:Q",
        y2="Close:Q",
        color=alt.condition("datum.Open <= datum.Close", up, dn)
    )
    price_base = alt.layer(*(bands + [rule, bar, price_lines]))
else:
    price_base = alt.layer(*(bands + [price_lines]))

price_chart = (
    price_base.properties(height=360)
    .configure(padding={"top": 8, "left": 8, "right": 8, "bottom": 90})
    .configure_view(clip=False)
    .configure_axis(labelLimit=0)
    .configure_legend(orient="bottom")
    .interactive()
)
st.altair_chart(price_chart, use_container_width=True)

# -----------------------
# RSI
# -----------------------
st.subheader("RSI")
rsi_line = alt.Chart(df_plot).mark_line().encode(
    x=alt.X("dt:T", axis=axis_x),
    y=alt.Y("RSI:Q", title="RSI"),
    color=alt.value("#2563eb"),
    tooltip=[alt.Tooltip("dt:T", title="Data"),
             alt.Tooltip("RSI:Q", format=".1f")]
)
rsi_thresh = pd.DataFrame({"y": [30, 70], "soglia": ["RSI 30", "RSI 70"]})
rsi_rules = alt.Chart(rsi_thresh).mark_rule().encode(
    y="y:Q", color=alt.Color("soglia:N", title="Soglie RSI")
)
rsi_chart = (rsi_line + rsi_rules).properties(height=190).configure(padding={"bottom": 40}).configure_view(clip=False).interactive()
st.altair_chart(rsi_chart, use_container_width=True)

# -----------------------
# MACD
# -----------------------
st.subheader("MACD")
macd_long = pd.concat([
    df_plot[["dt","MACD"]].rename(columns={"MACD":"value"}).assign(series="MACD"),
    df_plot[["dt","MACD_signal"]].rename(columns={"MACD_signal":"value"}).assign(series="MACD signal"),
], ignore_index=True)
macd_lines = alt.Chart(macd_long).mark_line().encode(
    x=alt.X("dt:T", axis=axis_x),
    y=alt.Y("value:Q", title="MACD"),
    color=alt.Color("series:N", title="Linee MACD"),
    tooltip=[alt.Tooltip("dt:T", title="Data"),
             alt.Tooltip("series:N", title="Linea"),
             alt.Tooltip("value:Q", title="Valore", format=".4f")]
)
macd_hist = alt.Chart(df_plot).transform_calculate(
    sign="datum.MACD_hist >= 0 ? '≥ 0' : '< 0'"
).mark_bar().encode(
    x=alt.X("dt:T", axis=axis_x),
    y="MACD_hist:Q",
    color=alt.Color("sign:N", title="Istogramma MACD", scale=alt.Scale(domain=["≥ 0","< 0"]))
).properties(height=170)
st.altair_chart((macd_hist + macd_lines).configure(padding={"bottom": 50}).configure_view(clip=False).interactive(), use_container_width=True)

# -----------------------
# Volumi
# -----------------------
if show_volume and "Volume" in df_plot.columns:
    st.subheader("Volumi")
    vol_bar = alt.Chart(df_plot).mark_bar().encode(
        x=alt.X("dt:T", axis=axis_x),
        y=alt.Y("Volume:Q", title="Volume"),
        color=alt.value("#9ca3af"),
        tooltip=[alt.Tooltip("dt:T", title="Data"),
                 alt.Tooltip("Volume:Q", format=",")]
    )
    vol_layers = [vol_bar]
    vser = []
    if "Vol_MA20" in df_plot.columns:
        vser.append(df_plot[["dt","Vol_MA20"]].rename(columns={"Vol_MA20":"value"}).assign(series="Vol MA20"))
    if "Vol_MA50" in df_plot.columns:
        vser.append(df_plot[["dt","Vol_MA50"]].rename(columns={"Vol_MA50":"value"}).assign(series="Vol MA50"))
    if vser:
        vol_long = pd.concat(vser, ignore_index=True)
        vol_lines = alt.Chart(vol_long).mark_line().encode(
            x=alt.X("dt:T", axis=axis_x),
            y="value:Q",
            color=alt.Color("series:N", title="Medie Volume"),
            tooltip=[alt.Tooltip("dt:T", title="Data"),
                     alt.Tooltip("series:N"),
                     alt.Tooltip("value:Q", format=",")]
        )
        vol_layers.append(vol_lines)
    vol_chart = alt.layer(*vol_layers).properties(height=190).configure(padding={"bottom": 40}).configure_view(clip=False).interactive()
    st.altair_chart(vol_chart, use_container_width=True)

# -----------------------
# Segnali + Tabella
# -----------------------
st.subheader("Segnali sintetici")
for txt in summarize_signals(df_ind, sma_list, ema_list):
    st.info(txt)

st.subheader("Valori e distanze attuali")
tab = latest_table(df_ind, sma_list, ema_list)
if not tab.empty:
    def _styler(df):
        def colorize(v):
            try:
                if pd.isna(v): return ""
                v = float(v)
                return "color: green;" if v > 0 else ("color: red;" if v < 0 else "")
            except Exception:
                return ""
        sty = df.style.applymap(colorize, subset=["Δ% vs Close"])
        sty = sty.format({"Value": "{:.4f}", "Δ% vs Close": "{:.2f}%"})
        return sty
    st.dataframe(_styler(tab), use_container_width=True, height=280)
else:
    st.caption("Nessun dato disponibile per la tabella.")

with st.expander("Glossario acronimi"):
    st.markdown(glossary_md(), unsafe_allow_html=True)

# -----------------------
# Auto refresh locale (debounced)
# -----------------------
if st.session_state.auto_refresh:
    time.sleep(int(st.session_state.refresh_secs))
    st.rerun()
