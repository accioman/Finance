# pages/3_Analisi_tecnica.py — Analisi avanzata + autorefresh locale
from __future__ import annotations
import time
import pandas as pd
import altair as alt
import streamlit as st
from src.tech import get_history, add_indicators, summarize_signals, latest_table
from src.utils import ensure_state, require_data, reload_portfolio_from_state, glossary_md

st.set_page_config(page_title="Analisi tecnica", page_icon="📈", layout="wide")

ensure_state()
reload_portfolio_from_state()
require_data()

st.title("📈 Analisi tecnica (avanzata)")

symbols = st.session_state.df["Symbol"].unique().tolist()
col1, col2, col3, col4 = st.columns([2,1,1,1])
symbol = col1.selectbox("Simbolo", symbols)
period = col2.selectbox("Periodo", ["1mo","3mo","6mo","1y","2y","5y"], index=3)
interval = col3.selectbox("Intervallo", ["1d","1h","30m","15m"], index=0)
show_volume = col4.checkbox("Mostra volumi", value=True)

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

hist = get_history(symbol, period=period, interval=interval)
if hist.empty:
    st.warning("Dati storici non disponibili per questo simbolo.")
    st.stop()

# -----------------
# Calcolo indicatori
# -----------------
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

# ------------
# Impostazioni comuni per assi/legende (anti-taglio)
# ------------
axis_x = alt.Axis(
    title="",
    labelAngle=-30,
    labelFlush=False,      # NON allineare al bordo (evita taglio)
    labelOverlap=True      # gestione overlap
    # (niente tickCount="auto": non valido)
)
legend_bottom = alt.Legend(
    title="Serie",
    orient="bottom",
    direction="horizontal",
    columns=4,
    labelLimit=1000,       # spazio ampio per i label
    symbolSize=120
)

# =========================
#   CHART: Prezzo + MA/EMA
# =========================
series_frames = [
    df_plot[["dt", "Close"]].rename(columns={"Close": "value"}).assign(series="Close")
]

# SMA
for n in sma_list:
    col = f"SMA_{n}"
    if col in df_plot.columns:
        series_frames.append(
            df_plot[["dt", col]].rename(columns={col: "value"}).assign(series=f"SMA {n}")
        )

# EMA
for n in ema_list:
    col = f"EMA_{n}"
    if col in df_plot.columns:
        series_frames.append(
            df_plot[["dt", col]].rename(columns={col: "value"}).assign(series=f"EMA {n}")
        )

# Keltner (linee opzionali in legenda)
for col, label in [("KC_low", "KC low"), ("KC_up", "KC up"), ("KC_mid", "KC mid")]:
    if show_kc and col in df_plot.columns:
        series_frames.append(
            df_plot[["dt", col]].rename(columns={col: "value"}).assign(series=label)
        )

price_long = pd.concat(series_frames, ignore_index=True)

price_lines = alt.Chart(price_long).mark_line().encode(
    x=alt.X("dt:T", axis=axis_x),
    y=alt.Y("value:Q", title="Prezzo"),
    color=alt.Color("series:N", legend=legend_bottom),
)

# Bande (non in legenda)
bands = []
if show_kc and all(c in df_plot.columns for c in ["KC_low", "KC_up"]):
    bands.append(
        alt.Chart(df_plot).mark_area(opacity=0.12).encode(
            x=alt.X("dt:T", axis=axis_x), y="KC_low:Q", y2="KC_up:Q"
        )
    )
if show_bb and all(c in df_plot.columns for c in ["BB_low", "BB_up"]):
    bands.append(
        alt.Chart(df_plot).mark_area(opacity=0.15).encode(
            x=alt.X("dt:T", axis=axis_x), y="BB_low:Q", y2="BB_up:Q"
        )
    )

price_chart = (
    alt.layer(*(bands + [price_lines]))
    .properties(height=340)
    .configure(padding={"top": 8, "left": 8, "right": 8, "bottom": 80})
    .configure_view(clip=False)         # evita ritaglio
    .configure_axis(labelLimit=0)       # nessun limite caratteri etichette
    .configure_legend(orient="bottom")
    .interactive()
)

st.altair_chart(price_chart, use_container_width=True)

# =========
#   RSI
# =========
st.subheader("RSI")
rsi_line = alt.Chart(df_plot).mark_line().encode(
    x=alt.X("dt:T", axis=axis_x),
    y=alt.Y("RSI:Q", title="RSI"),
    color=alt.value("#2563eb"),
)
rsi_thresh = pd.DataFrame({"y": [30, 70], "soglia": ["RSI 30", "RSI 70"]})
rsi_rules = alt.Chart(rsi_thresh).mark_rule().encode(
    y="y:Q", color=alt.Color("soglia:N", title="Soglie RSI")
)
rsi_chart = (
    (rsi_line + rsi_rules)
    .properties(height=190)
    .configure(padding={"bottom": 40})
    .configure_view(clip=False)
    .interactive()
)
st.altair_chart(rsi_chart, use_container_width=True)

# =========
#   MACD
# =========
st.subheader("MACD")
macd_long = pd.concat(
    [
        df_plot[["dt", "MACD"]].rename(columns={"MACD": "value"}).assign(series="MACD"),
        df_plot[["dt", "MACD_signal"]]
        .rename(columns={"MACD_signal": "value"})
        .assign(series="MACD signal"),
    ],
    ignore_index=True,
)
macd_lines = alt.Chart(macd_long).mark_line().encode(
    x=alt.X("dt:T", axis=axis_x),
    y=alt.Y("value:Q", title="MACD"),
    color=alt.Color("series:N", title="Linee MACD"),
)
macd_hist = (
    alt.Chart(df_plot)
    .transform_calculate(sign="datum.MACD_hist >= 0 ? '≥ 0' : '< 0'")
    .mark_bar()
    .encode(
        x=alt.X("dt:T", axis=axis_x),
        y="MACD_hist:Q",
        color=alt.Color(
            "sign:N", title="Istogramma MACD", scale=alt.Scale(domain=["≥ 0", "< 0"])
        ),
    )
    .properties(height=170)
)
st.altair_chart(
    (macd_hist + macd_lines)
    .configure(padding={"bottom": 50})
    .configure_view(clip=False)
    .interactive(),
    use_container_width=True,
)

# =========
#   Volumi
# =========
if show_volume and "Volume" in df_plot.columns:
    st.subheader("Volumi")
    vol_bar = alt.Chart(df_plot).mark_bar().encode(
        x=alt.X("dt:T", axis=axis_x),
        y=alt.Y("Volume:Q", title="Volume"),
        color=alt.value("#9ca3af"),
    )
    vol_layers = [vol_bar]
    vol_series = []
    if "Vol_MA20" in df_plot.columns:
        vol_series.append(
            df_plot[["dt", "Vol_MA20"]]
            .rename(columns={"Vol_MA20": "value"})
            .assign(series="Vol MA20")
        )
    if "Vol_MA50" in df_plot.columns:
        vol_series.append(
            df_plot[["dt", "Vol_MA50"]]
            .rename(columns={"Vol_MA50": "value"})
            .assign(series="Vol MA50")
        )
    if vol_series:
        vol_long = pd.concat(vol_series, ignore_index=True)
        vol_lines = alt.Chart(vol_long).mark_line().encode(
            x=alt.X("dt:T", axis=axis_x),
            y="value:Q",
            color=alt.Color("series:N", title="Medie Volume"),
        )
        vol_layers.append(vol_lines)

    vol_chart = (
        alt.layer(*vol_layers)
        .properties(height=190)
        .configure(padding={"bottom": 40})
        .configure_view(clip=False)
        .interactive()
    )
    st.altair_chart(vol_chart, use_container_width=True)

# =========
#   Segnali + Tabella
# =========
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

# Auto refresh locale
if st.session_state.auto_refresh:
    time.sleep(int(st.session_state.refresh_secs))
    st.rerun()
