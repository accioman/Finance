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
    st.session_state.setdefault("refresh_secs", 30)

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


def render_glossary_tabs():

    # ---- Helpers ------------------------------------------------------------
    def _h(txt: str):
        st.markdown(f"**{txt}**")

    def _eq(latex: str):
        # Renderizza una formula singola in modo affidabile
        st.latex(latex)

    def _p(txt: str):
        st.markdown(txt)

    st.caption("Suggerimento: usa le tab come sotto-menu. Supporta LaTeX e link interni.")
    search = st.text_input(
        "Filtra voci (testo libero)",
        value="",
        placeholder="es. RSI, ATR, squeeze..."
    ).strip().lower()

    tabs = st.tabs([
        "SMA", "EMA", "RSI", "MACD", "Bollinger", "ATR",
        "Keltner", "Squeeze", "Gap", "52-Week", "Volumi",
        "Candele", "Crossovers", "P/L & Day P/L",
        "Drawdown", "Volatilità", "Sharpe", "Peso", "Best Practice",
        "Combo & Workflow"
    ])

    def match(*keywords: str) -> bool:
        if not search:
            return True
        hay = " ".join(keywords).lower()
        return search in hay

    # -------------------- SMA ------------------------------------------------
    with tabs[0]:
        if match("sma simple moving average media mobile"):
            st.header("SMA — Simple Moving Average")
            _h("Cos’è")
            _p("Media aritmetica dei **close** degli ultimi *n* periodi; riduce il rumore ma è lenta.")
            _h("Formula")
            _eq(r"\mathrm{SMA}_n(t)=\frac{1}{n}\sum_{i=0}^{n-1} \mathrm{Close}_{t-i}")
            _h("A cosa serve")
            _p("• Smoothing; • **Supporti/resistenze dinamiche**; • Riferimento di **trend di fondo** (es. SMA200).")
            _h("Come usarla")
            _p("- Bias: prezzo sopra SMA200 ⇒ bias **long**; sotto ⇒ **cautela**.\n"
               "- **Slope**: SMA inclinata in su/down conferma direzione.\n"
               "- Filtra i breakout: richiedi chiusure **sopra** la SMA chiave + volumi.")
            _h("In pratica")
            _p("Usa **SMA50/200** per regime, **SMA20** per pullback su daily; su intraday preferisci EMA (più reattiva).")

    # -------------------- EMA ------------------------------------------------
    with tabs[1]:
        if match("ema exponential media mobile esponenziale"):
            st.header("EMA — Exponential Moving Average")
            _h("Cos’è")
            _p("Media mobile che **pesa di più i dati recenti** ⇒ più reattiva della SMA.")
            _h("Formula")
            _eq(r"\mathrm{EMA}_n(t)=\alpha\, \mathrm{Close}_t+(1-\alpha)\,\mathrm{EMA}_n(t-1),\quad \alpha=\frac{2}{n+1}")
            _h("A cosa serve")
            _p("Segnali **crossover** (8/21/50), trailing dinamico, timing su pullback.")
            _h("Come usarla")
            _p("- **Trend-following**: sequenza EMA8>21>50 in uptrend.\n"
               "- **Trigger**: rientro sopra EMA8/21 dopo pullback.\n"
               "- Evita whipsaw: conferma con MACD/RSI o struttura HH/HL.")
            _h("In pratica")
            _p("Intraday: **EMA8/21** ottime per ritmo; swing: **EMA21/50** per trend intermedio.")

    # -------------------- RSI ------------------------------------------------
    with tabs[2]:
        if match("rsi relative strength index oscillatore ipercomprato ipervenduto"):
            st.header("RSI — Relative Strength Index")
            _h("Cos’è")
            _p("Oscillatore 0-100 (Wilder) che confronta guadagni/perdite medie.")
            _h("Calcolo (Wilder, n)")
            _eq(r"\Delta_t=C_t-C_{t-1},\quad U_t=\max(\Delta_t,0),\quad D_t=\max(-\Delta_t,0)")
            _eq(r"AU=\text{MA}(U,n),\quad AD=\text{MA}(D,n),\quad RS=\frac{AU}{AD},\quad RSI=100-\frac{100}{1+RS}")
            _h("A cosa serve")
            _p("• **Momentum**; • identificare **eccessi** (70/30) e **divergenze**.")
            _h("Come usarlo")
            _p("- **Range rules**: in uptrend RSI raramente <40; in downtrend raramente >60.\n"
               "- **Divergenze**: prezzo fa HH ma RSI non conferma ⇒ momentum in esaurimento.")
            _h("In pratica")
            _p("Usalo per **filtrare breakout** (RSI>50) o per timing **mean-reversion** vicino a 30/70, con volume/price action.")

    # -------------------- MACD ----------------------------------------------
    with tabs[3]:
        if match("macd moving average convergence divergence momentum"):
            st.header("MACD — Moving Average Convergence/Divergence")
            _h("Componenti (daily tipico 12,26,9)")
            _eq(r"\text{MACD}=\mathrm{EMA}_{12}-\mathrm{EMA}_{26}")
            _eq(r"\text{Signal}=\mathrm{EMA}_9(\text{MACD}),\quad \text{Histogram}=\text{MACD}-\text{Signal}")
            _h("A cosa serve")
            _p("Misura il **momentum di trend**. L’istogramma dà la **derivata** del momentum.")
            _h("Come usarlo")
            _p("- **Cross MACD/Signal** per timing; meglio se sopra lo zero in long, sotto in short.\n"
               "- **Istogramma crescente** ⇒ slancio che aumenta (buono per breakout).")
            _h("In pratica")
            _p("Non usarlo da solo su intraday rumoroso; combina con struttura prezzi e volumi.")

    # -------------------- Bollinger -----------------------------------------
    with tabs[4]:
        if match("bollinger bb bandwidth percent b"):
            st.header("Bollinger Bands — BB")
            _h("Definizione")
            _eq(r"MB=\mathrm{SMA}_n,\quad UP=MB+k\sigma_n,\quad LOW=MB-k\sigma_n")
            _h("%B e Bandwidth")
            _eq(r"\%B=\frac{C-LOW}{UP-LOW},\quad \text{Bandwidth}=\frac{UP-LOW}{MB}\times 100")
            _h("A cosa serve")
            _p("Misura **dispersione**; le **compressioni** spesso precedono espansioni (breakout).")
            _h("Come usarle")
            _p("Cerca **bandwidth bassi** e price action coerente; conferma con volumi/RSI/MACD.")

    # -------------------- ATR ------------------------------------------------
    with tabs[5]:
        if match("atr average true range volatilità true range"):
            st.header("ATR — Average True Range")
            _h("True Range e ATR")
            _eq(r"TR_t=\max\{H-L, |H-C_{t-1}|, |L-C_{t-1}|\}")
            _eq(r"ATR=\text{MA}(TR,n),\quad ATR\%=\frac{ATR}{C}\times 100")
            _h("A cosa serve")
            _p("• **Volatilità realizzata**; • **position sizing**; • **trailing stop** dinamici.")
            _h("Come usarlo")
            _p("Stop a 1.5–3×ATR dal prezzo; confronta **ATR%** tra titoli per parità di rischio.")

    # -------------------- Keltner -------------------------------------------
    with tabs[6]:
        if match("keltner kc channels atr ema canali"):
            st.header("Keltner Channels — KC")
            _h("Definizione")
            _eq(r"KC_{mid}=\mathrm{EMA}_n,\quad KC_{up}=KC_{mid}+m\cdot ATR,\quad KC_{low}=KC_{mid}-m\cdot ATR")
            _h("A cosa serve")
            _p("Canale basato su **range (ATR)**: più stabile ai picchi outlier rispetto alle BB.")
            _h("Come usarli")
            _p("Break/close **oltre KCup** con volumi ⇒ slancio; **pullback** su KCmid come buy-the-dip in trend.")

    # -------------------- Squeeze -------------------------------------------
    with tabs[7]:
        if match("squeeze ttm bb kc compressione breakout"):
            st.header("TTM Squeeze — BB vs KC")
            _h("Regole")
            _p("**ON**: BB *dentro* KC (compressione). **OFF**: BB *fuori* KC (rilascio).")
            _h("Uso pratico")
            _p("Trade il **rilascio** nella direzione confermata da **volume + RSI/MACD**; evita rotture senza partecipazione.")

    # -------------------- Gap -----------------------------------------------
    with tabs[8]:
        if match("gap open close percentuale news"):
            st.header("Gap %")
            _eq(r"\text{Gap}\%=\left(\frac{Open_t}{Close_{t-1}}-1\right)\times 100")
            _h("A cosa serve")
            _p("Misura l’**impulso da news/aste**. Gap > ±2% sono “material”.")
            _h("Come usarli")
            _p("Cerca **gap + run** con volumi (continuation) o **fade** su livelli chiave (reversal), sempre con stop chiari.")

    # -------------------- 52-Week -------------------------------------------
    with tabs[9]:
        if match("52w high low massimo minimo annuale distanza forza relativa"):
            st.header("52-Week High/Low & Distanze")
            _eq(r"High_{52w}=\max(C_{t-251..t}),\quad Low_{52w}=\min(C_{t-251..t})")
            _eq(r"\Delta_{High}\%=\left(\frac{C}{High_{52w}}-1\right)\times 100,\quad \Delta_{Low}\%=\left(\frac{C}{Low_{52w}}-1\right)\times 100")
            _h("A cosa serve")
            _p("Forza relativa di medio periodo; **breakout 52w** spesso catalizzano trend con flussi/attenzione.")
            _h("Come usarli")
            _p("Richiedi close **sopra** i massimi annuali + volumi; usa ATR% per dimensionare lo stop.")

    # -------------------- Volumi --------------------------------------------
    with tabs[10]:
        if match("volumi media volume ma20 ma50 conferma breakout partecipazione"):
            st.header("Medie Volume (Vol MA20/MA50)")
            _h("Cos’è")
            _p("Medie mobili del **volume** per valutare la **partecipazione**.")
            _h("Come usarle")
            _p("Breakout/impulsi con volume **> 1.5× MA20** ⇒ maggiore probabilità di follow-through.")

    # -------------------- Candele -------------------------------------------
    with tabs[11]:
        if match("candele candlestick ohlc pattern hammer doji engulfing"):
            st.header("Candele (OHLC)")
            _h("Cos’è")
            _p("Corpo tra **Open** e **Close**; ombre = range fino a **High/Low**.")
            _h("A cosa serve")
            _p("Lettura del **sentiment intrabar** e dei rifiuti di prezzo (rejection).")
            _h("Come usarle")
            _p("Pattern **solo con contesto**: trend, livelli, volumi. Un hammer su supporto + volumi vale, da solo no.")

    # -------------------- Crossovers ----------------------------------------
    with tabs[12]:
        if match("crossover cross sma ema golden death incroci segnali"):
            st.header("Crossovers (SMA/EMA)")
            _h("Cos’è")
            _p("Incrocio della media **veloce** sopra/sotto la **lenta** ⇒ segnali trend-following **ritardati**.")
            _h("Come usarli")
            _p("Meglio se **allineati** con struttura HH/HL, **volumi** e **momentum** (MACD>0, RSI>50).")

    # -------------------- P/L -----------------------------------------------
    with tabs[13]:
        if match("pl p/l profit loss day giornaliero costo medio"):
            st.header("P/L % vs Cost & Day P/L")
            _eq(r"P/L\%=\left(\frac{Prezzo_{att}}{Costo_{medio}}-1\right)\times 100")
            _eq(r"\text{P/L vs Prev}=\left(\frac{Prezzo_{att}}{Close_{t-1}}-1\right)\times 100")
            _h("Uso")
            _p("Controllo performance e rischio **per posizione**; evita di trasformare perdite in investimenti senza motivo.")

    # -------------------- Drawdown ------------------------------------------
    with tabs[14]:
        if match("drawdown maxdd picco calo rischio"):
            st.header("Drawdown / Max Drawdown")
            _eq(r"DD(t)=\frac{Equity(t)-\max_{\tau\le t}Equity(\tau)}{\max_{\tau\le t}Equity(\tau)}")
            _h("Uso")
            _p("Misura il **rischio vissuto**; **MaxDD** è il peggiore DD. Fondamentale per confrontare strategie.")

    # -------------------- Volatilità ----------------------------------------
    with tabs[15]:
        if match("volatilità annualizzata std sqrt 252"):
            st.header("Volatilità annualizzata (stima)")
            _eq(r"\sigma_{ann}=\mathrm{std}(r_t)\cdot\sqrt{252}")
            _h("Uso")
            _p("Rende confrontabili asset; non è costante ⇒ aggiorna finestre e adatta sizing.")

    # -------------------- Sharpe --------------------------------------------
    with tabs[16]:
        if match("sharpe mu sigma rischio rendimento sortino calmar"):
            st.header("Sharpe (naïve, rf≈0)")
            _eq(r"\text{Sharpe}=\frac{\mu_{ann}}{\sigma_{ann}}")
            _h("Uso & limiti")
            _p("Rendimento per unità di rischio; **non** cattura code grasse/asimmetrie ⇒ affianca **Sortino/Calmar**.")

    # -------------------- Peso ----------------------------------------------
    with tabs[17]:
        if match("peso weight portafoglio concentrazione rischio"):
            st.header("Weight % (peso in portafoglio)")
            _eq(r"Peso\%=\frac{Valore\ posizione}{Valore\ totale\ portafoglio}\times 100")
            _h("Uso")
            _p("Controllo **concentrazione**; imposta limiti per **titolo/settore** e usa sizing coerente con ATR%.")

    # -------------------- Best Practice -------------------------------------
    with tabs[18]:
        if match("best practice trend momentum volatilità volume contesto confluenze"):
            st.header("Best practice di lettura combinata")
            _p("1) **Trend**: prezzo vs SMA/EMA (bias di fondo)\n"
               "2) **Momentum**: MACD/RSI (slancio e cambi di pendenza)\n"
               "3) **Volatilità**: ATR% e BB Bandwidth (compressione→espansione)\n"
               "4) **Volume**: conferme su breakout\n"
               "5) **Contesto**: 52w, gap, candele, news\n"
               "Cerca **confluenze**, evita segnali singoli isolati, fai **backtest** con costi/slippage.")

    # -------------------- Combo & Workflow ----------------------------------
    with tabs[19]:
        if match("combo workflow processo segnali insieme strategia pattern set up"):
            st.header("Combo & Workflow operativo")
            _h("Setup trend-following (swing daily)")
            _p("- Bias **long**: Prezzo > **SMA200**, EMA21 in su\n"
               "- **Squeeze OFF** al rialzo (BB fuori KC) + **Bandwidth** che si espande\n"
               "- **MACD** sopra 0 e istogramma crescente; **RSI** > 50\n"
               "- **Breakout** dei massimi recenti **con volumi > 1.5×MA20**\n"
               "- **Stop**: 2×ATR sotto il minimo di setup; **Rischio**: sizing via ATR%\n"
               "- **Gestione**: trailing su KCmid/EMA21 o stop dinamico a 2×ATR")
            _h("Setup pullback nel trend")
            _p("- Trend up: sequenza **EMA8>21>50**\n"
               "- Pullback verso **EMA21/KCmid** con **RSI** che regge >40\n"
               "- **MACD** non deve girare decisamente sotto 0\n"
               "- Entrata su **re-claim** della EMA/trigger candlestick + volume")
            _h("Setup breakout 52-week")
            _p("- Base sotto il massimo annuale; **bandwidth** compresso\n"
               "- Close sopra **52w High** con **volume** elevato; **RSI** >55\n"
               "- **Stop**: sotto livello breakout o 1.5–2×ATR")
            _h("Mean-reversion controllata")
            _p("- Laterale/assenza trend: **RSI** vicino a 30/70 su livelli noti\n"
               "- Conferme da **pattern** e **volumi**; target verso MB/EMA21\n"
               "- Rischio: stop corti, accetta frequenti piccoli stop")
            _h("Note di rischio")
            _p("• Nessun indicatore è un oracolo: sono **trasformazioni del prezzo**.\n"
               "• Evita overfitting: parametri e timeframe cambiano molto i segnali.\n"
               "• In mercati laterali, i segnali di trend degradano: filtra con volatilità/ADX/volumi.")
