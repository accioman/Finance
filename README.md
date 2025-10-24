# Portfolio Stats & Alerts (CLI) — yfinance

Avvia un **programma a riga di comando** che legge un CSV stile Yahoo (come quello che hai postato),
calcola **statistiche** (valore, P/L, pesi, top movers) e mostra **alert** basati su soglie percentuali.
Opzionalmente può usare yfinance per aggiornare i prezzi live.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso
```bash
python app.py --csv path/al/tuo.csv --once   --upper 10 --lower 10 --use-live-prices
```
- `--upper / --lower`: soglie % di alert vs `Purchase Price` (prezzo medio).
- `--use-live-prices`: prova a prendere l'ultimo prezzo da yfinance; se fallisce usa `Current Price` dal CSV.
- `--plot`: mostra grafici (allocazione e P/L%%) con matplotlib.
- `--schedule 10`: ripete i controlli ogni N minuti (richiede APScheduler).

## Input CSV atteso
Deve contenere almeno le colonne:
`Symbol, Current Price, Purchase Price, Quantity`.
Sono accettate le righe di cassa `$$CASH_TX` (somma su `Quantity`).

## Esempi
- Esecuzione singola con grafici:
```bash
python app.py --csv sample.csv --once --use-live-prices --plot
```
- Scheduler ogni 15 minuti:
```bash
python app.py --csv sample.csv --schedule 15 --upper 12 --lower 8
```

> Nota: Solo informativo/educativo. Non è consulenza finanziaria.
