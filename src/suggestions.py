from __future__ import annotations
import pandas as pd

def derive_suggestions(df: pd.DataFrame, upper: float, lower: float):
    """Suggerimenti semplici (non consulenza): take profit, stop loss, rebalance, concentrazione, DCA."""
    hints = []
    if df.empty:
        return hints

    d = df.copy()
    d["Value"] = d["Price Used"] * d["Quantity"]
    d["PL_pct"] = (d["Price Used"] - d["Purchase Price"]) / d["Purchase Price"] * 100.0
    tot = d["Value"].sum()
    d["Weight_%"] = d["Value"] / tot * 100.0 if tot else 0.0

    tp = d[d["PL_pct"] >= upper * 1.5]
    if not tp.empty:
        syms = ", ".join(tp.sort_values("PL_pct", ascending=False)["Symbol"].tolist())
        hints.append(f"🏁 Valuta presa di profitto parziale su: {syms} (profitto elevato).")

    sl = d[d["PL_pct"] <= -abs(lower) * 1.5]
    if not sl.empty:
        syms = ", ".join(sl.sort_values("PL_pct")["Symbol"].tolist())
        hints.append(f"🛑 Valuta riduzione/uscita su: {syms} (perdita severa).")

    heavy = d[d["Weight_%"] > 15]
    if not heavy.empty:
        syms = ", ".join(heavy.sort_values("Weight_%", ascending=False)["Symbol"].tolist())
        hints.append(f"⚖️ Ribilanciamento: peso >15% su {syms}.")

    top3 = d.sort_values("Value", ascending=False).head(3)
    if top3["Value"].sum() / max(tot, 1) > 0.5:
        hints.append("🧩 Concentrazione alta: top 3 >50% del portafoglio.")

    zone = d[(d["PL_pct"] <= -abs(lower)) & (d["PL_pct"] > -abs(lower) * 1.5) & (d["Weight_%"] < 5)]
    if not zone.empty:
        syms = ", ".join(zone.sort_values("PL_pct")["Symbol"].tolist())
        hints.append(f"➕ Possibile DCA su: {syms} (drawdown moderato e peso contenuto).")

    return hints
