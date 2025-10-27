from __future__ import annotations
import pandas as pd

def compute_stats_tables(df: pd.DataFrame, cash_total: float):
    d = df.copy()

    # Valori EUR
    d["Value"]   = pd.to_numeric(d["Price EUR"], errors="coerce") * pd.to_numeric(d["Quantity"], errors="coerce")
    d["Avg EUR"] = pd.to_numeric(d["Purchase EUR"], errors="coerce")
    d["PL_abs"]  = (d["Price EUR"] - d["Avg EUR"]) * d["Quantity"]
    d["PL_pct"]  = (d["Price EUR"] - d["Avg EUR"]) / d["Avg EUR"] * 100.0

    # --- GIORNO: delta per posizione ---
    prev_pos_val = (pd.to_numeric(d["Prev Close EUR"], errors="coerce") * pd.to_numeric(d["Quantity"], errors="coerce"))
    d["Day_abs"] = (pd.to_numeric(d["Day_Delta_EUR"], errors="coerce") * pd.to_numeric(d["Quantity"], errors="coerce"))
    d["Day_pct"] = (d["Day_abs"] / prev_pos_val.replace(0, pd.NA) * 100.0).fillna(0.0)

    total_value = d["Value"].sum() + cash_total
    d["Weight_%"] = d["Value"] / total_value * 100.0 if total_value else 0.0

    by_weight = d.sort_values("Value", ascending=False)[
        ["Symbol","Quantity","Currency","Price EUR","Value","Weight_%","PL_abs","PL_pct","Day_abs","Day_pct"]
    ]
    by_pl = d.sort_values("PL_pct", ascending=True)[["Symbol","PL_pct","PL_abs"]]

    # --- Movers giornalieri ---
    by_day = d.sort_values("Day_pct", ascending=True)[["Symbol","Day_pct","Day_abs"]]

    # Totali (in EUR)
    cost_basis = (d["Avg EUR"] * d["Quantity"]).sum()
    pl_total   = d["PL_abs"].sum()
    pl_pct_vs_cost = (pl_total / cost_basis * 100.0) if cost_basis else 0.0

    # --- Totali giornalieri ---
    prev_port_value = prev_pos_val.sum() + cash_total
    day_total = d["Day_abs"].sum()
    day_pct_vs_prev = (day_total / prev_port_value * 100.0) if prev_port_value else 0.0

    totals = {
        "positions_value": d["Value"].sum(),     # EUR
        "cash_total": cash_total,                # EUR
        "portfolio_total": total_value,          # EUR
        "pl_total": pl_total,                    # EUR
        "pl_pct_vs_cost": pl_pct_vs_cost,
        "day_pl_total": day_total,               # EUR
        "day_pl_pct_vs_prev": day_pct_vs_prev,   # %
    }

    return {"by_weight": by_weight, "by_pl": by_pl, "by_day": by_day, "totals": totals, "detail": d}