from __future__ import annotations
import pandas as pd

def compute_stats_tables(df: pd.DataFrame, cash_total: float):
    d = df.copy()

    # Tutto in EUR
    d["Value"]   = pd.to_numeric(d["Price EUR"], errors="coerce") * pd.to_numeric(d["Quantity"], errors="coerce")
    d["Avg EUR"] = pd.to_numeric(d["Purchase EUR"], errors="coerce")
    d["PL_abs"]  = (d["Price EUR"] - d["Avg EUR"]) * d["Quantity"]
    d["PL_pct"]  = (d["Price EUR"] - d["Avg EUR"]) / d["Avg EUR"] * 100.0

    total_value = d["Value"].sum() + cash_total
    d["Weight_%"] = d["Value"] / total_value * 100.0 if total_value else 0.0

    by_weight = d.sort_values("Value", ascending=False)[
        ["Symbol","Quantity","Currency","Price EUR","Value","Weight_%","PL_abs","PL_pct"]
    ]
    by_pl = d.sort_values("PL_pct", ascending=True)[["Symbol","PL_pct","PL_abs"]]

    # Totali (in EUR)
    cost_basis = (d["Avg EUR"] * d["Quantity"]).sum()
    pl_total   = d["PL_abs"].sum()
    pl_pct_vs_cost = (pl_total / cost_basis * 100.0) if cost_basis else 0.0

    totals = {
        "positions_value": d["Value"].sum(),     # EUR
        "cash_total": cash_total,                # EUR (dalla cassa CSV)
        "portfolio_total": total_value,          # EUR
        "pl_total": pl_total,                    # EUR
        "pl_pct_vs_cost": pl_pct_vs_cost,
    }

    return {"by_weight": by_weight, "by_pl": by_pl, "totals": totals, "detail": d}
