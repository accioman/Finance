from __future__ import annotations
import pandas as pd

def find_alerts(df: pd.DataFrame, upper: float, lower: float):
    d = df.copy()
    # P/L% calcolato su valori in EUR
    d["PL_pct"] = (d["Price EUR"] - d["Purchase EUR"]) / d["Purchase EUR"] * 100.0

    if "Upper Alert" in d.columns:
        d["UpperEff"] = d["Upper Alert"].fillna(upper)
    else:
        d["UpperEff"] = upper
    if "Lower Alert" in d.columns:
        d["LowerEff"] = d["Lower Alert"].fillna(lower)
    else:
        d["LowerEff"] = lower

    upper_hits = d[d["PL_pct"] >= d["UpperEff"]]
    lower_hits = d[d["PL_pct"] <= -abs(d["LowerEff"])]
    return {"upper": upper_hits, "lower": lower_hits}
