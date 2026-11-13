"""Fixed-gamma design (run with LAB_FIXED_GAMMA=1): the three shorting rules without and with trading costs, per frequency.

Reads ANALYSIS/summary.csv, return_series.parquet, implied_costs.json, transfer_coefficient.csv and the lambda schedule
(all written by `lab tc`, return_series.py, implied_costs.py and `lab select-lambda`). Writes ANALYSIS/gamma_costs.csv.
Also a warm-up check: for the daily cost runs, turnover and net return in the first 21 trading days of each year
(after the first, which has no warm-up) against the rest of the year.
"""

import json
import math

import numpy as np
import polars as pl

from liquidity_collapse.optimize import ANALYSIS, FIXED, O, RUN_END, SCHEDULE, START, STEMS, load_run

if not FIXED:
    raise SystemExit("run with LAB_FIXED_GAMMA=1")

summary = {r["run"]: r for r in pl.read_csv(ANALYSIS / "summary.csv").iter_rows(named=True)}
tc = {r["constrained"]: r["ex_ante_tc"] for r in pl.read_csv(ANALYSIS / "transfer_coefficient.csv").iter_rows(named=True)}
series = pl.read_parquet(ANALYSIS / "return_series.parquet")
implied = json.loads((ANALYSIS / "implied_costs.json").read_text())
schedule = json.loads(SCHEDULE.read_text())
years = (RUN_END - START).days / 365.25
RULES = {"U": "No limits", "S": f"Limited shorting ({O['short_limit'] * 100:g}%)", "C": "No shorting"}

rows = []
for freq, tag, k in (("daily", "D", 252), ("monthly", "M", 12)):
    for rule, label in RULES.items():
        for costs in (False, True):
            n = f"{rule}C-{tag}" if costs else f"{rule}-{tag}"
            if n not in summary:
                continue
            s = summary[n]
            net = series.filter(pl.col("run") == n)["net"].to_numpy()
            risk = float(net.std(ddof=1) * math.sqrt(k))
            rows.append({
                "frequency": freq, "portfolio": label, "costs": "with costs" if costs else "no costs", "run": n,
                "gross_pct": s["gross_return"] * 100, "net_pct": s["net_return"] * 100,
                "ir": s["ir"], "t": s["ir"] * math.sqrt(years), "risk_realized_pct": risk * 100,
                "risk_model_pct": s["mean_active_risk_model"] * 100, "turnover_per_year": s["mean_turnover"] * k,
                "turnover_one_way_per_year": s["mean_turnover"] * k / 2,
                "spread_cost_pct": s["cost_per_year"] * 100 if costs else implied.get(n, {}).get("cost_pct_per_year"),
                "spread_cost_kind": "charged" if costs else "if traded alone",
                "tc_vs_no_limits": tc.get(n), "lambda_by_year": json.dumps(schedule.get(n)) if costs else None,
            })
table = pl.DataFrame(rows)
table.write_csv(ANALYSIS / "gamma_costs.csv")
with pl.Config(tbl_cols=30, tbl_width_chars=250, float_precision=3, tbl_rows=20):
    print(table.drop("lambda_by_year", "run"))
print(json.dumps(schedule, indent=1))

print("\nwarm-up check (daily cost runs, years after the first): first 21 trading days vs the rest")
for n in ("UC-D", "SC-D", "CC-D"):
    if n not in summary:
        continue
    _, d = load_run(STEMS[n])
    r = series.filter(pl.col("run") == n).select("date", "net")
    j = d.select("date", "turnover").join(r, on="date", how="left").with_columns(y=pl.col("date").dt.year()).sort("date")
    j = j.with_columns(day=pl.int_range(pl.len()).over("y")).filter(pl.col("y") > O["first_year"])
    first, rest = j.filter(pl.col("day") < 21), j.filter(pl.col("day") >= 21)
    print(f"  {n}: turnover/day {first['turnover'].mean():.4f} first 21 vs {rest['turnover'].mean():.4f} rest; "
          f"net bp/day {np.nanmean(first['net'].to_numpy()) * 1e4:.3f} vs {np.nanmean(rest['net'].to_numpy()) * 1e4:.3f}")
