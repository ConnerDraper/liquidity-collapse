"""Every backtest's return split by where its active weight sits. Development years only.

For each run: return and IR (as in analysis/summary.csv), mean model active risk, gross exposure, failed
solves, and total and specific return split into stocks with a recent event (ages 1-21), an older
event (22-504) and the rest. Writes output/optimizer/analysis/run_decomposition.json.
"""

import datetime as dt
import json

import polars as pl
import sf_quant.data as sfd

from liquidity_collapse.daily_signal import END
from liquidity_collapse.event_alpha import EventAlphas
from liquidity_collapse.optimize import ANALYSIS, O, RUNS, ir, load_run, performance, previous_day, trading_days
from liquidity_collapse.panel import features

from liquidity_collapse.optimize import available
lo, hi = O["start"], END
alphas = EventAlphas(lo, hi)
all_days = trading_days(lo - dt.timedelta(days=10), hi)
prev = previous_day(all_days)
known = pl.concat([g.select("barrid", "age").with_columns(sig=pl.lit(day)) for day, g in alphas.by_date.items()])
known = known.join(pl.DataFrame({"sig": list(prev.values()), "date": list(prev.keys())}), on="sig").select(
    "date", "barrid", (pl.col("age") + 1).alias("age"))
ret = sfd.load_assets(lo, hi, ["date", "barrid", "return"], in_universe=False).drop_nulls("return").select(
    "date", "barrid", (pl.col("return") / 100).alias("r"))
spec = features(["date", "barrid", "s"], years=range(lo.year, hi.year + 1)).filter(pl.col("date").is_between(lo, hi)).collect()

out = {}
for name, stem in available().items():
    frequency = RUNS[name][0]
    p = performance(name, stem)
    w, d = load_run(stem)
    w = w.filter(pl.col("active_weight").abs() > 0)
    gross = float(w.group_by("date").agg(pl.col("active_weight").abs().sum())["active_weight"].mean())
    if frequency == "monthly":
        reb = sorted(w["date"].unique().to_list())
        cal = pl.DataFrame({"day": [x for x in all_days if lo <= x <= hi]}).join_asof(
            pl.DataFrame({"date": reb}), left_on="day", right_on="date", strategy="backward")
        w = w.join(cal, on="date").drop("date").rename({"day": "date"})
    j = (w.join(spec, on=["date", "barrid"], how="left").join(ret, on=["date", "barrid"], how="left").join(known, on=["date", "barrid"], how="left")
         .with_columns(bin=pl.when(pl.col("age").is_between(1, 21)).then(pl.lit("ages 1-21"))
                       .when(pl.col("age").is_between(22, 504)).then(pl.lit("ages 22-504")).otherwise(pl.lit("no live event"))))
    n_days = j["date"].n_unique()
    split = {r["bin"]: {"total_pct_per_year": round(r["y"] / n_days * 252 * 100, 3),
                        "specific_pct_per_year": round(r["x"] / n_days * 252 * 100, 3), "mean_net_weight": round(r["n"] / n_days, 4)}
             for r in j.group_by("bin").agg((pl.col("active_weight") * pl.col("s").fill_null(0)).sum().alias("x"),
                                            (pl.col("active_weight") * pl.col("r").fill_null(0)).sum().alias("y"),
                                            pl.col("active_weight").sum().alias("n")).iter_rows(named=True)}
    per_year = 252 if frequency == "daily" else 12
    x = p["net"].to_numpy()
    out[name] = {"net_return_pct_per_year": round(float(x.mean() * per_year * 100), 3), "net_ir": round(ir(x, frequency), 2),
                 "mean_model_active_risk": round(float(d["active_risk"].mean()), 4), "mean_gross_exposure": round(gross, 2),
                 "failed": int((d["status"] == "failed").sum()), "specific_by_age": split}
ANALYSIS.mkdir(parents=True, exist_ok=True)
(ANALYSIS / "run_decomposition.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1), flush=True)
