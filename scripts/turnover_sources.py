"""Where does the daily turnover come from? (2011-2018, daily runs, no costs.)

For each day D the trade in each stock is |h_D - h_{D-1}| (target active weights, before drift). Each stock-day is put
in one group by what its alpha did between the two days:
  new event       alpha 0 -> nonzero
  event ended     alpha nonzero -> 0 (age passed the window)
  alpha changed   nonzero on both days but different (age bucket switch at day 22, January re-estimate)
  alpha same      nonzero and unchanged on both days (trades from rescaling / re-hedging only)
  no alpha        zero on both days (hedges in similar stocks, beta hedge)
Separately, the part of each day's trade that is a uniform rescale of yesterday's portfolio: c_D = <h_D, h_{D-1}> /
<h_{D-1}, h_{D-1}> - 1, scale turnover = |c_D| * sum|h_{D-1}|; the rest is reported by group as "after removing the rescale".
Also reported: logged turnover (includes drift and the January restart of each one-year task) and the restart share.
Writes ANALYSIS/turnover_sources.json (output/optimizer_fixed_gamma/analysis with LAB_FIXED_GAMMA=1).
"""

import datetime as dt
import json

import numpy as np
import polars as pl

from liquidity_collapse.config import connect
from liquidity_collapse.optimize import (ANALYSIS, END, FIXED, START, EventAlphas, load_run, previous_day, short_stem,
                                         trading_days)

connect()
RUNS = {"U-D": "U-D/year_*", "C-D": "C-D/year_*", "S-D 0.10%": short_stem("S-D", 0.001), "S-D 0.25%": short_stem("S-D", 0.0025)}
if FIXED:  # fixed-gamma design (LAB_FIXED_GAMMA=1): its own no-cost daily runs
    RUNS = {"U-D": "U-D/year_*", "S-D": "S-D/year_*", "C-D": "C-D/year_*"}
YEARS = (END - START).days / 365.25

alphas = EventAlphas(START, END)
days = trading_days(START - dt.timedelta(days=10), END)
prev = previous_day(days)
w0, _ = load_run(RUNS["U-D"])
frames = []
for (d,), g in w0.group_by(["date"]):
    ids = g["barrid"].to_numpy()
    a = alphas(prev[d], d.year, ids, "daily", False)[0]
    frames.append(pl.DataFrame({"date": [d] * len(ids), "barrid": ids, "alpha": a}))
alpha = pl.concat(frames)
order = {d: i for i, d in enumerate(sorted(alpha["date"].unique().to_list()))}
idx = pl.DataFrame({"date": list(order), "di": list(order.values())})


def sources(stem: str) -> dict:
    w, log = load_run(stem)
    w = w.join(idx, on="date").join(alpha, on=["date", "barrid"], how="left").with_columns(pl.col("alpha").fill_null(0.0))
    before = w.select("barrid", (pl.col("di") + 1).alias("di"), pl.col("active_weight").alias("h0"), pl.col("alpha").alias("a0"))
    j = (w.join(before, on=["barrid", "di"], how="full", coalesce=True)
         .filter(pl.col("di").is_in(list(range(1, len(order)))))
         .with_columns(pl.col("active_weight", "h0", "alpha", "a0").fill_null(0.0)))
    scale = j.group_by("di").agg(((pl.col("active_weight") * pl.col("h0")).sum() / (pl.col("h0") ** 2).sum() - 1).alias("c"),
                                 pl.col("h0").abs().sum().alias("gross0"))
    j = j.join(scale, on="di").with_columns(
        trade=(pl.col("active_weight") - pl.col("h0")).abs(),
        resid=(pl.col("active_weight") - pl.col("h0") * (1 + pl.col("c").fill_nan(0.0))).abs(),
        group=pl.when((pl.col("a0") == 0) & (pl.col("alpha") != 0)).then(pl.lit("new event"))
        .when((pl.col("a0") != 0) & (pl.col("alpha") == 0)).then(pl.lit("event ended"))
        .when((pl.col("a0") != 0) & ((pl.col("alpha") - pl.col("a0")).abs() > 1e-12)).then(pl.lit("alpha changed"))
        .when(pl.col("a0") != 0).then(pl.lit("alpha same"))
        .otherwise(pl.lit("no alpha")))
    total = float(j["trade"].sum())
    by = j.group_by("group").agg(pl.col("trade").sum(), pl.col("resid").sum()).sort("trade", descending=True)
    sc = scale.with_columns(st=pl.col("c").fill_nan(0.0).abs() * pl.col("gross0"))
    g = log.select("date", "gamma").join(idx, on="date").sort("di").with_columns(
        g_ratio=(pl.col("gamma").shift(1) / pl.col("gamma") - 1)).join(scale, on="di")
    ok = g.filter(pl.col("g_ratio").is_finite(), pl.col("c").is_finite(), pl.col("gamma") > 0)
    first = log.with_columns(y=pl.col("date").dt.year()).group_by("y").agg(pl.col("turnover").first())
    return {
        "logged_turnover_per_year": float(log["turnover"].sum() / YEARS),
        "trade_turnover_per_year": total / YEARS,
        "january_restart_share_of_logged": float(first["turnover"].sum() / log["turnover"].sum()),
        "by_group_per_year": {r["group"]: r["trade"] / YEARS for r in by.iter_rows(named=True)},
        "by_group_share": {r["group"]: r["trade"] / total for r in by.iter_rows(named=True)},
        "rescale_per_year": float(sc["st"].sum() / YEARS),
        "rescale_share": float(sc["st"].sum() / total),
        "after_rescale_by_group_per_year": {r["group"]: r["resid"] / YEARS for r in by.iter_rows(named=True)},
        "corr_rescale_vs_gamma_change": float(np.corrcoef(ok["c"].to_numpy(), ok["g_ratio"].to_numpy())[0, 1]) if ok.height > 2 else None,
        "median_abs_daily_gamma_change": float(ok["g_ratio"].abs().median()) if ok.height else None,
    }


out = {name: sources(stem) for name, stem in RUNS.items()}
ANALYSIS.mkdir(parents=True, exist_ok=True)
(ANALYSIS / "turnover_sources.json").write_text(json.dumps(out, indent=1))
for name, r in out.items():
    print(f"\n== {name}: logged {r['logged_turnover_per_year']:.1f}x/yr, day-to-day trades {r['trade_turnover_per_year']:.1f}x/yr, "
          f"January restart {r['january_restart_share_of_logged']:.1%} of logged")
    for grp, v in r["by_group_per_year"].items():
        print(f"  {grp:14s} {v:7.1f}x/yr  {r['by_group_share'][grp]:6.1%}   after removing rescale {r['after_rescale_by_group_per_year'][grp]:7.1f}x/yr")
    print(f"  uniform rescale {r['rescale_per_year']:.1f}x/yr ({r['rescale_share']:.1%} of trades); corr with gamma change "
          f"{r['corr_rescale_vs_gamma_change']}; median |daily gamma change| {r['median_abs_daily_gamma_change']}")
