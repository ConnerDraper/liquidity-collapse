"""Half-spread cost each no-cost run would pay if traded alone (a diagnostic, not charged). Development years only.
Writes output/optimizer/analysis/implied_costs.json: cost %/yr and one-way turnover per year."""

import json
import warnings

import polars as pl
import sf_quant.performance as sfp

from liquidity_collapse.optimize import ANALYSIS, RUN_END, START, STEMS, available, load_run

years = (RUN_END - START).days / 365.25
out = {}
for n in [n for n in ("U-D", "S-D", "C-D", "U-M", "S-M", "C-M") if n in available()]:
    w, _ = load_run(STEMS[n])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = sfp.generate_trading_costs(w.select("date", "barrid", pl.col("active_weight").alias("weight")).sort("date", "barrid"))
    out[n] = {"cost_pct_per_year": float(c["cost"].sum() / years * 100), "turnover_per_year": float(c["trade"].abs().sum() / years)}
    print(n, out[n], flush=True)
(ANALYSIS / "implied_costs.json").write_text(json.dumps(out, indent=1))
