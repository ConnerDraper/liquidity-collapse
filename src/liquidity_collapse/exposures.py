"""Portfolio-side checks for every backtest: active Barra exposures, factor vs specific return, market beta.

Active exposure x_t = B_t' h_t from the cached risk model. Factor part of a period's active
return = sum_k x_k * (factor k's compounded return over the period); the rest is specific.
Development years only.
"""

import datetime as dt
import json

import numpy as np
import polars as pl
import sf_quant.data as sfd
from sf_quant.data._factors import factors as FACTORS

from .optimize import RUN_END as END
from .optimize import ANALYSIS, O, RUNS, STEMS, load_cache, load_run, performance
from .signals.factor_momentum import COUNTRY, STYLE


def active_exposures(weights: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (date_,), w in weights.group_by(["date"], maintain_order=True):
        c = load_cache(date_)
        h = pl.DataFrame({"barrid": c["ids"]}).join(w, on="barrid", how="left", maintain_order="left")["active_weight"].fill_null(0.0).to_numpy()
        x = c["B"].astype(np.float64).T @ h
        rows.append({"date": date_, **dict(zip(FACTORS, x.tolist()))})
    return pl.DataFrame(rows).sort("date")


def period_compound(daily: pl.DataFrame, starts: pl.Series, cols: list[str]) -> pl.DataFrame:
    periods = pl.DataFrame({"date": starts.sort()}).with_columns(pl.col("date").alias("start"))
    return (
        daily.sort("date").join_asof(periods, on="date", strategy="backward").drop_nulls("start")
        .group_by("start").agg([(pl.col(c) + 1).product().sub(1).alias(c) for c in cols]).rename({"start": "date"})
    )


def run_checks(name: str) -> dict:
    frequency, _, costs = RUNS[name]
    weights, _ = load_run(STEMS[name])
    perf = performance(name, STEMS[name])
    x = active_exposures(weights)
    lo = perf["date"].min()
    f = sfd.load_factors(lo, END).select("date", *[(pl.col(k) / 100).alias(k) for k in FACTORS if k != "date"])
    m = sfd.load_benchmark_returns(lo, END).select("date", (pl.col("bmk_return") / 100).alias("market"))
    fp = period_compound(f, perf["date"], FACTORS)
    mp = period_compound(m, perf["date"], ["market"])
    j = perf.join(x, on="date").join(fp, on="date", suffix="_ret").join(mp, on="date")
    factor_part = sum(j[k] * j[f"{k}_ret"] for k in FACTORS if k != COUNTRY)
    j = j.with_columns(factor=factor_part, specific=pl.col("gross") - factor_part)
    k = 252 if frequency == "daily" else 12
    beta = float(np.cov(j["gross"], j["market"], ddof=1)[0, 1] / np.var(j["market"].to_numpy(), ddof=1))
    styles = sorted(STYLE)
    industries = [c for c in FACTORS if c not in STYLE and c != COUNTRY]
    mean_x = x.select([pl.col(c).mean() for c in FACTORS]).row(0, named=True)
    top_ind = sorted(industries, key=lambda c: -abs(mean_x[c]))[:5]
    return {
        "run": name, "periods": j.height, "realized_market_beta": beta,
        "factor_return_per_year": float(j["factor"].mean() * k), "specific_return_per_year": float(j["specific"].mean() * k),
        "factor_share_of_active_variance": float(j["factor"].var() / j["gross"].var()),
        "mean_style_exposure": {s.removeprefix("USSLOWL_"): round(mean_x[s], 4) for s in styles},
        "largest_industry_exposures": {s.removeprefix("USSLOWL_"): round(mean_x[s], 4) for s in top_ind},
        "mean_abs_style_exposure": float(np.mean([abs(mean_x[s]) for s in styles])),
    }


def run() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    out = [run_checks(n) for n in STEMS]
    (ANALYSIS / "exposures.json").write_text(json.dumps(out, indent=1, default=str))
    for r in out:
        print(json.dumps(r, default=str), flush=True)
