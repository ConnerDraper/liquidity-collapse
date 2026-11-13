"""Backup test: hold the benchmark but sell flagged stocks to zero. Development years only (2011-2018).

Four portfolios: {liquidity events, IVOL events} x {daily, monthly rebalancing}.
  event      = a stock enters the extreme 5% (worst s0 / highest ivol63) after 21 trading days unflagged
  avoided    = latest event of that kind 1-63 trading days before the weight date (age measured on the signal day)
  weights    = benchmark at the previous close; flagged stocks' weight goes to the other stocks in the same
               size x pre-shock-illiquidity fifth, in proportion to their benchmark weight (whole benchmark if none)
Active return = sum of active weight x return. Writes output/optimizer/avoid/avoid_summary.csv and avoid_returns.parquet.
"""

import datetime as dt
import math
import warnings

import numpy as np
import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp

from liquidity_collapse.daily_signal import HOLDOUT, OPT, SMOOTHED
from liquidity_collapse.evaluate import market_returns
from liquidity_collapse.event_alpha import EventAlphas, ages, events
from liquidity_collapse.optimize import O, RUN_END, gross_returns, previous_day, trading_days
from liquidity_collapse.panel import features

assert not HOLDOUT, "development years only"
LO, HI = O["start"], RUN_END
assert HI <= dt.date(2018, 12, 31)
WINDOW, SHARE = 63, 0.05
OUT = OPT / "avoid"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- events and ages (signal day)
start = dt.date(2010, 1, 1)
sm = pl.read_parquet(SMOOTHED, columns=["date", "barrid", "s0"]).filter(pl.col("date").is_between(start, HI))
f = (features(["date", "barrid", "eligible", "market_cap", "ivol63", "s"], years=range(start.year, HI.year + 1))
     .filter(pl.col("date").is_between(start, HI)).collect())
data = sm.join(f.filter(pl.col("eligible")).drop("eligible", "s"), on=["date", "barrid"], how="inner")
days = sorted(data["date"].unique().to_list())
di = {d: i for i, d in enumerate(days)}
kinds = {
    "Liquidity": events(data.select("date", "barrid", "s0", "market_cap"), "s0", SHARE),
    "IVOL": events(data.select("date", "barrid", (-pl.col("ivol63")).alias("nivol"), "market_cap"), "nivol", SHARE),
}
flags = {}
for kind, ev in kinds.items():
    a = ages(data.select("date", "barrid"), ev, di)
    # weights dated D use the signal day D-1: avoided if 1 <= age(D-1) + 1 <= WINDOW
    flags[kind] = a.filter((pl.col("age") + 1).is_between(1, WINDOW)).select(pl.col("date").alias("sig"), "barrid", pl.lit(True).alias("flag"))
    n = ev.filter("event", pl.col("date").is_between(LO, HI)).height
    print(kind, "events per year", round(n / 8, 1), flush=True)
events_per_year = {k: ev.filter("event", pl.col("date").is_between(LO, HI)).height / ((HI - LO).days / 365.25) for k, ev in kinds.items()}

cells = pl.concat([g.with_columns(sig=pl.lit(d)) for d, g in EventAlphas(LO, HI).cells_by_date.items()])

# ---------------------------------------------------------------- weights
all_days = trading_days(LO - dt.timedelta(days=10), HI)
prev = previous_day(all_days)
daily_dates = [d for d in all_days if LO <= d <= HI]
monthly_dates = [d for k, d in enumerate(daily_dates) if k == 0 or d.month != daily_dates[k - 1].month]
bench = sfd.load_benchmark(LO - dt.timedelta(days=10), HI).select(pl.col("date").alias("sig"), "barrid", pl.col("weight").alias("b"))


def weights(kind: str, dates: list[dt.date]) -> tuple[pl.DataFrame, float]:
    w = (pl.DataFrame({"date": dates, "sig": [prev[d] for d in dates]})
         .join(bench, on="sig")
         .join(flags[kind], on=["sig", "barrid"], how="left").with_columns(pl.col("flag").fill_null(False))
         .join(cells, on=["sig", "barrid"], how="left").with_columns(pl.col("cell").fill_null(25)))
    w = w.with_columns(b=pl.col("b") / pl.col("b").sum().over("date"))  # benchmark weights sum to 1 each day
    w = w.with_columns(
        removed=pl.col("b").filter(pl.col("flag")).sum().over("date", "cell"),
        receivers=pl.col("b").filter(~pl.col("flag")).sum().over("date", "cell"),
    ).with_columns(pl.col("removed").fill_null(0.0), pl.col("receivers").fill_null(0.0))
    # groups with nothing left to receive: their weight goes to the whole benchmark
    w = w.with_columns(leftover=pl.when(pl.col("flag") & (pl.col("receivers") <= 0)).then(pl.col("b")).otherwise(0.0))
    w = w.with_columns(
        leftover_total=pl.col("leftover").sum().over("date"),
        kept_total=pl.col("b").filter(~pl.col("flag")).sum().over("date"),
    ).with_columns(pl.col("leftover_total").fill_null(0.0))
    w = w.with_columns(active_weight=pl.when(pl.col("flag")).then(-pl.col("b")).otherwise(
        pl.when(pl.col("receivers") > 0).then(pl.col("b") * pl.col("removed") / pl.col("receivers")).otherwise(0.0)
        + pl.col("b") * pl.col("leftover_total") / pl.col("kept_total")))
    chk = w.group_by("date").agg(pl.col("active_weight").sum().abs().alias("s"), (pl.col("b") + pl.col("active_weight")).min().alias("m"),
                                 pl.col("b").filter(pl.col("flag")).sum().alias("avoided"))
    assert chk["s"].max() < 1e-9, chk["s"].max()
    assert chk["m"].min() > -1e-12, chk["m"].min()
    return w.select("date", "barrid", "active_weight", "b", "flag"), float(chk["avoided"].fill_null(0.0).mean())


# ---------------------------------------------------------------- returns and metrics
spec = features(["date", "barrid", "s"], years=range(LO.year, HI.year + 1)).filter(pl.col("date").is_between(LO, HI)).collect()
rows, series = [], []
for kind in kinds:
    for frequency, dates in (("daily", daily_dates), ("monthly", monthly_dates)):
        name = f"{kind}, {frequency}"
        w, avoided = weights(kind, dates)
        k = 252 if frequency == "daily" else 12
        g = gross_returns(w.select("date", "barrid", "active_weight"), frequency, HI).sort("date")
        x = g["gross"].to_numpy()
        vol = x.std(ddof=1) * math.sqrt(k)
        ir = x.mean() * k / vol
        cum = np.cumsum(x)
        dd = float((cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]).min())
        by_year = g.group_by(pl.col("date").dt.year()).agg(pl.col("gross").sum())["gross"]
        m = g.join(market_returns(g["date"]), on="date", how="left").drop_nulls("market")
        beta = float(np.cov(m["gross"], m["market"], ddof=1)[0, 1] / np.var(m["market"].to_numpy(), ddof=1))
        hedged = m["gross"].to_numpy() - beta * m["market"].to_numpy()
        # specific-return version: monthly weights held flat through the month
        held = w.select("date", "barrid", "active_weight")
        if frequency == "monthly":
            cal = pl.DataFrame({"day": daily_dates}).join_asof(pl.DataFrame({"date": dates}), left_on="day", right_on="date", strategy="backward")
            held = held.join(cal, on="date").drop("date").rename({"day": "date"})
        sp = held.join(spec, on=["date", "barrid"], how="left").group_by("date").agg((pl.col("active_weight") * pl.col("s").fill_null(0.0)).sum().alias("x")).sort("date")
        sx = sp["x"].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = sfp.generate_trading_costs(w.select("date", "barrid", pl.col("active_weight").alias("weight")).sort("date", "barrid"))
        years = (HI - LO).days / 365.25
        rows.append({
            "portfolio": name, "events_per_year": events_per_year[kind], "avoided_benchmark_weight_pct": avoided * 100,
            "active_return_pct": x.mean() * k * 100, "active_risk_pct": vol * 100, "ir": ir, "t": ir * math.sqrt(years),
            "max_drawdown_pct": dd * 100, "positive_years": f"{int((by_year > 0).sum())}/{len(by_year)}",
            "market_beta": beta, "hedged_ir": float(hedged.mean() / hedged.std(ddof=1) * math.sqrt(k)),
            "specific_return_pct": sx.mean() * 252 * 100, "specific_ir": float(sx.mean() / sx.std(ddof=1) * math.sqrt(252)),
            "turnover_per_year": float(c["trade"].abs().sum() / years), "spread_cost_pct_per_year": float(c["cost"].sum() / years * 100),
        })
        series.append(g.select(pl.lit(name).alias("portfolio"), "date", "gross"))
        if frequency == "daily":  # hand check: sum(h x same-day return) on one day matches the sf-quant return
            d0 = dates[100]
            r = sfd.load_assets(d0, d0, ["date", "barrid", "return"], in_universe=False).select("barrid", (pl.col("return") / 100).alias("r"))
            manual = float(w.filter(pl.col("date") == d0).join(r, on="barrid", how="left").select((pl.col("active_weight") * pl.col("r").fill_null(0.0)).sum()).item())
            print(name, "hand check", d0, round(manual * 1e4, 4), "bp vs", round(float(g.filter(pl.col("date") == d0)["gross"].item()) * 1e4, 4), "bp", flush=True)

out = pl.DataFrame(rows)
out.write_csv(OUT / "avoid_summary.csv")
pl.concat(series).write_parquet(OUT / "avoid_returns.parquet")
with pl.Config(tbl_cols=20, tbl_width_chars=250, float_precision=3):
    print(out, flush=True)
