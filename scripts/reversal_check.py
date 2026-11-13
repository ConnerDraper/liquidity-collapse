"""Is the liquidity event just short-term reversal? Development years only.

Monthly Fama-MacBeth at month-ends 2008-2018 of next-21-day specific return (target21) on:
  live        = a liquidity event 1-21 trading days ago (same events as the backtests: bottom 5% of s0,
                21 days unflagged); also 1-63 days as a check
  live_rose / live_fell = live split by the sign of the 5-day specific return up to the event
  controls    = ranks of size, pre-shock illiquidity, momentum (months 2-12), long-term reversal,
                and short-term reversal (specific return over the last 21 days)
Writes output/results/reversal_check.json.
"""

import datetime as dt
import json

import numpy as np
import polars as pl

from liquidity_collapse.config import RESULTS
from liquidity_collapse.evaluate import nw_t
from liquidity_collapse.event_alpha import COLUMN, FLAG_SHARE, events, panel

data = panel().sort("barrid", "date").with_columns(c=pl.col("s").fill_null(0.0).log1p().cum_sum().over("barrid")).with_columns(
    strev=pl.col("c") - pl.col("c").shift(21).over("barrid"),
    pre5=pl.col("c") - pl.col("c").shift(5).over("barrid"),
)
days = sorted(data["date"].unique().to_list())
di = pl.DataFrame({"date": days, "t": list(range(len(days)))})
ev = events(data.select("date", "barrid", COLUMN, "market_cap"), COLUMN, FLAG_SHARE)
e = (ev.filter("event").join(data.select("date", "barrid", "pre5"), on=["date", "barrid"]).join(di, on="date")
     .select("barrid", pl.col("date").alias("edate"), pl.col("t").alias("et"), (pl.col("pre5") > 0).alias("rose")).sort("edate"))
month_ends = pl.DataFrame({"date": days}).group_by(pl.col("date").dt.truncate("1mo").alias("month")).agg(pl.col("date").max())["date"]
m = (data.filter(pl.col("date").is_in(month_ends.implode()), pl.col("date") >= dt.date(2008, 1, 1)).join(di, on="date").sort("date")
     .join_asof(e, left_on="date", right_on="edate", by="barrid", strategy="backward")
     .with_columns(age=pl.col("t") - pl.col("et"))
     .with_columns(
         live=pl.col("age").is_between(1, 21).fill_null(False).cast(pl.Float64),
         live63=pl.col("age").is_between(1, 63).fill_null(False).cast(pl.Float64),
     ).with_columns(
         live_rose=pl.col("live") * pl.col("rose").fill_null(False).cast(pl.Float64),
         live_fell=pl.col("live") * (~pl.col("rose").fill_null(True)).cast(pl.Float64),
     )
     .drop_nulls(["target21", "market_cap", "level", "mom", "ltr", "strev"]))
m = m.with_columns([((pl.col(c).rank() / pl.len()).over("date") - 0.5).alias(f"r_{c}") for c in ["market_cap", "level", "mom", "ltr", "strev"]])
BASE = ["r_market_cap", "r_level", "r_mom", "r_ltr"]


def fm(cols):
    coefs = []
    for (_,), g in m.sort("date").group_by(["date"], maintain_order=True):
        x = np.column_stack([np.ones(g.height)] + [g[c].to_numpy() for c in cols])
        coefs.append(np.linalg.lstsq(x, g["target21"].to_numpy(), rcond=None)[0][1:])
    coefs = np.array(coefs)
    return {c: {"coef_pct_month": round(float(coefs[:, i].mean() * 100), 3), "t": round(nw_t(coefs[:, i]), 2)}
            for i, c in enumerate(cols) if not c.startswith("r_") or c == "r_strev"}


out = {
    "months": m["date"].n_unique(), "live_share": float(m["live"].mean()),
    "share_of_live_that_rose": float(m["live_rose"].sum() / m["live"].sum()),
    "corr_live_strev_rank": float(m.select(pl.corr("live", "r_strev")).item()),
    "note": "coefficients in %/month of next-21-day specific return; controls: ranks of size, pre-shock illiquidity, momentum, long-term reversal",
    "fama_macbeth": {
        "live + controls": fm(["live", *BASE]),
        "live + controls + short-term reversal": fm(["live", *BASE, "r_strev"]),
        "live 1-63 + controls + short-term reversal": fm(["live63", *BASE, "r_strev"]),
        "live split by pre-event move + controls": fm(["live_rose", "live_fell", *BASE]),
        "live split by pre-event move + controls + short-term reversal": fm(["live_rose", "live_fell", *BASE, "r_strev"]),
    },
}
(RESULTS / "reversal_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1), flush=True)
