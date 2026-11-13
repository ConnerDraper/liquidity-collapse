"""Daily liquidity shock with the fund's scores removed, and its walk-forward settings.

Development years only: nothing after [optimizer] end is read.

raw_t  = -(L_t - B_t), L_t = log mean(|return| / dollar volume) over the last 21 trading days
         (at least 15), B_t = mean of L at t-21, t-42, ..., t-252 (at least 9 of 12)
z_t    = residual of raw_t on the fund's six scores that day, z-scored across eligible stocks
s_t(h) = 0.5**(1/h) * s_{t-1} + (1 - 0.5**(1/h)) * z_t   (missing z counts as 0; h = 0 means s_t = z_t)
"""

import datetime as dt
import json
import os

import numpy as np
import polars as pl
import sf_quant.data as sfd

from .config import CFG, FUND_SCORES, OUTPUT, PANEL, SIGNALS
from .evaluate import nw_t
from .panel import features

O = CFG["optimizer"]
# LAB_HOLDOUT=1: the frozen design run through the holdout years, into a separate folder.
HOLDOUT = os.environ.get("LAB_HOLDOUT") == "1"
END = CFG["window"]["holdout_end"] if HOLDOUT else O["end"]
OPT = OUTPUT / ("optimizer_holdout" if HOLDOUT else "optimizer")
DAILY = OPT / "daily_signal.parquet"
SMOOTHED = OPT / "smoothed.parquet"
PARAMS = OPT / "signal_params.json"


def guard(date_: dt.date) -> None:
    if date_ > END:
        raise RuntimeError(f"refusing to read data after {END}: {date_}")


def raw_shock(barrids: pl.Series) -> pl.DataFrame:
    frames = []
    for year in range(2004, END.year + 1):
        lo, hi = dt.date(year, 1, 1), min(dt.date(year, 12, 31), END)
        guard(hi)
        d = sfd.load_assets(lo, hi, ["date", "barrid", "return", "price", "daily_volume"], in_universe=False)
        frames.append(
            d.filter(pl.col("barrid").is_in(barrids.implode())).select(
                "date", "barrid",
                pl.when((pl.col("daily_volume") > 0) & (pl.col("price") > 0) & pl.col("return").is_not_null())
                .then((pl.col("return") / 100).abs() / (pl.col("price") * pl.col("daily_volume"))).alias("x"),
            )
        )
        print(f"raw: loaded {year}", flush=True)
    d = pl.concat(frames).sort("barrid", "date").with_columns(
        L=pl.col("x").rolling_mean(21, min_samples=15).over("barrid").log()
    ).with_columns(pl.when(pl.col("L").is_finite()).then(pl.col("L")).alias("L"))
    shifted = [pl.col("L").shift(21 * k).over("barrid") for k in range(1, 13)]
    d = d.with_columns(
        n=pl.sum_horizontal([s.is_not_null().cast(pl.Int32) for s in shifted]),
        total=pl.sum_horizontal(shifted),
    )
    return d.select("date", "barrid", pl.when(pl.col("n") >= 9).then(-(pl.col("L") - pl.col("total") / pl.col("n"))).alias("raw")).drop_nulls()


def build() -> None:
    """Writes DAILY (date, barrid, raw, z) for eligible stock-days 2005-END."""
    OPT.mkdir(parents=True, exist_ok=True)
    eligible = (
        features(["date", "barrid", "eligible"], years=range(2005, END.year + 1))
        .filter(pl.col("eligible"), pl.col("date") <= END).select("date", "barrid").collect()
    )
    data = raw_shock(eligible["barrid"].unique()).join(eligible, on=["date", "barrid"])
    scores = []
    for year in range(2005, END.year + 1):
        lo, hi = dt.date(year, 1, 1), min(dt.date(year, 12, 31), END)
        s = sfd.load_scores(lo, hi, FUND_SCORES)
        scores.append(s.pivot(on="signal_name", index=["date", "barrid"], values="score"))
        print(f"scores: loaded {year}", flush=True)
    data = data.join(pl.concat(scores, how="diagonal_relaxed"), on=["date", "barrid"], how="inner").drop_nulls(
        ["raw", *FUND_SCORES]
    ).filter(pl.all_horizontal([pl.col(c).is_finite() for c in ["raw", *FUND_SCORES]]))

    def one(day: pl.DataFrame) -> pl.DataFrame:
        x = np.column_stack([np.ones(day.height), day.select(FUND_SCORES).to_numpy()])
        y = day["raw"].to_numpy()
        r = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        return day.select("date", "barrid", "raw").with_columns(z=pl.Series((r - r.mean()) / r.std()))

    out = data.group_by("date", maintain_order=True).map_groups(one).sort("date", "barrid")
    out.write_parquet(DAILY)
    print({"rows": out.height, "days": out["date"].n_unique(), "first": out["date"].min(), "last": out["date"].max()}, flush=True)


def smooth() -> pl.DataFrame:
    """EWMA of z for every half-life, on a full day x stock grid so absent days decay toward 0."""
    z = pl.read_parquet(DAILY, columns=["date", "barrid", "z"]).filter(pl.col("date") >= O["history_start"])
    days = z["date"].unique().sort()
    names = z["barrid"].unique().sort()
    di = pl.DataFrame({"date": days, "i": np.arange(len(days))})
    bi = pl.DataFrame({"barrid": names, "j": np.arange(len(names))})
    idx = z.join(di, on="date").join(bi, on="barrid")
    grid = np.zeros((len(days), len(names)))
    grid[idx["i"].to_numpy(), idx["j"].to_numpy()] = idx["z"].to_numpy()
    out = idx.select("date", "barrid", "i", "j")
    for h in O["half_lives"]:
        lam = 0.0 if h == 0 else 0.5 ** (1 / h)  # h = 0: no smoothing
        s, smoothed = np.zeros(len(names)), np.empty_like(grid)
        for t in range(len(days)):
            s = lam * s + (1 - lam) * grid[t]
            smoothed[t] = s
        out = out.with_columns(pl.Series(f"s{h}", smoothed[idx["i"].to_numpy(), idx["j"].to_numpy()]))
    out = out.drop("i", "j").sort("date", "barrid")
    out.write_parquet(SMOOTHED)
    return out


def bottom_gaps(frame: pl.DataFrame, column: str, share: float) -> pl.DataFrame:
    """Per month: mean target21 of the bottom `share` by `column` minus the mean of all."""
    d = frame.drop_nulls([column, "target21"]).with_columns(
        flagged=pl.col(column).rank("ordinal").over("date") <= (pl.len().over("date") * share).ceil()
    )
    return d.group_by("date").agg(
        (pl.col("target21").filter(pl.col("flagged")).mean() - pl.col("target21").mean()).alias("gap")
    ).drop_nulls().sort("date")


def params() -> None:
    """Walk-forward stage 1: each January pick (h, q) by the t-statistic of the bottom-q gap
    over month-ends from history_start to the prior November (their targets end before January),
    and set alpha a = that combination's mean gap."""
    smoothed = smooth()
    month_ends = (
        pl.read_parquet(PANEL, columns=["date", "barrid", "eligible", "target21"])
        .filter(pl.col("eligible"), pl.col("date").is_between(O["history_start"], END))
    )
    data = month_ends.join(smoothed, on=["date", "barrid"], how="inner")
    gaps = {(h, q): bottom_gaps(data, f"s{h}", q) for h in O["half_lives"] for q in O["flag_shares"]}
    chosen, table = {}, []
    for year in range(O["first_year"], END.year + 1):
        cutoff = dt.date(year - 1, 12, 1)
        best = None
        for (h, q), g in gaps.items():
            x = g.filter(pl.col("date") < cutoff)["gap"].to_numpy()
            t, mean = nw_t(x), float(x.mean())
            table.append({"year": year, "half_life": h, "flag_share": q, "months": len(x), "gap_pct_month": mean * 100, "t": t})
            if best is None or t < best["t"]:
                best = {"half_life": h, "flag_share": q, "alpha_monthly": mean, "t": t, "months": len(x)}
        chosen[str(year)] = best
    PARAMS.write_text(json.dumps(chosen, indent=1))
    pl.DataFrame(table).write_csv(OPT / "signal_params_grid.csv")
    # Month-end samples of the daily signal, for the robustness checks.
    sample = pl.read_parquet(DAILY).join(month_ends.select("date").unique(), on="date")
    sample.select("date", "barrid", pl.col("z").alias("score")).write_parquet(SIGNALS / "liquidity_daily.parquet")
    sample.select("date", "barrid", pl.col("raw").alias("score")).write_parquet(SIGNALS / "liquidity_daily_raw.parquet")
    print(json.dumps(chosen, indent=1), flush=True)
