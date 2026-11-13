"""The same report for every signal: coverage, IC, residual IC, deciles, two costed portfolios.

Every stage runs on every signal; nothing here stops a weak signal.
"""

import datetime as dt
import json
import math

import numpy as np
import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp

from .config import CFG, FUND_SCORES, PANEL, RESULTS, SIGNALS, windows

NW_LAGS = 3


def nw_t(x: np.ndarray, lags: int = NW_LAGS) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return math.nan
    e = x - x.mean()
    variance = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        variance += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / math.sqrt(variance / n))


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), x])
    return y - design @ np.linalg.lstsq(design, y, rcond=None)[0]


def ic_series(frame: pl.DataFrame, score: str = "score") -> pl.DataFrame:
    return (
        frame.drop_nulls([score, "target21"]).group_by("date")
        .agg(pl.corr(score, "target21", method="spearman").alias("ic"), pl.len().alias("names"))
        .drop_nulls().sort("date")
    )


def residual_ic(frame: pl.DataFrame) -> pl.DataFrame:
    usable = frame.drop_nulls(["score", "target21", *FUND_SCORES]).filter(
        pl.all_horizontal([pl.col(c).is_finite() for c in FUND_SCORES])
    )

    def one(day: pl.DataFrame) -> pl.DataFrame:
        if day.height <= len(FUND_SCORES) + 10:
            return day.head(0).select("date", "target21").with_columns(residual=pl.lit(None, pl.Float64))
        r = residualize(day["score"].to_numpy(), day.select(FUND_SCORES).to_numpy())
        return day.select("date", "target21").with_columns(residual=pl.Series(r))

    return ic_series(usable.group_by("date", maintain_order=True).map_groups(one), "residual")


def deciles(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    ranked = frame.drop_nulls(["score", "target21"]).with_columns(
        decile=((pl.col("score").rank("ordinal").over("date") - 1) * 10 // pl.len().over("date") + 1).cast(pl.Int8)
    )
    by_date = ranked.group_by("date", "decile").agg(pl.col("target21").mean())
    table = by_date.group_by("decile").agg(pl.col("target21").mean().alias("mean_target21")).sort("decile")
    spread = (
        by_date.pivot(on="decile", index="date", values="target21").sort("date")
        .select("date", (pl.col("10") - pl.col("1")).alias("spread")).drop_nulls()
    )
    return table, spread


def long_short_weights(frame: pl.DataFrame) -> pl.DataFrame:
    tail = CFG["portfolio"]["tail"]
    ranked = frame.filter(pl.col("benchmark_weight").is_not_null()).drop_nulls("score").with_columns(
        pct=pl.col("score").rank("average").over("weight_date") / pl.len().over("weight_date")
    )
    legs = ranked.with_columns(
        side=pl.when(pl.col("pct") > 1 - tail).then(1.0).when(pl.col("pct") <= tail).then(-1.0).otherwise(0.0)
    ).filter(pl.col("side") != 0)
    return legs.select(
        pl.col("weight_date").alias("date"), "barrid",
        (pl.col("side") * 0.5 / pl.len().over("weight_date", "side")).alias("weight"),
    )


def long_only_weights(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Benchmark tilted by the signal: weight = benchmark * (1 + tilt * clip(z)); unscored names keep z = 0."""
    p = CFG["portfolio"]
    bench = frame.filter(pl.col("benchmark_weight").is_not_null())
    z = (pl.col("score") - pl.col("score").mean().over("weight_date")) / pl.col("score").std().over("weight_date")
    tilted = bench.with_columns(z=z.clip(-p["z_clip"], p["z_clip"]).fill_null(0.0).fill_nan(0.0)).with_columns(
        raw=pl.col("benchmark_weight") * (1 + p["tilt"] * pl.col("z"))
    ).with_columns(weight=pl.col("raw") / pl.col("raw").sum().over("weight_date"))
    portfolio = tilted.select(pl.col("weight_date").alias("date"), "barrid", "weight")
    benchmark = bench.select(
        pl.col("weight_date").alias("date"), "barrid",
        (pl.col("benchmark_weight") / pl.col("benchmark_weight").sum().over("weight_date")).alias("weight"),
    )
    return portfolio, benchmark


def period_returns(weights: pl.DataFrame, costs: bool = True) -> pl.DataFrame:
    weights = weights.sort("date", "barrid")
    gross = sfp.generate_returns_from_weights(weights, costs=False, delistings=True).rename({"return": "gross"})
    if not costs:
        return gross
    trades = sfp.generate_trading_costs(weights)
    cost = trades.group_by("date").agg(pl.col("cost").sum(), pl.col("trade").abs().sum().alias("turnover"))
    return gross.join(cost, on="date", how="left").with_columns(pl.col("cost", "turnover").fill_null(0.0))


def market_returns(dates: pl.Series) -> pl.DataFrame:
    dates = dates.unique().sort()
    daily = sfd.load_benchmark_returns(dates.min(), dates.max()).select(
        "date", (pl.col("bmk_return") / 100).alias("market")
    ).sort("date")
    periods = pl.DataFrame({"date": dates}).with_columns(pl.col("date").alias("start"))
    return (
        daily.join_asof(periods, on="date", strategy="backward").group_by("start")
        .agg((pl.col("market") + 1).product().sub(1)).rename({"start": "date"})
    )


def stats(r: np.ndarray, prefix: str) -> dict:
    mean, sd = float(r.mean()), float(r.std(ddof=1))
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return {
        f"{prefix}_return": mean * 12, f"{prefix}_vol": sd * math.sqrt(12),
        f"{prefix}_ir": mean / sd * math.sqrt(12) if sd else math.nan,
        f"{prefix}_t": mean / sd * math.sqrt(len(r)) if sd else math.nan,
        f"{prefix}_max_drawdown": float((wealth / peak - 1).min()),
    }


def beta(frame: pl.DataFrame) -> float:
    both = frame.drop_nulls(["net", "market"])
    net, market = both["net"].to_numpy(), both["market"].to_numpy()
    return float(np.cov(net, market, ddof=1)[0, 1] / np.var(market, ddof=1))


def portfolio_frame(scored: pl.DataFrame, start: dt.date, end: dt.date) -> pl.DataFrame:
    """Formations whose weights trade inside [start, end], plus the one before (opening portfolio)
    and the one after (closes the last period). Opening and closing periods are dropped later."""
    dates = scored.select("date", "weight_date").unique().sort("weight_date")
    inside = dates.filter(pl.col("weight_date").is_between(start, end))
    before = dates.filter(pl.col("weight_date") < start).tail(1)
    after = dates.filter(pl.col("weight_date") > end).head(1)
    keep = pl.concat([before, inside, after])["date"]
    return scored.filter(pl.col("date").is_in(keep.implode()))


def run(name: str, sample: str) -> dict:
    win = windows(sample)
    panel = pl.read_parquet(PANEL)
    scores = pl.read_parquet(SIGNALS / f"{name}.parquet").unique(["date", "barrid"])
    data = panel.join(scores, on=["date", "barrid"], how="left")
    out = RESULTS / name / sample
    out.mkdir(parents=True, exist_ok=True)

    ic_lo, ic_hi = win["ic"]
    universe = data.filter(pl.col("eligible"), pl.col("date").is_between(ic_lo, ic_hi))
    coverage = universe.group_by("date").agg(
        pl.col("score").is_not_null().mean().alias("coverage"), pl.col("score").is_not_null().sum().alias("scored")
    ).sort("date")
    scored = universe.drop_nulls("score")
    ic = ic_series(scored)
    ric = residual_ic(scored)
    ic_year = ic.group_by(pl.col("date").dt.year().alias("year")).agg(
        pl.col("ic").mean(), pl.len().alias("months")
    ).sort("year")
    decile_table, spread = deciles(scored)

    p_lo, p_hi = win["portfolio"]
    frame = portfolio_frame(data.filter(pl.col("eligible") | pl.col("benchmark_weight").is_not_null()), p_lo, p_hi)
    keep = lambda f: f.filter(pl.col("date").is_between(p_lo, p_hi)).sort("date")
    ls = period_returns(long_short_weights(frame))
    ls = keep(ls.join(market_returns(ls["date"]), on="date", how="left").with_columns(net=pl.col("gross") - pl.col("cost")))
    tilt, bench = long_only_weights(frame)
    lo = period_returns(tilt).join(period_returns(bench, costs=False).rename({"gross": "benchmark"}), on="date")
    lo = keep(lo.with_columns(active_net=pl.col("gross") - pl.col("cost") - pl.col("benchmark")))

    summary = {
        "signal": name, "sample": sample,
        "ic_months": ic.height, "first_month": str(ic["date"].min()), "last_month": str(ic["date"].max()),
        "mean_coverage": float(coverage["coverage"].mean()), "mean_scored": float(coverage["scored"].mean()),
        "mean_ic": float(ic["ic"].mean()), "ic_t": nw_t(ic["ic"].to_numpy()),
        "positive_ic_years": int((ic_year["ic"] > 0).sum()), "ic_years": ic_year.height,
        "residual_ic": float(ric["ic"].mean()) if ric.height else math.nan,
        "residual_ic_t": nw_t(ric["ic"].to_numpy()) if ric.height else math.nan,
        "decile_spread_monthly": float(spread["spread"].mean()), "decile_spread_t": nw_t(spread["spread"].to_numpy()),
        "decile_monotonicity": float(np.corrcoef(decile_table["decile"], decile_table["mean_target21"])[0, 1]),
        "periods": ls.height,
        **stats(ls["gross"].to_numpy(), "ls_gross"), **stats(ls["net"].to_numpy(), "ls_net"),
        "ls_cost": float(ls["cost"].mean()) * 12, "ls_turnover": float(ls["turnover"].mean()),
        "ls_beta": beta(ls),
        **stats(lo["active_net"].to_numpy(), "lo_active_net"),
        "lo_cost": float(lo["cost"].mean()) * 12, "lo_turnover": float(lo["turnover"].mean()),
    }
    coverage.write_csv(out / "coverage.csv")
    ic.write_csv(out / "ic.csv")
    ric.write_csv(out / "residual_ic.csv")
    ic_year.write_csv(out / "ic_by_year.csv")
    decile_table.write_csv(out / "deciles.csv")
    ls.write_csv(out / "long_short.csv")
    lo.write_csv(out / "long_only.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str), flush=True)
    return summary


def collect() -> pl.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*/*/summary.json"))]
    table = pl.DataFrame(rows)
    table.write_csv(RESULTS.parent / "summary.csv")
    return table


def compare(base: str, candidate: str, sample: str) -> dict:
    """Month-by-month difference between two evaluated signals' portfolios (candidate minus base)."""
    rows = {}
    for kind, column in (("long_only", "active_net"), ("long_short", "net")):
        a = pl.read_csv(RESULTS / base / sample / f"{kind}.csv").select("date", pl.col(column).alias("base"))
        b = pl.read_csv(RESULTS / candidate / sample / f"{kind}.csv").select("date", pl.col(column).alias("candidate"))
        both = a.join(b, on="date").sort("date")
        d = (both["candidate"] - both["base"]).to_numpy()
        rows[kind] = {
            "months": len(d),
            "base_ir": float(both["base"].mean() / both["base"].std() * math.sqrt(12)),
            "candidate_ir": float(both["candidate"].mean() / both["candidate"].std() * math.sqrt(12)),
            "base_return": float(both["base"].mean() * 12), "candidate_return": float(both["candidate"].mean() * 12),
            "difference_return": float(d.mean() * 12), "difference_t": float(d.mean() / d.std(ddof=1) * math.sqrt(len(d))),
            "correlation": float(np.corrcoef(both["base"], both["candidate"])[0, 1]),
        }
    out = RESULTS / f"compare_{candidate}_vs_{base}_{sample}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows, indent=1), flush=True)
    return rows
