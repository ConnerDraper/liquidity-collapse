"""Daily (or monthly) return series of each backtest, for the cumulative-return charts. Development years only.
Writes output/optimizer/analysis/return_series.parquet: run, date, gross, cost, net."""

import polars as pl

from liquidity_collapse.optimize import ANALYSIS, available, performance

frames = [performance(n, s).select(pl.lit(n).alias("run"), "date", "gross", pl.col("cost").fill_null(0.0), "net") for n, s in available().items()]
out = pl.concat(frames)
out.write_parquet(ANALYSIS / "return_series.parquet")
print(out.group_by("run").agg(pl.len(), pl.col("net").sum()))
