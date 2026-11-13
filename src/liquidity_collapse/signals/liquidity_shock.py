"""Liquidity shock (Bali, Peng, Shen & Tang, 2014): stocks whose liquidity just improved keep outperforming.

Monthly Amihud illiquidity = mean over the month of |return| / dollar volume (at least 15 days).
Score = -(log illiquidity this month - mean log illiquidity over the previous 12 months, at least 9).
Logs keep the change comparable across large and small stocks.
"""

import polars as pl

from ._daily import months_by_year, to_formations


def monthly_illiquidity(daily: pl.DataFrame) -> pl.DataFrame:
    return (
        daily.filter(pl.col("daily_volume") > 0, pl.col("price") > 0, pl.col("return").is_not_null())
        .with_columns(illiq=(pl.col("return") / 100).abs() / (pl.col("price") * pl.col("daily_volume")))
        .group_by("barrid", pl.col("date").dt.truncate("1mo").alias("month"))
        .agg(pl.col("illiq").mean(), pl.len().alias("days"))
        .filter(pl.col("days") >= 15, pl.col("illiq") > 0)
        .select("barrid", "month", pl.col("illiq").log().alias("log_illiq"))
    )


def build(panel: pl.DataFrame) -> pl.DataFrame:
    first = panel["date"].min().year - 1
    monthly = months_by_year(panel, first, ["return", "price", "daily_volume"], monthly_illiquidity).sort("barrid", "month")
    shocked = monthly.with_columns(
        base=pl.col("log_illiq").rolling_mean(12, min_samples=9).shift(1).over("barrid")
    ).with_columns(score=-(pl.col("log_illiq") - pl.col("base"))).drop_nulls("score")
    return to_formations(shocked.select("barrid", "month", "score"), panel).select("date", "barrid", "score")


def build_residual(panel: pl.DataFrame) -> pl.DataFrame:
    """The part of the liquidity shock the fund's six scores don't explain.

    Each formation date, regress the score on the fund scores (with a constant) across eligible
    names that have all of them, and keep the residual. Uses only same-day information.
    """
    import numpy as np

    from ..config import FUND_SCORES

    data = (
        panel.filter(pl.col("eligible")).select("date", "barrid", *FUND_SCORES)
        .join(build(panel), on=["date", "barrid"])
        .drop_nulls(["score", *FUND_SCORES])
        .filter(pl.all_horizontal([pl.col(c).is_finite() for c in ["score", *FUND_SCORES]]))
    )

    def one(day: pl.DataFrame) -> pl.DataFrame:
        x = np.column_stack([np.ones(day.height), day.select(FUND_SCORES).to_numpy()])
        y = day["score"].to_numpy()
        residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        return day.select("date", "barrid").with_columns(score=pl.Series(residual))

    return data.group_by("date", maintain_order=True).map_groups(one)
