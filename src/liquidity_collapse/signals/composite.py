"""Equal-weight composites, to test whether a new signal improves what the fund already trades.

Each input is z-scored across eligible names on each formation date (missing -> 0), then averaged.
- fund_composite: the fund's six scores.
- fund_plus_liquidity: the same six plus the liquidity shock with fund scores removed, each 1/7.
"""

import polars as pl

from ..config import FUND_SCORES
from . import liquidity_shock


def zscore(column: str) -> pl.Expr:
    c = pl.when(pl.col(column).is_finite()).then(pl.col(column))
    return ((c - c.mean().over("date")) / c.std().over("date")).fill_null(0.0).fill_nan(0.0)


def composite(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    scored = frame.filter(pl.any_horizontal([pl.col(c).is_not_null() for c in columns]))
    return scored.select("date", "barrid", pl.mean_horizontal([zscore(c) for c in columns]).alias("score"))


def fund(panel: pl.DataFrame) -> pl.DataFrame:
    return composite(panel.filter(pl.col("eligible")), FUND_SCORES)


def fund_plus_liquidity(panel: pl.DataFrame) -> pl.DataFrame:
    liquidity = liquidity_shock.build_residual(panel).rename({"score": "liquidity"})
    frame = panel.filter(pl.col("eligible")).join(liquidity, on=["date", "barrid"], how="left")
    return composite(frame, [*FUND_SCORES, "liquidity"])
