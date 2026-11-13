"""Daily Barra data for signals that need more history or columns than the cached features."""

import datetime as dt

import polars as pl
import sf_quant.data as sfd


def months_by_year(panel: pl.DataFrame, first_year: int, columns: list[str], aggregate) -> pl.DataFrame:
    """Load daily assets one year at a time for every barrid ever eligible, and apply
    `aggregate(daily) -> frame` to each year so the full daily history is never in memory."""
    barrids = panel.filter(pl.col("eligible"))["barrid"].unique().implode()
    last_year = panel["date"].max().year
    frames = []
    for year in range(first_year, last_year + 1):
        daily = sfd.load_assets(dt.date(year, 1, 1), dt.date(year, 12, 31), ["date", "barrid", *columns], in_universe=False)
        frames.append(aggregate(daily.filter(pl.col("barrid").is_in(barrids))))
        print(f"loaded {year}", flush=True)
    return pl.concat(frames)


def to_formations(monthly: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Attach monthly values (keyed by `month`) to that month's formation date."""
    formations = panel.select("date").unique().with_columns(month=pl.col("date").dt.truncate("1mo"))
    return monthly.join(formations, on="month").drop("month")
