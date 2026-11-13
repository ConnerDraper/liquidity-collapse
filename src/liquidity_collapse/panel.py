"""Monthly panel shared by every signal: universe, forward return, fund scores, benchmark."""

import datetime as dt

import polars as pl
import sf_quant.data as sfd

from .config import CALENDAR, CFG, FEATURES, FUND_SCORES, OUTPUT, PANEL, YEARS


def features(columns: list[str], years=YEARS) -> pl.LazyFrame:
    return pl.scan_parquet([FEATURES / f"features_{y}.parquet" for y in years]).select(columns)


def build() -> None:
    """Formation dates are each month's last trading day; portfolios trade the next day."""
    days = features(["date"]).unique().collect()["date"]
    boundary = sfd.load_benchmark(dt.date(2025, 1, 1), dt.date(2025, 1, 10))["date"].min()
    days = pl.concat([days, pl.Series("date", [boundary])]).unique().sort()
    calendar = pl.DataFrame({"date": days, "next_date": days.shift(-1)})
    formations = (
        calendar.drop_nulls()
        .filter(pl.col("date") >= CFG["window"]["first_formation"], pl.col("date") <= CFG["window"]["holdout_end"])
        .group_by(pl.col("date").dt.truncate("1mo").alias("month")).agg(pl.col("date").max(), pl.col("next_date").max())
        .sort("date").drop("month")
    )
    # `max` of next_date is safe: the month's last day has the latest next day.
    panel = (
        features(["date", "barrid", "eligible", "price", "market_cap", "historical_beta", "ivol63", "target21"])
        .filter(pl.col("date").is_in(formations["date"].implode()))
        .collect()
        .join(formations.rename({"next_date": "weight_date"}), on="date")
    )
    frames = []
    for year in formations["date"].dt.year().unique().sort():
        dates = formations.filter(pl.col("date").dt.year() == year)["date"]
        scores = sfd.load_scores(dates.min(), dates.max(), FUND_SCORES).filter(pl.col("date").is_in(dates.implode()))
        frames.append(scores.pivot(on="signal_name", index=["date", "barrid"], values="score"))
    panel = panel.join(pl.concat(frames, how="diagonal_relaxed"), on=["date", "barrid"], how="left")
    benchmark = sfd.load_benchmark(formations["next_date"].min(), formations["next_date"].max()).filter(
        pl.col("date").is_in(formations["next_date"].implode())
    )
    panel = panel.join(
        benchmark.rename({"date": "weight_date", "weight": "benchmark_weight"}), on=["weight_date", "barrid"], how="left"
    )
    coverage = panel.group_by("date").agg(pl.col("benchmark_weight").sum())["benchmark_weight"]
    print({"benchmark_weight_covered_min": coverage.min(), "median": coverage.median()}, flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    calendar.write_parquet(CALENDAR)
    panel.sort("date", "barrid").write_parquet(PANEL)
    print({"formations": formations.height, "rows": panel.height, "first": formations["date"].min(),
           "last": formations["date"].max()}, flush=True)
