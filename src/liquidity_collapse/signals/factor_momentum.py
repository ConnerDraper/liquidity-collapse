"""Momentum in Barra factors, mapped to stocks through their exposures.

score_i = sum_k exposure_ik * f_k, where f_k is factor k's trailing return up to the
formation close.
- industry: Barra industry factors, f_k = cumulative 6-month (126-day) log return
  (Moskowitz & Grinblatt, 1999).
- style: the 17 Barra style factors except the country factor, f_k = sign of the 12-month
  (252-day) return (time-series factor momentum, Ehsani & Linnainmaa, 2022).
"""

import datetime as dt

import numpy as np
import polars as pl
import sf_quant.data as sfd

STYLE = {
    "USSLOWL_BETA", "USSLOWL_DIVYILD", "USSLOWL_EARNQLTY", "USSLOWL_EARNYILD", "USSLOWL_GROWTH",
    "USSLOWL_LEVERAGE", "USSLOWL_LIQUIDTY", "USSLOWL_LTREVRSL", "USSLOWL_MGMTQLTY", "USSLOWL_MIDCAP",
    "USSLOWL_MOMENTUM", "USSLOWL_PROFIT", "USSLOWL_PROSPECT", "USSLOWL_RESVOL", "USSLOWL_SIZE",
    "USSLOWL_SPLTYRET", "USSLOWL_VALUE",
}
COUNTRY = "USSLOWL_COUNTRY"


def trailing(formations: pl.Series, window: int) -> tuple[list[str], pl.DataFrame]:
    """Cumulative log return of every factor over the `window` trading days ending on each formation date."""
    start = formations.min() - dt.timedelta(days=int(window * 1.6) + 10)
    factors = sfd.load_factors(start, formations.max()).sort("date")
    names = [c for c in factors.columns if c != "date"]
    logs = factors.with_columns([(pl.col(c).fill_null(0.0) / 100).log1p().cum_sum().alias(c) for c in names])
    past = logs.with_columns([pl.col(c) - pl.col(c).shift(window) for c in names])
    return names, past.filter(pl.col("date").is_in(formations.implode())).drop_nulls()


def score(panel: pl.DataFrame, kind: str) -> pl.DataFrame:
    formations = panel["date"].unique().sort()
    window = 126 if kind == "industry" else 252
    names, past = trailing(formations, window)
    chosen = [c for c in names if (c in STYLE) == (kind == "style") and c != COUNTRY]
    rows = []
    for row in past.iter_rows(named=True):
        date_ = row["date"]
        f = np.array([row[c] for c in chosen])
        if kind == "style":
            f = np.sign(f)
        exposures = sfd.load_exposures_by_date(date_)
        present = [c for c in chosen if c in exposures.columns]
        x = exposures.select([pl.col(c).fill_null(0.0) for c in present]).to_numpy()
        weights = f[[chosen.index(c) for c in present]]
        rows.append(pl.DataFrame({"date": [date_] * exposures.height, "barrid": exposures["barrid"], "score": x @ weights}))
    return pl.concat(rows).filter(pl.col("score").is_finite())


def industry(panel: pl.DataFrame) -> pl.DataFrame:
    return score(panel, "industry")


def style(panel: pl.DataFrame) -> pl.DataFrame:
    return score(panel, "style")
