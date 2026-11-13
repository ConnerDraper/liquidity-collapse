"""Toy checks of the event clock: ages, restarts, size groups, controls and alpha sums."""

import datetime as dt

import numpy as np
import polars as pl

from liquidity_collapse.event_alpha import BUCKETS, REFRACTORY, ages, alpha_sum, events

DAYS = [dt.date(2011, 1, 3) + dt.timedelta(days=i) for i in range(400)]
INDEX = {d: i for i, d in enumerate(DAYS)}


def frame(scores: dict, caps: dict) -> pl.DataFrame:
    rows = [{"date": d, "barrid": b, "s": v[i], "market_cap": caps[b]}
            for b, v in scores.items() for i, d in enumerate(DAYS) if v[i] is not None]
    return pl.DataFrame(rows)


def test_ages_restart_groups_controls():
    n = len(DAYS)
    base = {f"Z{k}": [float(k)] * n for k in range(8)}
    a = [0.5] * n
    a[5] = -9.0
    a[5 + REFRACTORY + 10] = -9.0
    b = [0.5] * n
    b[8] = -9.0
    caps = {"A": 1.0, "B": 1e9, **{f"Z{k}": 10.0 + k for k in range(8)}}
    f = frame({"A": a, "B": b, **base}, caps)
    ev = events(f, "s", 0.1)
    flagged = ev.filter("event")
    assert flagged.filter(pl.col("barrid") == "A")["date"].to_list() == [DAYS[5], DAYS[5 + REFRACTORY + 10]]
    assert flagged.filter(pl.col("barrid") == "B")["group"].to_list() == [1]
    # A is not a control on its event day or in the REFRACTORY days after it
    ctrl = ev.filter(pl.col("barrid") == "A").sort("date")["control"].to_list()
    assert ctrl[5] is False and ctrl[6] is False and ctrl[5 + REFRACTORY] is False and ctrl[5 + REFRACTORY + 1] is True
    rows = ages(f.select("date", "barrid"), ev, INDEX).filter(pl.col("barrid") == "A").sort("date")
    age = dict(zip(rows["date"], rows["age"]))
    assert age[DAYS[4]] is None and age[DAYS[5]] == 0 and age[DAYS[6]] == 1
    assert age[DAYS[5 + REFRACTORY + 10]] == 0 and age[DAYS[5 + REFRACTORY + 11]] == 1


def test_refractory_blocks_quick_reflag():
    n = len(DAYS)
    base = {f"Z{k}": [float(k)] * n for k in range(9)}
    a = [0.5] * n
    a[5] = a[10] = -9.0
    f = frame({"A": a, **base}, {"A": 1.0, **{f"Z{k}": 10.0 + k for k in range(9)}})
    assert events(f, "s", 0.1).filter("event", pl.col("barrid") == "A")["date"].to_list() == [DAYS[5]]


def test_alpha_sums():
    # rates: -3 in bucket 0 (ages 1-21), -2 in bucket 1 (22-63), -1 in bucket 2 (64-126), window N = 126
    rates = np.zeros((1, len(BUCKETS)))
    rates[0, :3] = [-3.0, -2.0, -1.0]
    n = 126
    one = lambda age, length: float(alpha_sum(np.array([age], float), rates, n, length)[0])
    assert BUCKETS[0] == (1, 21) and BUCKETS[2] == (64, 126)
    assert one(1, 1) == -3.0 and one(21, 1) == -3.0 and one(22, 1) == -2.0 and one(64, 1) == -1.0
    assert one(126, 1) == -1.0 and one(127, 1) == 0.0 and one(0, 1) == 0.0 and one(np.nan, 1) == 0.0
    assert one(1, None) == -3.0 * 21 - 2.0 * 42 - 1.0 * 63
    assert one(126, None) == -1.0
    assert one(20, 21) == -3.0 * 2 - 2.0 * 19
    assert one(120, 21) == -1.0 * 7          # runs past N
