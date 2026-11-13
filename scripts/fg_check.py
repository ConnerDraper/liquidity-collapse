"""Checks before the fixed-gamma runs.

`python scripts/fg_check.py equivalence` (no LAB_FIXED_GAMMA): re-solve 10 saved days of the existing 5% runs U-D,
C-D and limited shorting 0.25% through sf-quant's cost_aware_mve_optimizer (optimize.solve), with each day's logged
gamma, and compare with the saved weights. Exits 1 if any day is off by more than 1% of its gross weight.

`LAB_FIXED_GAMMA=1 LAB_SMOKE=1 python scripts/fg_check.py smoke`: a no-cost and a cost run on January-March 2011, and a
cost run for January 2012 with its warm-up, into the smoke folders.
"""

import datetime as dt
import sys

import numpy as np
import polars as pl

from liquidity_collapse.config import connect
from liquidity_collapse import optimize as o

connect()


def equivalence() -> None:
    assert not o.FIXED, "run without LAB_FIXED_GAMMA"
    worst = 0.0
    for name, stem, floor in (("U-D", "U-D/year_*", None), ("C-D", "C-D/year_*", None),
                              ("S-D", o.short_stem("S-D", 0.0025), 0.0025)):
        w, d = o.load_run(stem)
        ok = d.filter(pl.col("status") == "ok")
        dates = ok["date"][:: max(1, ok.height // 10)][:10].to_list()
        constraints = [o.ZeroInvestment(), o.ZeroBeta()] + ([o.LongOnly()] if name == "C-D" else [])
        for date_ in dates:
            days = o.trading_days(date_ - dt.timedelta(days=10), date_)
            prev = days[-2]
            lo = max(dt.date(date_.year, 1, 1), o.START)
            alphas = o.EventAlphas(lo, min(dt.date(date_.year, 12, 31), o.RUN_END))
            c, p = o.load_cache(date_), o.load_cache(prev)
            ids = c["ids"]
            b = pl.DataFrame({"barrid": ids}).join(pl.DataFrame({"barrid": p["ids"], "b": p["b"]}), on="barrid", how="left",
                                                   maintain_order="left")["b"].fill_null(0.0).to_numpy()
            alpha = alphas(prev, date_.year, ids, "daily", False)[0]
            gamma = float(d.filter(pl.col("date") == date_)["gamma"].item())
            h = o.solve(alpha, c["B"].astype(np.float64), c["F"], c["D"], constraints, np.zeros(len(ids)), c["c"], 0.0, b,
                        c["beta"], gamma, alphas.cells(prev, ids), floor)
            saved = pl.DataFrame({"barrid": ids}).join(w.filter(pl.col("date") == date_), on="barrid", how="left",
                                                      maintain_order="left")["active_weight"].fill_null(0.0).to_numpy()
            if h is None:
                print(f"{name} {date_}: solve failed", flush=True)
                worst = max(worst, 1.0)
                continue
            rel = float(np.abs(h - saved).sum() / max(np.abs(saved).sum(), 1e-12))
            worst = max(worst, rel)
            print(f"{name} {date_}: max |diff| {np.abs(h - saved).max():.2e}, sum |diff| / gross {rel:.2e}", flush=True)
    print(f"worst relative difference {worst:.2e}", flush=True)
    if worst > 0.01:
        raise SystemExit(1)


def smoke() -> None:
    assert o.FIXED and o.SMOKE, "run with LAB_FIXED_GAMMA=1 LAB_SMOKE=1"
    tasks = o.variant_tasks()
    for want in (("U-D", 0.0, 2011), ("SC-D", 1.0, 2011)):
        o.variant(tasks.index(want))
    lo = dt.date(2012, 1, 3)
    warm = o.trading_days(lo - dt.timedelta(days=o.O["warm_up_days"] * 2), lo - dt.timedelta(days=1))[-o.O["warm_up_days"]]
    o.run("SC-D", 1.0, warm, dt.date(2012, 1, 31), "SC-D/warmup_test", floor=o.O["short_limit"], record_from=lo)
    for stem in ("U-D/year_2011", "SC-D/lambda_1/year_2011", "SC-D/warmup_test"):
        d = pl.read_parquet(o.RUNS_DIR / f"{stem}_days.parquet")
        print(stem, {"days": d.height, "first": str(d["date"].min()), "failed": int((d["status"] == "failed").sum()),
                     "max_risk": round(float(d["active_risk"].max()), 5), "min_total_weight": round(float(d["min_total_weight"].min()), 5),
                     "turnover_per_day": round(float(d["turnover"].mean()), 4), "gamma": sorted(set(d["gamma"].round(4).to_list()))[:3]},
              flush=True)


{"equivalence": equivalence, "smoke": smoke}[sys.argv[1]]()
