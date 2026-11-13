"""Why do the no-limits runs realize a quarter of their expected return? Development years only (2011-2018).

For U-D (and C-D for comparison), each day: the weights held, each stock's next-day alpha (the one the
transfer coefficient uses), Barra predicted specific risk from the risk-model cache, and the realized Barra
specific return. Stocks are put in fifths of predicted specific risk that day. Reports, by fifth and by
whether the stock carried alpha:
  expected  = mean daily sum(alpha * h) x 252, %/yr
  realized  = mean daily sum(h * specific return) x 252, %/yr
  gross     = mean daily sum |h|
  bias      = mean of (specific return^2 / predicted daily specific variance); 1.0 = Barra's risk is right,
              above 1 = Barra underestimates the stock's risk
Writes output/optimizer/analysis/no_limits_diagnostic.json.
"""

import datetime as dt
import json

import numpy as np
import polars as pl

from liquidity_collapse.daily_signal import HOLDOUT
from liquidity_collapse.event_alpha import EventAlphas
from liquidity_collapse.optimize import ANALYSIS, O, RUN_END, STEMS, load_cache, load_run, previous_day, trading_days
from liquidity_collapse.panel import features

assert not HOLDOUT, "development years only"
lo, hi = O["start"], RUN_END
alphas = EventAlphas(lo, hi)
prev = previous_day(trading_days(lo - dt.timedelta(days=10), hi))
spec = features(["date", "barrid", "s"], years=range(lo.year, hi.year + 1)).filter(pl.col("date").is_between(lo, hi)).collect()
spec = {d: g for (d,), g in spec.group_by(["date"])}

out = {}
for name in ("U-D", "C-D"):
    w, _ = load_run(STEMS[name])
    frames = []
    for (date_,), g in w.group_by(["date"]):
        if date_ not in spec:
            continue
        c = load_cache(date_)
        ids = c["ids"]
        base = pl.DataFrame({"barrid": ids})
        h = base.join(g.select("barrid", "active_weight"), on="barrid", how="left", maintain_order="left")["active_weight"].fill_null(0.0).to_numpy()
        _, a, _ = alphas(prev[date_], date_.year, ids, "daily", False)
        s = base.join(spec[date_].select("barrid", "s"), on="barrid", how="left", maintain_order="left")["s"].to_numpy()
        vol = np.sqrt(c["D"])  # annual predicted specific risk
        q = np.ceil(pl.Series(vol).rank().to_numpy() / len(vol) * 5).clip(1, 5).astype(int)
        frames.append(pl.DataFrame({"date": [date_] * len(ids), "h": h, "a": a, "s": s, "vol": vol, "q": q, "has_alpha": a != 0}))
    d = pl.concat(frames).filter(pl.col("s").is_not_null())
    n_days = d["date"].n_unique()
    by = (d.group_by("q", "has_alpha").agg(
        expected=(pl.col("a") * pl.col("h")).sum() / n_days * 252 * 100,
        realized=(pl.col("h") * pl.col("s")).sum() / n_days * 252 * 100,
        gross=pl.col("h").abs().sum() / n_days,
        net=pl.col("h").sum() / n_days,
        bias=(pl.col("s") ** 2 / (pl.col("vol") ** 2 / 252)).mean(),
        mean_predicted_vol=pl.col("vol").mean() * 100,
    ).sort("has_alpha", "q", descending=[True, False]))
    out[name] = {"days": n_days,
                 "expected_total": float((d["a"] * d["h"]).sum() / n_days * 252 * 100),
                 "realized_total": float((d["h"] * d["s"]).sum() / n_days * 252 * 100),
                 "by_risk_fifth": by.to_dicts()}
    with pl.Config(tbl_rows=20, tbl_cols=12, float_precision=3):
        print(name, {k: v for k, v in out[name].items() if k != "by_risk_fifth"}, flush=True)
        print(by, flush=True)
(ANALYSIS / "no_limits_diagnostic.json").write_text(json.dumps(out, indent=1, default=str))
