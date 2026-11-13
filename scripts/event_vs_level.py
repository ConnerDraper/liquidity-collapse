"""Is it the shock, or just being illiquid / small? Development years only (2006-2018).

1. Matched event study: each new flag (bottom 5% of s5, not flagged in the prior 21 days) vs unflagged
   stocks on the same day in the same pre-shock illiquidity quintile x size quintile. Pre-shock
   illiquidity = 63-day Amihud measured 63 trading days earlier. Returns = daily specific return minus
   the eligible average; missing days count as 0.
2. Monthly Fama-MacBeth: next-21-day specific return on cross-sectional ranks of the shock (fund scores
   removed), pre-shock illiquidity and size.
Writes output/results/event_vs_level.json.
"""

import datetime as dt
import json

import numpy as np
import polars as pl

from liquidity_collapse.config import OUTPUT, PANEL, SIGNALS
from liquidity_collapse.daily_signal import END, SMOOTHED
from liquidity_collapse.evaluate import nw_t
from liquidity_collapse.panel import features

Q, COL = 0.05, "s5"
KS = [21, 63, 126, 189, 252]
out = {}

f = (
    features(["date", "barrid", "s", "eligible", "market_cap", "amihud63"], years=range(2005, END.year + 1))
    .filter(pl.col("date") <= END).collect().sort("barrid", "date")
    .with_columns(level=pl.col("amihud63").shift(63).over("barrid"))
    .with_columns(x=pl.col("s") - pl.col("s").filter(pl.col("eligible")).mean().over("date"))
)
days = f["date"].unique().sort()
di = {d: i for i, d in enumerate(days.to_list())}
names = f["barrid"].unique().sort()
bj = {b: j for j, b in enumerate(names.to_list())}
X = np.zeros((len(days), len(names)))
X[f["date"].replace_strict(di).to_numpy(), f["barrid"].replace_strict(bj).to_numpy()] = np.nan_to_num(f["x"].to_numpy())
C = np.vstack([np.zeros(len(names)), np.cumsum(X, axis=0)])  # C[t+1] = sum of X[0..t]

# ---------------------------------------------------------------- 1. matched event study
sm = pl.read_parquet(SMOOTHED, columns=["date", "barrid", COL]).filter(pl.col("date") <= END)
d = sm.with_columns(flag=(pl.col(COL).rank("ordinal").over("date") <= (pl.len().over("date") * Q).ceil()).cast(pl.Int8)).sort("barrid", "date")
d = d.with_columns(recent=pl.col("flag").shift(1).rolling_max(21, min_samples=1).over("barrid").fill_null(0))
d = d.join(f.filter(pl.col("eligible")).select("date", "barrid", "market_cap", "level"), on=["date", "barrid"]).drop_nulls(["level", "market_cap"])
d = d.filter(pl.col("date") >= dt.date(2006, 3, 1)).with_columns(
    lq=((pl.col("level").rank() / pl.len()).over("date") * 5).ceil().clip(1, 5).cast(pl.Int32),
    sq=((pl.col("market_cap").rank() / pl.len()).over("date") * 5).ceil().clip(1, 5).cast(pl.Int32),
).with_columns(cell=(pl.col("lq") - 1) * 5 + pl.col("sq") - 1,
               event=(pl.col("flag") == 1) & (pl.col("recent") == 0),
               control=(pl.col("flag") == 0) & (pl.col("recent") == 0))
out["events_by_level_quintile"] = d.filter("event").group_by("lq").len().sort("lq").to_dicts()
out["events_by_size_quintile"] = d.filter("event").group_by("sq").len().sort("sq").to_dicts()

rows = []
for (date_,), g in d.group_by(["date"]):
    t = di[date_]
    if t + max(KS) >= len(days):
        continue
    ev, ct = g.filter("event"), g.filter("control")
    if ev.height == 0:
        continue
    jc = ct["barrid"].replace_strict(bj).to_numpy()
    cc = ct["cell"].to_numpy()
    je = ev["barrid"].replace_strict(bj).to_numpy()
    ce = ev["cell"].to_numpy()
    rec = {"date": date_, "lq": ev["lq"].to_numpy(), "sq": ev["sq"].to_numpy()}
    for k in KS:
        # return over days t+1 .. t+k = C[t+k+1] - C[t+1]
        pc = C[t + k + 1, jc] - C[t + 1, jc]
        cell_mean = np.bincount(cc, pc, minlength=25) / np.maximum(np.bincount(cc, minlength=25), 1)
        pe = C[t + k + 1, je] - C[t + 1, je]
        rec[f"raw{k}"] = pe
        rec[f"diff{k}"] = pe - cell_mean[ce]
    rows.append(rec)


def summarize(select=lambda r: np.ones(len(r["lq"]), bool)):
    res = {}
    for k in KS:
        per_date = [(r["date"], r[f"raw{k}"][m].mean(), r[f"diff{k}"][m].mean()) for r in rows if (m := select(r)).any()]
        per_date.sort()
        raw = np.array([p[1] for p in per_date]); diff = np.array([p[2] for p in per_date])
        res[k] = {"flagged_pct": round(raw.mean() * 100, 2), "vs_matched_pct": round(diff.mean() * 100, 2),
                  "t": round(nw_t(diff, lags=k), 2)}
    return res


out["matched"] = {"all": summarize()}
for q in (1, 3, 5):
    out["matched"][f"illiquidity_quintile_{q}"] = summarize(lambda r, q=q: r["lq"] == q)
    out["matched"][f"size_quintile_{q}"] = summarize(lambda r, q=q: r["sq"] == q)
for name, res in out["matched"].items():
    print(name, json.dumps(res), flush=True)

# ---------------------------------------------------------------- 2. monthly Fama-MacBeth
p = pl.read_parquet(PANEL, columns=["date", "barrid", "eligible", "target21", "market_cap"]).filter(
    pl.col("eligible"), pl.col("date").is_between(dt.date(2006, 1, 1), END), pl.col("target21").is_not_null())
m = (p.join(pl.read_parquet(SIGNALS / "liquidity_shock_resid.parquet").select("date", "barrid", "score"), on=["date", "barrid"])
     .join(f.select("date", "barrid", "level"), on=["date", "barrid"]).drop_nulls(["score", "level", "market_cap"]))
m = m.with_columns([((pl.col(c).rank() / pl.len()).over("date") - 0.5).alias(f"r_{c}") for c in ["score", "level", "market_cap"]])
m = m.with_columns(flag=((pl.col("score").rank() / pl.len()).over("date") <= 0.1).cast(pl.Float64))
out["rank_corr"] = {c: round(float(m.select(pl.corr("score", c, method="spearman")).item()), 3) for c in ["level", "market_cap"]}
print("rank corr of shock with", out["rank_corr"], flush=True)


def fm(cols):
    coefs = []
    for (_,), g in m.sort("date").group_by(["date"], maintain_order=True):
        Xm = np.column_stack([np.ones(g.height)] + [g[c].to_numpy() for c in cols])
        coefs.append(np.linalg.lstsq(Xm, g["target21"].to_numpy(), rcond=None)[0][1:])
    coefs = np.array(coefs)
    return {c: {"coef_pct": round(coefs[:, i].mean() * 100, 3), "t": round(nw_t(coefs[:, i]), 2)} for i, c in enumerate(cols)}


out["fama_macbeth"] = {
    "shock alone": fm(["r_score"]),
    "shock + illiquidity + size": fm(["r_score", "r_level", "r_market_cap"]),
    "bottom-decile flag alone": fm(["flag"]),
    "bottom-decile flag + illiquidity + size": fm(["flag", "r_level", "r_market_cap"]),
    "illiquidity + size only": fm(["r_level", "r_market_cap"]),
}
for k, v in out["fama_macbeth"].items():
    print(k, v, flush=True)
(OUTPUT / "results" / "event_vs_level.json").write_text(json.dumps(out, indent=1, default=str))
