"""Event-clock alpha for the liquidity shock, and its walk-forward settings (stage 1).

Event: a stock enters the bottom FLAG_SHARE of the unsmoothed score (s0 in smoothed.parquet) on day e,
having had no flag in the previous REFRACTORY trading days. A new event restarts the clock. Age on
day t = t - e in trading days, so the first day a portfolio can hold it has age 1.

Matched drift: each event is compared with controls chosen on the event day: scored stocks that are
not flagged and were not flagged in the previous REFRACTORY days, in the same size fifth and
pre-shock illiquidity fifth (63-day Amihud measured 63 trading days earlier). For each age bucket,
drift = the event's mean daily specific return over those ages minus its controls' mean.

Walk-forward, each January, from events whose bucket ended before that year:
  window N  = extend bucket by bucket while the pooled matched drift has t <= STOP_T (0: still negative);
              stop at the first bucket that fails (N = 0 means no alpha that year)
  alpha     = each live bucket's mean daily matched drift per unit of the event stock's volatility (63-day
              idiosyncratic vol on the event day), separately for events in the largest size fifth ("large")
              and the rest, capped at zero (the hypothesis is underperformance only). A stock's alpha is that
              rate times its own event-day volatility, so volatile stocks get proportionally larger alpha.
Before any backtest, the monthly regression of next-month specific return on "live" (with size,
illiquidity, momentum and long-term reversal controls) must agree in sign with the alpha.
Development years only.
"""

import datetime as dt
import json

import numpy as np
import polars as pl

from .config import CFG
from .daily_signal import END, HOLDOUT, OPT, SMOOTHED, guard
from .evaluate import nw_t
from .panel import features

E = CFG["event"]
REFRACTORY = E["refractory"]
FLAG_SHARE = E["flag_share"]
STOP_T = E["stop_t"]
EDGES = E["bucket_edges"]  # bucket b covers ages EDGES[b-1]+1 .. EDGES[b]
BUCKETS = [(lo + 1, hi) for lo, hi in zip([0, *EDGES[:-1]], EDGES)]
COLUMN = "s0"  # unsmoothed score
EVENT_PARAMS = OPT / "event_params.json"
GROUPS = ("rest", "large")


# ---------------------------------------------------------------- alpha by age
def cumulative(k: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """S(k) = sum of alpha(1..k) for piecewise-constant daily rates per bucket.
    k: (n,), rates: (n, len(BUCKETS)); rates are 0 for buckets past the window."""
    k = np.asarray(k, dtype=float)
    out = np.zeros(len(k))
    for b, (lo, hi) in enumerate(BUCKETS):
        days = np.clip(np.minimum(k, hi) - lo + 1, 0, hi - lo + 1)
        out += rates[:, b] * days
    return out


def alpha_sum(age: np.ndarray, rates: np.ndarray, n: int, length) -> np.ndarray:
    """Sum of alpha over ages age .. age + length - 1; length=None means through N (remaining drift).
    Stocks without a live event (age <= 0, NaN or > N) get 0."""
    age = np.nan_to_num(np.asarray(age, dtype=float), nan=0.0)
    live = (age >= 1) & (age <= n)
    end = np.minimum(np.full_like(age, n) if length is None else age + length - 1, n)
    return np.where(live, cumulative(end, rates) - cumulative(age - 1, rates), 0.0)


# ---------------------------------------------------------------- events and ages
def events(scores: pl.DataFrame, column: str, share: float) -> pl.DataFrame:
    """scores: date, barrid, <column>, market_cap (one row per scored stock-day).
    Returns date, barrid, event, control (unflagged now and in the prior REFRACTORY days) and
    group (0 rest, 1 largest size fifth) on every row."""
    return (
        scores.drop_nulls(column)
        .with_columns(
            flag=(pl.col(column).rank("ordinal").over("date") <= (pl.len().over("date") * share).ceil()).cast(pl.Int8),
            group=(pl.col("market_cap").rank().over("date") > 0.8 * pl.len().over("date")).fill_null(False).cast(pl.Int8),
        )
        .sort("barrid", "date")
        .with_columns(recent=pl.col("flag").shift(1).rolling_max(REFRACTORY, min_samples=1).over("barrid").fill_null(0))
        .select("date", "barrid",
                ((pl.col("flag") == 1) & (pl.col("recent") == 0)).alias("event"),
                ((pl.col("flag") == 0) & (pl.col("recent") == 0)).alias("control"), "group",
                *(["vol"] if "vol" in scores.columns else []))
    )


def ages(rows: pl.DataFrame, ev: pl.DataFrame, day_index: dict) -> pl.DataFrame:
    """rows: date, barrid (+ anything). Adds age = trading days since the stock's latest event on or
    before that date (0 = event that day, null = none yet) and event_group (size group at that event).
    A return earned on day t belongs to age(t) >= 1; weights for day D use age(signal day) + 1."""
    idx = pl.DataFrame({"date": list(day_index), "t": list(day_index.values())})
    e = ev.filter("event").join(idx, on="date").select("barrid", "t", pl.col("t").alias("e"), pl.col("group").alias("event_group"),
                                                        *([pl.col("vol").alias("event_vol")] if "vol" in ev.columns else []))
    r = pl.concat([
        rows.join(idx, on="date").with_columns(pl.lit(0, pl.Int8).alias("_src")),
        e.with_columns(pl.lit(1, pl.Int8).alias("_src")),
    ], how="diagonal_relaxed").sort("barrid", "t", "_src", descending=[False, False, True])  # an event sorts before its day's row
    r = r.with_columns(pl.col("e").forward_fill().over("barrid"), pl.col("event_group").forward_fill().over("barrid"),
                       *([pl.col("event_vol").forward_fill().over("barrid")] if "event_vol" in e.columns else []))
    return r.filter(pl.col("_src") == 0).with_columns(age=pl.col("t") - pl.col("e")).drop("e", "_src", "t")


# ---------------------------------------------------------------- stage 1
def panel() -> pl.DataFrame:
    """Scored, eligible stock-days 2006-END with specific return, size, pre-shock illiquidity,
    past returns, next-month target and the unsmoothed score."""
    guard(END)
    f = (
        features(["date", "barrid", "s", "eligible", "market_cap", "amihud63", "target21", "ivol63"], years=range(2005, END.year + 1))
        .filter(pl.col("date") <= END).collect().sort("barrid", "date")
        .with_columns(ls=pl.col("s").fill_null(0.0).log1p())
        .with_columns(c=pl.col("ls").cum_sum().over("barrid"))
        .with_columns(
            level=pl.col("amihud63").shift(63).over("barrid"),
            mom=pl.col("c").shift(21).over("barrid") - pl.col("c").shift(252).over("barrid"),
            ltr=pl.col("c").shift(252).over("barrid") - pl.col("c").shift(756).over("barrid"),
        )
        .filter(pl.col("eligible"), pl.col("date") >= dt.date(2006, 1, 1))
        .select("date", "barrid", "s", "market_cap", "level", "mom", "ltr", "target21", pl.col("ivol63").alias("vol"))
    )
    sm = pl.read_parquet(SMOOTHED, columns=["date", "barrid", COLUMN]).filter(pl.col("date") <= END)
    return f.join(sm, on=["date", "barrid"], how="inner").with_columns(
        sq=((pl.col("market_cap").rank() / pl.len()).over("date") * 5).ceil().clip(1, 5).cast(pl.Int32),
        lq=((pl.col("level").rank() / pl.len()).over("date") * 5).ceil().clip(1, 5).cast(pl.Int32),
    )


def matched_drift(data: pl.DataFrame, share: float) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Per event date, size group and bucket: mean matched daily drift of that day's events.
    Returns (drift rows, events frame, day index)."""
    days = sorted(data["date"].unique().to_list())
    di = {d: i for i, d in enumerate(days)}
    names = sorted(data["barrid"].unique().to_list())
    bj = {b: j for j, b in enumerate(names)}
    X = np.zeros((len(days), len(names)))
    X[data["date"].replace_strict(di).to_numpy(), data["barrid"].replace_strict(bj).to_numpy()] = data["s"].fill_null(0.0).to_numpy()
    C = np.vstack([np.zeros(len(names)), np.cumsum(X, axis=0)])  # C[t+1] = sum of X[0..t]
    ev = events(data.select("date", "barrid", COLUMN, "market_cap", "vol"), COLUMN, share)
    d = ev.join(data.select("date", "barrid", "sq", "lq"), on=["date", "barrid"]).drop_nulls(["sq", "lq"]).with_columns(
        cell=(pl.col("sq") - 1) * 5 + pl.col("lq") - 1)
    rows = []
    for (date_,), g in d.filter(pl.col("event") | pl.col("control")).group_by(["date"]):
        t = di[date_]
        ev_rows, ct = g.filter("event"), g.filter("control")
        if ev_rows.height == 0 or ct.height == 0:
            continue
        jc, cc = ct["barrid"].replace_strict(bj).to_numpy(), ct["cell"].to_numpy()
        je, ce, ge = ev_rows["barrid"].replace_strict(bj).to_numpy(), ev_rows["cell"].to_numpy(), ev_rows["group"].to_numpy()
        ve = ev_rows["vol"].fill_null(np.nan).to_numpy()
        counts = np.bincount(cc, minlength=25)
        ok = (counts[ce] > 0) & np.isfinite(ve) & (ve > 0)
        ve = np.where(ok, ve, 1.0)
        for b, (lo, hi) in enumerate(BUCKETS):
            if t + hi >= len(days):
                break
            span = hi - lo + 1
            pc = (C[t + hi + 1, jc] - C[t + lo, jc]) / span
            cell_mean = np.bincount(cc, pc, minlength=25) / np.maximum(counts, 1)
            diff = ((C[t + hi + 1, je] - C[t + lo, je]) / span - cell_mean[ce]) / ve  # per unit of event-day vol
            for g_ in (0, 1):
                m = ok & (ge == g_)
                if m.any():
                    rows.append({"date": date_, "end": days[t + hi], "group": g_, "bucket": b, "n": int(m.sum()), "drift": float(diff[m].mean())})
    return pl.DataFrame(rows), ev, di


def fit_year(drift: pl.DataFrame, before: dt.date) -> dict:
    """Window and alpha from events whose bucket ended before `before`."""
    known = drift.filter(pl.col("end") < before)
    stats, n_window = [], 0
    rates = {g: [0.0] * len(BUCKETS) for g in GROUPS}
    for b, (lo, hi) in enumerate(BUCKETS):
        x = known.filter(pl.col("bucket") == b)
        pooled = x.group_by("date").agg(((pl.col("drift") * pl.col("n")).sum() / pl.col("n").sum()).alias("d")).sort("date")
        t = nw_t(pooled["d"].to_numpy(), lags=min(hi, 252)) if pooled.height > 20 else float("nan")
        by_group = {}
        for g in (0, 1):
            xg = x.filter(pl.col("group") == g)
            v = float((xg["drift"] * xg["n"]).sum() / xg["n"].sum()) if xg.height else 0.0
            by_group[GROUPS[g]] = v if np.isfinite(v) else 0.0
        stats.append({"bucket": [lo, hi], "event_dates": pooled.height, "pooled_bp_per_day": float(pooled["d"].mean() * 1e4) if pooled.height else None,
                      "t": t, **{f"{k}_bp_per_day": v * 1e4 for k, v in by_group.items()}})
        if n_window == lo - 1 and np.isfinite(t) and t <= STOP_T:
            n_window = hi
            for k, v in by_group.items():
                rates[k][b] = min(v, 0.0)  # raw estimates stay in stats
    return {"half_life": 0, "flag_share": FLAG_SHARE, "window": n_window, "rates": rates, "buckets": stats}


def sign_check(data: pl.DataFrame, ev: pl.DataFrame, di: dict, chosen: dict) -> dict:
    """Monthly Fama-MacBeth at month-ends 2008-END: next-month specific return on live (that year's
    window), size, illiquidity, momentum and long-term reversal, all as cross-sectional ranks."""
    a = ages(data.select("date", "barrid", "market_cap", "level", "mom", "ltr", "target21"), ev, di)
    month_ends = pl.DataFrame({"date": list(di)}).group_by(pl.col("date").dt.truncate("1mo").alias("month")).agg(pl.col("date").max())["date"]
    windows = pl.DataFrame({"year": [int(y) for y in chosen], "n": [p["window"] for p in chosen.values()]})
    m = (a.filter(pl.col("date").is_in(month_ends.implode()), pl.col("date") >= dt.date(2008, 1, 1))
         .with_columns(year=pl.col("date").dt.year()).join(windows, on="year", how="left").with_columns(pl.col("n").fill_null(0))
         .with_columns(live=(pl.col("age").is_between(1, pl.col("n"))).fill_null(False).cast(pl.Float64))
         .drop_nulls(["target21", "market_cap", "level", "mom", "ltr"]))
    m = m.with_columns([((pl.col(c).rank() / pl.len()).over("date") - 0.5).alias(f"r_{c}") for c in ["market_cap", "level", "mom", "ltr"]])
    cols = ["live", "r_market_cap", "r_level", "r_mom", "r_ltr"]
    coefs = []
    for (_,), g in m.sort("date").group_by(["date"], maintain_order=True):
        if g["live"].sum() == 0:
            continue
        x = np.column_stack([np.ones(g.height)] + [g[c].to_numpy() for c in cols])
        coefs.append(np.linalg.lstsq(x, g["target21"].to_numpy(), rcond=None)[0][1])
    coefs = np.array(coefs)
    return {"months": len(coefs), "live_share": float(m["live"].mean()),
            "live_coef_pct_month": float(coefs.mean() * 100), "t": nw_t(coefs)}


def params() -> None:
    """Walk-forward stage 1; writes EVENT_PARAMS, event_drift.parquet and event_sensitivity.json."""
    data = panel()
    print(f"panel: {data.height} scored stock-days", flush=True)
    drift, ev, di = matched_drift(data, FLAG_SHARE)
    drift.write_parquet(OPT / "event_drift.parquet")
    chosen = {}
    for year in range(CFG["optimizer"]["first_year"], END.year + 1):
        chosen[str(year)] = fit_year(drift, dt.date(year, 1, 1))
        p = chosen[str(year)]
        print(year, "window", p["window"], {g: [round(r * 1e4, 2) for r in v] for g, v in p["rates"].items()},
              "t by bucket", [round(s["t"], 1) if s["t"] == s["t"] else None for s in p["buckets"]], flush=True)
    EVENT_PARAMS.write_text(json.dumps(chosen, indent=1, default=str))
    # The fixed flag share's neighbours, for the paper; not used by the backtests.
    sens = {}
    for share in E["sensitivity_shares"]:
        d, _, _ = matched_drift(data, share)
        sens[str(share)] = fit_year(d, dt.date(END.year + 1, 1, 1))
    (OPT / "event_sensitivity.json").write_text(json.dumps(sens, indent=1, default=str))
    check = sign_check(data, ev, di, chosen)
    (OPT / "event_sign_check.json").write_text(json.dumps(check, indent=1))
    print("sign check:", json.dumps(check), flush=True)
    if any(p["window"] == 0 for p in chosen.values()):
        print("note: some years have no significant first bucket and carry no alpha", flush=True)
    if not (check["live_coef_pct_month"] < 0) and not HOLDOUT:  # holdout: frozen design, report only
        raise SystemExit(f"sign check failed: live coefficient {check['live_coef_pct_month']:+.3f}%/month (t {check['t']:.2f}) "
                         "does not agree with the negative alpha; stopping before any backtest")


# ---------------------------------------------------------------- the optimizer's view
class EventAlphas:
    """Alpha for each backtest day from the event clock, with year Y's settings from EVENT_PARAMS."""

    def __init__(self, lo: dt.date, hi: dt.date):
        guard(hi)
        self.params = json.loads(EVENT_PARAMS.read_text())
        start = lo - dt.timedelta(days=int(max(EDGES) * 1.5) + 30)  # older events are still live
        sm = pl.read_parquet(SMOOTHED, columns=["date", "barrid", COLUMN]).filter(pl.col("date").is_between(start, hi))
        caps = (features(["date", "barrid", "market_cap", "eligible", "amihud63", "ivol63"], years=range(start.year, hi.year + 1))
                .filter(pl.col("date").is_between(start, hi)).collect().sort("barrid", "date")
                .with_columns(level=pl.col("amihud63").shift(63).over("barrid"))
                .filter(pl.col("eligible")).select("date", "barrid", "market_cap", "level", pl.col("ivol63").alias("vol")))
        data = sm.join(caps, on=["date", "barrid"], how="inner")
        # matched-drift cells (as in panel()): size fifth x pre-shock illiquidity fifth; 25 = no cell
        cells = data.with_columns(
            sq=((pl.col("market_cap").rank() / pl.len()).over("date") * 5).ceil().clip(1, 5).cast(pl.Int32),
            lq=((pl.col("level").rank() / pl.len()).over("date") * 5).ceil().clip(1, 5).cast(pl.Int32),
        ).select("date", "barrid", ((pl.col("sq") - 1) * 5 + pl.col("lq") - 1).fill_null(25).alias("cell"))
        self.cells_by_date = {d: g for (d,), g in cells.group_by(["date"])}
        days = sorted(data["date"].unique().to_list())
        self.day_index = {d: i for i, d in enumerate(days)}
        ev = events(data.select("date", "barrid", COLUMN, "market_cap", "vol"), COLUMN, FLAG_SHARE)
        a = ages(data.select("date", "barrid"), ev, self.day_index)
        self.by_date = {d: g for (d,), g in a.group_by(["date"])}

    def cells(self, signal_date: dt.date, ids: np.ndarray) -> np.ndarray:
        """Matched-drift cell (0-24) of each asset on signal_date; 25 = unscored or no illiquidity history."""
        day = self.cells_by_date.get(signal_date)
        if day is None:
            return np.full(len(ids), 25)
        return pl.DataFrame({"barrid": ids}).join(day, on="barrid", how="left", maintain_order="left")["cell"].fill_null(25).to_numpy()

    def __call__(self, signal_date: dt.date, year: int, ids: np.ndarray, frequency: str, costs: bool):
        """Returns (alpha for the optimizer, alpha for the transfer coefficient, flagged mask) for the
        weights dated the trading day after signal_date."""
        p = self.params[str(year)]
        n = p["window"]
        day = self.by_date.get(signal_date)
        zero = np.zeros(len(ids))
        if day is None or n == 0:
            return zero, zero, zero.astype(bool)
        j = pl.DataFrame({"barrid": ids}).join(day.select("barrid", "age", "event_group", "event_vol"), on="barrid", how="left", maintain_order="left")
        age = j["age"].cast(pl.Float64).to_numpy() + 1  # one day later than signal_date
        grp = j["event_group"].fill_null(0).to_numpy()
        rates = np.where((grp == 1)[:, None], np.array(p["rates"]["large"])[None, :], np.array(p["rates"]["rest"])[None, :])
        vol = j["event_vol"].fill_null(0.0).fill_nan(0.0).to_numpy()  # rates are per unit of event-day vol
        nxt = alpha_sum(age, rates, n, 1 if frequency == "daily" else CFG["optimizer"]["days_per_month"]) * vol
        opt = (alpha_sum(age, rates, n, None) * vol) if costs else nxt
        a = np.nan_to_num(age, nan=0)
        return opt, nxt, (a >= 1) & (a <= n)
