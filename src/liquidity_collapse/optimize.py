"""Optimizer backtests of the liquidity-event alpha: risk-model cache, runs, lambda walk-forward, transfer coefficient.

Active weights h (w = b + h) solve sf-quant's cost_aware_mve_optimizer:
    max  alpha'h - (gamma/2) h'Vh - lambda * sum_i c_i |h_i - h_prev_i|     (c = half-spread)
    s.t. h'V h <= target^2,  sum h = 0,  beta'h = 0,  net zero within each size x pre-shock-illiquidity fifth,
         and, by run, nothing (no limits), b + h >= -0.25% (limited shorting) or b + h >= 0 (no shorting).
Weights dated D use the signal from the previous trading day's close; risk model, beta and benchmark come from D.
Alpha comes from event_alpha.EventAlphas.

Final design (LAB_FIXED_GAMMA=1, output/optimizer_fixed_gamma[_holdout]): target 1%; gamma fixed each year at the
previous year's median calibrated gamma; runs U/S/C-D and U/S/C-M without costs, UC/SC/CC-D and -M with costs, lambda
picked each January from earlier years (frozen at its 2018 value for 2019-2024).
Earlier design (default, output/optimizer[_holdout]): target 5%, gamma recalibrated daily, no costs (U-D, C-D, U-M, C-M).
LAB_HOLDOUT=1 runs 2019-2024; LAB_SMOKE=1 runs January-March 2011 into separate smoke folders.
"""

import datetime as dt
import json
import math
import os
import time

import cvxpy as cp
import numpy as np
import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp
from sf_quant.data._factors import factors as FACTORS
from sf_quant.data.covariance_matrix import construct_factor_model_components
from sf_quant.optimizer import (GroupNeutral, LongOnly, MaxShort, ZeroBeta, ZeroInvestment, calibrate_gamma,
                                cost_aware_mve_optimizer)

from .config import CFG, OUTPUT
from .daily_signal import END, HOLDOUT, OPT, guard
from .event_alpha import EventAlphas
from .panel import features

O = CFG["optimizer"]
CACHE = OUTPUT / "optimizer" / "cache"  # shared: the risk model does not depend on the signal
SMOKE = os.environ.get("LAB_SMOKE") == "1"
FIXED = os.environ.get("LAB_FIXED_GAMMA") == "1"
TARGET = O["fixed_target_risk"] if FIXED else O["target_risk"]
RUN_END = dt.date(2011, 3, 31) if SMOKE else END
START = CFG["window"]["holdout_start"] if HOLDOUT else O["start"]
FIRST_YEAR = START.year if HOLDOUT else O["first_year"]
OUT = OUTPUT / ("optimizer_fixed_gamma" + ("_holdout" if HOLDOUT else "")) if FIXED else OPT
RUNS_DIR = OUT / ("runs_smoke" if SMOKE else "runs")
ANALYSIS = OUT / ("analysis_smoke" if SMOKE else "analysis")
SCHEDULE = OUT / ("lambda_schedule_smoke.json" if SMOKE else "lambda_schedule.json")
RUNS = {  # name: (frequency, long-only, charges costs)
    "U-D": ("daily", False, False), "C-D": ("daily", True, False), "CC-D": ("daily", True, True),
    "CC-M": ("monthly", True, True), "U-M": ("monthly", False, False),
    "C-M": ("monthly", True, False),  # run separately by extra_backtest()
    "S-D": ("daily", False, False), "S-M": ("monthly", False, False),  # limited shorting: b + h >= -s
    "UC-D": ("daily", False, True), "SC-D": ("daily", False, True),  # no limits / limited shorting with costs
    "UC-M": ("monthly", False, True), "SC-M": ("monthly", False, True),
}
LIMITED = {"S-D", "S-M", "SC-D", "SC-M"}  # names with the per-stock short limit O["short_limit"]
# Every run has zero net active weight within each size x pre-shock-illiquidity fifth (the matched-drift
# cells), so the hedge is drawn from the events' own controls.


# ---------------------------------------------------------------- calendar and inputs
def trading_days(lo: dt.date, hi: dt.date) -> list[dt.date]:
    guard(hi)
    years = range(lo.year, hi.year + 1)
    days = features(["date"], years=years).filter(pl.col("date").is_between(lo, hi)).unique().collect()["date"].sort()
    return days.to_list()


def previous_day(days: list[dt.date]) -> dict:
    return {d: p for p, d in zip(days[:-1], days[1:])}


def fill_spreads(spreads: pl.DataFrame) -> pl.DataFrame:
    """Blank spreads take the stock's last known spread, else that day's median (no look-ahead)."""
    return (
        spreads.sort("barrid", "date")
        .with_columns(pl.col("half_spread").forward_fill().over("barrid"))
        .with_columns(pl.col("half_spread").fill_null(pl.col("half_spread").median().over("date")))
    )


def risk_components(date_: dt.date, ids: list[str]):
    try:
        return ids, construct_factor_model_components(date_, ids)
    except ValueError:
        have = set(sfd.load_exposures_by_date(date_)["barrid"].to_list())
        kept = [b for b in ids if b in have]
        return kept, construct_factor_model_components(date_, kept)


def cache_year(year: int, smoke: bool = False) -> None:
    """One .npz per trading day: ids, B (exposures), F (factor cov), D (specific var), beta, b, c."""
    CACHE.mkdir(parents=True, exist_ok=True)
    lo, hi = max(dt.date(year, 1, 1), O["cache_start"]), min(dt.date(year, 12, 31), END)
    if smoke:
        hi = min(hi, dt.date(2011, 3, 31))
    days = trading_days(lo - dt.timedelta(days=10), hi)
    prev = previous_day(days)
    days = [d for d in days if d >= lo]
    assets = sfd.load_assets(lo, hi, ["date", "barrid", "predicted_beta"], in_universe=True).drop_nulls("predicted_beta")
    bench = sfd.load_benchmark(lo, hi).rename({"weight": "b"})
    spreads = fill_spreads(sfd.load_half_spreads(lo - dt.timedelta(days=60), hi)).rename({"date": "prev"})
    started = time.time()
    for k, date_ in enumerate(days):
        path = CACHE / f"{date_}.npz"
        if path.exists():
            continue
        day = (
            assets.filter(pl.col("date") == date_).join(bench.filter(pl.col("date") == date_).drop("date"), on="barrid", how="left")
            .join(spreads.filter(pl.col("prev") == prev[date_]).drop("prev"), on="barrid", how="left")
            .with_columns(pl.col("half_spread").fill_null(pl.col("half_spread").median()))
            .with_columns(pl.col("b").fill_null(0.0), pl.col("half_spread").fill_null(0.0)).sort("barrid")
        )
        ids, (B, F, D) = risk_components(date_, day["barrid"].to_list())
        day = pl.DataFrame({"barrid": ids}).join(day, on="barrid", how="left", maintain_order="left")
        tmp = path.with_name(path.stem + ".tmp.npz")  # write then rename, so an interrupted job leaves no broken file
        np.savez(tmp, ids=np.array(ids), B=B.astype(np.float32), F=F, D=D, beta=day["predicted_beta"].to_numpy(),
                 b=day["b"].to_numpy(), c=day["half_spread"].to_numpy())
        tmp.rename(path)
        if k % 50 == 0:
            print(f"cache {date_}: {len(ids)} names, {time.time() - started:.0f}s so far", flush=True)
    print(f"cache {year}: {len(days)} days in {time.time() - started:.0f}s", flush=True)


def load_cache(date_: dt.date) -> dict:
    z = np.load(CACHE / f"{date_}.npz")
    return {k: z[k] for k in z.files}


def daily_returns(lo: dt.date, hi: dt.date) -> dict:
    guard(hi)
    r = sfd.load_assets(lo, hi, ["date", "barrid", "return"], in_universe=False).drop_nulls("return").with_columns(
        pl.col("return") / 100)
    return {d: g.select("barrid", "return") for (d,), g in r.group_by(["date"])}


# ---------------------------------------------------------------- solve
def risk_aversion(alpha, B, F, D) -> float:
    """gamma = sqrt(alpha' V^-1 alpha) / target: the unconstrained optimum V^-1 alpha / gamma sits exactly at the target."""
    return calibrate_gamma(alpha, B, F, D, TARGET)


def solve(alpha, B, F, D, constraints, before, c, lam, b, beta, gamma, groups=None, floor=None):
    """sf-quant's cost_aware_mve_optimizer: max alpha'h - lam * c'|h - before| - (gamma/2) h'Vh under the risk cap
    TARGET and the constraints. groups (int per asset): net active weight is zero within each group (GroupNeutral).
    floor: total weight b + h >= -floor (MaxShort). Returns None when the solve fails."""
    cons = list(constraints) + ([GroupNeutral(groups)] if groups is not None else []) + ([MaxShort(floor)] if floor is not None else [])
    try:
        out = cost_aware_mve_optimizer(
            ids=[str(k) for k in range(len(alpha))], alphas=alpha, factor_exposures=B, factor_covariance=F, specific_risk=D,
            constraints=cons, previous_weights=before, half_spreads=c, cost_aversion=lam, target_active_risk=TARGET,
            betas=beta, benchmark_weights=b, gamma=gamma)
    except (ValueError, cp.error.SolverError):
        return None
    return out["weight"].to_numpy()


def gamma_schedule(years, frequency: str, costs: bool, alphas, hi: dt.date) -> dict:
    """Fixed gamma for each year: the median calibrated gamma over every 5th trading day of the previous year (the
    first development year, which has no previous year, uses its own). Only days with live alpha count."""
    out = {}
    for y in years:
        src = y - 1 if y - 1 >= O["first_year"] else y
        days = trading_days(dt.date(src, 1, 1) - dt.timedelta(days=10), min(dt.date(src, 12, 31), hi))
        prev = previous_day(days)
        g = []
        for d in [d for d in days if d.year == src][::5]:
            c = load_cache(d)
            a = alphas(prev[d], d.year, c["ids"], frequency, costs)[0]
            if np.any(a != 0):
                g.append(risk_aversion(a, c["B"].astype(np.float64), c["F"], c["D"]))
        if not g:
            raise ValueError(f"no live alpha in {src} to fix gamma for {y}")
        out[y] = float(np.median(g))
        print(f"gamma {y} ({frequency}, costs={costs}): {out[y]:.4g} from {len(g)} days of {src}", flush=True)
    return out


# ---------------------------------------------------------------- one run
def drift(held: pl.DataFrame, growth: pl.DataFrame) -> pl.DataFrame:
    """Total weights after compounded returns: w (1 + R) / (1 + w'R)."""
    j = held.join(growth, on="barrid", how="left").with_columns(pl.col("R").fill_null(0.0))
    return j.select("barrid", (pl.col("total") * (1 + pl.col("R")) / (1 + (pl.col("total") * pl.col("R")).sum())).alias("drifted"))


def growth(returns: dict, dates: list[dt.date]) -> pl.DataFrame:
    frames = [returns[d] for d in dates if d in returns]
    if not frames:
        return pl.DataFrame(schema={"barrid": pl.String, "R": pl.Float64})
    return pl.concat(frames).group_by("barrid").agg((pl.col("return") + 1).product().sub(1).alias("R"))


def run(name: str, lam, lo: dt.date, hi: dt.date, out: str, start=None, floor=None, record_from=None) -> None:
    """``lam`` is a number or {year: lambda}. ``start`` = (held total weights, their date, earlier frames,
    earlier log rows) resumes a run already computed up to lo. ``floor`` caps each stock's short at -floor
    total weight. ``record_from``: days before it are solved (a warm-up) but not saved. Writes RUNS_DIR/{out}.parquet and {out}_days.parquet."""
    frequency, long_only, costs = RUNS[name]
    constraints = [ZeroInvestment(), ZeroBeta()] + ([LongOnly()] if long_only else [])
    first = min(lo - dt.timedelta(days=10), start[1]) if start else lo - dt.timedelta(days=10)
    all_days = trading_days(first, hi)
    prev = previous_day(all_days)
    days = [d for d in all_days if d >= lo]
    rebalance = days if frequency == "daily" else [d for k, d in enumerate(days) if k == 0 or d.month != days[k - 1].month]
    returns = daily_returns(first, hi)
    record_from = record_from or lo
    years = range(record_from.year, hi.year + 1)
    alphas = EventAlphas(min(lo, dt.date(max(record_from.year - 1, O["first_year"]), 1, 1)) if FIXED else lo, hi)
    fixed = gamma_schedule(years, frequency, costs, alphas, hi) if FIXED else None
    gamma = None
    caps = features(["date", "barrid", "market_cap"], years=range(lo.year, hi.year + 1)).filter(
        pl.col("date").is_between(lo - dt.timedelta(days=10), hi)).collect()
    caps = {d: g for (d,), g in caps.group_by(["date"])}
    held, held_date, frames, rows = start if start else (None, None, [], [])
    started = time.time()
    for k, date_ in enumerate(rebalance):
        c = load_cache(date_)
        ids, beta = c["ids"], c["beta"]
        # Cached b on day D is end-of-day (it already includes D's return); the weights held during D
        # are the previous day's.
        p = load_cache(prev[date_])
        b = pl.DataFrame({"barrid": ids}).join(pl.DataFrame({"barrid": p["ids"], "b": p["b"]}), on="barrid", how="left",
                                               maintain_order="left")["b"].fill_null(0.0).to_numpy()
        alpha, alpha_tc, flagged = alphas(prev[date_], date_.year, ids, frequency, costs)
        if held is None:
            before, exits = np.zeros(len(ids)), 0.0
        else:
            span = [d for d in all_days if held_date <= d < date_]
            drifted = drift(held, growth(returns, span))
            before = pl.DataFrame({"barrid": ids}).join(drifted, on="barrid", how="left", maintain_order="left")[
                "drifted"].fill_null(0.0).to_numpy() - b
            exits = float(drifted.filter(~pl.col("barrid").is_in(list(ids)))["drifted"].abs().sum())
        lam_d = (lam[str(date_.year)] if isinstance(lam, dict) else lam) if costs else 0.0
        B64 = c["B"].astype(np.float64)
        if np.any(alpha != 0):
            gamma = fixed[max(date_.year, record_from.year)] if FIXED else risk_aversion(alpha, B64, c["F"], c["D"])
        try:
            if gamma is None:  # no live events yet: stay at the benchmark
                h, status = np.zeros(len(ids)), "no_alpha"
            else:
                groups = alphas.cells(prev[date_], ids)
                h, status = solve(alpha, B64, c["F"], c["D"], constraints, before, c["c"], lam_d, b, beta, gamma, groups, floor), "ok"
                if h is None:
                    h, status = before.copy(), "failed"
        except (ValueError, cp.error.SolverError, np.linalg.LinAlgError):
            h, status = before.copy(), "failed"
        x = B64.T @ h
        risk = float(np.sqrt(max(0.0, x @ c["F"] @ x + (c["D"] * h ** 2).sum())))
        trade = np.abs(h - before)
        cap = caps.get(prev[date_])
        small = np.zeros(len(ids), dtype=bool)
        if cap is not None:
            m = pl.DataFrame({"barrid": ids}).join(cap.select("barrid", "market_cap"), on="barrid", how="left", maintain_order="left")["market_cap"]
            cutoff = m.drop_nulls().quantile(1 / 3)
            small = (m.fill_null(0.0) <= cutoff).to_numpy()
        absh = np.abs(h).sum()
        exp = c["B"].astype(np.float64).T @ h
        factor_var = float(exp @ c["F"] @ exp)  # Barra covariances are annual
        specific_var = float((h ** 2 * c["D"]).sum())
        held, held_date = pl.DataFrame({"barrid": ids, "total": h + b}), date_
        if date_ < record_from:
            continue
        rows.append({
            "date": date_, "status": status, "lambda": lam_d, "names": len(ids), "flagged": int(flagged.sum()),
            "alpha_h": float(alpha_tc @ h), "alpha_opt_h": float(alpha @ h), "gamma": gamma or 0.0, "active_risk": risk, "turnover": float(trade.sum() + exits),
            "cost": float((c["c"] * trade).sum()) if costs else 0.0,
            "underweights": int((h < -1e-6).sum()), "min_total_weight": float((h + b).min()),
            "sum_h": float(h.sum()), "beta_h": float(beta @ h),
            "flagged_share_abs": float(np.abs(h[flagged]).sum() / absh) if absh > 0 else 0.0,
            "small_share_abs": float(np.abs(h[small]).sum() / absh) if absh > 0 else 0.0,
            "factor_risk_share": factor_var / (factor_var + specific_var) if factor_var + specific_var > 0 else float("nan"),
        })
        frames.append(pl.DataFrame({"date": [date_] * len(ids), "barrid": ids, "active_weight": h}))
        if k % 100 == 0:
            print(f"{out} {date_}: {status}, risk {risk:.4f}, {time.time() - started:.0f}s", flush=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / out).parent.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_parquet(RUNS_DIR / f"{out}.parquet")
    log = pl.DataFrame(rows)
    log.write_parquet(RUNS_DIR / f"{out}_days.parquet")
    print(json.dumps({"run": out, "rebalances": log.height, "failed": int((log["status"] == "failed").sum()),
                      "seconds": round(time.time() - started)}), flush=True)


# ---------------------------------------------------------------- tasks
def backtest_tasks() -> list[tuple]:
    """(run, lambda, year or None). No-cost daily runs split by year; cost runs one task per lambda."""
    years = range(FIRST_YEAR, END.year + 1)
    tasks = [("U-D", 0.0, y) for y in years] + [("C-D", 0.0, y) for y in years]
    tasks += [("U-M", 0.0, None)]
    if HOLDOUT:  # frozen design, no-cost runs only
        return tasks + [("C-M", 0.0, None)]
    tasks += [("CC-D", lam, None) for lam in O["lambdas"]] + [("CC-M", lam, None) for lam in O["lambdas"]]
    return tasks


def extra_backtest() -> None:
    """C-M (monthly, no shorting, no costs): the monthly twin of C-D, outside the task array."""
    run("C-M", 0.0, START, RUN_END, "C-M/lambda_0")


def short_stem(name: str, s: float) -> str:
    return f"{name}/limit_{s * 100:g}" + ("/year_*" if name == "S-D" else "/lambda_0")


def short_tasks() -> list[tuple]:
    """Limited shorting (b + h >= -s): every s in the grid on the development years; the chosen s on the holdout."""
    limits = [O["short_limit"]] if HOLDOUT else O["short_limits"]
    years = range(FIRST_YEAR, END.year + 1)
    return [("S-D", s, y) for s in limits for y in years] + [("S-M", s, None) for s in limits]


def short_backtest(task: int) -> None:
    name, s, year = short_tasks()[task]
    lo, hi = (dt.date(year, 1, 1), dt.date(year, 12, 31)) if year else (START, RUN_END)
    lo, hi = max(lo, START), min(hi, RUN_END)
    run(name, 0.0, lo, hi, short_stem(name, s).replace("year_*", f"year_{year}"), floor=s)


DEV_SCHEDULE = OUTPUT / "optimizer_fixed_gamma" / "lambda_schedule.json"


def frozen_lambdas() -> dict:
    """Holdout: each cost run keeps the lambda the walk-forward chose for the last development year."""
    return {n: v[str(O["end"].year)] for n, v in json.loads(DEV_SCHEDULE.read_text()).items()}


def variant_tasks() -> list[tuple]:
    """Fixed-gamma design (LAB_FIXED_GAMMA=1): (run, lambda, year or None). No-cost daily runs by year, no-cost monthly
    runs whole; daily cost runs by (lambda, year) with a warm-up, monthly cost runs by lambda. Holdout (LAB_HOLDOUT=1):
    the frozen design, each cost run at its frozen lambda only."""
    years = range(FIRST_YEAR, END.year + 1)
    tasks = [(n, 0.0, y) for n in ("U-D", "S-D", "C-D") for y in years] + [(n, 0.0, None) for n in ("U-M", "S-M", "C-M")]
    if HOLDOUT:
        lam = frozen_lambdas()
        tasks += [(n, lam[n], y) for n in ("UC-D", "SC-D", "CC-D") for y in years]
        return tasks + [(n, lam[n], None) for n in ("UC-M", "SC-M", "CC-M")]
    tasks += [(n, lam, y) for n in ("UC-D", "SC-D", "CC-D") for lam in O["lambdas"] for y in years]
    return tasks + [(n, lam, None) for n in ("UC-M", "SC-M", "CC-M") for lam in O["lambdas"]]


def variant_name(name: str, lam: float, year) -> str:
    costs = RUNS[name][2]
    if costs and HOLDOUT:  # frozen lambda: the run itself is the final run
        return f"{name}/final" + (f"/year_{year}" if year else "")
    return (f"{name}/lambda_{lam:g}" if costs else name) + (f"/year_{year}" if year else ("" if costs else "/lambda_0"))


def variant(task: int) -> None:
    """One fixed-gamma task. A daily cost run's year starts from the benchmark warm_up_days trading days before
    January 1 and records from January 1, so years are independent; the first development year has no warm-up."""
    if not FIXED:
        raise SystemExit("variant tasks need LAB_FIXED_GAMMA=1")
    name, lam, year = variant_tasks()[task]
    floor = O["short_limit"] if name in LIMITED else None
    if year is None:
        return run(name, lam, START, RUN_END, variant_name(name, lam, year), floor=floor)
    lo, hi = max(dt.date(year, 1, 1), START), min(dt.date(year, 12, 31), RUN_END)
    warm = lo
    if RUNS[name][2] and year > O["first_year"]:
        warm = trading_days(lo - dt.timedelta(days=O["warm_up_days"] * 2), lo - dt.timedelta(days=1))[-O["warm_up_days"]]
    run(name, lam, warm, hi, variant_name(name, lam, year), floor=floor, record_from=lo)


def smoke_tasks() -> list[int]:
    return [k for k, (_, _, year) in enumerate(backtest_tasks()) if year in (None, O["first_year"])]


def task_name(name: str, lam: float, year) -> str:
    return f"{name}/year_{year}" if year else f"{name}/lambda_{lam:g}"


def backtest(task: int) -> None:
    name, lam, year = backtest_tasks()[task]
    lo, hi = (dt.date(year, 1, 1), dt.date(year, 12, 31)) if year else (O["start"], RUN_END)
    lo, hi = max(lo, START), min(hi, RUN_END)
    run(name, lam, lo, hi, task_name(name, lam, year))


# ---------------------------------------------------------------- results
def gross_returns(weights: pl.DataFrame, frequency: str, hi: dt.date) -> pl.DataFrame:
    w = weights.select("date", "barrid", pl.col("active_weight").alias("weight"))
    if frequency == "monthly":  # zero-weight row so the last month's period ends at hi
        w = pl.concat([w, pl.DataFrame({"date": [hi], "barrid": [w["barrid"][0]], "weight": [0.0]})])
    r = sfp.generate_returns_from_weights(w.sort("date", "barrid"), costs=False, delistings=True)
    r = r.rename({"return": "gross"})
    return r.filter(pl.col("date") < hi) if frequency == "monthly" else r


def load_run(path_stem: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    files = sorted(f for f in RUNS_DIR.glob(f"{path_stem}.parquet") if not f.stem.endswith("_days")) if "*" in path_stem \
        else [RUNS_DIR / f"{path_stem}.parquet"]
    w = pl.concat([pl.read_parquet(f) for f in files])
    d = pl.concat([pl.read_parquet(f.with_name(f.stem + "_days.parquet")) for f in files]).sort("date")
    return w, d


def performance(name: str, stem: str, hi: dt.date = RUN_END) -> pl.DataFrame:
    frequency, _, costs = RUNS[name]
    w, d = load_run(stem)
    r = gross_returns(w, frequency, hi).join(d.select("date", "cost", "alpha_h", "active_risk"), on="date", how="left")
    return r.with_columns(net=pl.col("gross") - pl.col("cost").fill_null(0.0)).sort("date")


def ir(x: np.ndarray, frequency: str) -> float:
    k = 252 if frequency == "daily" else 12
    return float(x.mean() / x.std(ddof=1) * math.sqrt(k)) if len(x) > 2 and x.std() > 0 else float("nan")


def write_frozen_schedule() -> None:
    """Holdout: record the frozen lambdas in the holdout's schedule file (read by gamma_costs_summary.py)."""
    lam = frozen_lambdas()
    SCHEDULE.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE.write_text(json.dumps({n: {str(y): v for y in range(FIRST_YEAR, END.year + 1)} for n, v in lam.items()}, indent=1))
    print(json.dumps(lam), flush=True)


def select_lambda() -> None:
    """Walk-forward stage 2: from first_lambda_year, each January takes the lambda with the best net IR so far.
    Fixed-gamma design: the six cost runs, and the chosen run is stitched year by year from the grid runs into
    {name}/final (exact for daily, whose years are independent; monthly carries holdings across a switch)."""
    out = {}
    names = ("UC-D", "SC-D", "CC-D", "UC-M", "SC-M", "CC-M") if FIXED else ("CC-D", "CC-M")
    for name in names:
        grid, default = O["lambdas"], O["default_lambda"]
        frequency = RUNS[name][0]
        stem = lambda lam: f"{name}/lambda_{lam:g}" + ("/year_*" if FIXED and frequency == "daily" else "")
        perf = {lam: performance(name, stem(lam)) for lam in grid}
        schedule = {}
        for year in range(O["first_year"], RUN_END.year + 1):
            if year < O["first_lambda_year"]:
                schedule[str(year)] = default
                continue
            scores = {lam: ir(p.filter(pl.col("date") < dt.date(year, 1, 1))["net"].to_numpy(), frequency) for lam, p in perf.items()}
            schedule[str(year)] = max(scores, key=lambda lam: scores[lam])
        out[name] = schedule
        if FIXED:
            parts = {lam: load_run(stem(lam)) for lam in set(schedule.values())}
            pick = lambda df, year: df.filter(pl.col("date").dt.year() == year)
            w = pl.concat([pick(parts[schedule[str(y)]][0], y) for y in range(O["first_year"], RUN_END.year + 1)])
            d = pl.concat([pick(parts[schedule[str(y)]][1], y) for y in range(O["first_year"], RUN_END.year + 1)])
            (RUNS_DIR / name).mkdir(parents=True, exist_ok=True)
            w.write_parquet(RUNS_DIR / f"{name}/final.parquet")
            d.write_parquet(RUNS_DIR / f"{name}/final_days.parquet")
    SCHEDULE.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


def final(task: int) -> None:
    """Rerun with the lambda schedule. Until the schedule first leaves the default lambda the run is
    identical to the default-lambda grid run, so that part is copied and the run resumes from its weights."""
    name = ["CC-D", "CC-M"][task]
    schedule = json.loads(SCHEDULE.read_text())[name]
    default = O["default_lambda"]
    switch = next((int(y) for y, lam in sorted(schedule.items()) if lam != default), None)
    w, d = load_run(f"{name}/lambda_{default:g}")
    if switch is None:
        w.write_parquet(RUNS_DIR / f"{name}/final.parquet")
        d.write_parquet(RUNS_DIR / f"{name}/final_days.parquet")
        print(f"{name}: schedule never leaves {default}; final = grid run", flush=True)
        return
    lo = dt.date(switch, 1, 1)
    w, d = w.filter(pl.col("date") < lo), d.filter(pl.col("date") < lo)
    last = w["date"].max()
    c = load_cache(last)
    h = pl.DataFrame({"barrid": c["ids"]}).join(w.filter(pl.col("date") == last), on="barrid", how="left",
                                                 maintain_order="left")["active_weight"].fill_null(0.0).to_numpy()
    pc = load_cache(trading_days(last - dt.timedelta(days=10), last)[-2])  # benchmark held during `last`
    b = pl.DataFrame({"barrid": c["ids"]}).join(pl.DataFrame({"barrid": pc["ids"], "b": pc["b"]}), on="barrid", how="left",
                                                 maintain_order="left")["b"].fill_null(0.0).to_numpy()
    held = pl.DataFrame({"barrid": c["ids"], "total": h + b})
    frames = [g for _, g in w.group_by("date", maintain_order=True)]
    print(f"{name}: copied grid run before {lo}, resuming from {last}", flush=True)
    run(name, schedule, lo, RUN_END, f"{name}/final", start=(held, last, frames, d.to_dicts()))


STEMS = {"U-D": "U-D/year_*", "C-D": "C-D/year_*", "U-M": "U-M/lambda_0", "CC-D": "CC-D/final", "CC-M": "CC-M/final",
         "C-M": "C-M/lambda_0"}
if FIXED:  # fixed-gamma design: its own folder, no limit grid
    STEMS = {"U-D": "U-D/year_*", "S-D": "S-D/year_*", "C-D": "C-D/year_*", "U-M": "U-M/lambda_0", "S-M": "S-M/lambda_0",
             "C-M": "C-M/lambda_0"} | {n: f"{n}/final" for n in ("UC-D", "SC-D", "CC-D", "UC-M", "SC-M", "CC-M")}
    if HOLDOUT:  # daily cost runs are saved by year
        STEMS |= {n: f"{n}/final/year_*" for n in ("UC-D", "SC-D", "CC-D")}
elif "short_limit" in O:
    STEMS |= {n: short_stem(n, O["short_limit"]) for n in ("S-D", "S-M")}


def available() -> dict:
    """The runs that exist in RUNS_DIR (the holdout has no cost runs)."""
    return {n: s for n, s in STEMS.items() if any(f for f in RUNS_DIR.glob(f"{s}.parquet") if not f.stem.endswith("_days"))}


def tc() -> None:
    """Ex-ante TC = mean over dates of (alpha'h / risk) of the constrained run over the reference's;
    monthly alpha is divided by 21 to compare with daily. Realized TC = ratio of realized IRs."""
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    perf = {n: performance(n, s) for n, s in available().items()}
    summary = []
    for n, p in perf.items():
        frequency, _, costs = RUNS[n]
        _, d = load_run(STEMS[n])
        x = p["net" if costs else "gross"].to_numpy()
        summary.append({
            "run": n, "periods": len(x), "gross_return": float(p["gross"].mean() * (252 if frequency == "daily" else 12)),
            "net_return": float(p["net"].mean() * (252 if frequency == "daily" else 12)),
            "ir": ir(x, frequency), "ir_gross": ir(p["gross"].to_numpy(), frequency),
            "mean_turnover": float(d["turnover"].mean()), "cost_per_year": float(d["cost"].sum() / ((RUN_END - START).days / 365.25)),
            "mean_active_risk_model": float(d["active_risk"].mean()), "failed": int((d["status"] == "failed").sum()),
            "min_total_weight": float(d["min_total_weight"].min()), "max_abs_sum_h": float(d["sum_h"].abs().max()),
            "max_abs_beta_h": float(d["beta_h"].abs().max()), "mean_underweights": float(d["underweights"].mean()),
            "mean_flagged_share_abs": float(d["flagged_share_abs"].mean()), "mean_small_share_abs": float(d["small_share_abs"].mean()),
            "mean_factor_risk_share": float(d["factor_risk_share"].mean()),
        })
    pairs = [("C-D", "U-D"), ("CC-D", "U-D"), ("C-M", "U-M"), ("CC-M", "U-M"), ("S-D", "U-D"), ("S-M", "U-M"),
             ("SC-D", "U-D"), ("SC-M", "U-M"), ("UC-D", "U-D"), ("UC-M", "U-M")]
    years = (RUN_END - START).days / 365.25
    rows = []
    for con, ref in [p for p in pairs if p[0] in perf and p[1] in perf]:
        _, dc = load_run(STEMS[con])
        _, du = load_run(STEMS[ref])
        sc = 1 / O["days_per_month"] if RUNS[con][0] == "monthly" else 1.0
        su = 1 / O["days_per_month"] if RUNS[ref][0] == "monthly" else 1.0
        j = dc.select("date", (pl.col("alpha_h") * sc / pl.col("active_risk")).alias("c")).join(
            du.select("date", (pl.col("alpha_h") * su / pl.col("active_risk")).alias("u")), on="date"
        ).filter(pl.col("c").is_finite(), pl.col("u").is_finite(), pl.col("u") > 0)
        ic, iu = summary[[s["run"] for s in summary].index(con)]["ir"], summary[[s["run"] for s in summary].index(ref)]["ir"]
        rows.append({"constrained": con, "reference": ref, "ex_ante_tc": float((j["c"] / j["u"]).mean()), "dates": j.height,
                     "realized_ir_constrained": ic, "realized_ir_constrained_t": ic * math.sqrt(years),
                     "realized_ir_reference": iu, "realized_ir_reference_t": iu * math.sqrt(years)})
    pl.DataFrame(summary).write_csv(ANALYSIS / "summary.csv")
    pl.DataFrame(rows).write_csv(ANALYSIS / "transfer_coefficient.csv")
    with pl.Config(tbl_cols=30, tbl_width_chars=250, float_precision=4):
        print(pl.DataFrame(summary))
        print(pl.DataFrame(rows))


def check() -> None:
    """Hard checks on a finished set of runs (used after the smoke chain; also safe on the full runs)."""
    summary = pl.read_csv(ANALYSIS / "summary.csv")
    problems = []
    for r in summary.iter_rows(named=True):
        _, long_only, costs = RUNS[r["run"]]
        if r["failed"] > 0:
            problems.append(f"{r['run']}: {r['failed']} failed solves")
        if long_only and r["min_total_weight"] < -1e-8:
            problems.append(f"{r['run']}: shorting {r['min_total_weight']:.2e}")
        if r["run"] in LIMITED and r["min_total_weight"] < -O["short_limit"] - 1e-8:
            problems.append(f"{r['run']}: short beyond the limit {r['min_total_weight']:.2e}")
        if r["max_abs_sum_h"] > 1e-6 or r["max_abs_beta_h"] > 1e-6:
            problems.append(f"{r['run']}: zero-investment or zero-beta broken")
        if r["mean_active_risk_model"] > TARGET + 1e-4:
            problems.append(f"{r['run']}: risk above cap")
        if costs and (r["mean_turnover"] <= 0 or r["mean_flagged_share_abs"] < 0.2):
            problems.append(f"{r['run']}: cost run barely trades flagged names (turnover {r['mean_turnover']:.4f}, flagged share {r['mean_flagged_share_abs']:.2f})")
    for name, stem in available().items():
        _, d = load_run(stem)
        live = d.filter(pl.col("status") == "ok")
        if live.height and float(live["alpha_h"].mean()) <= 0:
            problems.append(f"{name}: mean alpha'h not positive")
    # benchmark timing: yesterday's b drifted by today's return should match today's cached b
    days = trading_days(O["start"], O["start"] + dt.timedelta(days=10))
    r = daily_returns(days[0], days[2])
    for a, b_ in zip(days[:2], days[1:3]):
        ca, cb = load_cache(a), load_cache(b_)
        j = (pl.DataFrame({"barrid": ca["ids"], "ba": ca["b"]}).join(pl.DataFrame({"barrid": cb["ids"], "bb": cb["b"]}), on="barrid")
             .join(r[b_].rename({"return": "rb"}), on="barrid", how="left").fill_null(0.0))
        p = j["ba"] * (1 + j["rb"])
        err = float((p / p.sum() - j["bb"] / j["bb"].sum()).abs().sum())
        if err > 0.005:
            problems.append(f"benchmark timing off on {b_}: {err:.4f}")
    print("\n".join(problems) if problems else "all checks passed", flush=True)
    if problems:
        raise SystemExit(1)


def pick_short() -> None:
    """Choose the short limit on the development years: among limits whose daily run averages at least
    min_middle_risk ex-ante active risk, the highest daily IR; if none gets there, the riskiest. Writes
    ANALYSIS/short_limit_grid.csv."""
    rows = []
    for s in O["short_limits"]:
        row = {"limit": s}
        for name in ("S-D", "S-M"):
            stem = short_stem(name, s)
            if not any(f for f in RUNS_DIR.glob(f"{stem}.parquet") if not f.stem.endswith("_days")):
                continue
            frequency = RUNS[name][0]
            _, d = load_run(stem)
            p = performance(name, stem)
            k = 252 if frequency == "daily" else 12
            tag = name[-1].lower()
            row |= {f"ir_{tag}": ir(p["gross"].to_numpy(), frequency), f"return_{tag}": float(p["gross"].mean() * k),
                    f"risk_ex_ante_{tag}": float(d["active_risk"].mean()), f"turnover_{tag}": float(d["turnover"].mean() * k),
                    f"min_total_weight_{tag}": float(d["min_total_weight"].min()), f"failed_{tag}": int((d["status"] == "failed").sum())}
        rows.append(row)
    grid = pl.DataFrame(rows)
    big = grid.filter(pl.col("risk_ex_ante_d") >= O["min_middle_risk"])
    chosen = (big.sort("ir_d", descending=True) if big.height else grid.sort("risk_ex_ante_d", descending=True))["limit"][0]
    grid = grid.with_columns(chosen=pl.col("limit") == chosen)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    grid.write_csv(ANALYSIS / "short_limit_grid.csv")
    with pl.Config(tbl_cols=30, tbl_width_chars=250, float_precision=4):
        print(grid)
    print(json.dumps({"short_limit": chosen, "rule": "highest daily IR among limits with mean ex-ante risk >= "
                      f"{O['min_middle_risk']}" if big.height else "no limit reached the risk floor: riskiest"}), flush=True)
