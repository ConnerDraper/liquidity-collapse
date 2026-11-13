"""Optimizer backtests of the liquidity event clock: cache, runs, lambda walk-forward, transfer coefficient.

Adapted from research/earnings-signals/src/earnings_signals/backtest.py (risk_components,
drift, solve_day, fill_spreads, the run loop). Active weights h with zero active investment,
zero active beta, zero net weight within each size x pre-shock-illiquidity fifth, 5% active-risk cap; constrained runs add w_b + w_a >= 0. Weights dated
D use the signal from the previous trading day's close; risk model, beta and benchmark come
from D; half-spreads from the previous trading day. Development years only.

Runs:  U-D, C-D (daily, no costs), CC-D (daily, costs), CC-M (monthly, costs), U-M (monthly, no costs).
Objective: max alpha'h - lambda * sum c|h - h_prev| - (gamma/2) h'Vh, gamma set daily so the
unconstrained no-cost portfolio sits at the 5% risk target. Alpha comes from event_alpha.EventAlphas.
Set LAB_SMOKE=1 to run every step on January-March 2011 into separate smoke directories.
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
from sf_quant.optimizer import LongOnly, ZeroBeta, ZeroInvestment
from sf_quant.optimizer.optimizers import _construct_constraints

from .config import CFG, OUTPUT
from .daily_signal import END, HOLDOUT, OPT, guard
from .event_alpha import EventAlphas
from .panel import features

O = CFG["optimizer"]
CACHE = OUTPUT / "optimizer" / "cache"  # shared: the risk model does not depend on the signal
SMOKE = os.environ.get("LAB_SMOKE") == "1"
RUN_END = dt.date(2011, 3, 31) if SMOKE else END
START = CFG["window"]["holdout_start"] if HOLDOUT else O["start"]
FIRST_YEAR = START.year if HOLDOUT else O["first_year"]
RUNS_DIR = OPT / ("runs_smoke" if SMOKE else "runs")
ANALYSIS = OPT / ("analysis_smoke" if SMOKE else "analysis")
SCHEDULE = OPT / ("lambda_schedule_smoke.json" if SMOKE else "lambda_schedule.json")
RUNS = {  # name: (frequency, long-only, charges costs)
    "U-D": ("daily", False, False), "C-D": ("daily", True, False), "CC-D": ("daily", True, True),
    "CC-M": ("monthly", True, True), "U-M": ("monthly", False, False),
    "C-M": ("monthly", True, False),  # run separately by extra_backtest()
}
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
    """gamma = sqrt(alpha' V^-1 alpha) / target, so the unconstrained optimum V^-1 alpha / gamma has
    exactly the target risk. V = B F B' + diag(D), inverted with the Woodbury identity."""
    u = alpha / D
    w = B.T @ u
    m = np.linalg.inv(F) + (B.T * (1 / D)) @ B
    q = float(alpha @ u - w @ np.linalg.solve(m, w))
    return math.sqrt(max(q, 0.0)) / O["target_risk"]


def solve(alpha, B, F, D, constraints, before, c, lam, b, beta, gamma, groups=None):
    """max alpha'h - lam * c'|h - before| - (gamma/2) h'Vh, with the risk cap and constraints.
    groups (int per asset): net active weight is zero within each group."""
    h = cp.Variable(len(alpha))
    risk = cp.quad_form(B.T @ h, F) + cp.sum_squares(cp.multiply(np.sqrt(D), h))
    objective = alpha @ h - gamma / 2 * risk - (lam * (c @ cp.abs(h - before)) if lam > 0 else 0)
    cons = [risk <= O["target_risk"] ** 2] + [k(h) for k in _construct_constraints(constraints, betas=beta, benchmark_weights=b)]
    if groups is not None:
        G = (groups[None, :] == np.unique(groups)[:, None]).astype(float)
        cons.append(G @ h == 0)
    problem = cp.Problem(cp.Maximize(objective), cons)
    try:
        problem.solve(solver="CLARABEL", max_threads=1)
    except cp.error.SolverError:
        problem.solve(solver="SCS", max_iters=20000)
    if problem.status not in ("optimal", "optimal_inaccurate") or h.value is None:
        return None
    return h.value


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


def run(name: str, lam, lo: dt.date, hi: dt.date, out: str, start=None) -> None:
    """``lam`` is a number or {year: lambda}. ``start`` = (held total weights, their date, earlier frames,
    earlier log rows) resumes a run already computed up to lo. Writes RUNS_DIR/{out}.parquet and {out}_days.parquet."""
    frequency, long_only, costs = RUNS[name]
    constraints = [ZeroInvestment(), ZeroBeta()] + ([LongOnly()] if long_only else [])
    first = min(lo - dt.timedelta(days=10), start[1]) if start else lo - dt.timedelta(days=10)
    all_days = trading_days(first, hi)
    prev = previous_day(all_days)
    days = [d for d in all_days if d >= lo]
    rebalance = days if frequency == "daily" else [d for k, d in enumerate(days) if k == 0 or d.month != days[k - 1].month]
    returns = daily_returns(first, hi)
    alphas = EventAlphas(lo, hi)
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
            gamma = risk_aversion(alpha, B64, c["F"], c["D"])
        try:
            if gamma is None:  # no live events yet: stay at the benchmark
                h, status = np.zeros(len(ids)), "no_alpha"
            else:
                groups = alphas.cells(prev[date_], ids)
                h, status = solve(alpha, B64, c["F"], c["D"], constraints, before, c["c"], lam_d, b, beta, gamma, groups), "ok"
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
        held, held_date = pl.DataFrame({"barrid": ids, "total": h + b}), date_
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


def select_lambda() -> None:
    """Walk-forward stage 2: from first_lambda_year, each January takes the lambda with the best net IR so far."""
    out = {}
    for name in ("CC-D", "CC-M"):
        grid, default = O["lambdas"], O["default_lambda"]
        frequency = RUNS[name][0]
        perf = {lam: performance(name, f"{name}/lambda_{lam:g}") for lam in grid}
        schedule = {}
        for year in range(O["first_year"], RUN_END.year + 1):
            if year < O["first_lambda_year"]:
                schedule[str(year)] = default
                continue
            scores = {lam: ir(p.filter(pl.col("date") < dt.date(year, 1, 1))["net"].to_numpy(), frequency) for lam, p in perf.items()}
            schedule[str(year)] = max(scores, key=lambda lam: scores[lam])
        out[name] = schedule
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
    pairs = [("C-D", "U-D"), ("CC-D", "U-D"), ("C-M", "U-M"), ("CC-M", "U-M")]
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
        if r["max_abs_sum_h"] > 1e-6 or r["max_abs_beta_h"] > 1e-6:
            problems.append(f"{r['run']}: zero-investment or zero-beta broken")
        if r["mean_active_risk_model"] > O["target_risk"] + 1e-4:
            problems.append(f"{r['run']}: risk above cap")
        if costs and (r["mean_turnover"] <= 0 or r["mean_flagged_share_abs"] < 0.2):
            problems.append(f"{r['run']}: cost run barely trades flagged names (turnover {r['mean_turnover']:.4f}, flagged share {r['mean_flagged_share_abs']:.2f})")
    for name, stem in STEMS.items():
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
