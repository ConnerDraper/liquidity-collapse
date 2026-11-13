"""Builds notebooks/liquidity_events.ipynb: the whole study as one notebook, in the format of the conditioned-momentum
notebook.

Cell kinds:
  define    the project's real code (functions and settings, inlined from src/ and scripts/), executed
  heavy     the calls that ran on the supercomputer; not executed here, output = the saved log of that run
  analysis  code that reads the saved results and makes the tables and charts; executed, output captured
Run from the project folder:  .venv/bin/python notebooks/build_notebook.py
"""

import ast
import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, SCRIPTS, LOGS = ROOT / "src/liquidity_collapse", ROOT / "scripts", ROOT / "output/logs"
CELLS = []


# ---------------------------------------------------------------- sources and logs
def names_of(node) -> set:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        out = set()
        for t in node.targets:
            out |= {e.id for e in ast.walk(t) if isinstance(e, ast.Name)}
        return out
    return set()


def source(path: Path, keep=None, drop=()) -> str:
    """Top-level code of a file without docstring and imports; only `keep` names if given, minus `drop`."""
    text = path.read_text()
    out = []
    for n in ast.parse(text).body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
            continue
        names = names_of(n)
        if keep is not None and not names & set(keep):
            continue
        if names & set(drop):
            continue
        out.append(ast.get_source_segment(text, n))
    return "\n\n\n".join(out)


def log(name: str) -> str:
    return (LOGS / name).read_text()


def final_line(name: str) -> str:
    return [l for l in log(name).splitlines() if l.startswith("{")][-1]


def md(text):
    CELLS.append({"kind": "md", "text": text.strip()})


def define(code):
    CELLS.append({"kind": "define", "code": code.strip()})


def heavy(code, output=""):
    CELLS.append({"kind": "heavy", "code": code.strip(), "log": output})


def analysis(code):
    CELLS.append({"kind": "analysis", "code": code.strip()})


signal_log = log("opt_signal_15232326.out")
signal_log = signal_log[: signal_log.index("}", signal_log.index("{'rows'")) + 1] + "\n"
dev_runs = "\n".join(final_line(f"v4_runs_15309604_{k}.out") for k in range(17)) + "\n" + final_line("v4_cm_15309605.out") + "\n"
hold_runs = "\n".join(final_line(f"v4h_runs_15309608_{k}.out") for k in range(14)) + "\n"

# =========================================================================================== notebook content
md(r"""
# When Trading Dries Up: a liquidity signal for a long-only fund

*Heavy steps ran on BYU's supercomputer with the same code shown here, and their saved outputs are shown under those
cells. Everything else runs in this notebook from the saved results.*

When trading in a stock suddenly dries up, the stock keeps falling for months: about 3% behind similar stocks over the next
year, beyond anything the fund's six existing scores capture. We turn this into a Grinold–Kahn alpha and trade it with the
fund's optimizer. Rebalanced daily, the long-only portfolio earns +0.19% a year in 2011–2018 and +0.51% a year (information
ratio 1.05) in 2019–2024, years we held back and ran only once. The effect is real but small, so it belongs as one input to
the fund's alpha.

## 1. The idea

**Two kinds of investors.** Most investors (mutual funds facing daily redemptions, institutions that size positions by
trading volume, active traders) want stocks they can buy and sell quickly and cheaply. A much smaller group with long
horizons will hold illiquid stocks, but only at a lower price, which earns them a slightly higher expected return (Amihud and
Mendelson 1986). Liquid stocks are held by the many, illiquid stocks by the patient few.

**A collapse forces a hand-off.** When trading in a stock suddenly collapses, it no longer suits the investors who own it, and
it has to pass to the patient group. Sellers can exit only a little at a time, because volume has dried up, and the few
patient buyers need a slightly lower price for each sale. The price walks down step by step until the stock is held by
investors content at the new, lower price.

**Why the drift is not traded away.** Anyone who saw the decline coming would short the stock, but shorting small, illiquid
stocks is expensive and often impossible, so the decline is predictable and persists (Bali et al. 2014).

**Predictions** (about the *change* in liquidity, not its level): (1) lower returns than similar stocks whose liquidity did not
collapse; (2) a gradual decline, strongest at first; (3) strongest in small stocks, where patient buyers are scarcest;
(4) driven by the fall in trading volume, which is what slows the exit; (5) stocks that are *already* illiquid earn slightly
more, the patient holders' reward.

**The question for the fund.** The fund is long-only and already trades six scores (idiosyncratic volatility, momentum,
reversal, Barra momentum, beta, Barra reversal). Does the signal add new information, and can a fund that can only hold
*less* of a stock than the benchmark make money from it?

## 2. Setup
""")
define(r"""
import datetime as dt, json, math, os, time
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp
from IPython.display import Image, display
from sf_quant.data.covariance_matrix import construct_factor_model_components
from sf_quant.optimizer import LongOnly, ZeroBeta, ZeroInvestment
from sf_quant.optimizer.optimizers import _construct_constraints

from liquidity_collapse import style
from liquidity_collapse.config import CFG, FUND_SCORES, OUTPUT, PANEL, RESULTS, SIGNALS, YEARS, connect
from liquidity_collapse.evaluate import nw_t

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
connect()
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)
pl.Config.set_float_precision(2)
""")
md("Settings from `config.toml` used below: the event definition and the optimizer.")
analysis(r"""
print({k: CFG["event"][k] for k in ("flag_share", "refractory", "bucket_edges", "stop_t")})
print({k: CFG["optimizer"][k] for k in ("start", "end", "target_risk")}, "| held back:", CFG["window"]["holdout_start"], "to", CFG["window"]["holdout_end"])
""")

md(r"""
## 3. The signal

**Liquidity shock.** Amihud illiquidity is the price move per dollar traded, $|r|/(P\cdot V)$, averaged over the last 21
trading days. The shock compares it with the stock's own average over the previous year, so a very low value means trading
just dried up *for that stock*. **Removing what the fund knows:** each day the shock is regressed across stocks on the fund's
six scores and the residual is standardized to a $Z$-score, so the signal only gets credit for new information.
""")
define(source(SRC / "panel.py", keep=["features"]) + "\n\n\n" +
       source(SRC / "daily_signal.py", drop=["bottom_gaps", "params", "PARAMS"]))
heavy("build()\nsmooth()", signal_log)

md(r"""
**Events.** A stock has an event when its $Z$ enters the worst 5% after 21 trading days outside it. Its age is the number of
trading days since the event; a new event restarts the clock. Everything is measured against **similar stocks**: same day,
same size fifth, same fifth of illiquidity three months earlier, no recent event. Returns are abnormal returns: the part of
each stock's return not explained by the market and Barra's industry and style factors (Barra's specific return).
""")
define(source(SRC / "event_alpha.py", keep=["E", "REFRACTORY", "FLAG_SHARE", "STOP_T", "EDGES", "BUCKETS", "COLUMN",
                                              "EVENT_PARAMS", "GROUPS", "events", "ages", "panel", "matched_drift"]))

md(r"""
## 4. Does the evidence fit?

**Predictions 1–3: lower returns, gradual, concentrated in small illiquid stocks.** The event study compares every event with
similar stocks over the following year (`scripts/event_vs_level.py`, run on the supercomputer).
""")
heavy(source(SCRIPTS / "event_vs_level.py"))
md("The chart below is made from the saved results of that run.")
define(source(SCRIPTS / "build_slides.py", keep=["RES", "OPT", "AN", "HOLD", "SLIDES", "PAPER_FIG", "BLUE", "ORANGE", "AQUA",
                                                   "GREY", "RUN_NAME", "PAIRS", "ROLE", "ROLE_COLOR", "PER_YEAR", "FREQ", "j",
                                                   "has_holdout", "figure", "finish", "zero_line", "num", "run_metrics"])
       .replace('os.environ.get("SHOW_HOLDOUT") == "1" and ', "") + "\n\n\nstyle.apply()")
analysis(source(SCRIPTS / "build_slides.py", keep=["fig_matched_drift"]) +
         '\n\n\nfig_matched_drift()\ndisplay(Image(filename=str(SLIDES / "figures/matched_drift.png"), width=800))')
analysis(source(SCRIPTS / "build_slides.py", keep=["fig_where"]) +
         '\n\n\nfig_where()\ndisplay(Image(filename=str(SLIDES / "figures/where.png"), width=800))')

md(r"""
**Is it a known effect?** Monthly regressions of next month's abnormal return on "had an event in the last month", with size,
prior illiquidity, momentum, long-term reversal and (second model) short-term reversal as controls.
""")
heavy(source(SCRIPTS / "reversal_check.py"), log("reversal_check_15274805.out"))
analysis(r"""
fm = json.loads((RESULTS / "reversal_check.json").read_text())["fama_macbeth"]
rows = [("event + controls", fm["live + controls"]["live"]),
        ("event + controls + last month's return", fm["live + controls + short-term reversal"]["live"])]
display(pl.DataFrame({"model": [r[0] for r in rows], "flagged stocks earn less by, %/month": [-r[1]["coef_pct_month"] for r in rows],
                      "t": [-r[1]["t"] for r in rows]}))
""")
md("""
**Prediction 4: the fall in trading.** The robustness run split the shock into its volume part and its price-move part
(results saved by `scripts/liquidity_checks.py`):
""")
analysis(r"""
checks = json.loads((RESULTS / "liquidity_checks_dev.json").read_text())["components"]
display(pl.DataFrame([{"part of the shock": k.split(",")[0], "IC": v["ic"], "worst tenth vs average, %/month": v["gap_pct_month"], "t": v["gap_t"]}
                      for k, v in checks.items() if isinstance(v, dict)]))
""")
md("**Prediction 5: already-illiquid stocks earn slightly more.** The event-study run also regressed next month's return on the ranks of illiquidity and size together:")
analysis(r"""
fm = json.loads((RESULTS / "event_vs_level.json").read_text())["fama_macbeth"]["illiquidity + size only"]
display(pl.DataFrame({"rank of": ["illiquidity", "size"], "most minus least, %/month": [fm["r_level"]["coef_pct"], fm["r_market_cap"]["coef_pct"]],
                      "t": [fm["r_level"]["t"], fm["r_market_cap"]["t"]]}))
""")
md("Holding size fixed, the most illiquid stocks earn about 0.8% a month more than the most liquid: the patient holders' reward. All five predictions hold. The effect lives in small stocks, which hold little benchmark weight, and that is the practical catch for a long-only fund.")

md(r"""
## 5. From signal to alpha

The Grinold–Kahn form says expected abnormal return = forecasting skill × the stock's volatility × the signal. Our signal is a
flag with an age, so `matched_drift` measures, for each age band (1–21 days, 22–63 days, ...), the average gap between
flagged and similar stocks **per unit of each stock's volatility**, and a stock's alpha is its volatility times that rate.
Each January `fit_year` re-estimates the rates from earlier years only, keeps a band only while its past drift is still
negative, never allows a positive rate, and estimates large stocks separately. `EventAlphas` turns the rates into each day's
alpha for the optimizer.
""")
define(source(SRC / "event_alpha.py", keep=["cumulative", "alpha_sum", "fit_year", "sign_check", "params", "EventAlphas"]))
md("For each year: the holding window, the rates by age band (per unit of volatility, ×10,000) for smaller and large stocks, and each band's t-statistic.")
heavy("params()      # 2011-2018", log("v4_event_15309603.out"))
heavy("# design frozen; same code with LAB_HOLDOUT=1, which extends the end date to 2024 and writes to output/optimizer_holdout\nparams()",
      log("v4h_event_15309607.out"))

md(r"""
**Why volatility belongs in the alpha.** An earlier version left the volatility term out and gave every flagged stock the
same alpha. The optimizer then favored calm stocks, where the same expected drift looks like a better deal. A diagnostic
compared expected and realized return by fifth of predicted volatility:
""")
heavy(source(SCRIPTS / "no_limits_diagnostic.py"), log("no_limits_diag_15309239.out"))
analysis(source(SCRIPTS / "build_slides.py", keep=["fig_vol_check"]) +
         '\n\n\nfig_vol_check()\ndisplay(Image(filename=str(SLIDES / "figures/vol_check.png"), width=800))')
md("Calm flagged stocks delivered about a third of the prediction and the most volatile half again as much. The drift scales with volatility, so the final alpha does too.")

md(r"""
## 6. Building the portfolio

Active weights $h = w - b$ maximize expected active return minus a risk penalty, $h^\top\alpha - \tfrac{\gamma}{2}h^\top V h$
with Barra's covariance matrix $V$, subject to zero net investment ($\sum h = 0$), zero beta to the benchmark, a 5% tracking
error limit, and zero net active weight within each similar-stock group. $\gamma$ is set each day so the portfolio without
the long-only rule sits at 5% tracking error. **Two portfolios:** *no limits*, and *no shorting* (a stock's weight can be cut
to zero but not below, $b + h \ge 0$), each rebalanced daily and monthly. Weights for day $t$ use only day $t-1$'s close.

The **transfer coefficient** (TC) measures how much of the forecast survives the long-only rule: the fundamental law
$IR = IC\sqrt{BR}$ becomes $IR = TC \times IC\sqrt{BR}$ (Clarke, de Silva and Thorley 2002). **No trading costs** are
charged, since the signal would be one input to the fund's alpha where trades net out; what each portfolio would pay in
bid–ask spreads alone is reported.
""")
define(source(SRC / "optimize.py", drop=["smoke_tasks", "select_lambda", "final", "check", "SMOKE", "RUN_END", "RUNS_DIR",
                                         "ANALYSIS", "SCHEDULE"]).replace("O = CFG[\"optimizer\"]",
       'O = CFG["optimizer"]\nRUN_END = END\nRUNS_DIR = OPT / "runs"\nANALYSIS = OPT / "analysis"'))
md("**2011–2018.** One task per year for the daily runs, one task for each monthly run. Output: each run's summary line (progress lines omitted), then the results summary, transfer coefficients and spread costs.")
heavy("for task in range(len(backtest_tasks())):\n    backtest(task)\nextra_backtest()      # monthly, no shorting\ntc()\n\n" +
      "# scripts/return_series.py\n" + source(SCRIPTS / "return_series.py") + "\n\n# scripts/implied_costs.py\n" + source(SCRIPTS / "implied_costs.py"),
      dev_runs + log("v4_results_15309606.out"))
md("**2019–2024, held back.** The same code with `LAB_HOLDOUT=1`, run once after the design was frozen.")
heavy("for task in range(len(backtest_tasks())):\n    backtest(task)\ntc()\n# then scripts/return_series.py and scripts/implied_costs.py, as above",
      hold_runs + log("v4h_results_15309609.out"))

md("## 7. Results\n\nActive return, information ratio and transfer coefficient, both periods (no trading costs charged):")
analysis(r"""
names = {"U-D": "daily, no limits", "C-D": "daily, no shorting", "U-M": "monthly, no limits", "C-M": "monthly, no shorting"}
rows = []
for run, label in names.items():
    row = {"portfolio": label}
    for period, folder in (("2011-18", AN), ("2019-24", HOLD)):
        m = run_metrics(run, folder)
        row |= {f"{period} return, %": m["net"], f"{period} IR": m["ir"], f"{period} TC": m["tc"]}
    rows.append(row)
display(pl.DataFrame(rows))
""")
analysis(source(SCRIPTS / "build_slides.py", keep=["fig_pair"]) +
         '\n\n\nfor freq in ("daily", "monthly"):\n    fig_pair(freq)\n    display(Image(filename=str(SLIDES / f"figures/pair_{freq}.png"), width=560))')
md(r"""
Each period is scaled to 1% risk; shaded years were held back.

- **Daily holds up.** Long-only earns +0.51% a year in 2019–2024 (IR 1.05, t = 2.6); the no-limits IR (1.21) matches 2011–2018.
- **Monthly does not.** A portfolio that trades once a month reaches most events after their strongest weeks.

**Is the optimizer needed?** The simplest long-only use: hold the benchmark, sell every stock with an event in the last 63
days to zero, and give its weight to similar stocks. The run also tested IVOL events; we leave those aside.
""")
heavy(source(SCRIPTS / "avoid_list.py"), log("avoid_list_15310015.out"))
analysis(r"""
avoid = pl.read_csv(OPT / "avoid/avoid_summary.csv").filter(pl.col("portfolio").str.starts_with("Liquidity"))
display(avoid.select("portfolio", pl.col("active_return_pct").alias("return, %/yr"), pl.col("active_risk_pct").alias("risk, %"), "ir", "t"))
""")
md("Avoiding flagged stocks earns almost nothing: dropping them tilts the portfolio toward large stocks, and that noise swamps the effect. The optimizer hedges it.")

md(r"""
## 8. How much to trust it

- **One honest test.** The design was refined on 2011–2018 in four rounds (below), then frozen; 2019–2024 was run once.
- **Small.** After bid–ask spreads (0.19% a year), daily long-only earns about +0.3% a year.
- **Short sample.** An IR's standard error is about $1/\sqrt{\text{years}}$, so six years leaves the true long-only IR between
  roughly 0.2 and 1.9.
- **Not yet checked.** Thinly traded stocks have stale prices, which can make daily risk look too small.

**What changed, and when** (daily runs, 2011–2018; version 1's result is from its own summary):
""")
analysis(r"""
versions = {"2. event clock": "v2_event_clock/analysis", "3. no positive alpha, hedge with similar stocks": "v3_event_clock/analysis",
            "4. alpha scaled by volatility (final)": "analysis"}
get = lambda f, run, col: f.filter(pl.col("run") == run)[col].item()
table = [{"version": "1. flag that turns over daily", "no limits, %/yr": -0.9, "no limits IR": -0.15, "no shorting, %/yr": None, "no shorting IR": None}]
for v, d in versions.items():
    f = pl.read_csv(OPT / d / "summary.csv")
    table.append({"version": v, "no limits, %/yr": get(f, "U-D", "net_return") * 100, "no limits IR": get(f, "U-D", "ir"),
                  "no shorting, %/yr": get(f, "C-D", "net_return") * 100, "no shorting IR": get(f, "C-D", "ir")})
display(pl.DataFrame(table))
""")
md("**Why the 2019–2024 long-only portfolio is bigger.** Until 2015 the model gave large stocks no alpha, so it could only sell small stocks with little benchmark weight; later, large stocks got a negative alpha too:")
analysis(r"""
rows = []
for folder, params_file, years in ((AN, OPT / "event_params.json", range(2011, 2019)),
                                   (HOLD, ROOT / "output/optimizer_holdout/event_params.json", range(2019, 2025))):
    p = json.loads(params_file.read_text())
    r = pl.read_parquet(folder / "return_series.parquet").filter(pl.col("run") == "C-D")
    for (year,), g in r.group_by(pl.col("date").dt.year()):
        if year in years:
            rows.append({"year": year, "large-stock alpha": p[str(year)]["rates"]["large"][0] * 1e4,
                         "long-only risk, %": g["net"].std() * 252 ** 0.5 * 100, "long-only return, %": g["net"].sum() * 100})
display(pl.DataFrame(rows).sort("year"))
""")
md(r"""
## 9. Conclusion

Stocks whose trading dries up keep falling for months: most investors no longer want them, and the hand-off to the few patient
buyers is slow. A long-only fund can harvest this,
but only with a volatility-scaled Grinold–Kahn alpha, the optimizer and daily trading. The gain is a few tenths of a percent a
year, so we recommend it as one input to the fund's combined alpha, after a check for stale prices.
""")


# =========================================================================================== build
class Capture:
    """Stands in for a notebook kernel's display system."""

    def __init__(self):
        self.outputs, self.buffer = [], io.StringIO()

    def flush(self):
        if self.buffer.getvalue():
            self.outputs.append({"output_type": "stream", "name": "stdout", "text": self.buffer.getvalue()})
            self.buffer = io.StringIO()

    def display(self, obj):
        self.flush()
        if hasattr(obj, "data") and hasattr(obj, "format") and type(obj).__name__ == "Image":
            data = {"image/png": base64.b64encode(obj.data).decode(), "text/plain": "<IPython.core.display.Image object>"}
        else:
            data = {"text/plain": str(obj)}
            if hasattr(obj, "_repr_html_"):
                data["text/html"] = obj._repr_html_()
        self.outputs.append({"output_type": "display_data", "data": data, "metadata": {}})


def main() -> None:
    os.chdir(ROOT)
    sys.path.insert(0, str(SRC.parent))
    import matplotlib
    matplotlib.use("Agg")
    ns: dict = {"__name__": "__main__"}
    cells, count = [], 0
    for cell in CELLS:
        if cell["kind"] == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": cell["text"]})
            continue
        count += 1
        cap = Capture()
        code = cell["code"]
        if cell["kind"] == "heavy":
            if cell["log"]:
                cap.outputs.append({"output_type": "stream", "name": "stdout", "text": cell["log"]})
        else:
            with contextlib.redirect_stdout(cap.buffer):
                ns["display"] = cap.display
                exec(compile(code.replace("from IPython.display import Image, display", "from IPython.display import Image"),
                             f"cell{count}", "exec"), ns)
            cap.flush()
        cells.append({"cell_type": "code", "metadata": {}, "source": code, "outputs": cap.outputs, "execution_count": count})
    for i, c in enumerate(cells):
        c["source"] = c["source"].splitlines(True)
        c["id"] = f"cell-{i:03d}"
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python", "version": "3.13"}},
          "nbformat": 4, "nbformat_minor": 5}
    (Path(__file__).parent / "liquidity_events.ipynb").write_text(json.dumps(nb, indent=1))
    print(count, "code cells;", len(cells), "cells")


main()
