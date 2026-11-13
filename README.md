# Liquidity Collapse

A liquidity signal for a long-only fund, built for the Silver Fund.

When trading in a stock suddenly dries up, the stock keeps falling for months. Most investors avoid illiquid stocks, so
after a collapse the stock has to pass to a small group of patient buyers, and that slow hand-off pushes the price down.
We turn this into a Grinold–Kahn alpha and trade it with the fund's mean–variance optimizer.

| | 2011–2018 | 2019–2024 (held back, run once) |
|---|---|---|
| Long-only, rebalanced daily | +0.19% a year, IR 1.82 | +0.51% a year, IR 1.05 |
| Same alphas without the long-only rule | IR 1.17 | IR 1.21 |
| Transfer coefficient (long-only) | 0.44 | 0.36 |

No trading costs are charged; traded on its own, the long-only portfolio would pay about 0.19% a year in spreads. The
effect is real but small, so we recommend it as one input to the fund's combined alpha.

## What's here

| Path | Contents |
|---|---|
| `paper/paper.pdf` | The write-up: idea, evidence, how the signal is built, results |
| `slides/slides.pdf` | The presentation |
| `notebooks/liquidity_events.ipynb` | The study as a notebook, with the code that produced each result |
| `question-bank.html` | A 20-question multiple-choice drill on the theory, optimization, Sharpe and information ratios, and our results; missed questions come back until you get them right (open in a browser) |
| `src/liquidity_collapse/` | The pipeline: signal (`daily_signal.py`), events and alpha (`event_alpha.py`), optimizer and backtests (`optimize.py`), command line (`cli.py`) |
| `scripts/` | Event study, robustness checks, the avoid-list comparison, and `build_slides.py` (charts and tables for the paper and slides) |
| `config.toml` | Data paths and the study's settings |

## Running it

The data is not included: it is Barra data used under license. The code runs on sf-quant
and was run on BYU's supercomputer through SLURM:

```bash
uv sync
lab daily-signal                 # the liquidity shock with the fund's scores removed
lab event-params                 # events and the alpha estimates, walk-forward
lab backtest --task N            # one backtest task (see optimize.backtest_tasks)
lab tc                           # results summary and transfer coefficients
LAB_HOLDOUT=1 lab event-params   # the same pipeline through 2024, for the held-back test
```

Charts, tables and the notebook are rebuilt from saved results:

```bash
SHOW_HOLDOUT=1 python scripts/build_slides.py
python notebooks/build_notebook.py
```
