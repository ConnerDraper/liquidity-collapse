# Liquidity Collapse

A liquidity signal for a long-only fund, built for the Silver Fund.

When trading in a stock suddenly dries up, the stock keeps falling for months. Most investors avoid illiquid stocks, so
after a collapse the stock has to pass to a small group of patient buyers, and that slow hand-off pushes the price down.
We turn this into a Grinold–Kahn alpha and trade it with a mean–variance optimizer: risk aversion fixed each year, active
risk capped at 1%, and half the bid–ask spread charged on every trade.

| Daily rebalancing | 2011–2018 | 2019–2024 (held back, run once) | Turnover, one-way |
|---|---|---|---|
| No limits, no costs (the signal's full strength) | +1.30% a year, IR 1.17 | +1.44% a year, IR 1.20 | ~17× a year |
| Limited shorting (0.25% per stock), after costs | +0.57% a year, IR 0.59 | +0.35% a year, IR 0.31 | ~6× a year |
| Long-only, after costs | +0.12% a year, IR 1.04 | +0.17% a year, IR 0.56 | ~0.5× a year |

The effect is real but small, so we recommend it as one input to the fund's combined alpha.

## What's here

| Path | Contents |
|---|---|
| `slides/slides_fixed_gamma.pdf` | The presentation (final design) |
| `slides/speaker-notes.md` | The speaker script, slide by slide, with likely questions |
| `slides/slides.pdf`, `paper/` | The earlier design (5% risk target, no trading costs) |
| `question-bank.html` | A 20-question multiple-choice drill on the theory and our results (open in a browser) |
| `src/liquidity_collapse/` | The pipeline: signal (`daily_signal.py`), events and alpha (`event_alpha.py`), optimizer and backtests (`optimize.py`), command line (`cli.py`) |
| `scripts/` | Event study, robustness checks, the avoid-list comparison, turnover breakdown, results summary (`gamma_costs_summary.py`), and `build_slides.py` |
| `config.toml` | Data paths and the study's settings |

## Running it

The data is not included: it is Barra data used under license. The code runs on sf-quant (fork branch `08-gamma`) and was
run on BYU's supercomputer through SLURM:

```bash
uv sync
lab daily-signal                          # the liquidity shock with the fund's scores removed
lab event-params                          # events and the alpha estimates, walk-forward
LAB_FIXED_GAMMA=1 lab variant --task N    # one backtest task (see optimize.variant_tasks)
LAB_FIXED_GAMMA=1 lab select-lambda       # cost aversion chosen each January from earlier years
LAB_FIXED_GAMMA=1 lab tc                  # results summary and transfer coefficients
LAB_FIXED_GAMMA=1 LAB_HOLDOUT=1 ...       # the same steps on 2019-2024, design frozen
```

Charts, tables and slides are rebuilt from saved results:

```bash
SHOW_HOLDOUT=1 FIXED_GAMMA=1 python scripts/build_slides.py
```
