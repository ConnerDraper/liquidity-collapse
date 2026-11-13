"""Figures for the slides and the paper, and the slide deck, from saved results only (development years).

Slide figures: slides/figures/*.png (16:9, banner). Paper figures: paper/figures/*.pdf (no banner).
Deck: slides/slides.tex. Light enough for the login node.
"""

import datetime as dt
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from liquidity_collapse import style
from liquidity_collapse.config import CFG

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "output/results"
OPT = ROOT / "output/optimizer"
AN = OPT / "analysis"
HOLD = ROOT / "output/optimizer_holdout/analysis"  # frozen design run on 2019-2024
# FIXED_GAMMA=1: the fixed-gamma design (1% cap, trading costs) on the backtest slides, written to
# slides/slides_fixed_gamma.tex with its own pair figures; the current deck and figures are left alone.
FIXED_DECK = os.environ.get("FIXED_GAMMA") == "1"
PAN = ROOT / "output/optimizer_fixed_gamma/analysis" if FIXED_DECK else AN  # runs on the backtest slides
if FIXED_DECK:
    HOLD = ROOT / "output/optimizer_fixed_gamma_holdout/analysis"  # the fixed-gamma design, frozen, on 2019-2024
SUFFIX = "_fg" if FIXED_DECK else ""


def has_holdout() -> bool:
    """Holdout results go into the slides only when asked for: SHOW_HOLDOUT=1."""
    files = ("return_series.parquet", "summary.csv", "transfer_coefficient.csv", "implied_costs.json")
    return os.environ.get("SHOW_HOLDOUT") == "1" and all((HOLD / f).exists() for f in files)
SLIDES = ROOT / "slides"
PAPER_FIG = ROOT / "paper/figures"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a8a"
RUN_COLOR = {"U-D": BLUE, "U-M": ORANGE, "C-D": BLUE, "CC-D": ORANGE, "CC-M": AQUA}
RUN_NAME = {"U-D": "Daily, no limits", "U-M": "Monthly, no limits", "C-D": "Daily, no shorting",
            "CC-D": "Daily, no shorting, costs", "CC-M": "Monthly, no shorting, costs", "C-M": "Monthly, no shorting",
            "S-D": "Daily, limited shorting", "S-M": "Monthly, limited shorting",
            "SC-D": "Daily, limited shorting, costs", "SC-M": "Monthly, limited shorting, costs",
            "UC-D": "Daily, no limits, costs", "UC-M": "Monthly, no limits, costs"}
SHORT = CFG["optimizer"].get("short_limit")  # per-stock short limit of the middle portfolio, chosen on 2011-2018
if not FIXED_DECK and not ((AN / "summary.csv").exists() and "S-D" in pl.read_csv(AN / "summary.csv")["run"].to_list()):
    SHORT = None  # the current deck has no limited-shorting runs
SHORT_PCT = f"{SHORT * 100:g}" if SHORT else None
# The two backtest slides: same roles at each frequency, same colors by role.
if FIXED_DECK:
    PAIRS = {"daily": ["U-D", "SC-D", "CC-D"], "monthly": ["U-M", "SC-M", "CC-M"]}
    ROLE = ["No limits", "Limited shorting + costs", "No shorting + costs"]
    ROLE_COLOR = [BLUE, AQUA, ORANGE]
elif SHORT:
    PAIRS = {"daily": ["U-D", "S-D", "C-D"], "monthly": ["U-M", "S-M", "C-M"]}
    ROLE = ["No limits", "Limited shorting", "No shorting"]
    ROLE_COLOR = [BLUE, AQUA, ORANGE]
else:
    PAIRS = {"daily": ["U-D", "C-D"], "monthly": ["U-M", "C-M"]}
    ROLE = ["No limits", "No shorting"]
    ROLE_COLOR = [BLUE, ORANGE]

j = lambda p: json.loads(Path(p).read_text())
style.apply()


def figure(title: str, wide: bool = False, paper: bool = False):
    if not paper:
        return style.frame(title, wide=wide)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.grid(True, color=style.GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=style.MUTED_INK, labelsize=9)
    return fig, ax


def finish(fig, ax, name: str, paper: bool, note: str, xlabel: str | None, ylabel: str, legend: bool = True):
    if paper:
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        if legend:
            ax.legend(frameon=False, fontsize=9, loc="best")
        fig.tight_layout()
        PAPER_FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(PAPER_FIG / f"{name}.pdf")
        plt.close(fig)
    else:
        style.labels(ax, xlabel=xlabel, ylabel=ylabel)
        if legend:
            style.legend(ax)
        style.footnote(fig, note)
        style.save(fig, SLIDES / "figures" / f"{name}.png")


def zero_line(ax):
    ax.axhline(0, color=style.MUTED_INK, linewidth=1)


# ---------------------------------------------------------------- figures
def fig_ic_by_year(paper=False):
    d = pl.read_csv(RES / "liquidity_shock_resid/dev/ic_by_year.csv")
    fig, ax = figure(f"Appendix: new information beyond the fund's scores? IC by year", wide=True, paper=paper)
    ax.bar(d["year"], d["ic"] * 100, color=BLUE, width=0.7)
    zero_line(ax)
    ax.set_xticks(d["year"].to_list())
    ax.tick_params(axis="x", labelrotation=0)
    finish(fig, ax, "ic_by_year", paper, "IC: how well our score ranks next month's stock-specific returns, after removing the fund's six scores. Positive in 12 of 13 years.",
           None, "Mean monthly IC (x100)", legend=False)


def fig_matched_drift(paper=False):
    m = j(RES / "event_vs_level.json")["matched"]
    ks = [0, 21, 63, 126, 189, 252]
    fig, ax = figure("After trading dries up, the stock keeps falling", paper=paper)
    for key, label, color, dy in (("all", "All events", BLUE, -9), ("size_quintile_1", "Smallest fifth", ORANGE, 9),
                                  ("size_quintile_5", "Largest fifth", GREY, 0)):
        y = [0.0] + [m[key][str(k)]["vs_matched_pct"] for k in ks[1:]]
        ax.plot(ks, y, color=color, linewidth=2.5, marker="o", markersize=6, label=label)
        ax.annotate(f"{y[-1]:+.1f}%".replace("-", "\u2212"), (ks[-1], y[-1]), xytext=(6, dy), textcoords="offset points", va="center",
                    fontsize=10 if not paper else 8, color=style.INK)
    zero_line(ax)
    ax.set_xticks(ks)
    finish(fig, ax, "matched_drift", paper,
           "Flagged stocks minus similar stocks (same size, same liquidity before the event), 2006–2018.",
           "Trading days since the event", "Cumulative gap (%)")


def fig_where(paper=False):
    m = j(RES / "event_vs_level.json")["matched"]
    rows = [("Smallest fifth", "size_quintile_1"), ("Middle size fifth", "size_quintile_3"), ("Largest fifth", "size_quintile_5"),
            ("Most liquid fifth", "illiquidity_quintile_1"), ("Middle liquidity fifth", "illiquidity_quintile_3"),
            ("Least liquid fifth", "illiquidity_quintile_5")]
    fig, ax = figure("Mostly in small, illiquid stocks", wide=True, paper=paper)
    y = np.arange(len(rows))[::-1]
    vals = [m[k]["252"]["vs_matched_pct"] for _, k in rows]
    ts = [m[k]["252"]["t"] for _, k in rows]
    ax.barh(y, vals, color=[BLUE] * 3 + [ORANGE] * 3, height=0.6)
    for yi, v, t in zip(y, vals, ts):
        ax.annotate(f"{v:+.1f}%  (t {t:.1f})", (min(v, 0), yi), xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=10 if not paper else 8, color=style.INK)
    ax.set_yticks(y, [r for r, _ in rows])
    ax.axvline(0, color=style.MUTED_INK, linewidth=1)
    ax.set_xlim(min(vals) * 1.9, 1)
    ax.grid(False, axis="y")
    if not paper:
        x0, y0, w0, h0 = style.WIDE_RECT
        ax.set_position((0.22, y0, w0 + x0 - 0.22, h0))
    finish(fig, ax, "where", paper, "One year after the event, 2006–2018. Blue: by size. Orange: by how illiquid the stock already was.",
           "Gap against similar stocks after one year (%)", "", legend=False)


def fig_rose_fell(paper=False):
    d = j(RES / "bounce_checks.json")["1_matched_drift_by_day"]
    fig, ax = figure("Appendix: is it just a bounce? Stocks that jumped on thin volume give it back", paper=paper)
    days = list(range(0, 22))
    for key, label, color in (("rose", "Rose in the 5 days before", ORANGE), ("fell", "Fell in the 5 days before", BLUE)):
        daily = [d[key][str(k)]["bp_per_day"] for k in range(1, 11)] + [d[key]["11-21"]["bp_per_day"]] * 11
        y = np.r_[0, np.cumsum(daily)]
        ax.plot(days, y, color=color, linewidth=2.5, label=label)
        ax.annotate(f"{y[-1]:+.0f} bp", (days[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=10 if not paper else 8, color=style.INK)
    zero_line(ax)
    finish(fig, ax, "rose_fell", paper,
           "Return against similar stocks, from the day after the event (event day: +274 bp for 'rose', -12 bp for 'fell'). Days 11–21 use their average. 2006–2018.",
           "Trading days since the event", "Cumulative gap (bp)")


def fig_cumulative(group: str, paper=False):
    r = pl.read_parquet(AN / "return_series.parquet")
    runs = ["U-D", "U-M"] if group == "unconstrained" else ["C-D", "CC-D", "CC-M"]
    title = "Backtests without limits" if group == "unconstrained" else "Long-only backtests"
    fig, ax = figure(f"{title}: cumulative active return, {r['date'].min().year}–{r['date'].max().year}", paper=paper)
    for n in runs:
        x = r.filter(pl.col("run") == n).sort("date")
        y = x["net"].cum_sum() * 100
        ax.plot(x["date"], y, color=RUN_COLOR[n], linewidth=2.2, label=RUN_NAME[n])
    zero_line(ax)
    note = ("Active weights with zero net investment and zero beta, hedged within size x illiquidity fifths, 5% active-risk target; no costs."
            if group == "unconstrained" else
            "No shorting below zero total weight; costs = half-spread on every trade; cost aversion chosen each January from earlier years.")
    finish(fig, ax, f"cumulative_{group}", paper, note, None, "Cumulative active return (%)")


PER_YEAR = {"daily": 252, "monthly": 12}
FREQ = {"U-D": "daily", "U-M": "monthly", "C-D": "daily", "CC-D": "daily", "CC-M": "monthly", "C-M": "monthly",
        "S-D": "daily", "S-M": "monthly", "SC-D": "daily", "SC-M": "monthly", "UC-D": "daily", "UC-M": "monthly"}
is_cost = lambda n: len(n.split("-")[0]) == 2  # UC-D, SC-D, CC-D, ...
LO_RUN = "CC-D" if FIXED_DECK else "C-D"  # the long-only portfolio quoted in the text


def fig_pair(freq: str):
    """Slide only, no banner: the three runs of one frequency, each scaled to 1% annual realized active risk."""
    r = pl.read_parquet(PAN / "return_series.parquet")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.grid(True, color=style.GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=style.MUTED_INK, labelsize=12)
    ends = []
    hold = pl.read_parquet(HOLD / "return_series.parquet") if has_holdout() else None
    if hold is not None:
        ax.axvspan(dt.date(2019, 1, 1), hold["date"].max(), color="#efefef", zorder=0)
        ax.text(dt.date(2019, 3, 1), 1, "Holdout", transform=ax.get_xaxis_transform(), va="top", fontsize=12, color=style.MUTED_INK)
    for n, label, color in zip(PAIRS[freq], ROLE, ROLE_COLOR):
        parts = [r.filter(pl.col("run") == n).sort("date").select("date", "net")]
        if hold is not None:
            parts.append(hold.filter(pl.col("run") == n).sort("date").select("date", "net"))
        # each period scaled to 1% of its own realized risk, so neither period dominates the picture
        parts = [p_.with_columns(pl.col("net") * 0.01 / (p_["net"].std() * PER_YEAR[freq] ** 0.5)) for p_ in parts]
        x = pl.concat(parts)
        y = np.cumsum(x["net"].to_numpy()) * 100
        ax.plot(x["date"], y, color=color, linewidth=2.0, label=label)
        ends.append([x["date"][-1], y[-1], y[-1]])
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    ends.sort(key=lambda e: e[1])
    for a, b in zip(ends, ends[1:]):  # push end labels apart
        b[2] = max(b[2], a[2] + 0.07 * span)
    for d, v, pos in ends:
        ax.text(d + dt.timedelta(days=45), pos, f"{v:.1f}%", va="center", ha="left", fontsize=12, color=style.INK)
    last = 2025 if hold is not None else 2019
    step = 4 if hold is not None else 2
    ax.set_xticks([f"{yr}-01-01" for yr in range(2011, last + 1, step)], [str(yr) for yr in range(2011, last + 1, step)])
    ax.axhline(0, color=style.MUTED_INK, linewidth=1)
    ax.set_ylabel("Cumulative return at 1% risk (%)", fontsize=12, color=style.INK)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2 if len(ROLE) in (2, 4) else 3, frameon=False,
              fontsize=13 if len(ROLE) == 2 else 10.5, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    (SLIDES / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(SLIDES / "figures" / f"pair_{freq}{SUFFIX}.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_vol_check(paper=False):
    """Slide: in the version with the same alpha for every flagged stock, realized vs expected return by volatility fifth."""
    d = j(OPT / "v3_event_clock/analysis/no_limits_diagnostic.json")["U-D"]["by_risk_fifth"]
    rows = sorted([r for r in d if r["has_alpha"]], key=lambda r: r["q"])
    ratio = [r["realized"] / r["expected"] * 100 for r in rows]
    vol = [r["mean_predicted_vol"] for r in rows]
    fig, ax = figure("Calm stocks fell less than we expected, volatile stocks more", wide=True, paper=paper)
    labels = [f"{i + 1}: ~{v:.0f}% vol" for i, v in enumerate(vol)]
    ax.bar(range(5), ratio, color=BLUE, width=0.6)
    ax.axhline(100, color=style.MUTED_INK, linewidth=1.2, linestyle="--")
    ax.text(-0.45, 103, "realized = expected", ha="left", va="bottom", fontsize=11, color=style.MUTED_INK)
    for i, v in enumerate(ratio):
        ax.text(i, v + 3, f"{v:.0f}%", ha="center", va="bottom", fontsize=9 if paper else 13, color=style.INK)
    ax.set_xticks(range(5), labels)
    ax.set_ylim(0, max(ratio) * 1.18)
    ax.grid(False, axis="x")
    finish(fig, ax, "vol_check", paper,
           "Actual drift as a share of what we predicted, by volatility group (earlier version, 2011–2018).",
           "Stocks grouped by volatility (1 = calmest fifth)", "Actual / predicted (%)", legend=False)


def fig_long_only_years(paper=False):
    d = pl.read_csv(AN / "long_only_by_year.csv").filter(pl.col("run") == "C-D")
    years = sorted(d["year"].unique().to_list())
    fig, ax = figure("Long-only: what the flagged-stock positions earned each year", paper=paper)
    w = 0.38
    for off, g, label, color in ((-w / 2, "rest", "Smaller stocks", BLUE), (w / 2, "large", "Largest fifth", ORANGE)):
        v = [d.filter(pl.col("year") == y, pl.col("group") == g)["specific_bp_per_year"].item() for y in years]
        ax.bar(np.array(years) + off, v, width=w, color=color, label=label)
    zero_line(ax)
    ax.set_xticks(years)
    finish(fig, ax, "long_only_years", paper,
           "Daily long-only run without costs: specific return from stocks flagged 1–21 days earlier, by size group of the event.",
           None, "Specific return (bp per year)")


# ---------------------------------------------------------------- tables
def num(x: float, fmt: str) -> str:
    """Signed number for LaTeX with a real minus sign."""
    t = format(x, fmt)
    return t.replace("-", "$-$") if t.startswith("-") else t


def tex_escape(s: str) -> str:
    return s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def backtest_rows() -> list[dict]:
    s = pl.read_csv(AN / "summary.csv")
    tc = {r["constrained"]: r["ex_ante_tc"] for r in pl.read_csv(AN / "transfer_coefficient.csv").iter_rows(named=True)}
    years = (pl.read_parquet(AN / "return_series.parquet")["date"].max() - pl.read_parquet(AN / "return_series.parquet")["date"].min()).days / 365.25
    out = []
    for n in ["U-D", "U-M", "C-D", "CC-D", "CC-M"]:
        r = s.filter(pl.col("run") == n).row(0, named=True)
        out.append({"run": RUN_NAME[n], "net": r["net_return"] * 100, "ir": r["ir"], "t": r["ir"] * years ** 0.5,
                    "risk": r["mean_active_risk_model"] * 100, "cost": r["cost_per_year"] * 100, "tc": tc.get(n)})
    return out


def backtest_table(size=r"\small") -> str:
    lines = [size, r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Run & Net return & IR ($t$) & Active risk & Costs & TC \\", r"\midrule"]
    for r in backtest_rows():
        tc = f"{r['tc']:.2f}" if r["tc"] is not None else "ref."
        cost = f"{r['cost']:.3f}\\%/yr" if r["cost"] > 0 else "---"
        lines.append(f"{r['run']} & {num(r['net'], '+.2f')}\\%/yr & {num(r['ir'], '.2f')} ({num(r['t'], '.1f')}) & {r['risk']:.2f}\\% & {cost} & {tc} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def run_metrics(n: str, an: Path = AN) -> dict:
    """Realized metrics of one run from its return series, plus turnover, costs and TC from the summary files."""
    r = pl.read_parquet(an / "return_series.parquet")
    s = pl.read_csv(an / "summary.csv")
    tc = {x["constrained"]: x["ex_ante_tc"] for x in pl.read_csv(an / "transfer_coefficient.csv").iter_rows(named=True)}
    years = (r["date"].max() - r["date"].min()).days / 365.25
    x = r.filter(pl.col("run") == n).sort("date")
    k = PER_YEAR[FREQ[n]]
    net, gross = x["net"].to_numpy(), x["gross"].to_numpy()
    vol = net.std(ddof=1) * k ** 0.5
    ir = net.mean() * k / vol
    cum = np.cumsum(net)
    by_year = x.group_by(pl.col("date").dt.year()).agg(pl.col("net").sum())["net"]
    row = s.filter(pl.col("run") == n).row(0, named=True)
    return {"gross": gross.mean() * k * 100, "net": net.mean() * k * 100, "risk": vol * 100, "ir": ir, "t": ir * years ** 0.5,
            "dd": float((cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]).min()) * 100,
            "pos": f"{int((by_year > 0).sum())}/{len(by_year)}", "turnover": row["mean_turnover"] * k / 2,  # one-way: half of buys + sells
            "cost": row["cost_per_year"] * 100, "tc": tc.get(n)}


def portfolio_table(size=r"\scriptsize") -> str:
    """All runs: gross and net return, realized risk, IR (t), drawdown, positive years, turnover, costs, TC (paper)."""
    lines = [size, r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r" & Gross & Net & Active & & Max & Positive & Turnover, & Costs & \\",
             r"Run & \%/yr & \%/yr & risk & IR ($t$) & drawdown & years & one-way $\times$/yr & \%/yr & TC \\", r"\midrule"]
    have = set(pl.read_csv(AN / "summary.csv")["run"].to_list())
    for group, runs in (("No limits", ["U-D", "U-M"]), ("Long-only", ["C-D", "CC-D", "C-M", "CC-M"])):
        lines.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{group}}}}} \\")
        for n in [r for r in runs if r in have]:
            m = run_metrics(n)
            cost = f"{m['cost']:.3f}" if m["cost"] > 0 else "---"
            t = f"{m['tc']:.2f}" if m["tc"] is not None else "ref."
            lines.append(f"{RUN_NAME[n]} & {num(m['gross'], '+.2f')} & {num(m['net'], '+.2f')} & {m['risk']:.2f}\\% & "
                         f"{num(m['ir'], '.2f')} ({num(m['t'], '.1f')}) & {num(m['dd'], '.2f')}\\% & {m['pos']} & "
                         f"{m['turnover']:.1f} & {cost} & {t} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(lines)


def pair_table(freq: str, size=r"\scriptsize") -> str:
    """Slide table beside the chart: metrics as rows, the two runs of one frequency as columns. No costs charged;
    the spread cost each would pay if traded alone is shown as a diagnostic."""
    folders = [PAN, HOLD] if has_holdout() else [PAN]
    spread = lambda n, an, m: m["cost"] if is_cost(n) else j(an / "implied_costs.json")[n]["cost_pct_per_year"]
    ms = [(m := run_metrics(n, an)) | {"spread": spread(n, an, m)} for n in PAIRS[freq] for an in folders]
    rows = [("Active return, \\%/yr", lambda m: num(m["net"], "+.2f")),
            ("Active risk (realized)", lambda m: f"{m['risk']:.2f}\\%"),
            ("IR ($t$)", lambda m: f"{num(m['ir'], '.2f')} ({num(m['t'], '.1f')})"),
            ("Max drawdown", lambda m: f"{num(m['dd'], '.2f')}\\%"),
            ("Positive years", lambda m: m["pos"]),
            ("Turnover (one-way), $\\times$/yr", lambda m: f"{m['turnover']:.1f}"),
            ("TC vs no limits", lambda m: f"{m['tc']:.2f}" if m["tc"] is not None else "ref."),
            (("Spread cost, \\%/yr" if FIXED_DECK else "Spread cost if traded alone, \\%/yr"), lambda m: f"{m['spread']:.2f}")]
    k = len(PAIRS[freq])
    if len(folders) == 2:
        lines = [size, r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l" + "rr" * k + "}", r"\toprule",
                 " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{r}}}" for r in ROLE) + r" \\",
                 "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(k)),
                 " & " + " & ".join(["2011--18 & 2019--24"] * k) + r" \\", r"\midrule"]
    else:
        lines = [size, r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l" + "r" * k + "}", r"\toprule",
                 " & " + " & ".join(ROLE) + r" \\", r"\midrule"]
    for label, f in rows:
        lines.append(f"{label} & " + " & ".join(f(m) for m in ms) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(lines)


def avoid_table(size=r"\small") -> str:
    """The simple way (avoid flagged stocks for 63 days) next to the event-clock optimizer, 2011-2018."""
    a = pl.read_csv(OPT / "avoid/avoid_summary.csv")
    lines = [size, r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"2011--2018 & Weight avoided & Return & Risk & IR ($t$) & IR, market removed \\", r"\midrule"]
    for r in a.filter(pl.col("portfolio").str.starts_with("Liquidity")).iter_rows(named=True):
        lines.append(f"Just avoid flagged stocks, {r['portfolio'].split(', ')[1]} & {r['avoided_benchmark_weight_pct']:.1f}\\% & "
                     f"{num(r['active_return_pct'], '+.2f')}\\%/yr & {r['active_risk_pct']:.2f}\\% & {num(r['ir'], '.2f')} ({num(r['t'], '.1f')}) & {num(r['hedged_ir'], '.2f')} \\\\")
    m = run_metrics(LO_RUN, PAN)
    lines += [r"\midrule", f"Optimizer, daily, no shorting{', costs' if FIXED_DECK else ''} & --- &{num(m['net'], '+.2f')}\\%/yr & {m['risk']:.2f}\\% & "
              f"{num(m['ir'], '.2f')} ({num(m['t'], '.1f')}) & --- \\\\", r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(lines)


def avoid_compare_table(size=r"\small") -> str:
    """Slide: just avoiding flagged stocks vs the optimizer. The avoid list's active return is split into the part earned
    on the stocks themselves (stock-specific) and the part from factor tilts the reweighting creates."""
    a = pl.read_csv(OPT / "avoid/avoid_summary.csv").filter(pl.col("portfolio").str.starts_with("Liquidity"))
    lines = [size, r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"2011--2018 & Weight & Return & Stock- & Factor & Risk & IR ($t$) \\",
             r" & moved & \%/yr & specific & tilts & & \\", r"\midrule"]
    for r in a.iter_rows(named=True):
        lines.append(f"Just avoid them, {r['portfolio'].split(', ')[1]} & {r['avoided_benchmark_weight_pct']:.1f}\\% & "
                     f"{num(r['active_return_pct'], '+.2f')} & {num(r['specific_return_pct'], '+.2f')} & "
                     f"{num(r['active_return_pct'] - r['specific_return_pct'], '+.2f')} & {r['active_risk_pct']:.2f}\\% & "
                     f"{num(r['ir'], '.2f')} ({num(r['t'], '.1f')}) \\\\")
    m = run_metrics(LO_RUN, PAN)
    lines += [r"\midrule", f"Optimizer, long-only{' + costs' if FIXED_DECK else ''}, daily & --- & {num(m['net'], '+.2f')} & --- & --- & "
              f"{m['risk']:.2f}\\% & {num(m['ir'], '.2f')} ({num(m['t'], '.1f')}) \\\\", r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(lines)


def regression_table(size=r"\small") -> str:
    d = j(RES / "reversal_check.json")["fama_macbeth"]
    c = lambda m, v: f"{-d[m][v]['coef_pct_month']:.2f} ({-d[m][v]['t']:.1f})"  # shown as a shortfall: positive = flagged stocks earn less
    rows = [("Had a liquidity event in the last month", c("live + controls", "live")),
            ("... also controlling for last month's return", c("live + controls + short-term reversal", "live"))]
    lines = [size, r"\begin{tabular}{lr}", r"\toprule", r" & How much less flagged stocks earn, \%/month ($t$) \\", r"\midrule"]
    lines += [f"{a} & {b} \\\\" for a, b in rows] + [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def age_table(size=r"\small") -> str:
    d = j(AN / "run_decomposition.json")
    lines = [size, r"\begin{tabular}{lrrr}", r"\toprule",
             r"Return, \%/yr & Event 1--21 days ago & Event 22--504 days ago & No recent event \\", r"\midrule"]
    for n in ["U-D", "C-D", "U-M", "C-M"]:
        b = d[n]["specific_by_age"]
        lines.append(f"{RUN_NAME[n]} & {num(b['ages 1-21']['specific_pct_per_year'], '+.2f')} & {num(b['ages 22-504']['specific_pct_per_year'], '+.2f')} & "
                     f"{num(b['no live event']['specific_pct_per_year'], '+.2f')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def versions_table(size=r"\small") -> str:
    v2 = pl.read_csv(OPT / "v2_event_clock/analysis/summary.csv")
    v3 = pl.read_csv(OPT / "v3_event_clock/analysis/summary.csv")
    v4 = pl.read_csv(AN / "summary.csv")
    g = lambda f, n, k: f.filter(pl.col("run") == n)[k].item()
    lines = [size, r"\begin{tabular}{lrr}", r"\toprule",
             r"Daily runs, \%/yr (IR), no costs & No limits & No shorting \\", r"\midrule",
             r"1. Flag that turns over daily & $-0.9$ ($-0.15$) & --- \\"]
    for label, f in (("2. Event clock", v2), ("3. + no positive alpha, hedge with similar stocks", v3), ("4. + alpha scaled by volatility", v4)):
        cells = [f"{num(g(f, n, 'net_return') * 100, '+.2f')} ({num(g(f, n, 'ir'), '.2f')})" for n in ("U-D", "C-D")]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- deck
def size_table(size=r"\scriptsize") -> str:
    """Backup: the model's large-stock alpha and the long-only portfolio's size, by year."""
    sources = [(AN, OPT / "event_params.json", lambda y: y <= 2018)]
    if has_holdout():
        sources.append((HOLD, ROOT / "output/optimizer_holdout/event_params.json", lambda y: y >= 2019))
    rows = []
    for an, params, keep in sources:
        p = j(params)
        r = pl.read_parquet(an / "return_series.parquet").filter(pl.col("run") == "C-D")
        by = r.group_by(pl.col("date").dt.year().alias("y")).agg(
            (pl.col("net").std() * 252 ** 0.5 * 100).alias("risk"), (pl.col("net").sum() * 100).alias("ret")).sort("y")
        rows += [(y, p[str(y)]["rates"]["large"][0] * 1e4, risk, ret) for y, risk, ret in by.iter_rows() if keep(y) and str(y) in p]
    lines = [size, r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrr}", r"\toprule",
             r" & Large-stock & Long-only & Long-only \\", r"Year & alpha & risk & return \\", r"\midrule"]
    for y, a, risk, ret in rows:
        if y == 2019:
            lines.append(r"\midrule \multicolumn{4}{l}{\textit{Holdout}} \\")
        lines.append(f"{y} & {num(a, '.1f')} & {risk:.2f}\\% & {num(ret, '+.2f')}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(lines)


def paper_results_table(size=r"\small") -> str:
    """Paper: the four portfolios, return / IR / TC in 2011-2018 and in the held-back 2019-2024."""
    names = {n: RUN_NAME[n] for n in PAIRS["daily"] + PAIRS["monthly"]}
    folders = [AN, HOLD] if has_holdout() else [AN]
    cols = "l" + "rrr" * len(folders)
    lines = [size, r"\begin{tabular}{" + cols + "}", r"\toprule",
             " & " + " & ".join(r"\multicolumn{3}{c}{" + p + "}" for p in (["2011--2018", "2019--2024 (held back)"][:len(folders)])) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(len(folders))),
             "Portfolio" + " & Return & IR & TC" * len(folders) + r" \\", r"\midrule"]
    for n, label in names.items():
        cells = []
        for f in folders:
            m = run_metrics(n, f)
            cells += [num(m["net"], "+.2f") + r"\%", f"{num(m['ir'], '.2f')}", f"{m['tc']:.2f}" if m["tc"] is not None else "---"]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def short_grid_table(size=r"\small") -> str:
    """Appendix: the short limits tried on 2011-2018 and the one kept."""
    g = pl.read_csv(AN / "short_limit_grid.csv").sort("limit")
    lines = [size, r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Short limit per stock & Active risk (model) & IR & Turnover (one-way), $\times$/yr & Monthly IR \\", r"\midrule"]
    for r in g.iter_rows(named=True):
        cells = [rf"{r['limit'] * 100:g}\%", rf"{r['risk_ex_ante_d'] * 100:.2f}\%", num(r["ir_d"], ".2f"), f"{r['turnover_d'] / 2:.0f}",
                 num(r["ir_m"], ".2f")]
        if SHORT and abs(r["limit"] - SHORT) < 1e-12:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def ir_puzzle_table(size=r"\small") -> str:
    """Fixed-gamma deck appendix: limited shorting vs long-only, before and after trading costs, both periods."""
    import glob
    folders = [("2011--18", PAN)] + ([("2019--24", HOLD)] if has_holdout() else [])
    lines = [size, r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Daily & & Risk & Return before & Spreads & Return after & IR before & IR after & Turnover \\",
             r" & & & costs, \%/yr & paid & costs, \%/yr & costs & costs & (one-way) \\", r"\midrule"]
    ir = lambda x: x.mean() / x.std() * 252 ** 0.5
    for k, (label, an) in enumerate(folders):
        if k:
            lines.append(r"\midrule")
        series = pl.read_parquet(an / "return_series.parquet")
        summ = {r["run"]: r for r in pl.read_csv(an / "summary.csv").iter_rows(named=True)}
        for n, name in (("SC-D", "Limited shorting"), ("CC-D", "No shorting")):
            x = series.filter(pl.col("run") == n)
            g, net = x["gross"], x["net"]
            lines.append(f"{label if n == 'SC-D' else ''} & {name} & {net.std() * 252 ** 0.5 * 100:.2f}\\% & {num(g.mean() * 25200, '+.2f')} & "
                         f"{summ[n]['cost_per_year'] * 100:.2f} & {num(net.mean() * 25200, '+.2f')} & {num(ir(g), '.2f')} & {num(ir(net), '.2f')} & "
                         f"{summ[n]['mean_turnover'] * 126:.1f}$\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(lines)


def constraints_table() -> str:
    """Slide 8: which constraints each portfolio on the backtest slides has."""
    ck, no = r"$\checkmark$", "--"
    runs = PAIRS["daily"]
    rows = [("Fully invested: active weights sum to zero", lambda n: True),
            ("Zero beta to the benchmark", lambda n: True),
            (r"Net zero within each size $\times$ liquidity group", lambda n: True)]
    if SHORT:
        rows.append((rf"Short at most {SHORT_PCT}\% per stock ($b + h \ge -{SHORT_PCT}\%$)", lambda n: n[0] == "S"))
    rows.append((r"Long-only: no weight below zero ($b + h \ge 0$)", lambda n: n[0] == "C"))
    if FIXED_DECK:
        rows.append((r"Pays half the bid--ask spread on every trade", is_cost))
    lines = [r"\begin{tabular}{l" + "c" * len(runs) + "}", r"\toprule", "Constraint & " + " & ".join(ROLE) + r" \\", r"\midrule"]
    for label, has in rows:
        lines.append(label + " & " + " & ".join(ck if has(n) else no for n in runs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def gamma_costs_table(size=r"\scriptsize") -> str:
    """Appendix of the fixed-gamma deck: every portfolio with and without trading costs, 2011-2018."""
    g = pl.read_csv(PAN / "gamma_costs.csv")
    lines = [size, r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{llrrrrr}", r"\toprule",
             r" & & Return, \%/yr & IR ($t$) & Risk & Turnover (one-way), $\times$/yr & Spread cost, \%/yr \\", r"\midrule"]
    for k, freq in enumerate(("daily", "monthly")):
        if k:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{freq.capitalize()} rebalancing}}}} \\")
        for r in g.filter(pl.col("frequency") == freq).iter_rows(named=True):
            cost = f"{r['spread_cost_pct']:.2f}" + ("" if r["costs"] == "with costs" else "*")
            lines.append(f"{tex_escape(r['portfolio'])} & {r['costs']} & {num(r['net_pct'], '+.2f')} & {num(r['ir'], '.2f')} ({num(r['t'], '.1f')}) & "
                         f"{r['risk_realized_pct']:.2f}\\% & {r['turnover_per_year'] / 2:.1f} & {cost} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(lines)


def deck() -> str:
    period = "2011--2018 and holdout 2019--2024" if has_holdout() else "2011--2018"
    dv = {n: run_metrics(n, PAN) for n in set(PAIRS["daily"] + PAIRS["monthly"] + [LO_RUN])}
    pct = lambda x: num(x, "+.2f") + r"\%"
    av = pl.read_csv(OPT / "avoid/avoid_summary.csv").filter(pl.col("portfolio").str.starts_with("Liquidity"))
    av_lo, av_hi = av["active_return_pct"].min(), av["active_return_pct"].max()
    short_note = rf" Limited shorting = total weight $\ge -{SHORT_PCT}\%$." if SHORT else ""
    if FIXED_DECK:
        short_note += r" + costs = pays half the spread on each trade; returns are after those costs. Fixed $\gamma$, risk $\le 1\%$."
    hold_note = (r" Holdout: design frozen before it was run; settings re-estimated each January from earlier years only."
                 if has_holdout() else "")
    if has_holdout():
        hd = {n: run_metrics(n, HOLD) for n in set(PAIRS["daily"] + PAIRS["monthly"] + [LO_RUN])}
        after = hd[LO_RUN]["net"] if FIXED_DECK else hd["C-D"]["net"] - j(HOLD / "implied_costs.json")["C-D"]["cost_pct_per_year"]
        se = 6 ** -0.5
        bottom = (r"\item \textbf{Real:} flagged stocks lag similar stocks by about 3\% over a year" + "\n"
                  + r"\item \textbf{Just avoiding them:} right direction, too noisy" + "\n"
                  + r"\item \textbf{Optimizer, long-only" + (" after costs" if FIXED_DECK else "") + r":} " + pct(hd[LO_RUN]["net"])
                  + f" a year, IR {hd[LO_RUN]['ir']:.2f}, on held-back 2019--24" + "\n"
                  + r"\item \textbf{Small:} one input to the fund's alpha")
        trust = (r"\item Built on 2011--18, frozen, then run on 2019--24 \textbf{once}" + "\n"
                 + r"\item Long-only after costs, 2019--24: " + pct(after) + f" a year, IR {hd[LO_RUN]['ir']:.2f}" + "\n"
                 + r"\item Six years is short: true IR likely between " + num(hd[LO_RUN]['ir'] - 2 * se, ".1f") + " and " + num(hd[LO_RUN]['ir'] + 2 * se, ".1f"))
    else:
        bottom = (r"\item \textbf{Real:} flagged stocks lag similar stocks by about 3\% over a year" + "\n"
                  + r"\item \textbf{Just avoiding them:} right direction, too noisy" + "\n"
                  + r"\item \textbf{Optimizer, long-only" + (" after costs" if FIXED_DECK else "") + r":} " + pct(dv[LO_RUN]["net"])
                  + f" a year, IR {dv[LO_RUN]['ir']:.2f}" + "\n"
                  + r"\item \textbf{Small:} one input to the fund's alpha")
        trust = r"\item Built on 2011--18; 2019--24 held back for a final test"
    return r"""% Generated by scripts/build_slides.py from saved results.
\documentclass[aspectratio=169,11pt]{beamer}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amssymb}
\definecolor{banner}{HTML}{1B3A6B}
\definecolor{accent}{HTML}{2E6DB4}
\setbeamercolor{frametitle}{bg=banner,fg=white}
\setbeamercolor{title}{fg=banner}
\setbeamercolor{structure}{fg=banner}
\setbeamercolor{item}{fg=accent}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}[frame number]
\newcommand{\slide}[1]{\begin{frame}[plain]\makebox[\textwidth]{\includegraphics[width=\paperwidth]{figures/#1}}\end{frame}}

\title{When Trading Dries Up}
\subtitle{A liquidity signal for a long-only fund}
\author{Group 7 \\ Silver Fund}
\date{Built on 2011--2018""" + (r", tested on held-back 2019--2024" if has_holdout() else "") + r"""}

\begin{document}
\begin{frame}\titlepage\end{frame}

\begin{frame}{The idea}
\begin{itemize}
\item Most investors avoid illiquid stocks
\item When a stock's trading collapses, it must pass to a few patient buyers
\item That hand-off is slow, so the price \textbf{drifts down for months}
\item Shorting these stocks is costly, so nobody trades the drift away
\end{itemize}
\end{frame}

\begin{frame}{What is a liquidity event?}
\textbf{Illiquidity} (Amihud): price move per dollar traded, last 21 days
\[ \mathrm{ILLIQ}_{i,t} = \frac{1}{21}\sum_{d=t-20}^{t} \frac{|r_{i,d}|}{\text{dollar volume}_{i,d}} \]
\textbf{Shock}: against the stock's own past year
\[ x_{i,t} = \log \mathrm{ILLIQ}_{i,t} - \frac{1}{12}\sum_{k=1}^{12} \log \mathrm{ILLIQ}_{i,t-21k} \]
\textbf{Remove the fund's six scores} $f_{i,t}$, then standardize across stocks
\[ x_{i,t} = b_t^\top f_{i,t} + e_{i,t}, \qquad z_{i,t} = \frac{e_{i,t} - \bar e_t}{\mathrm{sd}_t(e)} \]
\textbf{Event}: $z_{i,t}$ in the day's top 5\%, with no flag in the previous 21 trading days
\end{frame}

\begin{frame}{Bottom line}
\begin{itemize}
""" + bottom + r"""
\end{itemize}
\end{frame}

\slide{matched_drift.png}
\slide{where.png}

\begin{frame}{It is not something we already know}
\centering
""" + regression_table() + r"""

\medskip
\raggedright\footnotesize Positive = flagged stocks earn less. Monthly regressions, 2008--2018, controlling for size, prior illiquidity, momentum and long-term reversal.
\end{frame}

\begin{frame}{Just avoiding them points the right way, but is mostly noise}
\centering
""" + avoid_compare_table(r"\small") + r"""

\medskip
\raggedright
\begin{itemize}
\item Stock-specific part is positive: the flagged stocks did lag
\item Reweighting 8--9\% of the index adds factor tilts: about half the risk, about no return
\item The optimizer hedges the tilts and sizes each bet by expected drift
\end{itemize}
\end{frame}

\begin{frame}{How we bet on it}
\textbf{Our alpha} (Grinold--Kahn form): \quad $\alpha_i = \sigma_i \times \bar d_{\,\text{age of event}}$
\begin{itemize}\small
\item $\sigma_i$: volatility \quad $\bar d$: past drift vs similar stocks, per unit of volatility \quad no recent event: $\alpha = 0$
\end{itemize}
\medskip
""" + (r"""\textbf{Optimizer:} \quad maximize $h^\top\alpha - \tfrac{\gamma}{2}\,h^\top V h - \lambda\cdot\text{spread costs}$, \ active risk $\le 1\%$""" if FIXED_DECK else r"""\textbf{Optimizer:} \quad maximize $h^\top\alpha$ \ at a fixed active risk of 5\%""") + r"""
\medskip

\centering\small
""" + (r"\resizebox{0.92\linewidth}{!}{" + constraints_table() + "}" if FIXED_DECK else constraints_table()) + r"""

\medskip
\end{frame}

\slide{vol_check.png}

\begin{frame}{Daily rebalancing, """ + period + r"""}
\begin{columns}[T]
\begin{column}{""" + ("0.40" if len(PAIRS["daily"]) > 2 else "0.50") + r"""\textwidth}
\includegraphics[width=\linewidth]{figures/pair_daily""" + SUFFIX + r""".png}
\end{column}
\begin{column}{""" + ("0.59" if len(PAIRS["daily"]) > 2 else "0.48") + r"""\textwidth}
\vspace{10mm}
""" + pair_table("daily") + r"""
\end{column}
\end{columns}
\vspace{1mm}
{\scriptsize Lines scaled to equal risk so their shapes compare; the table has the real numbers.""" + (
    r" Long-only runs at only 0.1--0.3\% risk: the chart scales it up about 5--10$\times$, which it can't do in practice." if FIXED_DECK else "") + r"""\par}
\end{frame}

\begin{frame}{Monthly rebalancing, """ + period + r"""}
\begin{columns}[T]
\begin{column}{""" + ("0.40" if len(PAIRS["daily"]) > 2 else "0.50") + r"""\textwidth}
\includegraphics[width=\linewidth]{figures/pair_monthly""" + SUFFIX + r""".png}
\end{column}
\begin{column}{""" + ("0.59" if len(PAIRS["daily"]) > 2 else "0.48") + r"""\textwidth}
\vspace{10mm}
""" + pair_table("monthly") + r"""
\end{column}
\end{columns}
\vspace{1mm}
{\scriptsize Lines scaled to equal risk so their shapes compare; the table has the real numbers.""" + (
    r" Long-only runs at only 0.1--0.3\% risk: the chart scales it up about 5--10$\times$, which it can't do in practice." if FIXED_DECK else "") + r"""\par}
\end{frame}

\begin{frame}{How much to trust it}
\begin{itemize}
""" + trust + r"""
\end{itemize}
\end{frame}

\begin{frame}{Conclusion}
\begin{itemize}
\item When trading collapses, the stock lags for months
\item A long-only fund can capture it, with the optimizer
\item It is small: one input to the fund's alpha
\item Next: combine it with the fund's six scores
\end{itemize}
\end{frame}

\begin{frame}[plain]\vfill\centering{\Large\textbf{\color{banner}Appendix}}\vfill\end{frame}

""" + (r"""\begin{frame}{Appendix: why does long-only have the higher IR?}
\centering
""" + ir_puzzle_table() + r"""

\medskip
\raggedright
\begin{itemize}
\item Before costs, the two have about the same IR
\item Limited shorting trades about 6$\times$ a year, mostly in small stocks: spreads take about half its return
\item Long-only trades about 0.5$\times$ a year in larger stocks: spreads cost almost nothing
\item Long-only takes only 0.1--0.3\% risk: higher IR after costs, but less return
\end{itemize}
\end{frame}

""" if FIXED_DECK else "") + r"""
\slide{ic_by_year.png}

\begin{frame}{Appendix: what did we change, and when?}
\centering
""" + versions_table(r"\scriptsize") + r"""

\medskip
\raggedright\footnotesize
\begin{itemize}
\item v2: each event gets an age \quad v3: no positive alpha; hedge with similar stocks \quad v4: alpha scales with volatility
\end{itemize}
\end{frame}

""" + (r"""\begin{frame}{Appendix: how we picked the short limit (2011--2018)}
\centering
""" + short_grid_table() + r"""

\medskip
\raggedright\footnotesize IRs are within noise of each other, so we picked the limit by risk (bold). Earlier design, 5\% risk.
\end{frame}

""" if SHORT else "") + (r"""\begin{frame}{Appendix: every portfolio with and without trading costs (2011--2018)}
\centering
""" + gamma_costs_table() + r"""

\medskip
\raggedright\footnotesize * = spread cost if it were traded, not charged.
\end{frame}

""" if FIXED_DECK else "") + (r"""\begin{frame}{Appendix: why is the long-only result bigger in 2019--24?}
\begin{columns}[T]
\begin{column}{0.48\textwidth}
\centering
""" + size_table() + r"""
\end{column}
\begin{column}{0.48\textwidth}
\footnotesize
\begin{itemize}
\item Large stocks had no alpha until 2015
\item Later they did, so there was more weight to sell
\end{itemize}
\end{column}
\end{columns}
\end{frame}
""" if not FIXED_DECK else "") + r"""
\begin{frame}{Appendix: where does the return come from? (2011--2018)}
\centering
""" + age_table() + r"""

\medskip
\raggedright\footnotesize Return by the age of the event behind each position.
\end{frame}
\end{document}
"""


def extend_pairs() -> None:
    """Fixed-gamma deck: if long-only with costs earns about nothing (t < 1 at either frequency), add long-only without
    costs as a fourth portfolio, so the slides show both that the signal works and what costs take."""
    if not any(run_metrics(n, PAN)["t"] < 1 for n in ("CC-D", "CC-M")):
        return
    PAIRS["daily"].append("C-D")
    PAIRS["monthly"].append("C-M")
    ROLE.append("No shorting, no costs")
    ROLE_COLOR.append(GREY)


if __name__ == "__main__" and FIXED_DECK:
    extend_pairs()
    fig_pair("daily")
    fig_pair("monthly")
    (SLIDES / "slides_fixed_gamma.tex").write_text(deck())
    print("built slides_fixed_gamma.tex", PAIRS, flush=True)
elif __name__ == "__main__":
    for paper in (False, True):
        fig_ic_by_year(paper)
        fig_matched_drift(paper)
        fig_where(paper)
        fig_vol_check(paper)
    fig_pair("daily")
    fig_pair("monthly")
    tables = ROOT / "paper/tables"
    tables.mkdir(parents=True, exist_ok=True)
    (SLIDES / "slides.tex").write_text(deck())
    (tables / "regressions.tex").write_text(regression_table())
    (tables / "versions.tex").write_text(versions_table())
    (tables / "size.tex").write_text(size_table(r"\small"))
    (tables / "results.tex").write_text(paper_results_table())
    print("built", flush=True)
