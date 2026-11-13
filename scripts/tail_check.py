"""Where does a signal's IC come from: one tail, or the whole distribution?

Monthly rank ICs (score vs next 21-day specific return) on subsets of the score
distribution, plus tail-only indicators. Development years only by default.

usage: tail_check.py NAME [NAME ...]   NAME = a signal file in output/signals, or
       panel:COLUMN (a panel column, e.g. the fund's ivol score), or neg:COLUMN (sign flipped).
"""

import json
import sys

import numpy as np
import polars as pl

from liquidity_collapse.config import PANEL, RESULTS, SIGNALS, windows
from liquidity_collapse.evaluate import nw_t

def monthly_ic(frame: pl.DataFrame, score: str = "score") -> np.ndarray:
    ic = frame.group_by("date").agg(pl.corr(score, "target21", method="spearman").alias("ic")).drop_nulls()
    return ic.sort("date")["ic"].to_numpy()


def stat(x: np.ndarray) -> dict:
    return {"mean": float(np.mean(x)), "t": nw_t(x), "months": int(len(x))}


def scores(name: str) -> tuple[list[str], pl.Expr | None, pl.DataFrame | None]:
    if ":" in name:
        kind, column = name.split(":")
        return [column], (-pl.col(column) if kind == "neg" else pl.col(column)).alias("score"), None
    return [], None, pl.read_parquet(SIGNALS / f"{name}.parquet")


def run(name: str, sample: str) -> dict:
    lo, hi = windows(sample)["ic"]
    extra, expression, signal = scores(name)
    data = pl.read_parquet(PANEL, columns=["date", "barrid", "eligible", "target21", *extra]).filter(
        pl.col("eligible"), pl.col("date").is_between(lo, hi), pl.col("target21").is_not_null()
    )
    data = data.with_columns(expression).drop_nulls("score") if signal is None else data.join(signal, on=["date", "barrid"])
    data = (
        data.filter(pl.col("score").is_finite())
        .with_columns(decile=((pl.col("score").rank("ordinal").over("date") - 1) * 10 // pl.len().over("date") + 1))
        .with_columns(tail=-(pl.col("decile") == 1).cast(pl.Float64), top=(pl.col("decile") == 10).cast(pl.Float64))
    )
    out = {
        "all stocks": stat(monthly_ic(data)),
        "without bottom decile (2-10)": stat(monthly_ic(data.filter(pl.col("decile") > 1))),
        "without top decile (1-9)": stat(monthly_ic(data.filter(pl.col("decile") < 10))),
        "middle 80% (2-9)": stat(monthly_ic(data.filter(pl.col("decile").is_between(2, 9)))),
        "inside bottom decile": stat(monthly_ic(data.filter(pl.col("decile") == 1))),
        "inside top decile": stat(monthly_ic(data.filter(pl.col("decile") == 10))),
        "tail-only indicator (-1 bottom decile, else 0)": stat(monthly_ic(data, "tail")),
        "top-only indicator (+1 top decile, else 0)": stat(monthly_ic(data, "top")),
    }
    means = data.group_by("date").agg(
        (pl.col("target21").filter(pl.col("decile") == 1).mean() - pl.col("target21").mean()).alias("bottom"),
        (pl.col("target21").filter(pl.col("decile") == 10).mean() - pl.col("target21").mean()).alias("top"),
    ).sort("date")
    out["bottom decile return vs average, %/month"] = {k: v * 100 if k == "mean" else v for k, v in stat(means["bottom"].to_numpy()).items()}
    out["top decile return vs average, %/month"] = {k: v * 100 if k == "mean" else v for k, v in stat(means["top"].to_numpy()).items()}
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        rows = run(name, "dev")
        (RESULTS / f"tail_check_{name.replace(':', '_')}_dev.json").write_text(json.dumps(rows, indent=1))
        print(f"== {name} (development years)")
        for label, r in rows.items():
            print(f"{label:52s} {r['mean']:+.4f}  t={r['t']:+.2f}  months={r['months']}")
