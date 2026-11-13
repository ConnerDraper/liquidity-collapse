"""Robustness checks for the liquidity shock (fund scores removed). Development years only (2006-2018).

1. Size: benchmark-weighted bottom-decile gap; IC and gap within size terciles.
2. Overlap: residualize on last month's return and |return|; earnings vs non-earnings months.
3. Decay: IC and bottom-decile gap 1-12 months ahead (targets capped at Dec 2018).
4. Components: price-impact part vs volume part of the shock.
5. Barra residualization: on Barra style + industry exposures, alone and with the fund's scores.
"""

import datetime as dt
import json
import sys

import numpy as np
import polars as pl
import sf_quant.data as sfd

from liquidity_collapse.config import DOWNLOADS, FUND_SCORES, PANEL, RESULTS, SIGNALS, connect
from liquidity_collapse.evaluate import nw_t
from liquidity_collapse.links import permno_to_barrid
from liquidity_collapse.signals.factor_momentum import COUNTRY, STYLE

DEV_START, DEV_END = dt.date(2006, 1, 1), dt.date(2018, 12, 31)
# usage: liquidity_checks.py [RESID RAW]  (signal files in output/signals; defaults = the monthly signal)
RESID = sys.argv[1] if len(sys.argv) > 1 else "liquidity_shock_resid"
RAW = sys.argv[2] if len(sys.argv) > 2 else "liquidity_shock"
OUT = RESULTS / ("liquidity_checks_dev.json" if RESID == "liquidity_shock_resid" else f"liquidity_checks_{RESID}_dev.json")


# ---------- helpers ----------
def finite(cols: list[str]) -> pl.Expr:
    return pl.all_horizontal([pl.col(c).is_finite() for c in cols])


def residual(frame: pl.DataFrame, y: str, xs: list[str], name: str) -> pl.DataFrame:
    """Per-date OLS residual of y on xs (with constant); rows missing any input are dropped."""
    usable = frame.drop_nulls([y, *xs]).filter(finite([y, *xs]))

    def one(day: pl.DataFrame) -> pl.DataFrame:
        if day.height < len(xs) + 20:
            return day.head(0).with_columns(pl.lit(None, pl.Float64).alias(name))
        x = np.column_stack([np.ones(day.height), day.select(xs).to_numpy()])
        v = day[y].to_numpy()
        return day.with_columns(pl.Series(name, v - x @ np.linalg.lstsq(x, v, rcond=None)[0]))

    return usable.group_by("date", maintain_order=True).map_groups(one)


def ic(frame: pl.DataFrame, score: str, target: str = "target21") -> dict:
    s = frame.drop_nulls([score, target]).group_by("date").agg(
        pl.corr(score, target, method="spearman").alias("ic")).drop_nulls().sort("date")["ic"].to_numpy()
    return {"ic": float(s.mean()) if len(s) else None, "t": nw_t(s) if len(s) > 2 else None, "months": int(len(s))}


def gap(frame: pl.DataFrame, score: str, target: str = "target21", weight: str | None = None) -> dict:
    """Bottom-decile mean target minus the universe mean, per month (in %/month), equal- or weight-weighted."""
    d = frame.drop_nulls([score, target]).with_columns(
        bottom=((pl.col(score).rank("ordinal").over("date") - 1) * 10 // pl.len().over("date")) == 0,
        _w=pl.col(weight) if weight else pl.lit(1.0),
    )
    w = pl.col("_w")
    m = d.group_by("date").agg(
        ((pl.col(target) * w).filter(pl.col("bottom")).sum() / w.filter(pl.col("bottom")).sum()).alias("b"),
        ((pl.col(target) * w).sum() / w.sum()).alias("a"),
        (w.filter(pl.col("bottom")).sum() / w.sum()).alias("share"),
    ).drop_nulls().sort("date")
    if m.height == 0:
        return {"gap_pct_month": None, "t": None, "months": 0, "bottom_decile_weight_share": None}
    x = (m["b"] - m["a"]).to_numpy()
    return {"gap_pct_month": float(x.mean() * 100), "t": nw_t(x), "months": int(len(x)),
            "bottom_decile_weight_share": float(m["share"].mean())}


def both(frame: pl.DataFrame, score: str, **kw) -> dict:
    if frame.drop_nulls([score, kw.get("target", "target21")]).height == 0:
        return {"rows": 0}
    return {**ic(frame, score, kw.get("target", "target21")), **{f"gap_{k}" if k != "gap_pct_month" else k: v for k, v in gap(frame, score, **kw).items()}}


# ---------- data ----------
def base() -> pl.DataFrame:
    panel = pl.read_parquet(PANEL, columns=["date", "barrid", "eligible", "target21", "market_cap", "benchmark_weight", *FUND_SCORES])
    raw = pl.read_parquet(SIGNALS / f"{RAW}.parquet").rename({"score": "raw"})
    res = pl.read_parquet(SIGNALS / f"{RESID}.parquet")
    return (
        panel.filter(pl.col("eligible"))
        .join(raw, on=["date", "barrid"], how="left").join(res, on=["date", "barrid"], how="left")
    )


def monthly_components(barrids: pl.Series) -> pl.DataFrame:
    """Per barrid-month: log mean |return|, log mean dollar volume, month log return."""
    frames = []
    for year in range(2004, DEV_END.year + 1):
        d = sfd.load_assets(dt.date(year, 1, 1), dt.date(year, 12, 31), ["date", "barrid", "return", "price", "daily_volume"],
                            in_universe=False).filter(pl.col("barrid").is_in(barrids.implode()))
        frames.append(
            d.filter(pl.col("daily_volume") > 0, pl.col("price") > 0, pl.col("return").is_not_null())
            .with_columns(r=pl.col("return") / 100, dv=pl.col("price") * pl.col("daily_volume"))
            .group_by("barrid", pl.col("date").dt.truncate("1mo").alias("month"))
            .agg(pl.col("r").abs().mean().alias("abs_r"), pl.col("dv").mean().alias("dv"),
                 pl.col("r").clip(lower_bound=-0.99).log1p().sum().alias("month_ret"), pl.len().alias("days"))
            .filter(pl.col("days") >= 15, pl.col("abs_r") > 0)
        )
        print(f"loaded {year}", flush=True)
    m = pl.concat(frames).sort("barrid", "month").with_columns(
        log_abs=pl.col("abs_r").log(), log_dv=pl.col("dv").log())
    return m.with_columns(
        price_part=-(pl.col("log_abs") - pl.col("log_abs").rolling_mean(12, min_samples=9).shift(1).over("barrid")),
        volume_part=pl.col("log_dv") - pl.col("log_dv").rolling_mean(12, min_samples=9).shift(1).over("barrid"),
        abs_month_ret=pl.col("month_ret").abs(),
    ).select("barrid", "month", "price_part", "volume_part", "month_ret", "abs_month_ret")


def earnings_months(dates: pl.Series) -> pl.DataFrame:
    """(date, barrid) pairs where the firm reported earnings (Compustat rdq) in the formation month."""
    fq = (
        pl.read_csv(DOWNLOADS / "compustat" / "compustat_fundq.csv", infer_schema=False, columns=["gvkey", "rdq"])
        .with_columns(pl.col("gvkey").str.zfill(6), pl.col("rdq").str.to_date(strict=False)).drop_nulls().unique()
    )
    ccm = (
        pl.read_csv(DOWNLOADS / "crsp" / "ccm_link.csv", infer_schema=False)
        .filter(pl.col("LINKTYPE").is_in(["LU", "LC"]), pl.col("LINKPRIM").is_in(["P", "C"]))
        .select("gvkey", pl.col("LPERMNO").cast(pl.Int64).alias("permno"),
                pl.col("LINKDT").str.to_date(strict=False).alias("s"),
                pl.col("LINKENDDT").str.to_date(strict=False).fill_null(pl.date(2099, 12, 31)).alias("e"))
    )
    linked = fq.join(ccm, on="gvkey").filter(pl.col("rdq").is_between(pl.col("s"), pl.col("e"))).select(
        pl.col("rdq").alias("date"), "permno")
    linked = permno_to_barrid(linked).select(pl.col("date").dt.truncate("1mo").alias("month"), "barrid").unique()
    formations = pl.DataFrame({"date": dates}).with_columns(month=pl.col("date").dt.truncate("1mo"))
    return linked.join(formations, on="month").select("date", "barrid").with_columns(earnings=pl.lit(True))


def barra_exposures(dates: pl.Series) -> tuple[pl.DataFrame, list[str], list[str]]:
    frames, styles, industries = [], set(), set()
    for date_ in dates:
        e = sfd.load_exposures_by_date(date_)
        s = [c for c in e.columns if c in STYLE]
        i = [c for c in e.columns if c.startswith("USSLOWL_") and c not in STYLE and c != COUNTRY]
        styles.update(s); industries.update(i)
        frames.append(e.select("barrid", *s, *i).with_columns(date=pl.lit(date_)))
    exp = pl.concat(frames, how="diagonal_relaxed")
    styles, industries = sorted(styles), sorted(industries)
    # Missing style exposure = no reading; exposures are standardized, so 0 is the average.
    return exp.with_columns([pl.col(c).fill_null(0.0) for c in [*styles, *industries]]), styles, industries


# ---------- checks ----------
def main() -> None:
    connect()
    data = base().filter(pl.col("date").is_between(DEV_START, DEV_END))
    scored = data.filter(pl.col("score").is_not_null(), pl.col("target21").is_not_null())
    result: dict = {"signal": RESID, "baseline": both(scored, "score"),
                    "baseline_raw_signal": both(data.filter(pl.col("raw").is_not_null(), pl.col("target21").is_not_null()), "raw")}

    if RESID != "liquidity_shock_resid":
        monthly = pl.read_parquet(SIGNALS / "liquidity_shock_resid.parquet").rename({"score": "monthly"})
        both_ = scored.join(monthly, on=["date", "barrid"])
        result["rank_corr_with_monthly_signal"] = float(both_.group_by("date").agg(
            pl.corr("score", "monthly", method="spearman").alias("r"))["r"].mean())

    # 1. size
    b = scored.filter(pl.col("benchmark_weight").is_not_null())
    sized = scored.with_columns(
        size=((pl.col("market_cap").rank("ordinal").over("date") - 1) * 3 // pl.len().over("date")).cast(pl.Int8))
    result["size"] = {
        "benchmark_weighted (benchmark names)": gap(b, "score", weight="benchmark_weight"),
        "equal_weighted (benchmark names)": gap(b, "score"),
        **{f"tercile_{['small', 'mid', 'large'][k]}": both(sized.filter(pl.col("size") == k), "score") for k in range(3)},
    }
    print("size done", flush=True)
    OUT.write_text(json.dumps(result, indent=1, default=str))

    # 2. last month's move, 4. components
    comps = monthly_components(data["barrid"].unique())
    data2 = scored.with_columns(month=pl.col("date").dt.truncate("1mo")).join(comps, on=["barrid", "month"], how="left")
    r1 = residual(data2, "score", ["month_ret", "abs_month_ret"], "score_ex_move")
    r2 = residual(data2, "raw", [*FUND_SCORES, "month_ret", "abs_month_ret"], "raw_ex_fund_move")
    result["last_month_move"] = {
        "signal_vs_month_return_rank_corr": float(data2.drop_nulls(["score", "month_ret"]).select(
            pl.corr("score", "month_ret", method="spearman")).item()),
        "signal_vs_abs_month_return_rank_corr": float(data2.drop_nulls(["score", "abs_month_ret"]).select(
            pl.corr("score", "abs_month_ret", method="spearman")).item()),
        "after removing last month's return and |return|": both(r1, "score_ex_move"),
        "raw shock after fund scores + last month's move": both(r2, "raw_ex_fund_move"),
    }
    earn = earnings_months(data["date"].unique().sort())
    data3 = scored.join(earn, on=["date", "barrid"], how="left").with_columns(pl.col("earnings").fill_null(False))
    result["earnings"] = {
        "share_of_stock_months_with_earnings": float(data3["earnings"].mean()),
        "earnings months": both(data3.filter(pl.col("earnings")), "score"),
        "non-earnings months": both(data3.filter(~pl.col("earnings")), "score"),
    }
    print("overlap done", flush=True)
    OUT.write_text(json.dumps(result, indent=1, default=str))
    pp = residual(data2, "price_part", FUND_SCORES, "price_resid")
    vp = residual(data2, "volume_part", FUND_SCORES, "volume_resid")
    result["components"] = {
        "price-impact part (bigger moves), fund scores removed": both(pp, "price_resid"),
        "volume part (more trading = good), fund scores removed": both(vp, "volume_resid"),
        "rank_corr_signal_vs_price_part": float(data2.drop_nulls(["score", "price_part"]).select(
            pl.corr("score", "price_part", method="spearman")).item()),
        "rank_corr_signal_vs_volume_part": float(data2.drop_nulls(["score", "volume_part"]).select(
            pl.corr("score", "volume_part", method="spearman")).item()),
    }
    print("components done", flush=True)
    OUT.write_text(json.dumps(result, indent=1, default=str))

    # 3. decay: target k months ahead = target21 at the formation k-1 months later (same barrid), capped at DEV_END
    panel = pl.read_parquet(PANEL, columns=["date", "barrid", "target21"]).filter(pl.col("date") <= DEV_END).sort("barrid", "date")
    dates = panel["date"].unique().sort()
    index = pl.DataFrame({"date": dates, "i": np.arange(len(dates))})
    future = panel.join(index, on="date")
    decay = {}
    for k in (1, 2, 3, 6, 12):
        ahead = future.with_columns((pl.col("i") - (k - 1)).alias("i")).rename({"target21": "ahead"}).drop("date")
        d = scored.join(index, on="date").join(ahead, on=["barrid", "i"], how="left")
        decay[f"{k} month(s) ahead"] = both(d, "score", target="ahead")
    result["decay"] = decay
    print("decay done", flush=True)
    OUT.write_text(json.dumps(result, indent=1, default=str))

    # 5. Barra residualization of the raw shock
    exp, styles, industries = barra_exposures(data["date"].unique().sort())
    data5 = data.filter(pl.col("raw").is_not_null(), pl.col("target21").is_not_null()).join(exp, on=["date", "barrid"], how="inner")
    rb = residual(data5, "raw", [*styles, *industries], "raw_ex_barra")
    rbf = residual(data5, "raw", [*styles, *industries, *FUND_SCORES], "raw_ex_barra_fund")
    rs = residual(data5, "score", [*styles, *industries], "resid_ex_barra")
    print({"data5": data5.height, "data5_with_score": data5["score"].is_not_null().sum(), "rb": rb.height, "rbf": rbf.height,
           "rs": rs.height, "rs_dates": rs["date"].n_unique() if rs.height else 0,
           "styles_null_share": float(data5.select(pl.mean_horizontal([pl.col(c).is_null().cast(pl.Float64) for c in styles])).mean().item())},
          flush=True)
    result["barra"] = {
        "styles": len(styles), "industries": len(industries),
        "raw shock after Barra styles + industries": both(rb, "raw_ex_barra"),
        "raw shock after Barra + fund scores": both(rbf, "raw_ex_barra_fund"),
        "signal (fund scores removed) after Barra too": both(rs, "resid_ex_barra"),
    }
    print("barra done", flush=True)

    OUT.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
