"""Identifier links into Barra barrids, as of a date."""

import polars as pl
import sf_quant.data as sfd

from .config import DOWNLOADS


def permno_to_barrid(frame: pl.DataFrame, date: str = "date") -> pl.DataFrame:
    """Add `barrid` for each (permno, date); rows whose permno maps to two barrids that day are dropped."""
    link = sfd.load_barrid_permno_link()
    out = (
        frame.join(link, on="permno", how="inner")
        .filter(pl.col(date).is_between(pl.col("start_date"), pl.col("end_date")))
        .drop("start_date", "end_date")
    )
    return out.filter(pl.len().over("permno", date) == 1, pl.len().over("barrid", date) == 1)


def gvkey_to_permno(frame: pl.DataFrame, date: str = "date") -> pl.DataFrame:
    """CRSP/Compustat link on gvkey and issue id (iid), primary links only."""
    ccm = (
        pl.read_csv(DOWNLOADS / "crsp" / "ccm_link.csv", infer_schema=False)
        .filter(pl.col("LINKTYPE").is_in(["LU", "LC"]), pl.col("LINKPRIM").is_in(["P", "C"]))
        .select(
            pl.col("gvkey"), pl.col("LIID").alias("iid"), pl.col("LPERMNO").cast(pl.Int64).alias("permno"),
            pl.col("LINKDT").str.to_date(strict=False).alias("link_start"),
            pl.col("LINKENDDT").str.to_date(strict=False).fill_null(pl.date(2099, 12, 31)).alias("link_end"),
        )
    )
    return (
        frame.join(ccm, on=["gvkey", "iid"], how="inner")
        .filter(pl.col(date).is_between(pl.col("link_start"), pl.col("link_end")))
        .drop("link_start", "link_end")
    )
