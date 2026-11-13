import os
import tomllib
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
CFG = tomllib.loads((PROJECT / "config.toml").read_text())
OUTPUT = PROJECT / CFG["data"]["output"]
FEATURES = Path(CFG["data"]["features"])
RESEARCH = Path(CFG["data"]["research_database"])
DOWNLOADS = Path(CFG["data"]["downloads"])
PANEL = OUTPUT / "panel.parquet"
CALENDAR = OUTPUT / "calendar.parquet"
SIGNALS = OUTPUT / "signals"
RESULTS = OUTPUT / "results"
LOGS = OUTPUT / "logs"
YEARS = range(2005, 2025)
FUND_SCORES = ["ivol", "momentum", "reversal", "barra_momentum", "beta", "barra_reversal"]


def connect() -> None:
    import sf_quant.data as sfd
    sfd.env(root=CFG["data"]["root"], database=CFG["data"]["database"])
    os.environ["ROOT"] = CFG["data"]["root"]
    os.environ["DATABASE"] = CFG["data"]["database"]


def windows(sample: str) -> dict:
    w = CFG["window"]
    if sample == "dev":
        return {"ic": (w["ic_start"], w["dev_end"]), "portfolio": (w["portfolio_start"], w["dev_end"])}
    if sample == "holdout":
        return {"ic": (w["holdout_start"], w["holdout_end"]), "portfolio": (w["holdout_start"], w["holdout_end"])}
    raise ValueError(f"unknown sample {sample}")
