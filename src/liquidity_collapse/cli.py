"""lab panel | lab build NAME | lab evaluate NAME [--sample] | lab summary | lab submit NAME... [--panel] [--sample]"""

import argparse
import datetime as dt
import subprocess

from .config import CFG, LOGS, OUTPUT, PROJECT, SIGNALS


def sbatch(job: str, commands: list[str]) -> str:
    s = CFG["slurm"]
    LOGS.mkdir(parents=True, exist_ok=True)
    return "\n".join([
        "#!/bin/bash", f"#SBATCH --job-name=lab_{job}",
        f"#SBATCH --output={LOGS}/{job}_%j.out", f"#SBATCH --error={LOGS}/{job}_%j.err",
        "#SBATCH --ntasks=1", f"#SBATCH --cpus-per-task={s['cpus']}", f"#SBATCH --mem={s['mem']}",
        f"#SBATCH --time={s['time']}", "", "set -euo pipefail", "ulimit -c 0",
        *[f"export {v}={s['cpus']}" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS")],
        f"cd {PROJECT}", *commands, "",
    ])


def submit(names: list[str], sample: str, panel: bool, dry_run: bool) -> None:
    def send(job: str, commands: list[str], after: str | None) -> str:
        path = PROJECT / "slurm" / f"{job}.sbatch"
        path.parent.mkdir(exist_ok=True)
        path.write_text(sbatch(job, commands))
        cmd = ["sbatch", "--parsable", *([f"--dependency=afterok:{after}"] if after else []), str(path)]
        if dry_run:
            print(" ".join(cmd))
            return f"<{job}>"
        job_id = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
        print(f"{job}: {job_id}", flush=True)
        return job_id

    first = send("panel", [".venv/bin/lab panel"], None) if panel else None
    for name in names:
        send(f"{name}_{sample}", [f".venv/bin/lab build {name}", f".venv/bin/lab evaluate {name} --sample {sample}",
                                  ".venv/bin/lab summary"], first)




def submit_optimizer(smoke: bool, dry_run: bool, no_cache: bool = False) -> None:
    """Chains the optimizer study (plan: make-a-plan-for-hidden-russell.md). Smoke = Jan-Mar 2011 only."""
    from .optimize import backtest_tasks, smoke_tasks
    big = {"cpus": 4, "mem": "128G", "time": "04:00:00"}
    mid = {"cpus": 2, "mem": "48G", "time": "06:00:00"}
    run = {"cpus": 2, "mem": "48G", "time": "04:00:00" if smoke else "36:00:00"}

    def send(job, commands, res, after=None, array=None):
        text = sbatch(job, commands)
        text = text.replace(f"--cpus-per-task={CFG['slurm']['cpus']}", f"--cpus-per-task={res['cpus']}")
        text = text.replace(f"--mem={CFG['slurm']['mem']}", f"--mem={res['mem']}").replace(f"--time={CFG['slurm']['time']}", f"--time={res['time']}")
        if array:
            text = text.replace("#SBATCH --ntasks=1", f"#SBATCH --ntasks=1\n#SBATCH --array={array}").replace("_%j.", "_%A_%a.")
        path = PROJECT / "slurm" / f"{job}.sbatch"
        path.write_text(text)
        cmd = ["sbatch", "--parsable", *([f"--dependency=afterok:{after}"] if after else []), str(path)]
        if dry_run:
            print(" ".join(cmd)); return f"<{job}>"
        job_id = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
        print(f"{job}: {job_id}", flush=True)
        return job_id

    tag = "opt_smoke" if smoke else "opt"
    env = "LAB_SMOKE=1 " if smoke else ""
    signal = send(f"{tag}_event", [f"{env}.venv/bin/lab event-params"], big)
    after = signal
    if not no_cache:
        cache = send(f"{tag}_cache", [f".venv/bin/lab cache --task $SLURM_ARRAY_TASK_ID"], mid, array="0-8")
        after = f"{signal}:{cache}"
    tasks = ",".join(map(str, smoke_tasks())) if smoke else f"0-{len(backtest_tasks()) - 1}"
    runs = send(f"{tag}_runs", [f"{env}.venv/bin/lab backtest --task $SLURM_ARRAY_TASK_ID"], run, after=after, array=tasks)
    pick = send(f"{tag}_lambda", [f"{env}.venv/bin/lab select-lambda"], mid, after=runs)
    final = send(f"{tag}_final", [f"{env}.venv/bin/lab final --task $SLURM_ARRAY_TASK_ID"], run, after=pick, array="0-1")
    send(f"{tag}_results", [f"{env}.venv/bin/lab tc", f"{env}.venv/bin/lab exposures", f"{env}.venv/bin/lab check"], big, after=final)


def main() -> None:
    parser = argparse.ArgumentParser(prog="lab")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("panel")
    sub.add_parser("build").add_argument("name")
    p = sub.add_parser("evaluate"); p.add_argument("name"); p.add_argument("--sample", default="dev")
    sub.add_parser("summary")
    sub.add_parser("daily-signal"); sub.add_parser("params"); sub.add_parser("select-lambda")
    sub.add_parser("tc"); sub.add_parser("exposures")
    p = sub.add_parser("cache"); p.add_argument("--task", type=int, required=True)
    p = sub.add_parser("backtest"); p.add_argument("--task", type=int, required=True)
    sub.add_parser("event-params"); sub.add_parser("check")
    p = sub.add_parser("final"); p.add_argument("--task", type=int, required=True)
    p = sub.add_parser("submit-optimizer"); p.add_argument("--smoke", action="store_true"); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-cache", action="store_true", help="risk-model cache already built")
    p = sub.add_parser("compare"); p.add_argument("base"); p.add_argument("candidate"); p.add_argument("--sample", default="dev")
    p = sub.add_parser("submit")
    p.add_argument("names", nargs="+"); p.add_argument("--sample", default="dev")
    p.add_argument("--panel", action="store_true"); p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "submit":
        return submit(args.names, args.sample, args.panel, args.dry_run)
    if args.command == "submit-optimizer":
        return submit_optimizer(args.smoke, args.dry_run, args.no_cache)

    from .config import connect
    connect()
    if args.command == "panel":
        from .panel import build
        build()
    elif args.command == "build":
        import polars as pl
        from .config import PANEL
        from .signals import REGISTRY
        scores = REGISTRY[args.name](pl.read_parquet(PANEL))
        SIGNALS.mkdir(parents=True, exist_ok=True)
        scores.write_parquet(SIGNALS / f"{args.name}.parquet")
        print({"signal": args.name, "rows": scores.height, "dates": scores["date"].n_unique()}, flush=True)
    elif args.command == "evaluate":
        if args.sample == "holdout":
            with open(OUTPUT / "holdout_openings.log", "a") as log:
                log.write(f"{dt.datetime.now().isoformat()} {args.name}\n")
        from .evaluate import run
        run(args.name, args.sample)
    elif args.command == "compare":
        from .evaluate import compare
        compare(args.base, args.candidate, args.sample)
    elif args.command == "daily-signal":
        from .daily_signal import build as daily_build
        daily_build()
    elif args.command == "params":
        from .daily_signal import params as daily_params
        daily_params()
    elif args.command == "cache":
        from .optimize import cache_year
        from .daily_signal import END
        cache_year(list(range(2010, END.year + 1))[args.task])
    elif args.command == "backtest":
        from .optimize import backtest
        backtest(args.task)
    elif args.command == "select-lambda":
        from .optimize import select_lambda
        select_lambda()
    elif args.command == "final":
        from .optimize import final
        final(args.task)
    elif args.command == "event-params":
        from .event_alpha import params as event_params
        event_params()
    elif args.command == "check":
        from .optimize import check
        check()
    elif args.command == "tc":
        from .optimize import tc
        tc()
    elif args.command == "exposures":
        from .exposures import run as exposures_run
        exposures_run()
    elif args.command == "summary":
        from .evaluate import collect
        print(collect(), flush=True)
