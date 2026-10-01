"""Repository-rooted MLflow backend for the Phase-4 scripts.

Each of these scripts used to hard-code its own tracking URI, in two different
CWD-relative spellings, so the experiment history split across two databases. See
`utils/mlflow_uri.py` for the full story. This module exists so the 16 call sites need one
line each and only this file carries the path bootstrap.

Run-from-`openai_walker/` is the documented convention (see `run_all.bat`), so the repo
root is not on sys.path by default.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.mlflow_uri import tracking_uri  # noqa: E402


def set_uri(experiment=None):
    """Point mlflow at the repo-root mlruns.db, optionally setting the experiment."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    if experiment is not None:
        mlflow.set_experiment(experiment)
    return mlflow


def find_stale_runs(days=7):
    """List runs still marked RUNNING after `days`.

    Every Phase-4 trainer wraps its work in `with mlflow.start_run(...)`, which does close
    the run on a normal exit and on a Python exception. What it cannot do is close it when
    the process is killed - and two of those runs are months old, one with 653 metrics
    already logged (the interrupted session the README describes as a 20-hour restart).
    So this is not a missing end_run bug; it is bookkeeping for killed processes, and it
    matters because an abandoned run that never reached `torch.save` is exactly why an
    artifact is missing while its metrics are in the database.
    """
    import time

    import mlflow

    from mlflow.entities import RunStatus

    mlflow.set_tracking_uri(tracking_uri())
    client = mlflow.MlflowClient()
    cutoff = (time.time() - days * 86400) * 1000
    stale = []
    for exp in client.search_experiments():
        active = RunStatus.to_string(RunStatus.RUNNING)
        for run in client.search_runs([exp.experiment_id], filter_string=f"status = '{active}'"):
            if run.info.start_time and run.info.start_time < cutoff:
                stale.append((exp.name, run.info.run_id, run.info.run_name,
                              run.data.log_metrics_count if hasattr(run.data, "log_metrics_count")
                              else len(run.data.metrics)))
    return stale


def mark_stale_runs_finished(days=7, apply=False):
    """Set stale RUNNING runs to FINISHED. Dry-run by default; these are real records."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    client = mlflow.MlflowClient()
    stale = find_stale_runs(days=days)
    for exp_name, run_id, name, metrics in stale:
        print(f"{'closing' if apply else 'would close'}: {exp_name}/{name} "
              f"({run_id[:8]}, {metrics} metrics logged)")
        if apply:
            client.set_terminated(run_id, status="FINISHED")
    return stale


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Find MLflow runs left RUNNING by killed processes.")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--apply", action="store_true", help="Actually close them (default only reports).")
    a = p.parse_args()
    found = mark_stale_runs_finished(days=a.days, apply=a.apply)
    print(f"\n{len(found)} stale run(s){' closed' if a.apply else ' (dry run; use --apply)'}")


__all__ = ["set_uri", "tracking_uri", "find_stale_runs", "mark_stale_runs_finished"]
