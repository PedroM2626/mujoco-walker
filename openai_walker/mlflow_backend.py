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


__all__ = ["set_uri", "tracking_uri"]
