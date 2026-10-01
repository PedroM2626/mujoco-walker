"""Single source of truth for the MLflow tracking URI.

The trainers each set their own URI at import time, and two different CWD-relative
 spellings were in use: `sqlite:///mlruns.db` in 11 scripts and `sqlite:///../mlruns.db` in
 5. Run from the repo root those resolve to different files, and run from `openai_walker/`
they resolve differently again - which is why the experiment history ended up split
between a 54 MB `mlruns.db` at the root and a 868 KB one inside `openai_walker/`, with
`mlflow ui --backend-store-uri sqlite:///mlruns.db` showing only half of it.

Anchoring the path to the repository root makes it independent of the working directory.
Set `MLFLOW_TRACKING_URI` to override, e.g. to point at a real tracking server.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "mlruns.db"


def tracking_uri():
    override = os.environ.get("MLFLOW_TRACKING_URI")
    if override:
        return override
    # sqlite:/// expects a path with forward slashes, absolute on every platform.
    return "sqlite:///" + DEFAULT_DB.as_posix()


def set_uri(mlflow_module):
    mlflow_module.set_tracking_uri(tracking_uri())
    return mlflow_module
