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


def _client(uri=None):
    """MlflowClient against `uri` (default: the repo-root mlruns.db).

    The history is currently split across two sqlite files, so bookkeeping has to be able
    to address either one without a subprocess dance with MLFLOW_TRACKING_URI.
    """
    import mlflow

    mlflow.set_tracking_uri(uri or tracking_uri())
    return mlflow.MlflowClient()


def find_stale_runs(days=7, uri=None):
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

    from mlflow.entities import RunStatus

    client = _client(uri)
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


def mark_stale_runs_finished(days=7, apply=False, uri=None):
    """Set stale RUNNING runs to FINISHED. Dry-run by default; these are real records."""
    import time

    client = _client(uri)
    stale = find_stale_runs(days=days, uri=uri)
    for exp_name, run_id, name, metric_keys in stale:
        print(f"{'closing' if apply else 'would close'}: {exp_name}/{name} "
              f"({run_id[:8]}, {metric_keys} distinct metric keys)")
        if apply:
            client.set_terminated(run_id, status="FINISHED")
            # set_terminated stamps end_time with *now*, which would otherwise make a run that
            # died in June look like an 87-day training session. Say so in the record.
            client.set_tag(run_id, "closed_as_stale",
                           f"left RUNNING by a killed process; closed by mlflow_backend "
                           f"{time.strftime('%Y-%m-%d')}")
    return stale


def _sqlite_path(uri):
    """Resolve a `sqlite:///` tracking URI to a database path."""
    if not uri.startswith("sqlite:///"):
        raise ValueError(f"only sqlite tracking URIs can be merged, got {uri!r}")
    path = uri[len("sqlite:///"):]
    if not os.path.exists(path):
        # mlflow resolves a relative sqlite path against the *current* directory.
        path = os.path.join(_ROOT, path)
    return os.path.abspath(path)


def merge_dbs(source_uri, target_uri=None, apply=False, workspace="default"):
    """Copy every run of `source_uri` into the repo-root database, keeping history.

    The split this repairs is documented in `utils/mlflow_uri.py`: 11 trainers hard-coded
    `sqlite:///mlruns.db` and 5 used `sqlite:///../mlruns.db`, so depending on the working
    directory the same experiment series landed in two different files. Merging is a plain
    row move - the run tables are keyed by run_uuid (globally unique), and both files
    already point their artifact URIs at `openai_walker/mlruns/`, so nothing on disk moves.

    Experiments match by name; a source experiment with no target twin is created with a
    fresh id and an artifact location under the repo root, so future runs of the merged
    series log beside the database rather than in whichever folder the script started from.

    Dry-run by default, like the stale-run helpers: these are the only surviving records of
    runs whose checkpoints were never saved.
    """
    import sqlite3

    src = _sqlite_path(source_uri)
    dst = _sqlite_path(target_uri or tracking_uri())
    if src == dst:
        raise ValueError("source and target are the same database")

    con = sqlite3.connect(dst)
    con.execute("ATTACH DATABASE ? AS src", (src,))
    try:
        rev_dst = con.execute("select version_num from main.alembic_version").fetchone()
        rev_src = con.execute("select version_num from src.alembic_version").fetchone()
        if rev_dst != rev_src:
            raise RuntimeError(
                f"schema mismatch: target={rev_dst} source={rev_src}; upgrade mlflow and "
                "run `mlflow db upgrade` on the source before merging"
            )

        def cols(table):
            return [r[1] for r in con.execute(f"pragma main.table_info({table})")]

        src_tables = {r[0] for r in con.execute("select name from src.sqlite_master where type='table'")}
        dst_tables = {r[0] for r in con.execute("select name from main.sqlite_master where type='table'")}
        if src_tables != dst_tables:
            raise RuntimeError(f"table sets differ; only in source={src_tables - dst_tables}, "
                               f"only in target={dst_tables - src_tables}")

        def count(table, schema="main"):
            return con.execute(f"select count(*) from {schema}.{table}").fetchone()[0]

        # Anything keyed by experiment rather than run would need the same remapping as `runs`;
        # nothing in this schema carries such rows, so refuse rather than drop them silently.
        keyed_by_experiment = [
            t for t in sorted(dst_tables)
            if t not in ("runs", "experiments") and "experiment_id" in cols(t) and count(t, "src")
        ]
        if keyed_by_experiment:
            raise RuntimeError(f"source has rows in experiment-keyed tables this merge does not "
                               f"handle: {keyed_by_experiment}")

        run_tables = [t for t in sorted(dst_tables)
                      if t != "runs" and "run_uuid" in cols(t) and count(t, "src")]

        dst_exps = {name: eid for eid, name in
                    con.execute("select experiment_id, name from main.experiments")}
        next_id = con.execute("select max(experiment_id) from main.experiments").fetchone()[0] + 1
        root_artifacts = os.path.join(_ROOT, "mlruns").replace("\\", "/")

        plan, mapping = [], {}
        for eid, name in con.execute(
                "select experiment_id, name from src.experiments order by experiment_id"):
            if name in dst_exps:
                mapping[eid] = dst_exps[name]
                plan.append(f"experiment {eid} {name!r} -> existing id {dst_exps[name]}")
            else:
                mapping[eid] = next_id
                plan.append(f"experiment {eid} {name!r} -> new id {next_id}")
                next_id += 1
                if apply:
                    con.execute(
                        "insert into main.experiments (experiment_id, name, artifact_location, "
                        "lifecycle_stage, creation_time, last_update_time, workspace) "
                        "values (?, ?, ?, 'active', "
                        "(select creation_time from src.experiments where experiment_id=?), "
                        "(select last_update_time from src.experiments where experiment_id=?), ?)",
                        (mapping[eid], name, f"file:///{root_artifacts}/{mapping[eid]}", eid, eid,
                         workspace),
                    )

        new_runs = con.execute(
            "select count(*) from src.runs where run_uuid not in (select run_uuid from main.runs)"
        ).fetchone()[0]
        plan.append(f"{new_runs} new run(s) of {count('runs', 'src')} in source")

        if apply:
            for t in run_tables:
                tcols = ", ".join(cols(t))
                con.execute(f"insert or ignore into main.{t} ({tcols}) select {tcols} from src.{t}")
            run_cols = ", ".join(cols("runs"))
            for eid, mapped in mapping.items():
                select = ", ".join(str(mapped) if c == "experiment_id" else f"r.{c}"
                                   for c in cols("runs"))
                con.execute(
                    f"insert or ignore into main.runs ({run_cols}) select {select} from src.runs r "
                    "where r.experiment_id=? and r.run_uuid not in "
                    "(select run_uuid from main.runs)", (eid,))
            con.commit()

        print(f"{'MERGED' if apply else 'DRY RUN - would merge'}: {src} -> {dst}")
        for line in plan:
            print("  " + line)
        for t in run_tables:
            print(f"  {t}: +{count(t, 'src')} rows in source -> {count(t)} in target")
        counts = {t: count(t) for t in ("runs", "metrics", "latest_metrics", "params", "tags")}
        print(f"  target rows after: {counts}")
    finally:
        con.execute("DETACH DATABASE src")
        con.close()
    return plan


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    stale = sub.add_parser("stale-runs", help="find runs left RUNNING by killed processes")
    stale.add_argument("--days", type=int, default=7)
    stale.add_argument("--uri", default=None,
                       help="Tracking URI to inspect (default: the repo-root mlruns.db).")
    stale.add_argument("--all", action="store_true",
                       help="Inspect both sqlite backends that hold history.")
    stale.add_argument("--apply", action="store_true",
                       help="Actually close them (default only reports).")

    merge = sub.add_parser("merge", help="copy one mlruns.db into another, keeping history")
    merge.add_argument("--source", required=True, help="sqlite:/// URI of the database to drain")
    merge.add_argument("--target", default=None, help="sqlite:/// URI (default: repo-root)")
    merge.add_argument("--apply", action="store_true",
                       help="Actually write (default only reports the plan).")

    a = p.parse_args()
    if a.cmd == "stale-runs":
        legacy = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlruns.db")
        uris = [None] + (["sqlite:///" + legacy.replace("\\", "/")]
                         if a.all and os.path.exists(legacy) else [])
        found = []
        for uri in uris:
            found += mark_stale_runs_finished(days=a.days, apply=a.apply, uri=uri)
        print(f"\n{len(found)} stale run(s){' closed' if a.apply else ' (dry run; use --apply)'}")
    else:
        merge_dbs(a.source, a.target, apply=a.apply)


if __name__ == "__main__":
    main()


__all__ = ["set_uri", "tracking_uri", "find_stale_runs", "mark_stale_runs_finished", "merge_dbs"]
