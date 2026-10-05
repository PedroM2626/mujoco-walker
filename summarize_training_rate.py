"""What the trainers themselves logged about their own speed, as one comparable table.

The question this answers is not "how fast is the update" - the bench scripts measure that in
isolation - but "how fast did a real training run advance, on the day it ran". Every trainer logs
`sps`, and `train_dreamer.py` computes it as

    "sps": int(global_step / (time.time() - start_time))

a *cumulative* average since process start. The first sample is logged at `learning_starts`, before
any gradient update has run, so it reports the collection-only rate and the series then decays for
the rest of the run toward the rate the run actually costs. Read off a chart early, a run that
settles at 13 env-steps/s shows 235; that is the shape of the number, not a speedup.

So this script ignores the logged values as rates and recomputes one per run from the metric's own
timestamps: (last step - first step) / (last timestamp - first timestamp). That is the run's
wall-clock progress between the first and the last sample, and it is comparable across runs, across
code generations and across machines, because it needs no knowledge of what the trainer was doing.

    python summarize_training_rate.py                 # writes benchmarks/training_rate_history.json
    python summarize_training_rate.py --min-steps 50000

The database is read-only and is not in git; the artifact is, and it carries the rows it was built
from so the arithmetic can be checked without the database.
"""

import argparse
import datetime
import json
import os
import re
import sqlite3

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, "mlruns.db")
OUT = os.path.join(ROOT, "benchmarks", "training_rate_history.json")

SPS_DEFINITION = ("train_dreamer.py: \"sps\": int(global_step / (time.time() - start_time)) - a "
                  "cumulative average since process start, first logged at learning_starts before "
                  "any update runs, so it decays toward the run's real rate")


def _params(con, run_uuid):
    return dict(con.execute("SELECT key, value FROM params WHERE run_uuid=?", (run_uuid,)))


def collect(min_steps):
    con = sqlite3.connect("file:" + DB.replace(os.sep, "/") + "?mode=ro", uri=True, timeout=10)
    try:
        runs = con.execute(
            "SELECT run_uuid, name, start_time, status, lifecycle_stage FROM runs "
            "WHERE EXISTS (SELECT 1 FROM metrics m WHERE m.run_uuid = runs.run_uuid "
            "AND m.key = 'sps') ORDER BY start_time").fetchall()
        rows = []
        for run_uuid, name, start_ms, status, lifecycle in runs:
            samples = con.execute(
                "SELECT step, value, timestamp FROM metrics WHERE run_uuid=? AND key='sps' "
                "ORDER BY timestamp", (run_uuid,)).fetchall()
            if len(samples) < 2:
                continue
            (s0, v0, t0), (s1, v1, t1) = samples[0], samples[-1]
            span_s = (t1 - t0) / 1000.0
            steps = s1 - s0
            if steps < min_steps or span_s <= 0:
                continue
            params = _params(con, run_uuid)
            logged = sorted(float(v) for _, v, _ in samples)
            rows.append({
                "run_uuid": run_uuid,
                "name": name,
                "started": datetime.datetime.fromtimestamp(start_ms / 1000.0).isoformat(
                    timespec="seconds"),
                "start_time_ms": start_ms,
                "status": status,
                "lifecycle_stage": lifecycle,
                "algo": params.get("algo"),
                "num_envs": params.get("num_envs"),
                "task_phase": params.get("task_phase"),
                # Recorded so a comparison between two runs can be checked for being matched, rather
                # than asserted: the README claims the two cited Dreamer runs differ only in the code.
                "seed": params.get("seed"),
                "reset_mode": params.get("reset_mode"),
                "total_timesteps": params.get("total_timesteps"),
                "update_graph": params.get("update_graph"),
                "sps_samples": len(samples),
                "first_step": s0,
                "last_step": s1,
                "steps_between_samples": steps,
                "span_s": round(span_s, 1),
                "rate_steps_per_s": round(steps / span_s, 1),
                "hours_per_1m_steps": round(1_000_000 / (steps / span_s) / 3600.0, 2),
                "first_logged_sps": v0,
                "last_logged_sps": v1,
                "median_logged_sps": (logged[len(logged) // 2] if len(logged) % 2
                                      else round(sum(logged[len(logged) // 2 - 1:
                                                            len(logged) // 2 + 1]) / 2.0, 1)),
            })
        return rows
    finally:
        con.close()


def furthest(rows, algo, started_before=None, started_on_or_after=None):
    """The run of one algorithm that got furthest, inside an optional start-time window.

    "Furthest" is the selector the comparison needs: it picks the run that shows what that
    generation of the trainer could actually sustain, rather than a smoke test that stopped early.
    """
    pool = [r for r in rows if r["algo"] == algo]
    if started_before:
        pool = [r for r in pool if r["started"] < started_before]
    if started_on_or_after:
        pool = [r for r in pool if r["started"] >= started_on_or_after]
    return max(pool, key=lambda r: r["last_step"]) if pool else None


def normalizer_residue(path):
    """`obs_rms.count - global_step` from a full checkpoint, which is the run's `--num-envs`.

    `gymnasium.wrappers.NormalizeObservation` updates its `RunningMeanStd` once per step with the
    whole vector of observations, and it also updates once on the initial reset, so after a run that
    started from step zero the count is `global_step + num_envs` (plus the 1e-4 the counter starts
    at). Measured: 4.0001 and 16.0001 for the two Dreamer arms below, 16.0001 for two checkpoints of
    the `num_envs=16` REDQ run.

    It is a corroboration and not a general method. It assumes one uninterrupted run, and the
    assumption is checkable rather than trusted: `checkpoints/walker_target_v1/sac_ckpt_9000000.pt`
    gives a residue of 99,375,792, because that counter accumulated across resumed runs. Any arm
    whose residue disagrees with the flag it was started with is reported, not silently used.
    """
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    rms, step = state.get("obs_rms"), state.get("global_step")
    if rms is None or step is None:
        return None
    count = getattr(rms, "count", None)
    if count is None:
        return None
    return float(np.asarray(count).mean()) - float(step)


def evidence_log(run_id, root):
    """The trainer's own stdout for one arm, if it was kept: the two lines a rate needs as context.

    A wall clock without the GPU window it ran in is half a measurement on this machine (see the
    README's "A third way to lose a run"), and the window is only in the run's stdout.
    """
    path = os.path.join(root, f"{run_id}_evidence.log")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        text = handle.read()
    window = re.search(r"GPU window: (.+)", text)
    update = re.search(r"\[DREAMER\] update: (.+)", text)
    return {"path": os.path.relpath(path, root).replace(os.sep, "/"),
            "gpu_window": window.group(1).strip() if window else None,
            "update_mode": update.group(1).strip() if update else None}


def checkpoint_wall_clock(specs, checkpoints_root, root=ROOT):
    """Each run's own record of how fast it advanced: the mtimes of the checkpoints it wrote.

    A trainer that saves every `--checkpoint-interval` steps has already logged its progress
    against the wall clock, in the filesystem, with no instrumentation and no metric definition to
    interpret. Two runs compared over the *same* step window this way need no agreement about what
    `sps` means, and neither window contains process startup or graph capture, because both start at
    the first checkpoint rather than at the first step.

    `specs` is a list of (run_id, num_envs). The declared num_envs is cross-checked against the
    observation normalizer's own counter, so a mislabelled arm fails loudly instead of quietly
    changing what the comparison means.
    """
    runs = []
    for run_id, num_envs in specs:
        directory = os.path.join(checkpoints_root, run_id)
        if not os.path.isdir(directory):
            continue
        points = []
        for name in os.listdir(directory):
            match = re.match(r"dreamer_actor_(\d+)\.pt$", name)
            if match:
                path = os.path.join(directory, name)
                points.append((int(match.group(1)), os.path.getmtime(path)))
        points.sort()
        if len(points) < 2:
            continue
        intervals = [
            {"from_step": a[0], "to_step": b[0], "steps": b[0] - a[0],
             "seconds": round(b[1] - a[1], 2),
             "rate_steps_per_s": round((b[0] - a[0]) / (b[1] - a[1]), 1)}
            for a, b in zip(points, points[1:])
        ]
        full = os.path.join(directory, f"dreamer_ckpt_{points[-1][0]}.pt")
        residue = normalizer_residue(full) if os.path.exists(full) else None
        runs.append({
            "run_id": run_id,
            "num_envs": num_envs,
            "num_envs_from_normalizer": None if residue is None else round(residue),
            "normalizer_residue": residue,
            "evidence_log": evidence_log(run_id, root),
            "checkpoints": [{"step": s, "mtime": datetime.datetime.fromtimestamp(t).isoformat(
                timespec="seconds")} for s, t in points],
            "intervals": intervals,
        })
        if residue is not None and round(residue) != num_envs:
            print(f"[WARN] {run_id}: declared --num-envs {num_envs} but the observation normalizer's "
                  f"count implies {round(residue)} (residue {residue:.4f}); the run was resumed, or "
                  f"one of the two labels is wrong, and this comparison means nothing until that is "
                  f"settled")

    # The common window: the step range every arm actually has checkpoints for, so no arm is
    # credited with a stretch it was not measured over.
    if len(runs) >= 2:
        lo = max(r["checkpoints"][0]["step"] for r in runs)
        hi = min(r["checkpoints"][-1]["step"] for r in runs)
        for r in runs:
            points = sorted((c["step"], datetime.datetime.fromisoformat(c["mtime"]).timestamp())
                            for c in r["checkpoints"] if lo <= c["step"] <= hi)
            seconds = points[-1][1] - points[0][1]
            steps = points[-1][0] - points[0][0]
            r["window"] = {"from_step": points[0][0], "to_step": points[-1][0], "steps": steps,
                           "seconds": round(seconds, 2),
                           "rate_steps_per_s": round(steps / seconds, 1),
                           "interval_rates": [i["rate_steps_per_s"] for i in r["intervals"]
                                              if lo <= i["from_step"] and i["to_step"] <= hi]}
        rates = [r["window"]["rate_steps_per_s"] for r in runs]
        common = {"from_step": lo, "to_step": hi,
                  "fastest_over_slowest": round(max(rates) / min(rates), 2),
                  "arms": {r["run_id"]: r["window"]["rate_steps_per_s"] for r in runs}}
        return {"source": "mtime of checkpoints/<run-id>/dreamer_actor_<step>.pt",
                "note": ("the window excludes process startup and graph capture because it starts at "
                         "the first checkpoint both arms wrote; num_envs is the flag each run was "
                         "started with, cross-checked against the observation normalizer's own "
                         "counter (see normalizer_residue) - a disagreement prints a warning and "
                         "means the arm is mislabelled or was resumed"),
                "runs": runs, "common_window": common}
    return {"source": "mtime of checkpoints/<run-id>/dreamer_actor_<step>.pt", "runs": runs}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-steps", type=int, default=2000,
                   help="drop runs that advanced less than this between their first and last sample")
    p.add_argument("--runs", default="dreamer_v3_1m:4,dreamer_n16_250k:16",
                   help="comma-separated RUN_ID:NUM_ENVS arms to time from their checkpoint mtimes")
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    rows = collect(args.min_steps)
    # The two generations of the Dreamer trainer this repository has run to a large budget: before
    # the CUDA-graph capture shipped (2026-10-02 23:28, 55e1e59) and after it. Same algorithm, same
    # task_phase, same --num-envs, same seed - the only difference is the code.
    before = furthest(rows, "dreamer", started_before="2026-10-03")
    after = furthest(rows, "dreamer", started_on_or_after="2026-10-04")
    cited = {"before_graph_capture": before, "after_graph_capture": after}
    if before and after:
        cited["ratio_after_over_before"] = round(
            after["rate_steps_per_s"] / before["rate_steps_per_s"], 2)

    specs = []
    for item in args.runs.split(","):
        item = item.strip()
        if not item:
            continue
        run_id, _, num_envs = item.rpartition(":")
        specs.append((run_id, int(num_envs)))
    wall = checkpoint_wall_clock(specs, os.path.join(ROOT, "checkpoints"))

    payload = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": os.path.relpath(DB, ROOT),
        "query": ("runs having >=2 'sps' samples and >= --min-steps between the first and last; "
                  "rate = (last step - first step) / (last timestamp - first timestamp)"),
        "min_steps": args.min_steps,
        "sps_definition": SPS_DEFINITION,
        "cited": cited,
        "runs": rows,
        "checkpoint_wall_clock": wall,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"{'started':17s} {'run':32s} {'algo':7s} {'nenv':>4s} {'steps':>9s} "
          f"{'span_s':>8s} {'rate':>7s} {'h/1M':>6s} {'1st sps':>8s} {'last sps':>9s}")
    for r in rows:
        print(f"{r['started']:17s} {str(r['name'])[:32]:32s} {str(r['algo']):7s} "
              f"{str(r['num_envs']):>4s} {r['steps_between_samples']:9d} {r['span_s']:8.0f} "
              f"{r['rate_steps_per_s']:7.1f} {r['hours_per_1m_steps']:6.2f} "
              f"{r['first_logged_sps']:8.0f} {r['last_logged_sps']:9.0f}")
    if before and after:
        print(f"\ncited: {before['name']} ({before['started']}) {before['rate_steps_per_s']}/s "
              f"-> {after['name']} ({after['started']}) {after['rate_steps_per_s']}/s "
              f"= {cited['ratio_after_over_before']}x")

    for r in wall.get("runs", []):
        w = r.get("window")
        if not w:
            print(f"\n{r['run_id']} (n={r['num_envs']}): {len(r['checkpoints'])} checkpoints, "
                  f"no common window")
            continue
        print(f"\n{r['run_id']} (n={r['num_envs']}), steps {w['from_step']}-{w['to_step']}: "
              f"{w['seconds']:.1f} s = {w['rate_steps_per_s']} env-steps/s "
              f"(per-interval {w['interval_rates']})")
    if "common_window" in wall:
        cw = wall["common_window"]
        print(f"common window {cw['from_step']}-{cw['to_step']}: "
              f"{cw['fastest_over_slowest']}x between arms {cw['arms']}")
    print("wrote", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
