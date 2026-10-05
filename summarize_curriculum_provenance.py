"""Provenance of the manual phase curriculum: what the two pre-existing policies' metadata says.

This project's curriculum is manual - a phase is something a person changes between runs or at a
restart - and the only durable record of that is inside the checkpoints themselves. This script
reads metadata (never weights-as-evidence: no tensor is compared or quoted) out of
`checkpoints/walker_recovery_v1/` and `checkpoints/walker_target_v1/` and writes
`benchmarks/curriculum_provenance.json`, so the README's claims about the order of the two runs, the
46-to-49 observation widening and the mid-run change of nominal walking speed are gated against a
committed file rather than against a `checkpoints/` tree that `.gitignore` keeps out of clones.

    python summarize_curriculum_provenance.py

The three things this can and cannot establish, stated because the README repeats them:

- file mtime and the `env_version`/actor-width stamps establish **order**: recovery's checkpoints are
  six days older and 46-wide against target's 49, so recovery cannot have been initialised from
  weights that did not exist yet.
- they do **not** establish that target was initialised *from* recovery; that would need a
  weight-level comparison, and nothing here makes one.
- `target_forward_velocity` moving 1.2 -> 10.0 between consecutive checkpoints of one run-id
  establishes that the run was restarted with a different task parameter, not who restarted it or
  with what command line.
"""

import datetime
import json
import os
import re

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "benchmarks", "curriculum_provenance.json")

RUNS = ("walker_recovery_v1", "walker_target_v1")
STEP_IN_NAME = re.compile(r"_(\d+)\.pt$")


def actor_width(state):
    """Input width the actor was built for: the first backbone layer's column count."""
    sd = state.get("actor_state_dict") or {}
    weight = sd.get("backbone.0.weight")
    return None if weight is None else int(weight.shape[1])


def inspect_run(run_id):
    directory = os.path.join(ROOT, "checkpoints", run_id)
    if not os.path.isdir(directory):
        return {"run_id": run_id, "absent": True,
                "note": "checkpoints/ is gitignored; rerun this script where the weights exist"}
    files = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".pt"):
            continue
        match = STEP_IN_NAME.search(name)
        if not match:
            continue
        path = os.path.join(directory, name)
        state = torch.load(path, map_location="cpu", weights_only=False)
        mtime = os.path.getmtime(path)
        files.append({
            "file": name,
            "step": int(match.group(1)),
            "written": datetime.datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
            "actor_obs_width": actor_width(state),
            "env_version": state.get("env_version"),
            "task_phase": state.get("task_phase"),
            "target_forward_velocity": state.get("target_forward_velocity"),
            "records_reward_kwargs": "reward_kwargs" in state,
        })
    files.sort(key=lambda f: f["step"])
    tfv_switches = [
        {"from_step": a["step"], "to_step": b["step"],
         "from_velocity": a["target_forward_velocity"], "to_velocity": b["target_forward_velocity"],
         "from_written": a["written"], "to_written": b["written"]}
        for a, b in zip(files, files[1:])
        if a["target_forward_velocity"] != b["target_forward_velocity"]
        and a["target_forward_velocity"] is not None
        and b["target_forward_velocity"] is not None
    ]
    return {
        "run_id": run_id,
        "absent": False,
        "checkpoints": files,
        # The trainers write a full and an actor-only file per save, so the file count is double the
        # number of step positions; quote both rather than one number a reader has to guess at.
        "distinct_steps": len({f["step"] for f in files}),
        "first_step": files[0]["step"] if files else None,
        "last_step": files[-1]["step"] if files else None,
        "first_written": files[0]["written"] if files else None,
        "last_written": files[-1]["written"] if files else None,
        "actor_obs_width": files[0]["actor_obs_width"] if files else None,
        "env_version": files[0]["env_version"] if files else None,
        "task_phase": files[0]["task_phase"] if files else None,
        "target_forward_velocity_switches": tfv_switches,
    }


def main():
    runs = [inspect_run(r) for r in RUNS]
    present = sorted((r for r in runs if not r["absent"]), key=lambda r: r["first_written"])
    gap_days = first = second = None
    if len(present) == 2:
        first, second = present
        later = datetime.datetime.fromisoformat(second["first_written"])
        earlier = datetime.datetime.fromisoformat(first["first_written"])
        gap_days = round((later - earlier).total_seconds() / 86400.0, 2)

    payload = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": "mtime and metadata of checkpoints/<run-id>/sac_ckpt_*.pt (no weights compared)",
        "runs": runs,
        "order": {
            "first": first and first["run_id"],
            "second": second and second["run_id"],
            "gap_days": gap_days,
            "widths": {r["run_id"]: r["actor_obs_width"] for r in present},
            "env_versions": {r["run_id"]: r["env_version"] for r in present},
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    for r in runs:
        if r["absent"]:
            print(f"{r['run_id']:22s} absent")
            continue
        print(f"{r['run_id']:22s} {r['distinct_steps']:3d} step positions ({len(r['checkpoints'])} "
              f"files)  steps {r['first_step']}-{r['last_step']}  width={r['actor_obs_width']}  "
              f"env={r['env_version']}  phase={r['task_phase']}  "
              f"written {r['first_written']} -> {r['last_written']}")
        for s in r["target_forward_velocity_switches"]:
            print(f"{'':22s}   target_forward_velocity {s['from_velocity']} at {s['from_step']:,} "
                  f"({s['from_written']}) -> {s['to_velocity']} at {s['to_step']:,} "
                  f"({s['to_written']})")
    if gap_days is not None:
        print(f"order: {first['run_id']} first, {second['run_id']} {gap_days} days later")
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
