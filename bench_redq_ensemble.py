"""A/B the batched vs the loop REDQ critic ensemble, inside the real trainer.

`BatchedSoftQEnsemble` (train_walker.py) replaces the nn.ModuleList of N critics with one bmm
stack. The unit-level claim is measured in tests/test_redq_ensemble.py; this measures the thing
that matters, which is whether it moves the training loop. Two budgets, because they answer
different questions:

  n=16 parallel (the shipped configuration): 10k env steps, updates from 2k. Most of the wall
      clock is gradient work, but every run also pays ~2 s per worker to boot the vector backend,
      so the numbers that mean something are the deltas against the collection floor.
  n=4 sync: 4k env steps, updates from 400. Four processes, no per-worker boot inside the timed
      phase, so the update share is unobstructed.

The floor arms are the same command with the update gate closed (--learning-starts above the
budget), one per implementation: they should agree, and if they don't the comparison is
contended rather than informative.

    python bench_redq_ensemble.py
"""

import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
DB = "sqlite:///" + os.path.join(os.environ.get("TEMP", "."), "bench_redq_ensemble.db").replace(os.sep, "/")

# (tag, impl, num_envs, vec, learning_starts, steps, floor_tag)
#
# Keep the budgets small but update-dominated: utd_ratio=20 means 20 gradient steps per *vector*
# step, so at n=1 a 20k env-step budget is 400k updates and nothing finishes. n=16 with 10k steps
# is 500 vector steps x 20 = 10k updates, and n=4 with 4k steps is 900 x 20 = 18k updates.
RUNS = [
    ("b16", "batched", 16, "parallel", 2000, 10000, "f16b"),
    ("l16", "loop", 16, "parallel", 2000, 10000, "f16l"),
    ("b16r", "batched", 16, "parallel", 2000, 10000, "f16b"),
    ("l16r", "loop", 16, "parallel", 2000, 10000, "f16l"),
    ("f16b", "batched", 16, "parallel", 999999, 10000, None),
    ("f16l", "loop", 16, "parallel", 999999, 10000, None),
    ("b4", "batched", 4, "sync", 400, 4000, "f4b"),
    ("l4", "loop", 4, "sync", 400, 4000, "f4l"),
    ("f4b", "batched", 4, "sync", 999999, 4000, None),
    ("f4l", "loop", 4, "sync", 999999, 4000, None),
]
BY_TAG = {spec[0]: spec for spec in RUNS}


def one(tag, impl, num_envs, vec, starts, steps):
    args = ["train_redq.py", "--run-id", f"bre_{tag}", "--seed", "7",
            "--num-envs", str(num_envs), "--vec-backend", vec,
            "--total-timesteps", str(steps), "--learning-starts", str(starts),
            "--checkpoint-interval", "100000000", "--task-phase", "target",
            "--reset-mode", "upright", "--ensemble-impl", impl, "--device", "cuda"]
    env = dict(os.environ, MLFLOW_TRACKING_URI=DB)
    t0 = time.perf_counter()
    proc = subprocess.run([PY, "-u"] + args, cwd=ROOT, capture_output=True, text=True, env=env,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    dt = time.perf_counter() - t0
    print(f"{tag:6} {impl:8} n={num_envs:<3}{vec:9} updates_from={starts:7}  {dt:7.1f} s  "
          f"{steps / dt:7.1f} env-steps/s  exit={proc.returncode}", flush=True)
    if proc.returncode:
        print("\n".join(proc.stderr.splitlines()[-8:]), flush=True)
        return None
    return round(dt, 1)


def clean():
    for tag in BY_TAG:
        path = os.path.join(ROOT, "checkpoints", f"bre_{tag}")
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    runs_dir = os.path.join(ROOT, "runs")
    if os.path.isdir(runs_dir):
        for entry in os.scandir(runs_dir):
            if entry.is_dir() and entry.name.startswith("bre_"):
                shutil.rmtree(entry.path, ignore_errors=True)


def main():
    results = {}
    try:
        for spec in RUNS:
            results[spec[0]] = one(*spec[:6])
    finally:
        clean()

    def gradient(tag):
        floor = BY_TAG[tag][6]
        if not floor or results.get(floor) is None or results.get(tag) is None:
            return None
        return round(results[tag] - results[floor], 1)

    print("\narm     n   vec        total s  gradient s  env-steps/s")
    for spec in RUNS:
        tag, _impl, num_envs, vec, _starts, steps, _floor = spec
        if results.get(tag) is None:
            continue
        g = gradient(tag)
        print(f"{tag:6} n={num_envs:<3}{vec[:5]:6} {results[tag]:9.1f}"
              + (f"{g:12.1f}" if g is not None else "           -")
              + f" {steps / results[tag]:11.1f}")

    summary = {}
    for bat, loop, label in (("b16", "l16", "n=16 parallel"), ("b4", "l4", "n=4 sync")):
        gb, gl = gradient(bat), gradient(loop)
        if gb and gl and gb > 0:
            summary[label] = {"loop_gradient_s": gl, "batched_gradient_s": gb,
                              "update_phase_speedup": round(gl / gb, 2),
                              "whole_run_ratio": round(results[loop] / results[bat], 2),
                              "floors_s": [results[BY_TAG[bat][6]], results[BY_TAG[loop][6]]]}
            print(f"\n{label}: gradient work loop {gl:.1f} s vs batched {gb:.1f} s = "
                  f"{gl / gb:.2f}x on the update phase; whole-run ratio "
                  f"{results[loop] / results[bat]:.2f}x")

    out = os.path.join(ROOT, "benchmarks", "redq_ensemble_ab.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"runs_s": results, "gradient_s": {t: gradient(t) for t in BY_TAG},
                   "summary": summary,
                   "note": "10k/4k env-step budgets with utd_ratio=20, ensemble N=10, on a laptop "
                           "that also had a 1M-step REDQ run on it; floors are the same command "
                           "with the update gate closed",
                   "config": {"utd_ratio": 20, "ensemble_size": 10, "num_min_critics": 2,
                              "batch_size": 256, "policy_frequency": 2}}, handle, indent=2)
    print("wrote", os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
