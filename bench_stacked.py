"""End-to-end stacked throughput ladder: today's stack against the stack that was committed.

`bench_env.py` measures the environment in steady state. That is not what a training run
waits on, and multiplying steady-state factors is not a measurement (the device factor already
contains collection, the parallel factor does not). This script runs whole trainers instead,
and arms each change on top of the committed baseline:

  n=8 ladder, 20k-step SAC budget, updates from step 4000 (learner-dominated):
    E  original env + original trainer + sync + cpu   <- what commit 5920805 actually ran
    D  original env + current trainer + sync + cpu    <- the correctness fixes alone
    A  rewritten env + sync + cpu                     <- the hot-path rewrite alone
    B  rewritten env + parallel + cpu                 <- + the process-parallel backend
    C  rewritten env + parallel + cuda                <- + the GPU learner
    A2 rewritten env + sync + cuda                    <- today's default at 8 envs

  n=32 ladder, 600k steps with the update gate closed (collection-dominated):
    D32 original env + sync, A32 rewritten env + sync, B32 rewritten env + parallel

The legacy env and trainer are extracted from commit 5920805 with `git show` and written one
directory deep, because the env resolves its model as dirname(dirname(__file__))/walker_ragdoll.xml.
The old trainer chose its device with torch.cuda.is_available(); this venv has a CUDA build, so
arm E patches that call to False to reproduce the CPU-only wheel of the original .venv
(CUDA_VISIBLE_DEVICES=-1 segfaults mid-training instead).

Worker startup is part of the answer, not noise: 32 worker processes cost ~200 s here, which is
why the steady-state 6.5x collapses to 1.86x over 600k steps, and why --vec-backend auto does
not enable the parallel backend below 16 envs.

    python bench_stacked.py --ladder n8
    python bench_stacked.py --ladder n32 --reps 1
    python bench_stacked.py --ladder n32_real --reps 1
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
BASE_COMMIT = "5920805"
LEGACY_ENV = os.path.join(ROOT, "envs", "_legacy_bench_env.py")
LEGACY_TRAINER = os.path.join(ROOT, "_legacy_bench_train_walker.py")
MLFLOW_DB = os.path.join(os.environ.get("TEMP", "."), "bench_stacked_mlflow.db").replace(os.sep, "/")

RUNNER = """\
import sys, runpy
{prelude}
sys.argv = ["train_walker.py"] + {args!r}
runpy.run_path({trainer!r}, run_name="__main__")
"""

ENV_PRELUDE = """\
import importlib.util
spec = importlib.util.spec_from_file_location("envs.walker_ragdoll_env", {env!r})
mod = importlib.util.module_from_spec(spec)
sys.modules["envs.walker_ragdoll_env"] = mod
spec.loader.exec_module(mod)
"""
CPU_PRELUDE = "import torch\ntorch.cuda.is_available = lambda: False\n"

# old_env: the 5920805 environment class. old_trainer: the 5920805 train_walker.py, which has
# neither --vec-backend nor --device and always over-fires its actor gate.
LADDERS = {
    "n8": {
        "label": "n=8 learner-bound SAC",
        "steps": 20000, "envs": 8, "learning_starts": 4000, "phase": "recovery",
        "arms": {
            "E_orig_env_orig_trainer_sync_cpu": dict(old_env=True, old_trainer=True),
            "D_orig_env_current_trainer_sync_cpu": dict(old_env=True, vec="sync", device="cpu"),
            "A_new_env_sync_cpu": dict(vec="sync", device="cpu"),
            "B_new_env_parallel_cpu": dict(vec="parallel", device="cpu"),
            "C_new_env_parallel_cuda": dict(vec="parallel", device="cuda"),
            "A2_new_env_sync_cuda": dict(vec="sync", device="cuda"),
        },
    },
    "n32": {
        "label": "n=32 collection only",
        "steps": 600000, "envs": 32, "learning_starts": 600000, "phase": "recovery",
        "arms": {
            "D32_orig_env_sync": dict(old_env=True, vec="sync", device="cpu"),
            "A32_new_env_sync": dict(vec="sync", device="cpu"),
            "B32_new_env_parallel": dict(vec="parallel", device="cpu"),
        },
    },
    # The committed 40M-step run (checkpoints/walker_target_v1) was num_envs=32,
    # learning_starts=10000 (the repo default), task_phase=target. These two arms replay that
    # configuration for 400k steps: one on the 5920805 stack, one on today's default. The mtime
    # deltas between its checkpoints already say what it cost: 14.5 min per 1M env steps.
    "n32_real": {
        "label": "n=32 target-phase SAC, the 40M run's own configuration",
        "steps": 400000, "envs": 32, "learning_starts": 10000, "phase": "target",
        "arms": {
            "E32_orig_env_orig_trainer_sync_cpu": dict(old_env=True, old_trainer=True),
            "G32_new_env_parallel_cuda": dict(vec="parallel", device="cuda"),
        },
    },
}


def materialize():
    for path, dest in (("envs/walker_ragdoll_env.py", LEGACY_ENV),
                       ("train_walker.py", LEGACY_TRAINER)):
        out = subprocess.run(["git", "-c", f"safe.directory={ROOT}", "show",
                              f"{BASE_COMMIT}:{path}"], cwd=ROOT, capture_output=True, text=True)
        assert out.returncode == 0 and out.stdout, out.stderr[:300]
        with open(dest, "w", encoding="utf-8", newline="") as handle:
            handle.write(out.stdout)


def cleanup():
    import glob
    import shutil
    for path in (glob.glob(os.path.join(ROOT, "checkpoints", "benchstack_*")) +
                 glob.glob(os.path.join(ROOT, "runs", "benchstack_*")) +
                 [LEGACY_ENV, LEGACY_TRAINER]):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)


def build_script(cfg, run_args):
    prelude = ""
    if cfg.get("old_env"):
        prelude += ENV_PRELUDE.format(env=LEGACY_ENV)
    if cfg.get("old_trainer"):
        prelude += CPU_PRELUDE
    trainer = LEGACY_TRAINER if cfg.get("old_trainer") else os.path.join(ROOT, "train_walker.py")
    return RUNNER.format(prelude=prelude, args=run_args, trainer=trainer)


def run_one(arm, rep, cfg, spec):
    run_args = ["--algo", "sac", "--run-id", f"benchstack_{arm}_r{rep}", "--seed", "7",
                "--total-timesteps", str(spec["steps"]), "--num-envs", str(spec["envs"]),
                "--learning-starts", str(spec["learning_starts"]),
                "--checkpoint-interval", "100000000", "--task-phase", spec["phase"],
                "--reset-mode", "mixed"]
    if not cfg.get("old_trainer"):
        run_args += ["--vec-backend", cfg["vec"], "--device", cfg["device"]]
    env = dict(os.environ, MLFLOW_TRACKING_URI="sqlite:///" + MLFLOW_DB)
    t0 = time.perf_counter()
    p = subprocess.run([PY, "-c", build_script(cfg, run_args)], cwd=ROOT,
                       capture_output=True, text=True, env=env)
    dt = time.perf_counter() - t0
    if p.returncode:
        print(f"{arm} rep {rep}: FAILED exit={p.returncode}\n"
              + "\n".join(p.stderr.splitlines()[-6:]), flush=True)
        return None
    print(f"{arm:32} rep {rep}: {dt:6.1f} s  {spec['steps'] / dt:6.1f} env-steps/s", flush=True)
    return round(dt, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ladder", choices=list(LADDERS) + ["all"], default="all")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default=os.path.join("benchmarks", "throughput_stacked_ladder.json"))
    args = ap.parse_args()

    materialize()
    results = {}
    names = list(LADDERS) if args.ladder == "all" else [args.ladder]
    try:
        for name in names:
            spec = LADDERS[name]
            print(f"\n--- {spec['label']}: {spec['steps']} steps, {spec['envs']} envs, "
                  f"updates from {spec['learning_starts']} ---", flush=True)
            order = list(spec["arms"])
            reps = {arm: [] for arm in order}
            for rep in range(1, args.reps + 1):
                for arm in (order if rep % 2 else order[::-1]):
                    dt = run_one(arm, rep, spec["arms"][arm], spec)
                    if dt is not None:
                        reps[arm].append(dt)
            results[name] = {"label": spec["label"], "config": {
                "steps": spec["steps"], "num_envs": spec["envs"],
                "learning_starts": spec["learning_starts"], "task_phase": spec["phase"],
                "reps": args.reps}, "arms": reps}
    finally:
        cleanup()

    import json
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwrote {args.out} (check {os.path.exists(args.out)})")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
