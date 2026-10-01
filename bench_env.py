"""Measure WalkerRagdoll simulation throughput and split the cost between
MuJoCo physics and the Python environment code around it.

Usage:
    python bench_env.py                      # all modes, default settings
    python bench_env.py --mode vec --n 32
    python bench_env.py --seconds 5 --task-phase target
    python bench_env.py --json bench.json    # machine-readable, for A/B diffs

Throughput is reported as env steps/s. The budget a trainer pays for is the
env steps it can *collect*, so this is the number that matters for
"same training budget, less wall clock".
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import envs.walker_ragdoll_env  # noqa: F401,E402  (registers WalkerRagdoll-v0)
import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402

# One policy action = frame_skip mj_steps, matching WalkerRagdollEnv's default.
FRAME_SKIP = 5


def _time_fn(fn, make_args, seconds, warmup_steps=200):
    """Time `fn(*args)` for roughly `seconds`, returning steps/s and mean us/step."""
    args = make_args()
    for _ in range(warmup_steps):
        fn(*args)
    samples = []
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        t0 = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - t0)
    total = sum(samples)
    return len(samples) / total, total / len(samples) * 1e6


def bench_physics(seconds, task_phase, reset_mode):
    """Raw MuJoCo cost: one mj_step per policy call, i.e. frame_skip substeps."""
    xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walker_ragdoll.xml")
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def step():
        mujoco.mj_step(model, data)

    sps, us = _time_fn(step, lambda: (), seconds)
    return {
        "label": "physics only (mj_step x1)",
        "steps_per_s": sps,
        "us_per_step": us,
        "note": "integrator=%s timestep=%g frame_skip=%d"
        % (mujoco.mjtIntegrator(model.opt.integrator).name, model.opt.timestep, FRAME_SKIP),
        "frame_skip": FRAME_SKIP,
    }


def bench_single_env(seconds, task_phase, reset_mode):
    env = gym.make("WalkerRagdoll-v0", task_phase=task_phase, reset_mode=reset_mode)
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32) + 0.1

    def step():
        env.step(action)

    sps, us = _time_fn(step, lambda: (), seconds)
    result = {"label": "gym env.step (n=1)", "steps_per_s": sps, "us_per_step": us}
    env.close()
    return result


def bench_vec_env(seconds, n, task_phase, reset_mode, copy):
    def thunk():
        e = gym.make("WalkerRagdoll-v0", task_phase=task_phase, reset_mode=reset_mode)
        e = gym.wrappers.FlattenObservation(e)
        return e

    # gymnasium <=0.29 calls this `copy`, >=1.0 calls it `copy_observations`.
    try:
        vec = gym.vector.SyncVectorEnv([thunk for _ in range(n)], copy_observations=copy)
    except TypeError:
        vec = gym.vector.SyncVectorEnv([thunk for _ in range(n)], copy=copy)
    vec.reset(seed=0)
    actions = np.zeros(vec.action_space.shape, dtype=np.float32) + 0.1

    def step():
        vec.step(actions)

    sps, us = _time_fn(step, lambda: (), seconds)
    result = {
        "label": f"SyncVectorEnv (n={n}, copy={copy})",
        "steps_per_s": sps,
        "us_per_step": us,
        "note": "steps/s counts env steps across all n envs",
    }
    vec.close()
    return result


def bench_parallel_vec(seconds, n, task_phase, reset_mode, sparse_info=False,
                      envs_per_worker=None):
    from envs.parallel_vector_env import ParallelVectorEnv

    specs = [
        ("envs.parallel_vector_env", "make_walker_thunk", (), {
            "index": i, "task_phase": task_phase, "reset_mode": reset_mode, "flatten": True,
        })
        for i in range(n)
    ]
    start = time.perf_counter()
    vec = ParallelVectorEnv(specs, sparse_info=sparse_info, envs_per_worker=envs_per_worker)
    startup = time.perf_counter() - start
    vec.reset(seed=0)
    actions = np.zeros(vec.action_space.shape, dtype=np.float32) + 0.1

    def step():
        vec.step(actions)

    sps, us = _time_fn(step, lambda: (), seconds)
    result = {
        "label": f"ParallelVectorEnv (n={n}, per_worker={envs_per_worker or 'auto'}, sparse={sparse_info})",
        "steps_per_s": sps,
        "us_per_step": us,
        "note": f"{len(vec._groups)} workers, startup {startup:.1f}s",
    }
    vec.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["all", "physics", "env", "vec", "parallel"], default="all")
    p.add_argument("--n", type=int, default=32, help="Number of vectorized envs.")
    p.add_argument("--seconds", type=float, default=3.0, help="Wall clock per measurement.")
    p.add_argument("--task-phase", default="recovery", choices=["recovery", "balance", "walk", "target"])
    p.add_argument("--reset-mode", default="mixed", choices=["fixed", "mixed", "fallen", "upright"])
    p.add_argument("--json", type=str, default=None, help="Write results to this path.")
    p.add_argument("--envs-per-worker", type=int, default=None,
                   help="Group N envs into one worker process (default: auto).")
    p.add_argument("--sweep", action="store_true",
                   help="Sweep envs-per-worker 1..8 at --n and print the scaling curve.")
    args = p.parse_args()

    common = dict(seconds=args.seconds, task_phase=args.task_phase, reset_mode=args.reset_mode)
    results = []
    if args.sweep:
        for size in (1, 2, 3, 4, 6, 8):
            results.append(bench_parallel_vec(sparse_info=True, envs_per_worker=size,
                                              n=args.n, **common))
    else:
        if args.mode in ("all", "physics"):
            results.append(bench_physics(**common))
        if args.mode in ("all", "env"):
            results.append(bench_single_env(**common))
        if args.mode in ("all", "vec"):
            results.append(bench_vec_env(copy=True, n=args.n, **common))
            results.append(bench_vec_env(copy=False, n=args.n, **common))
        if args.mode in ("all", "parallel"):
            results.append(bench_parallel_vec(sparse_info=False, n=args.n, **common))
            results.append(bench_parallel_vec(sparse_info=True, n=args.n, **common))

    print(f"\nWalkerRagdoll-v0 throughput  (task_phase={args.task_phase}, {args.seconds}s per measurement)")
    print("-" * 78)
    for r in results:
        note = r.get("note", "")
        print(f"{r['label']:34s} {r['steps_per_s']:>11,.0f} steps/s  {r['us_per_step']:>9.1f} us/step  {note}")
    print("-" * 78)

    if args.mode in ("all", "physics") and args.mode in ("all", "env"):
        physics, env_only = results[0], [r for r in results if r["label"].startswith("gym env.step")]
        if env_only:
            # One env.step is frame_skip mj_steps, so put both on the same unit before
            # comparing: 5 mj_step calls at X us each is the physics part of the action.
            physics_per_action = physics["us_per_step"] * physics.get("frame_skip", FRAME_SKIP)
            python_share = 1.0 - physics_per_action / env_only[0]["us_per_step"]
            print(
                f"Of one env.step: physics {physics_per_action:.0f} us "
                f"({100 * (1 - python_share):.0f}%), Python around it {env_only[0]['us_per_step'] - physics_per_action:.0f} us "
                f"({100 * python_share:.0f}%)"
            )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
