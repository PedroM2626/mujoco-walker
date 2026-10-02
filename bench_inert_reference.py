"""The inert-robot reference the Phase-1 rows are judged against, in a chosen env revision.

Writes benchmarks/phase1_inert_reference_<tag>.json: one seeded 300-step rollout per behaviour
(commanding zero, uniform random, and the three evidence checkpoints), under the environment
defaults and with the deterministic policy each checkpoint defines. `--env-commit` aliases that
revision of `envs/walker_ragdoll_env.py` in as `envs.walker_ragdoll_env`, the same way
`eval_phase1.py --env-commit` does, so a claim about v8 can be re-measured instead of quoted from
a note. Run it twice to get the pair:

    python bench_inert_reference.py --tag v9
    python bench_inert_reference.py --tag v8 --env-commit 2d59b7b
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = "D:/mujoco-walker"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--env-commit", default=None)
ap.add_argument("--tag", required=True)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

tmp = None
if args.env_commit:
    out = subprocess.run(["git", "-c", f"safe.directory={ROOT}", "show",
                          f"{args.env_commit}:envs/walker_ragdoll_env.py"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and "WalkerRagdollEnv" in out.stdout, out.stderr[:200]
    tmp = os.path.join(ROOT, "envs", f"_inert_env_{args.env_commit[:8]}.py")
    open(tmp, "w", encoding="utf-8", newline="").write(out.stdout)
    spec = importlib.util.spec_from_file_location("envs.walker_ragdoll_env", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["envs.walker_ragdoll_env"] = mod
    spec.loader.exec_module(mod)
    envs_module = mod
else:
    import envs.walker_ragdoll_env as envs_module  # noqa: E402

import gymnasium as gym  # noqa: E402
import eval_phase1 as ev  # noqa: E402

EVIDENCE = {"redq": "redq_evidence/redq_actor_*.pt",
            "dreamer": "dreamer_smoke/dreamer_actor_*.pt",
            "ars": "ars_evidence/ars_ckpt_*.pt"}
STEPS, SEED = args.steps, args.seed


def latest(pattern):
    import glob
    hits = sorted(glob.glob(os.path.join(ROOT, "checkpoints", pattern)))
    return hits[-1] if hits else None


def rollout(policy, reset=lambda: None, seed=None):
    env = gym.make("WalkerRagdoll-v0", reset_mode="mixed", task_phase="target")
    try:
        obs, _ = env.reset(seed=SEED)
        # The Dreamer actor samples its stochastic state, so an unseeded rollout is a different
        # trajectory every time (-3425.86, -3637.40 and -3278.44 came from the same checkpoint in
        # three runs of this script). Seed per behaviour or the table is not reproducible.
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        reset()
        total, gates, vels, xs = 0.0, [], [], []
        for _ in range(STEPS):
            obs, reward, term, trunc, info = env.step(np.asarray(policy(obs), dtype=np.float64))
            total += float(reward)
            if "standing_gate" in info:
                gates.append(float(info["standing_gate"]))
            vels.append(float(info.get("x_velocity", 0.0)))
            xs.append(float(info.get("x_position", 0.0)))
            if term or trunc:
                break
        # v8 has no standing_gate in its step info; record None rather than a
        # default that would read as "the robot never stood".
        gate = float(np.mean(gates)) if gates else None
        return total, gate, float(np.mean(vels)), float(xs[-1] - xs[0])
    finally:
        env.close()


rng = np.random.default_rng(1)
rows = {"commanding_zero": rollout(lambda obs: np.zeros(17), seed=SEED),
        "uniform_random": rollout(lambda obs: rng.uniform(-1, 1, 17), seed=SEED)}
for label, pattern in EVIDENCE.items():
    path = latest(pattern)
    _, _, _, policy, reset, _ = ev.build_policy(path, torch.device("cpu"))
    rows[label] = rollout(policy, reset, seed=SEED) + (os.path.basename(path),)

print(f"env {envs_module.ENV_VERSION} ({args.tag}), {STEPS} steps, seed {SEED}")
print(f"{'behaviour':18} {'return':>10} {'gate':>7} {'vx':>8} {'dx':>7}")
for name, value in rows.items():
    gate_txt = "  n/a " if value[1] is None else f"{value[1]:7.3f}"
    print(f"{name:18} {value[0]:10.2f} {gate_txt} {value[2]:+8.4f} {value[3]:+7.3f}")

out = {"env_version": envs_module.ENV_VERSION, "env_commit": args.env_commit, "tag": args.tag,
       "steps": STEPS, "seed": SEED, "task_phase": "target",
       "reward": "environment defaults (as tests/test_phase1_behaviour.py runs it)",
       "checkpoints": {k: v[4] for k, v in rows.items() if len(v) > 4},
       "returns": {k: round(v[0], 2) for k, v in rows.items()},
       "standing_gate": {k: (round(v[1], 4) if v[1] is not None else None)
                           for k, v in rows.items()},
       "mean_x_velocity": {k: round(v[2], 5) for k, v in rows.items()},
       "x_displacement_m": {k: round(v[3], 4) for k, v in rows.items()}}
target = os.path.join(ROOT, "benchmarks", f"phase1_inert_reference_{args.tag}.json")
with open(target, "w", encoding="utf-8") as handle:
    json.dump(out, handle, indent=2)
print("wrote", os.path.relpath(target, ROOT))
if tmp and os.path.exists(tmp):
    os.remove(tmp)
