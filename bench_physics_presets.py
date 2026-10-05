"""How much does each physics preset actually buy, and how far does it move the robot?

`envs/walker_ragdoll_env.py` grew two screening worlds beside the published one - `euler` (Euler
instead of RK4, same timestep) and `fast` (Euler plus self-collision dropped, keeping every
floor contact because the standing gate counts them). A cheaper world is only useful for benchmark
work if two things are known and no third is assumed:

  1. the throughput it buys, measured back to back with the published one in the same window
     (absolute rates here move by more than 2x between windows, so a preset timed an hour from now
     is not comparable to a preset timed now);
  2. how far the dynamics drift. Same initial state, same actions, different integrator or
     collision set: the two trajectories separate, and the question a reader needs answered is
     whether they separate in a way that changes what the task measures - does the robot still
     stand, do the feet stay the only body on the floor, does the episode return keep its meaning.

Nothing here is a published result about a policy. The artifact records the cost and the gap of the
worlds themselves, which is what makes it legitimate to screen in one of them.

Usage:
    python bench_physics_presets.py --episodes 5 --steps 1000 --seconds 3 --reps 3
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym  # noqa: E402

import bench_env  # noqa: E402
from envs.walker_ragdoll_env import ENV_VERSION, PHYSICS_PRESETS  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
REFERENCE = "v9"


def _stats(values):
    arr = np.asarray(values, dtype=float)
    return {"median": round(float(np.median(arr)), 1), "min": round(float(arr.min()), 1),
            "max": round(float(arr.max()), 1)}


def throughput(presets, seconds, reps, n, task_phase, reset_mode):
    """Single-env, sync-vector and raw-mj_step rates, every preset measured in one loop.

    `_repeated` is bench_env's own repetition helper, so the median-and-spread convention here is
    the same one every throughput number in the README was produced with.
    """
    out = {}
    common = dict(seconds=seconds, task_phase=task_phase, reset_mode=reset_mode)
    for preset in presets:
        single = bench_env._repeated(bench_env.bench_single_env, reps,
                                     dict(common, physics_preset=preset))
        vec = bench_env._repeated(bench_env.bench_vec_env, reps,
                                  dict(common, physics_preset=preset, n=n, copy=True))
        physics = bench_env._repeated(bench_env.bench_physics, reps,
                                      dict(common, physics_preset=preset))
        out[preset] = {
            "env_step_per_s": _stats([r["steps_per_s"] for r in single]),
            "env_step_us": _stats([r["us_per_step"] for r in single]),
            f"vec{n}_steps_per_s": _stats([r["steps_per_s"] for r in vec]),
            "mj_step_us": _stats([r["us_per_step"] for r in physics]),
            "integrator_note": physics[0].get("note", ""),
        }
    ratios = {}
    base = out[REFERENCE]
    for preset, values in out.items():
        ratios[preset] = {
            "env_step_vs_v9": round(values["env_step_per_s"]["median"]
                                    / base["env_step_per_s"]["median"], 3),
            f"vec{n}_vs_v9": round(values[f"vec{n}_steps_per_s"]["median"]
                                   / base[f"vec{n}_steps_per_s"]["median"], 3),
            "mj_step_vs_v9": round(base["mj_step_us"]["median"] / values["mj_step_us"]["median"], 3),
        }
    return out, ratios


def _roll(env, seed, actions, steps):
    """One seeded episode under a fixed action sequence, with the quantities the task reads."""
    obs, _ = env.reset(seed=seed)
    unwrapped = env.unwrapped
    z, upright, healthy, contacts, rewards, obs_rows = [], [], [], [], [], []
    for t in range(steps):
        obs, reward, terminated, truncated, _ = env.step(actions[t])
        z.append(float(unwrapped.data.qpos[2]))
        upright.append(float(unwrapped.upright_factor))
        healthy.append(bool(unwrapped.is_healthy))
        bad, feet = unwrapped.floor_contact_counts
        contacts.append((bad, feet))
        rewards.append(float(reward))
        obs_rows.append(np.asarray(obs, dtype=float))
        if terminated or truncated:
            break
    env.close()
    return {"z": np.asarray(z), "upright": np.asarray(upright), "healthy": np.asarray(healthy),
            "contacts": contacts, "rewards": np.asarray(rewards),
            "obs": np.asarray(obs_rows), "steps": len(z)}


def divergence(presets, episodes, steps, seed, task_phase, reset_mode):
    """Same resets, same actions: how far each preset drifts from the published world."""
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(episodes):
        # One action sequence per episode, shared by every preset: uniform in the action bounds,
        # which is the harshest fair input - a trained gait would agree more, a scripted push less.
        actions = rng.uniform(-1.0, 1.0, size=(steps, 17)).astype(np.float64)
        reference = None
        for preset in [REFERENCE] + [p for p in presets if p != REFERENCE]:
            env = gym.make("WalkerRagdoll-v0", task_phase=task_phase, reset_mode=reset_mode,
                           physics_preset=preset)
            rolled = _roll(env, seed + ep, actions, steps)
            if preset == REFERENCE:
                reference, ref = rolled, rolled["z"][-1]
                continue
            n = min(reference["steps"], rolled["steps"])
            dz = np.abs(reference["z"][:n] - rolled["z"][:n])
            dob = np.linalg.norm(reference["obs"][:n] - rolled["obs"][:n], axis=1)
            healthy_flip = int(np.sum(reference["healthy"][:n] != rolled["healthy"][:n]))
            contact_flip = int(sum(1 for a, b in zip(reference["contacts"][:n],
                                                     rolled["contacts"][:n]) if a != b))
            rows.append({
                "episode": ep, "preset": preset, "compared_steps": int(n),
                "max_abs_torso_z_delta": round(float(dz.max()), 6),
                "mean_abs_torso_z_delta": round(float(dz.mean()), 6),
                "mean_abs_obs_l2": round(float(dob.mean()), 4),
                "max_abs_obs_l2": round(float(dob.max()), 4),
                "healthy_disagreements": healthy_flip,
                "floor_contact_disagreements": contact_flip,
                "reference_final_z": round(float(reference["z"][n - 1]), 4),
                "preset_final_z": round(float(rolled["z"][n - 1]), 4),
                "reference_return": round(float(reference["rewards"][:n].sum()), 2),
                "preset_return": round(float(rolled["rewards"][:n].sum()), 2),
                "return_delta": round(float(rolled["rewards"][:n].sum()
                                            - reference["rewards"][:n].sum()), 2),
                "reference_steps_standing": int(sum(
                    1 for bad, feet in reference["contacts"][:n] if bad == 0 and feet > 0)),
                "preset_steps_standing": int(sum(
                    1 for bad, feet in rolled["contacts"][:n] if bad == 0 and feet > 0)),
            })
    summary = {}
    for preset in presets:
        sub = [r for r in rows if r["preset"] == preset]
        if not sub:
            continue
        summary[preset] = {
            "episodes": len(sub),
            "max_abs_torso_z_delta": round(max(r["max_abs_torso_z_delta"] for r in sub), 6),
            "mean_abs_obs_l2": round(float(np.mean([r["mean_abs_obs_l2"] for r in sub])), 4),
            "healthy_disagreements": int(sum(r["healthy_disagreements"] for r in sub)),
            "floor_contact_disagreements": int(sum(r["floor_contact_disagreements"] for r in sub)),
            "abs_return_delta_mean": round(float(np.mean(
                [abs(r["return_delta"]) for r in sub])), 2),
            "steps_standing_total": int(sum(r["preset_steps_standing"] for r in sub)),
            "reference_steps_standing_total": int(sum(r["reference_steps_standing"] for r in sub)),
        }
    return rows, summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--presets", default="euler,fast",
                   help=f"comma separated presets to compare against {REFERENCE}; "
                        f"one of {list(PHYSICS_PRESETS)}")
    p.add_argument("--task-phase", default="target")
    p.add_argument("--reset-mode", default="mixed")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--n", type=int, default=8, help="vector size for the throughput arm.")
    p.add_argument("--skip-throughput", action="store_true")
    p.add_argument("--skip-divergence", action="store_true")
    p.add_argument("--out", default=os.path.join("benchmarks", "physics_presets.json"))
    args = p.parse_args()

    wanted = [a.strip() for a in args.presets.split(",") if a.strip()]
    bad = [a for a in wanted if a not in PHYSICS_PRESETS]
    if bad:
        p.error(f"unknown presets {bad}, expected one of {list(PHYSICS_PRESETS)}")
    presets = [REFERENCE] + [a for a in wanted if a != REFERENCE]

    out = {
        "protocol": (f"presets {presets} (reference {REFERENCE} = {ENV_VERSION}) in one process; "
                     f"throughput = bench_env._repeated at --seconds {args.seconds} x --reps "
                     f"{args.reps} on n=1, SyncVectorEnv n={args.n} and raw mj_step; divergence = "
                     f"{args.episodes} seeded episodes of {args.steps} steps under one shared "
                     f"action sequence drawn from a uniform(-1,1) rng seeded {args.seed}, "
                     f"task_phase={args.task_phase}, reset_mode={args.reset_mode}"),
        "rate_caveat": ("absolute steps/s belong to the window they were measured in - the same "
                        "build has measured 15.6 and 68 env-steps/s here - so only the ratios, "
                        "which are back-to-back in this one process, are reusable"),
        "std_convention": "median of --reps measurements, with min and max reported",
    }
    if not args.skip_throughput:
        rates, ratios = throughput(presets, args.seconds, args.reps, args.n,
                                   args.task_phase, args.reset_mode)
        out["throughput"] = rates
        out["throughput_ratio"] = ratios
    if not args.skip_divergence:
        rows, summary = divergence(presets, args.episodes, args.steps, args.seed,
                                   args.task_phase, args.reset_mode)
        out["divergence"] = summary
        out["divergence_per_episode"] = rows

    target = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}")
    if "throughput_ratio" in out:
        for preset, ratio in out["throughput_ratio"].items():
            print(f"  {preset:6s} env.step x{ratio['env_step_vs_v9']:.2f}  "
                  f"vec{args.n} x{ratio[f'vec{args.n}_vs_v9']:.2f}  "
                  f"mj_step x{ratio['mj_step_vs_v9']:.2f}")
    for preset, values in out.get("divergence", {}).items():
        print(f"  {preset:6s} max |dz| {values['max_abs_torso_z_delta']:.6f}  "
              f"obs L2 {values['mean_abs_obs_l2']:.4f}  "
              f"healthy flips {values['healthy_disagreements']}  "
              f"contact flips {values['floor_contact_disagreements']}  "
              f"|dR| {values['abs_return_delta_mean']}")


if __name__ == "__main__":
    main()
