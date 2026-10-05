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


def _arm(row):
    """One bench_env repetition summary, in the units this artifact quotes.

    `bench_env._repeated` already reduces its reps to a median plus the window's min/max/spread, so
    those fields are carried through rather than recomputed - the spread is the point of the column.
    """
    out = {"median_steps_per_s": round(float(row["steps_per_s"]), 1),
           "us_per_step": round(float(row["us_per_step"]), 2),
           "reps": int(row.get("reps", 1))}
    for key in ("min_steps_per_s", "max_steps_per_s"):
        if key in row:
            out[key] = round(float(row[key]), 1)
    if "spread_pct" in row:
        out["spread_pct"] = round(float(row["spread_pct"]), 1)
    if "note" in row:
        out["model_note"] = row["note"]
    return out


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
        out[preset] = {"env_step_n1": _arm(single),
                       f"env_step_n{n}": _arm(vec),
                       f"vec{n}_steps_per_s": round(float(vec["steps_per_s"]), 1),
                       "mj_step_raw": _arm(physics)}
    base = out[REFERENCE]
    ratios = {}
    for preset, values in out.items():
        ratios[preset] = {
            "env_step_n1_vs_v9": round(values["env_step_n1"]["median_steps_per_s"]
                                       / base["env_step_n1"]["median_steps_per_s"], 3),
            f"env_step_n{n}_vs_v9": round(values[f"env_step_n{n}"]["median_steps_per_s"]
                                          / base[f"env_step_n{n}"]["median_steps_per_s"], 3),
            # Physics cost is a time, not a rate: a third of the microseconds per mj_step is a 3x
            # cheaper step, so this ratio runs the other way from the throughput ones.
            "mj_step_time_vs_v9": round(base["mj_step_raw"]["us_per_step"]
                                        / values["mj_step_raw"]["us_per_step"], 3),
        }
    return out, ratios


def _roll(env, seed, actions, steps):
    """One seeded episode under a fixed action sequence, with the quantities the task reads."""
    obs, _ = env.reset(seed=seed)
    unwrapped = env.unwrapped
    z, upright, healthy, contacts, rewards, obs_rows = [], [], [], [], [], []
    foot_steps = bad_steps = 0
    for t in range(steps):
        obs, reward, terminated, truncated, _ = env.step(actions[t])
        z.append(float(unwrapped.data.qpos[2]))
        upright.append(float(unwrapped.upright_factor))
        healthy.append(bool(unwrapped.is_healthy))
        bad, feet = unwrapped.floor_contact_counts
        contacts.append((bad, feet))
        foot_steps += int(feet > 0)
        bad_steps += int(bad > 0)
        rewards.append(float(reward))
        obs_rows.append(np.asarray(obs, dtype=float))
        if terminated or truncated:
            break
    env.close()
    return {"z": np.asarray(z), "upright": np.asarray(upright), "healthy": np.asarray(healthy),
            "contacts": contacts, "rewards": np.asarray(rewards),
            "obs": np.asarray(obs_rows), "steps": len(z),
            "foot_steps": foot_steps, "bad_steps": bad_steps}


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
                # The step-by-step contact "flip" count above measures when contacts happened, and
                # two integrators disagree about that constantly. These two totals are the question
                # the standing gate actually asks: how many steps had a foot on the floor, and how
                # many had some other body on it.
                "reference_foot_contact_steps": reference["foot_steps"],
                "preset_foot_contact_steps": rolled["foot_steps"],
                "reference_bad_support_steps": reference["bad_steps"],
                "preset_bad_support_steps": rolled["bad_steps"],
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
            "foot_contact_steps_total": int(sum(r["preset_foot_contact_steps"] for r in sub)),
            "reference_foot_contact_steps_total": int(sum(r["reference_foot_contact_steps"]
                                                          for r in sub)),
            "bad_support_steps_total": int(sum(r["preset_bad_support_steps"] for r in sub)),
            "reference_bad_support_steps_total": int(sum(r["reference_bad_support_steps"]
                                                         for r in sub)),
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
    target = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out

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
    elif os.path.exists(target):
        # Re-running the divergence in another window must not silently drop the ratios that were
        # measured back-to-back: they belong to their window and cannot be recomputed later.
        previous = json.load(open(target, encoding="utf-8"))
        carried = []
        for key in ("throughput", "throughput_ratio"):
            if key in previous:
                out[key] = previous[key]
                carried.append(key)
        if carried:
            out["carried_from_first_run"] = carried
    if not args.skip_divergence:
        rows, summary = divergence(presets, args.episodes, args.steps, args.seed,
                                   args.task_phase, args.reset_mode)
        out["divergence"] = summary
        out["divergence_per_episode"] = rows

    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}")
    if "throughput_ratio" in out:
        for preset, ratio in out["throughput_ratio"].items():
            print(f"  {preset:6s} env.step(n=1) x{ratio['env_step_n1_vs_v9']:.2f}  "
                  f"n={args.n} x{ratio[f'env_step_n{args.n}_vs_v9']:.2f}  "
                  f"mj_step time x{ratio['mj_step_time_vs_v9']:.2f}")
    for preset, values in out.get("divergence", {}).items():
        print(f"  {preset:6s} max |dz| {values['max_abs_torso_z_delta']:.6f}  "
              f"obs L2 {values['mean_abs_obs_l2']:.4f}  "
              f"healthy flips {values['healthy_disagreements']}  "
              f"contact flips {values['floor_contact_disagreements']}  "
              f"|dR| {values['abs_return_delta_mean']}")


if __name__ == "__main__":
    main()
