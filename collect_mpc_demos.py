"""Collect the sampling MPC's trajectories as demonstrations for a behaviour-cloning student.

The question this feeds is not "can a network copy a planner" but "does the planner have anything worth
copying": its arrival column beats every learned arm and its gait column is the worst in the comparison,
so a student trained on its raw actions inherits the falling, and a student trained only on the steps it
took inside the standing band is the experiment that says whether the band-filtered part of the teacher
is the useful part.

Two things the file is deliberate about:

  * **Seed separation.** The scoring protocol every published row uses is seeds 11..30. Demonstrations are
    collected from a disjoint range (101.. by default), so a student that merely memorises its training
    episodes cannot score well on the eval episodes. A BC arm scored on the seeds it was trained on would
    be the same category of error as a checkpoint credited with answering a question it had already seen.
  * **One definition of the band.** The per-step `in_band` flag is the same z>1.0 and upright>0.7 test that
    `bench_approach_mechanism` splits the trace by and that `eval_phase1` uses for an upright arrival, so
    the filter and the scorer cannot drift apart into two different notions of "standing".

    python collect_mpc_demos.py --episodes 40 --samples 48 --iterations 3 --horizon 100 --replan 20
"""
import argparse
import json
import os

import numpy as np
import gymnasium as gym

import envs.walker_ragdoll_env  # noqa: F401  (registers WalkerRagdoll-v0)
from envs.reward_shaping import TRAINING_REWARD_KWARGS
import bench_mpc as mpc
import bench_approach_mechanism as bam

ROOT = os.path.dirname(os.path.abspath(__file__))


def collect(args):
    planner = mpc.CrossEntropyPlanner(horizon=args.horizon, samples=args.samples,
                                      iterations=args.iterations, elite=args.elite,
                                      replan=args.replan, seed=args.seed, action=args.action)
    env = gym.make("WalkerRagdoll-v0", reset_mode="mixed", task_phase="target",
                   **TRAINING_REWARD_KWARGS)
    unw = env.unwrapped
    radius = float(unw._target_radius)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(np.prod(env.action_space.shape))

    obs_all, act_all, band_all, dist_all, rew_all, val_all, episodes = [], [], [], [], [], [], []
    try:
        for ep in range(args.episodes):
            seed = args.demo_seed_base + ep
            obs, info = env.reset(seed=seed)
            planner.bind(env)
            o, a, b, d, rw, pv = [], [], [], [], [], []
            min_dist = float(info["target_distance"])
            min_dist_band = float("inf")
            n_steps = n_falls = 0
            was_healthy = unw.is_healthy
            for _ in range(args.steps):
                action = np.asarray(planner(obs), dtype=np.float64).reshape(-1)
                o.append(np.asarray(obs, dtype=np.float32))
                a.append(action.astype(np.float32))
                obs, reward, terminated, truncated, info = env.step(action)
                n_steps += 1
                dist = float(info["target_distance"])
                min_dist = min(min_dist, dist)
                in_band = (float(unw.data.qpos[2]) > bam.BAND_Z
                           and float(unw.upright_factor) > bam.BAND_UPRIGHT)
                if in_band:
                    min_dist_band = min(min_dist_band, dist)
                b.append(np.float32(in_band))
                d.append(np.float32(dist))
                # The reward the teacher's action actually earned, and the value the teacher had
                # assigned to the state it acted from. Together they turn the file from an
                # imitation dataset into a preference dataset: the second column is what a critic
                # would have to learn, and it is scored on the same rows as the first.
                rw.append(np.float32(reward))
                pv.append(np.float32(planner.last_value))
                healthy = unw.is_healthy
                if was_healthy and not healthy:
                    n_falls += 1
                was_healthy = healthy
                if terminated:
                    break
            o, a, b, d, rw, pv = (np.stack(o), np.stack(a), np.stack(b), np.stack(d),
                                  np.stack(rw), np.stack(pv))
            obs_all.append(o); act_all.append(a); band_all.append(b); dist_all.append(d)
            rew_all.append(rw); val_all.append(pv)
            episodes.append({
                "episode": ep, "seed": seed, "steps": int(n_steps),
                "min_target_distance": round(min_dist, 3),
                "reached_target": bool(min_dist <= radius),
                "reached_target_upright": bool(min_dist_band <= radius),
                "falls": int(n_falls),
                "pct_steps_in_band": round(100.0 * float(b.mean()), 1),
            })
            print(f"ep{ep:3d} seed{seed} steps={n_steps:4d} closest={min_dist:6.3f} "
                  f"upright_arr={episodes[-1]['reached_target_upright']} "
                  f"band={100.0 * float(b.mean()):5.1f}%", flush=True)
    finally:
        env.close()

    obs_cat = np.concatenate(obs_all)
    rms = {"mean": obs_cat.mean(axis=0).astype(np.float64),
           "var": obs_cat.var(axis=0).astype(np.float64),
           "count": int(obs_cat.shape[0])}
    return {
        "obs": np.concatenate(obs_all), "action": np.concatenate(act_all),
        "in_band": np.concatenate(band_all), "distance": np.concatenate(dist_all),
        "reward": np.concatenate(rew_all), "plan_value": np.concatenate(val_all),
    }, rms, episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--demo-seed-base", type=int, default=101,
                    help="first demo seed; the scored protocol uses 11..30, so these must not overlap")
    ap.add_argument("--seed", type=int, default=11, help="planner RNG seed (not the episode seeds)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--elite", type=int, default=12)
    ap.add_argument("--replan", type=int, default=20)
    ap.add_argument("--action", choices=["argmin", "elite-mean"], default="argmin",
                    help="passed through to the planner; elite-mean records the smooth target "
                         "(mean of the elite set's first action) instead of the search winner")
    ap.add_argument("--out", default="benchmarks/demos/mpc_demos.npz")
    args = ap.parse_args()

    arrays, rms, episodes = collect(args)
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **arrays)
    meta = {
        "protocol": (f"cross-entropy MPC (action={args.action}, samples={args.samples}, "
                     f"iterations={args.iterations}, "
                     f"horizon={args.horizon}, elite={args.elite}, replan={args.replan}) rolled out on "
                     f"{args.episodes} episodes with reset_mode=mixed, task_phase=target, "
                     "WalkerRagdoll-v0 in the published world, scored with "
                     "envs.reward_shaping.TRAINING_REWARD_KWARGS"),
        "demo_seeds": [e["seed"] for e in episodes],
        "eval_seeds_are_disjoint": not (set(e["seed"] for e in episodes) & set(range(11, 31))),
        "band_definition": f"torso z > {bam.BAND_Z} and upright > {bam.BAND_UPRIGHT} - the same test "
                           "bench_approach_mechanism splits the trace by",
        "arrays": {k: [int(v.shape[0]), int(v.shape[1]) if v.ndim > 1 else None]
                   for k, v in arrays.items()},
        "obs_rms": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rms.items()},
        "episodes": episodes,
        "summary": {
            "transitions": int(arrays["obs"].shape[0]),
            "transitions_in_band": int(arrays["in_band"].sum()),
            "pct_in_band": round(100.0 * float(arrays["in_band"].mean()), 2),
            "episodes_reached": sum(1 for e in episodes if e["reached_target"]),
            "episodes_upright_arrival": sum(1 for e in episodes if e["reached_target_upright"]),
            # Spread of the teacher's own opinion about its states. A constant would do if this were
            # near zero, and the predictability probe in bench_bc_teacher.py would be moot.
            "plan_value_std": round(float(arrays["plan_value"].std()), 3),
            "plan_value_min": round(float(arrays["plan_value"].min()), 3),
            "plan_value_max": round(float(arrays["plan_value"].max()), 3),
            "reward_mean": round(float(arrays["reward"].mean()), 4),
        },
    }
    with open(out + ".json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    print(json.dumps(meta["summary"], indent=1))
    print(f"wrote {out} and {out}.json")


if __name__ == "__main__":
    main()
