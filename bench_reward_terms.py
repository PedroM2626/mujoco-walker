"""What each reward term actually pays the 40M-step SAC actor, under both weight sets.

The evaluation scripts call `gym.make("WalkerRagdoll-v0", ...)` with the environment's default
reward weights, while `train_walker.py` trains with its own shaping kwargs (standing_reward=0.0,
target_direction_reward_weight=200, target_progress_reward_weight=300, stability_reward_weight=20,
stillness_penalty_weight=5, lateral_drift_penalty_weight=3). That makes the returns in the Phase-1
and Phase-3 tables a different quantity from the one the agents optimised, so the comparison has
to be measured rather than asserted. Two hand-written behaviours (command zero, hold every
actuator at full extension) are scored the same way, so the ranking of "does nothing" against
"tries hard" is visible in the same units.

    python bench_reward_terms.py --episodes 10 --out benchmarks/reward_term_breakdown.json
"""

import argparse
import json
import os

import numpy as np
import torch

import envs.walker_ragdoll_env  # noqa: F401  (registers WalkerRagdoll-v0)
import gymnasium as gym
import eval_phase1 as ev

TRAIN_KWARGS = dict(
    standing_reward=0.0, target_direction_reward_weight=200.0,
    target_progress_reward_weight=300.0, stand_height_reward_weight=100.0,
    stability_reward_weight=20.0, stillness_penalty_weight=5.0,
    lateral_drift_penalty_weight=3.0)
DEFAULT_KWARGS = {}
CKPT = os.path.join("checkpoints", "walker_target_v1", "sac_actor_40000000.pt")


def episode(fn, kwargs, seed, steps, reset):
    env = gym.make("WalkerRagdoll-v0", reset_mode="mixed", task_phase="target", **kwargs)
    obs, info = env.reset(seed=seed)
    reset()
    total, terms, dists, vels = 0.0, {}, [], []
    for _ in range(steps):
        obs, reward, terminated, _truncated, info = env.step(fn(obs))
        total += float(reward)
        for k, value in info.items():
            if k.startswith("reward_"):
                terms[k] = terms.get(k, 0.0) + float(value)
        dists.append(float(info["target_distance"]))
        vels.append(float(info["x_velocity"]))
        if terminated:
            break
    env.close()
    return {"return": total, "terms": terms, "closest_approach": min(dists),
            "mean_x_velocity": float(np.mean(vels)), "steps": len(dists)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--out", default=os.path.join("benchmarks", "reward_term_breakdown.json"))
    args = ap.parse_args()

    _, _, _, policy, reset = ev.build_policy(args.checkpoint, torch.device("cpu"))
    behaviours = {
        "sac_policy": (lambda obs: np.asarray(policy(obs), dtype=np.float64).reshape(-1), reset),
        "zero_action": (lambda obs: np.zeros(17), lambda: None),
        "full_extension": (lambda obs: np.ones(17), lambda: None),
    }
    out = {"checkpoint": args.checkpoint, "episodes": args.episodes, "seed": args.seed,
           "steps_per_episode": args.steps,
           "training_kwargs": TRAIN_KWARGS, "eval_default_kwargs": DEFAULT_KWARGS,
           "weights": {}}
    for label, kwargs in (("training_weights", TRAIN_KWARGS), ("eval_defaults", DEFAULT_KWARGS)):
        rows = {}
        for name, (fn, rst) in behaviours.items():
            eps = [episode(fn, kwargs, args.seed + i, args.steps, rst)
                   for i in range(args.episodes)]
            terms = {k: float(np.mean([e["terms"].get(k, 0.0) for e in eps])) for k in eps[0]["terms"]}
            rows[name] = {
                "return_per_step": round(float(np.mean([e["return"] for e in eps])) / args.steps, 3),
                "closest_approach_m": round(float(np.mean([e["closest_approach"] for e in eps])), 3),
                "mean_x_velocity": round(float(np.mean([e["mean_x_velocity"] for e in eps])), 4),
                "terms_per_step": {k: round(v / args.steps, 3)
                                   for k, v in sorted(terms.items(), key=lambda kv: -abs(kv[1]))},
            }
        out["weights"][label] = rows
        print(f"\n===== {label} =====")
        for name, row in rows.items():
            top = list(row["terms_per_step"].items())[:4]
            print(f"{name:16} per-step {row['return_per_step']:+8.2f} "
                  f"closest {row['closest_approach_m']:5.2f} m  vx {row['mean_x_velocity']:+.4f}  "
                  + " ".join(f"{k.replace('reward_', '')}={v:+.2f}" for k, v in top))

    os.makedirs(os.path.dirname(os.path.join(os.getcwd(), args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
