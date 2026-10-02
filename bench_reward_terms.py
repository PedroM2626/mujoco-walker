"""What each reward term actually pays the 40M-step SAC actor, under both weight sets.

The evaluation scripts used to call `gym.make("WalkerRagdoll-v0", ...)` with the environment's
default reward weights while `train_walker.py` trained with its own shaping kwargs, so the returns
in the Phase-1 and Phase-3 tables were a different quantity from the one the agents optimised.
`envs.walker_ragdoll_env.TRAINING_REWARD_KWARGS` is now the single source for that shaping, and
this script scores the same policy under both weight sets so the size of the discrepancy is
measured rather than asserted.

Hand-written behaviours are scored the same way, so the ranking of "does nothing" against "tries
hard" is visible in the same units - and one of them is a deliberate exploit probe. The env's
posture shaping changed in v9: below z=0.65 the height term paid nothing while
`low_upright_penalty` charged 20/step for being low, i.e. posture entered the return as a
punishment and a prone robot had no posture gradient. Now posture is a bonus graded from the
floor up and that penalty defaults to zero, so `topple_forward` checks the risk that comes with
it: collapsing toward the target must not outscore staying upright.

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
from envs.reward_shaping import TRAINING_REWARD_KWARGS

DEFAULT_KWARGS = {}
CKPT = os.path.join("checkpoints", "walker_target_v1", "sac_actor_40000000.pt")


def episode(fn, kwargs, seed, steps, reset):
    env = gym.make("WalkerRagdoll-v0", reset_mode="mixed", task_phase="target", **kwargs)
    obs, info = env.reset(seed=seed)
    reset()
    total, terms, dists, vels, zs, gates = 0.0, {}, [], [], [], []
    for _ in range(steps):
        obs, reward, terminated, _truncated, info = env.step(fn(obs))
        total += float(reward)
        for k, value in info.items():
            if k.startswith("reward_"):
                terms[k] = terms.get(k, 0.0) + float(value)
        dists.append(float(info["target_distance"]))
        vels.append(float(info["x_velocity"]))
        zs.append(float(info["z_position"]))
        gates.append(float(info.get("standing_gate", 0.0)))
        if terminated:
            break
    env.close()
    return {"return": total, "terms": terms, "closest_approach": min(dists),
            "mean_x_velocity": float(np.mean(vels)), "steps": len(dists),
            "mean_z": float(np.mean(zs)), "mean_standing_gate": float(np.mean(gates))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--out", default=os.path.join("benchmarks", "reward_term_breakdown.json"))
    args = ap.parse_args()

    _, _, _, policy, reset, _rinfo = ev.build_policy(args.checkpoint, torch.device("cpu"))

    def single(idx, value):
        def fn(obs):
            a = np.zeros(17)
            a[idx] = value
            return a
        return fn

    behaviours = {
        "sac_policy": (lambda obs: np.asarray(policy(obs), dtype=np.float64).reshape(-1), reset),
        "zero_action": (lambda obs: np.zeros(17), lambda: None),
        "full_extension": (lambda obs: np.ones(17), lambda: None),
        # The two signs of the pitch actuator: whichever one topples the robot toward the target
        # is the exploit candidate under a bonus-only posture shaping, and both are reported.
        "abdomen_y_plus": (single(0, 1.0), lambda: None),
        "abdomen_y_minus": (single(0, -1.0), lambda: None),
    }
    out = {"checkpoint": args.checkpoint, "episodes": args.episodes, "seed": args.seed,
           "steps_per_episode": args.steps, "env_version": envs.walker_ragdoll_env.ENV_VERSION,
           "training_kwargs": TRAINING_REWARD_KWARGS, "eval_default_kwargs": DEFAULT_KWARGS,
           "weights": {}}
    for label, kwargs in (("training_weights", TRAINING_REWARD_KWARGS),
                          ("eval_defaults", DEFAULT_KWARGS)):
        rows = {}
        for name, (fn, rst) in behaviours.items():
            eps = [episode(fn, kwargs, args.seed + i, args.steps, rst)
                   for i in range(args.episodes)]
            terms = {k: float(np.mean([e["terms"].get(k, 0.0) for e in eps])) for k in eps[0]["terms"]}
            rows[name] = {
                "return_per_step": round(float(np.mean([e["return"] for e in eps])) / args.steps, 3),
                "closest_approach_m": round(float(np.mean([e["closest_approach"] for e in eps])), 3),
                "mean_x_velocity": round(float(np.mean([e["mean_x_velocity"] for e in eps])), 4),
                "mean_z_m": round(float(np.mean([e["mean_z"] for e in eps])), 3),
                "mean_standing_gate": round(
                    float(np.mean([e["mean_standing_gate"] for e in eps])), 3),
                "terms_per_step": {k: round(v / args.steps, 3)
                                   for k, v in sorted(terms.items(), key=lambda kv: -abs(kv[1]))},
            }
        out["weights"][label] = rows
        print(f"\n===== {label} =====")
        for name, row in rows.items():
            top = list(row["terms_per_step"].items())[:4]
            print(f"{name:16} per-step {row['return_per_step']:+8.2f} "
                  f"closest {row['closest_approach_m']:5.2f} m  vx {row['mean_x_velocity']:+.4f}  "
                  f"gate {row['mean_standing_gate']:.3f}  "
                  + " ".join(f"{k.replace('reward_', '')}={v:+.2f}" for k, v in top))

    os.makedirs(os.path.dirname(os.path.join(os.getcwd(), args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
