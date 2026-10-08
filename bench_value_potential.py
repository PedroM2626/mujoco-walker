"""Measure what a potential-based shaping term is actually worth, before spending a training run on it.

The shaping term is `alpha * (gamma * Phi(s') - Phi(s))`. Two things about it cannot be seen from the
regressor's held-out R2, and both decide whether the arm is informative:

  * **scale.** Phi lives in the planner's lookahead-value units, which range over thousands, while the
    published task reward is a few dozen per step. A weight of 1 can therefore be worth fifty task
    rewards, and the run would measure the potential and not the task;
  * **coverage.** Phi was fitted on the states the *planner* visits. A policy that spends its episodes on
    the floor is asking the regressor about states it never saw, and every clamped value is an opinion the
    fit was never asked to hold.

So this script rolls a checkpoint through the published environment, scores each visited state with the
frozen potential, and reports the per-step magnitudes and the clamp rate. The shaping weight the arm uses
is then fixed by a rule stated here rather than by a number chosen after seeing the result: the largest
alpha whose mean shaping magnitude is at most `--reward-fraction` of the mean task reward the same
trajectory earned.

    python bench_value_potential.py --model control=checkpoints/.../sac_actor_5000000.pt \
      --potential checkpoints/planner_potential/potential_best8.npz --episodes 20
"""

import argparse
import json
import os

import gymnasium as gym
import numpy as np

import envs.walker_ragdoll_env  # noqa: F401  (registers WalkerRagdoll-v0)
import eval_phase1
from envs.normalize_compat import RunningMeanStd  # noqa: F401  (pickle shim for the checkpoint's obs_rms)
from envs.reward_shaping import TRAINING_REWARD_KWARGS
from envs.value_potential import ValuePotential

ROOT = os.path.dirname(os.path.abspath(__file__))


def roll(env, policy, potential, seed, steps, gamma):
    """Walk one seeded episode, scoring every state the policy actually visits."""
    obs, info = env.reset(seed=seed)
    rows = []
    phi_prev = potential(obs)
    for _ in range(steps):
        action = np.asarray(policy(obs), dtype=np.float64).reshape(-1)
        obs, reward, terminated, truncated, info = env.step(action)
        value, hit_low, hit_high = potential.clamped(obs)
        successor = 0.0 if terminated else gamma * value
        rows.append({"reward": float(reward), "phi": float(phi_prev), "phi_next": value,
                     "unit_shaping": float(successor - phi_prev), "hit_low": bool(hit_low),
                     "hit_high": bool(hit_high), "terminated": bool(terminated),
                     "distance": float(info.get("target_distance", np.nan)),
                     "x_velocity": float(info.get("x_velocity", np.nan))})
        phi_prev = value
        if terminated or truncated:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], metavar="NAME=CHECKPOINT",
                    help="policy to roll, repeatable. The states it visits are the states the shaped "
                         "arm will be trained on, so the control arm is the honest thing to measure.")
    ap.add_argument("--potential", default=os.path.join("checkpoints", "planner_potential",
                                                        "potential_best8.npz"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--task-phase", default="target")
    ap.add_argument("--shaping-gamma", type=float, default=0.99,
                    help="the discount the shaped arm will use inside the term")
    ap.add_argument("--reward-fraction", type=float, default=0.5,
                    help="the rule: cap the mean shaping magnitude at this fraction of the mean task "
                         "reward. Stated before the arm runs; a weight picked after seeing a result is "
                         "tuning, not design.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join("benchmarks", "planner_potential_coverage.json"))
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("nothing to measure: pass --model NAME=CHECKPOINT (repeatable)")

    path = os.path.join(ROOT, args.potential)
    potential = ValuePotential.load(path)
    sidecar = json.load(open(path + ".json", encoding="utf-8"))
    cells = {}
    for pair in args.model:
        name, _, ckpt = pair.partition("=")
        label, phase, width, policy, reset, rinfo = eval_phase1.build_policy(
            os.path.join(ROOT, ckpt), args.device)
        kwargs = dict(rinfo["reward_kwargs"] or TRAINING_REWARD_KWARGS)
        env = gym.make("WalkerRagdoll-v0", reset_mode="mixed", task_phase=args.task_phase, **kwargs)
        per_unit, phis, step_rewards, returns = [], [], [], []
        clamp_low, clamp_high, states = 0, 0, 0
        episode_lengths = []
        try:
            for ep in range(args.episodes):
                reset()
                rows = roll(env, policy, potential, args.seed + ep, args.steps, args.shaping_gamma)
                per_unit.extend(r["unit_shaping"] for r in rows)
                phis.extend(r["phi"] for r in rows)
                step_rewards.extend(abs(r["reward"]) for r in rows)
                returns.append(sum(r["reward"] for r in rows))
                clamp_low += sum(r["hit_low"] for r in rows)
                clamp_high += sum(r["hit_high"] for r in rows)
                states += len(rows)
                episode_lengths.append(len(rows))
        finally:
            env.close()
        unit = np.asarray(per_unit, dtype=np.float64)
        # The task reward's own per-step magnitude, measured on the same trajectory the shaping would
        # ride on - an episode-level mean would divide a fixed number of terms by a varying count.
        mean_reward_per_step = float(np.mean(step_rewards)) if step_rewards else 0.0
        mean_unit_shaping = float(np.mean(np.abs(unit)))
        cap = args.reward_fraction * mean_reward_per_step
        cells[name] = {
            "checkpoint": ckpt, "algo": label, "task_phase_trained": phase,
            "obs_width": width, "episodes": args.episodes, "seed": args.seed,
            "steps": states, "mean_episode_steps": round(float(np.mean(episode_lengths)), 1),
            "mean_abs_task_reward_per_step": round(mean_reward_per_step, 4),
            "mean_abs_shaping_at_weight_1": round(mean_unit_shaping, 4),
            "shaping_to_reward_ratio_at_weight_1": round(mean_unit_shaping / mean_reward_per_step, 3)
            if mean_reward_per_step > 0 else None,
            # The weight the rule allows: the largest alpha whose mean shaping magnitude stays under
            # `reward_fraction` of the mean task reward on the same trajectory.
            "weight_by_rule": round(min(1.0, cap / mean_unit_shaping), 4) if mean_unit_shaping > 0
            else 1.0,
            "pct_states_clamped": round(100.0 * (clamp_low + clamp_high) / max(states, 1), 2),
            "pct_states_clamped_low": round(100.0 * clamp_low / max(states, 1), 2),
            "pct_states_clamped_high": round(100.0 * clamp_high / max(states, 1), 2),
            "unit_shaping_range": [round(float(unit.min()), 3), round(float(unit.max()), 3)],
            "phi_range_on_this_policy": [round(float(np.min(phis)), 3), round(float(np.max(phis)), 3)],
            "mean_return_published": round(float(np.mean(returns)), 2),
        }
    data = {
        "protocol": (f"the checkpoint is rolled through WalkerRagdoll-v0 in task_phase={args.task_phase} "
                     f"on the published seeds ({args.seed}..{args.seed + args.episodes - 1}), one env "
                     f"step per action, and every state it visits is scored with the frozen potential "
                     f"{args.potential}; the shaping term is measured at weight 1 so the scale is the "
                     f"arm's own, not an artefact of the weight"),
        "potential": {"checkpoint": args.potential, "obs_dim": sidecar["obs_dim"],
                      "r2_holdout": sidecar["r2_holdout"], "clamp_value": sidecar["clamp_value"],
                      "demos": sidecar["demos"], "selected_epoch": sidecar["selected_epoch"]},
        "reward_fraction": args.reward_fraction,
        "shaping_gamma": args.shaping_gamma,
        "cells": cells,
        "reading": ("the weight is set by the rule stated in the script - the largest alpha whose mean "
                    "shaping magnitude stays under the given fraction of the mean task reward on the "
                    "control arm's own trajectory - so the arm's strength is a design decision recorded "
                    "before its outcome is known"),
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1)
    for name, cell in cells.items():
        print(f"{name:12s} |r|={cell['mean_abs_task_reward_per_step']:8.3f} "
              f"shaping@1={cell['mean_abs_shaping_at_weight_1']:10.3f} "
              f"ratio={cell['shaping_to_reward_ratio_at_weight_1']} "
              f"weight_by_rule={cell['weight_by_rule']} "
              f"clamped={cell['pct_states_clamped']}%")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
