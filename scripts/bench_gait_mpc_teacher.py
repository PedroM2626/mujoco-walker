"""Route 4 (corrected): Gait-objective + knot-held MPC teacher benchmark across all three layers.

Why this exists:
  1. The baseline sampling MPC (`benchmarks/mpc_target_rows.json`) reaches the marker by lunging below
     the standing band (`share_while_standing = 0.279` on the budget cell, `0.216` on the baseline cell,
     `0/20` episodes holding the standing band for 2 continuous seconds). Why? Two mechanisms in the
     planner's objective:
       * `envs/walker_ragdoll_env.py` defines `stand_height = clip((z - 0.65) / 0.60, 0, 1)`, so
         `standing_gate` pays up to 58.3% of full locomotion reward (`target_progress` weight 300 +
         `target_direction` weight 200) while torso height is in `0.65 < z <= 1.0 m` - below the
         `z > 1.0 m` standing band. Paying locomotion below the band incentivises diving and crawling;
         a gait objective must *zero* locomotion below `z = 1.0 m`, not pay more of it.
       * `scripts/bench_mpc.py` and `scripts/collect_mpc_demos.py` instantiate `WalkerRagdoll-v0` with
         the default `terminate_when_unhealthy=False`, so the `-500` fall penalty and `if terminated:
         break` inside `CrossEntropyPlanner._evaluate` never fire when a candidate rollout dives onto
         the floor. Conversely, enabling `terminate_when_unhealthy=True` unconditionally while the
         robot is recovering from the floor arms `_ever_healthy=True` the first frame `z` crosses
         `1.001 m` and punishes the stand-up wobble by `-500` on the next frame. The two-state rule
         (`settled_fall_termination`: enable `terminate_when_unhealthy` inside `_evaluate` only when the
         plan starts already settled in standing posture, `z0 > 1.12 m` and `upright0 > 0.80`) closes
         the lunge exploit without booby-trapping recovery from the floor.
  2. The baseline CEM samples independent 100 Hz white noise across `horizon * 17` dimensions (`hold=1`)
     and resets `std = 0.6` at every replan with zero penalty on `||a_t - a_{t-1}||^2`. That makes
     consecutive executed actions differ by MSE `0.7567` against action energy `0.3806` (ratio `1.988`),
     and leaves the first action `a_0` (1 step = 0.01 s) accounting for ~`1/H = 1%` of the variance of a
     100-step random rollout return. Holding each sampled control knot for `hold=5` steps with a
     slew-rate penalty `slew_weight * ||a_t - a_{t-1}||^2` and decaying continuation variance cuts
     action jitter by 6x-11x (below `0.40x` action energy) and makes held-out teacher actions
     predictable by a linear ridge regressor over the zero baseline.
  3. In `scripts/bench_planner_q.py`, `within_state_metrics` documented removing the per-group mean from
     both target and prediction, but line 136 computed `ss_res = sum((y - p)**2)` without removing
     `p.mean()`, and `fit_ridge` fitted uncentered returns whose between-state variance (`std ~ 3833`)
     swamped within-state action variance. Reconciling that metric on the committed 28-episode
     `benchmarks/demos/mpc_candidates.npz` puts `state_only` at `0.0000` by construction (instead of
     `-2.095`) and moves `joint_interactions` from `within_r2 = -0.3507, pearson = 0.0043` to
     `within_r2_centered = +0.0130, pearson = +0.1152` - matching the theoretical `~1/H = 1%` variance
     share of a 1-step action in a 100-step random rollout, and confirming that the pre-registered
     `Q(s, a)` gate (`within_r2 >= 0.30, pearson >= 0.50, elite_hit >= 0.50`) stays closed.

Usage:
    python scripts/bench_gait_mpc_teacher.py --reconcile-only
    python scripts/bench_gait_mpc_teacher.py --run-all
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import bench_approach_mechanism as bam  # noqa: E402
import bench_planner_q as bpq  # noqa: E402
from bench_mpc import CrossEntropyPlanner  # noqa: E402


def _published_4ep_summary(artifact_path):
    """Pool the first 4 episodes (seeds 11..14) of a committed 20-episode MPC artifact."""
    with open(os.path.join(ROOT, artifact_path), encoding="utf-8") as handle:
        doc = json.load(handle)
    eps = doc["per_episode"][:4]
    pooled = bam.pooled(eps)
    return {
        "artifact": artifact_path,
        "episodes": len(eps),
        "seeds": [e["seed"] for e in eps],
        "reach": sum(1 for e in eps if e["reached_target"]),
        "upright_arrivals": sum(1 for e in eps if e["reached_target_upright"]),
        "pct_steps_in_band": pooled["mean_pct_steps_in_band"],
        "share_while_standing": pooled["share_while_standing"],
        "closing_while_standing_m": pooled["closing_while_standing_m"],
        "closing_while_down_m": pooled["closing_while_down_m"],
        "max_longest_band_run_s": pooled["max_longest_band_run_s"],
        "episodes_with_band_run_ge_2s": pooled["episodes_with_band_run_ge_2s"],
        "all_20ep_share_while_standing": doc["all_episodes"]["share_while_standing"],
        "all_20ep_pct_steps_in_band": doc["all_episodes"]["mean_pct_steps_in_band"],
        "all_20ep_max_longest_band_run_s": doc["all_episodes"]["max_longest_band_run_s"],
    }


def _clean_row(row):
    out = {}
    for m, v in row.items():
        if isinstance(v, (float, np.floating)):
            out[m] = None if not np.isfinite(v) else round(float(v), 4)
        else:
            out[m] = v
    return out


def reconcile_planner_q():
    """Compare uncentered vs within-plan centered Q(s,a) probe on committed mpc_candidates.npz."""
    cand_path = os.path.join(ROOT, "benchmarks", "demos", "mpc_candidates.npz")
    data, train_mask, holdout_mask, holdout = bpq.load_candidates(cand_path, holdout_episodes=3)
    uncentered_rows, uncentered_won = bpq.probe(data, train_mask, holdout_mask)
    centered_rows, centered_won = bpq.probe_centered(data, train_mask, holdout_mask, alpha=10.0)
    return {
        "candidates_artifact": "benchmarks/demos/mpc_candidates.npz",
        "training_plans": int(train_mask.sum()),
        "holdout_plans": int(holdout_mask.sum()),
        "holdout_episodes": holdout.tolist(),
        "horizon_steps": 100,
        "action_hold_steps": 1,
        "theoretical_first_step_variance_share": 0.01,
        "uncentered_published": {k: _clean_row(row) for k, row in uncentered_rows.items()},
        "centered_within_plan": {k: _clean_row(row) for k, row in centered_rows.items()},
        "gate_passed_uncentered": uncentered_won,
        "gate_passed_centered": centered_won,
        "verdict": "close",
    }


def evaluate_gait_arm(label, **planner_kwargs):
    """Run a 4-episode probe (seeds 11..14) of a Gait-MPC configuration and score all 3 layers."""
    import gymnasium as gym
    import envs.walker_ragdoll_env  # noqa: F401
    from envs.reward_shaping import TRAINING_REWARD_KWARGS

    env = gym.make("WalkerRagdoll-v0", reset_mode="mixed", task_phase="target",
                   **TRAINING_REWARD_KWARGS)
    radius = float(env.unwrapped._target_radius)
    planner = CrossEntropyPlanner(**planner_kwargs)
    planner.record_candidates = True
    traces = []
    obs_all, act_all, ep_all = [], [], []
    plan_obs, plan_cands, plan_ep = [], [], []

    for ep in range(4):
        obs_ep, act_ep = [], []

        def wrapped_policy(obs):
            a = planner(obs)
            obs_ep.append(np.asarray(obs, dtype=np.float32))
            act_ep.append(np.asarray(a, dtype=np.float32))
            if planner.just_replanned and planner.last_candidates is not None:
                plan_obs.append(planner.last_obs)
                plan_cands.append(planner.last_candidates)
                plan_ep.append(ep)
            return a

        tr = bam.trace_episode(env, wrapped_policy, lambda: None, 11, ep, 1000, radius,
                               on_env=planner.bind)
        traces.append(tr)
        obs_all.append(np.stack(obs_ep))
        act_all.append(np.stack(act_ep))
        ep_all.append(np.full(len(obs_ep), ep, dtype=np.int32))
    env.close()

    pooled = bam.pooled(traces)
    diffs = np.concatenate([np.diff(a, axis=0) for a in act_all], axis=0)
    stacked_act = np.concatenate(act_all, axis=0)
    stacked_obs = np.concatenate(obs_all, axis=0)
    stacked_ep = np.concatenate(ep_all, axis=0)
    jitter = float(np.mean(diffs ** 2))
    energy = float(np.mean(stacked_act ** 2))

    # Layer 1: Episode-level held-out action predictability (train eps 0..2, val ep 3)
    train_m = stacked_ep < 3
    val_m = stacked_ep == 3
    mu = stacked_obs[train_m].mean(axis=0)
    sd = np.sqrt(stacked_obs[train_m].var(axis=0) + 1e-8)
    x_tr = np.clip((stacked_obs[train_m] - mu) / sd, -10.0, 10.0)
    x_va = np.clip((stacked_obs[val_m] - mu) / sd, -10.0, 10.0)
    y_tr, y_va = stacked_act[train_m], stacked_act[val_m]
    mse_zero = float(np.mean(y_va ** 2))
    xb_tr = np.concatenate([x_tr, np.ones((len(x_tr), 1))], axis=1)
    xb_va = np.concatenate([x_va, np.ones((len(x_va), 1))], axis=1)
    reg = 100.0 * np.eye(xb_tr.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(xb_tr.T @ xb_tr + reg, xb_tr.T @ y_tr)
    mse_ridge = float(np.mean((xb_va @ w - y_va) ** 2))

    # Layer 2: Within-state Q(s, a) probe on candidate knots
    cand_data = {
        "obs": np.stack(plan_obs).astype(np.float32),
        "candidates": np.stack(plan_cands).astype(np.float32),
        "episode": np.asarray(plan_ep, dtype=np.int32),
    }
    q_rows, q_won = bpq.probe_centered(cand_data, cand_data["episode"] < 3,
                                       cand_data["episode"] == 3, alpha=10.0)
    return {
        "label": label,
        "config": planner_kwargs,
        "gait": {
            "episodes": 4,
            "reach": sum(1 for t in traces if t["reached_target"]),
            "upright_arrivals": sum(1 for t in traces if t["reached_target_upright"]),
            "pct_steps_in_band": round(pooled["mean_pct_steps_in_band"], 2),
            "share_while_standing": round(pooled["share_while_standing"], 3),
            "closing_while_standing_m": round(pooled["closing_while_standing_m"], 2),
            "closing_while_down_m": round(pooled["closing_while_down_m"], 2),
            "max_longest_band_run_s": round(pooled["max_longest_band_run_s"], 2),
            "episodes_with_band_run_ge_2s": int(pooled["episodes_with_band_run_ge_2s"]),
            "per_episode": [
                {
                    "seed": t["seed"],
                    "reached_target": t["reached_target"],
                    "reached_target_upright": t["reached_target_upright"],
                    "pct_steps_in_band": t["pct_steps_in_band"],
                    "longest_band_run_s": t["longest_band_run_s"],
                    "closing_while_standing_m": round(t["closing_while_standing_m"], 2),
                    "closing_while_down_m": round(t["closing_while_down_m"], 2),
                    "n_falls": t["n_falls"],
                }
                for t in traces
            ],
        },
        "jitter": {
            "consecutive_action_mse": round(jitter, 4),
            "action_energy": round(energy, 4),
            "jitter_to_energy_ratio": round(jitter / max(energy, 1e-9), 3),
        },
        "action_predictability_val_mse": {
            "zero": round(mse_zero, 4),
            "ridge_alpha100": round(mse_ridge, 4),
            "ridge_beats_zero": bool(mse_ridge < mse_zero),
            "relative_mse_reduction_pct": round(100.0 * (1.0 - mse_ridge / mse_zero), 2),
        },
        "q_probe_centered": {
            k: {m: round(float(v), 4) if isinstance(v, float) else v for m, v in r.items()}
            for k, r in q_rows.items()
        },
        "q_gate_passed": q_won,
    }


# Measured 4-episode probe cells (seeds 11..14, horizon=60, samples=24, iterations=2, elite=6,
# replan=5, hold=5, action="elite-mean", objective="gait").
MEASURED_GAIT_ARMS = {
    "gait_smooth_v08": {
        "label": "Gait MPC, smooth in-band gate (no rollout-termination latch), v_ref=0.8 m/s",
        "config": {
            "horizon": 60, "samples": 24, "iterations": 2, "elite": 6, "replan": 5,
            "hold": 5, "init_std": 0.6, "min_std": 0.05, "cont_decay": 1.0, "seed": 11,
            "action": "elite-mean", "objective": "gait", "slew_weight": 0.30,
            "stability_weight": 60.0, "strict_band_gate": True,
            "settled_fall_termination": False, "overspeed_weight": 80.0,
            "target_vel": 0.8, "band_bonus": 40.0,
        },
        "gait": {
            "episodes": 4,
            "reach": 3,
            "upright_arrivals": 3,
            "pct_steps_in_band": 43.22,
            "share_while_standing": 0.346,
            "closing_while_standing_m": 4.62,
            "closing_while_down_m": 8.70,
            "max_longest_band_run_s": 1.84,
            "episodes_with_band_run_ge_2s": 0,
            "per_episode": [
                {"seed": 11, "reached_target": False, "reached_target_upright": False,
                 "pct_steps_in_band": 27.0, "longest_band_run_s": 1.84,
                 "closing_while_standing_m": 1.88, "closing_while_down_m": 1.36, "n_falls": 3},
                {"seed": 12, "reached_target": True, "reached_target_upright": True,
                 "pct_steps_in_band": 51.5, "longest_band_run_s": 0.95,
                 "closing_while_standing_m": 1.44, "closing_while_down_m": 1.72, "n_falls": 3},
                {"seed": 13, "reached_target": True, "reached_target_upright": True,
                 "pct_steps_in_band": 53.8, "longest_band_run_s": 1.70,
                 "closing_while_standing_m": 2.31, "closing_while_down_m": 1.77, "n_falls": 3},
                {"seed": 14, "reached_target": True, "reached_target_upright": True,
                 "pct_steps_in_band": 40.6, "longest_band_run_s": 1.31,
                 "closing_while_standing_m": -1.01, "closing_while_down_m": 3.85, "n_falls": 3},
            ],
        },
        "jitter": {
            "consecutive_action_mse": 0.1252,
            "action_energy": 0.3191,
            "jitter_to_energy_ratio": 0.392,
        },
    },
    "gait_settled_v11": {
        "label": "Gait MPC, settled-standing fall termination + v_ref=1.1 m/s (cont_decay=0.90)",
        "config": {
            "horizon": 60, "samples": 24, "iterations": 2, "elite": 6, "replan": 5,
            "hold": 5, "init_std": 0.5, "min_std": 0.05, "cont_decay": 0.90, "seed": 11,
            "action": "elite-mean", "objective": "gait", "slew_weight": 0.35,
            "stability_weight": 60.0, "strict_band_gate": True,
            "settled_fall_termination": True, "overspeed_weight": 40.0,
            "target_vel": 1.1, "band_bonus": 20.0,
        },
        "gait": {
            "episodes": 4,
            "reach": 1,
            "upright_arrivals": 1,
            "pct_steps_in_band": 41.25,
            "share_while_standing": 0.612,
            "closing_while_standing_m": 4.87,
            "closing_while_down_m": 3.09,
            "max_longest_band_run_s": 2.98,
            "episodes_with_band_run_ge_2s": 1,
            "per_episode": [
                {"seed": 11, "reached_target": False, "reached_target_upright": False,
                 "pct_steps_in_band": 47.8, "longest_band_run_s": 1.30,
                 "closing_while_standing_m": 2.28, "closing_while_down_m": 1.19, "n_falls": 5},
                {"seed": 12, "reached_target": True, "reached_target_upright": True,
                 "pct_steps_in_band": 61.5, "longest_band_run_s": 2.98,
                 "closing_while_standing_m": 1.58, "closing_while_down_m": 1.58, "n_falls": 3},
                {"seed": 13, "reached_target": False, "reached_target_upright": False,
                 "pct_steps_in_band": 17.0, "longest_band_run_s": 0.69,
                 "closing_while_standing_m": 0.00, "closing_while_down_m": 0.12, "n_falls": 4},
                {"seed": 14, "reached_target": False, "reached_target_upright": False,
                 "pct_steps_in_band": 38.7, "longest_band_run_s": 0.80,
                 "closing_while_standing_m": 1.01, "closing_while_down_m": 0.20, "n_falls": 4},
            ],
        },
        "jitter": {
            "consecutive_action_mse": 0.1036,
            "action_energy": 0.2662,
            "jitter_to_energy_ratio": 0.389,
        },
        "action_predictability_val_mse": {
            "zero": 0.2674,
            "ridge_alpha100": 0.2384,
            "ridge_beats_zero": True,
            "relative_mse_reduction_pct": 10.85,
        },
        "q_probe_centered": {
            "state_only": {"features": 49, "within_r2_centered": 0.0, "mean_pearson": None,
                           "elite_hit_rate": 0.075},
            "joint_linear": {"features": 66, "within_r2_centered": 0.0062, "mean_pearson": 0.0952,
                             "elite_hit_rate": 0.105},
            "joint_interactions": {"features": 916, "within_r2_centered": -0.0706,
                                   "mean_pearson": 0.0547, "elite_hit_rate": 0.065},
        },
        "q_gate_passed": {"within_r2": False, "mean_pearson": False, "elite_hit_rate": False},
    },
    "gait_settled_v08": {
        "label": "Gait MPC, settled-standing fall termination + strict v_ref=0.8 m/s (cont_decay=0.80)",
        "config": {
            "horizon": 60, "samples": 24, "iterations": 2, "elite": 6, "replan": 5,
            "hold": 5, "init_std": 0.5, "min_std": 0.05, "cont_decay": 0.80, "seed": 11,
            "action": "elite-mean", "objective": "gait", "slew_weight": 0.50,
            "stability_weight": 60.0, "strict_band_gate": True,
            "settled_fall_termination": True, "overspeed_weight": 80.0,
            "target_vel": 0.8, "band_bonus": 15.0,
        },
        "gait": {
            "episodes": 4,
            "reach": 0,
            "upright_arrivals": 0,
            "pct_steps_in_band": 54.17,
            "share_while_standing": 1.018,
            "closing_while_standing_m": 11.95,
            "closing_while_down_m": -0.21,
            "max_longest_band_run_s": 3.02,
            "episodes_with_band_run_ge_2s": 2,
            "per_episode": [
                {"seed": 11, "reached_target": False, "reached_target_upright": False,
                 "pct_steps_in_band": 40.1, "longest_band_run_s": 1.66,
                 "closing_while_standing_m": 3.04, "closing_while_down_m": 0.82, "n_falls": 5},
                {"seed": 12, "reached_target": False, "reached_target_upright": False,
                 "pct_steps_in_band": 46.6, "longest_band_run_s": 1.30,
                 "closing_while_standing_m": 2.88, "closing_while_down_m": 0.05, "n_falls": 6},
                {"seed": 13, "reached_target": False, "reached_target_upright": False,
                 "pct_steps_in_band": 63.4, "longest_band_run_s": 3.02,
                 "closing_while_standing_m": 3.67, "closing_while_down_m": -0.68, "n_falls": 3},
                {"seed": 14, "reached_target": False, "reached_target_upright": False,
                 "pct_steps_in_band": 66.6, "longest_band_run_s": 2.29,
                 "closing_while_standing_m": 2.36, "closing_while_down_m": -0.40, "n_falls": 4},
            ],
        },
        "jitter": {
            "consecutive_action_mse": 0.0703,
            "action_energy": 0.1847,
            "jitter_to_energy_ratio": 0.380,
        },
        "action_predictability_val_mse": {
            "zero": 0.1835,
            "ridge_alpha100": 0.1603,
            "ridge_beats_zero": True,
            "relative_mse_reduction_pct": 12.64,
        },
        "q_probe_centered": {
            "state_only": {"features": 49, "within_r2_centered": 0.0, "mean_pearson": None,
                           "elite_hit_rate": 0.035},
            "joint_linear": {"features": 66, "within_r2_centered": 0.0123, "mean_pearson": 0.1500,
                             "elite_hit_rate": 0.060},
            "joint_interactions": {"features": 916, "within_r2_centered": -0.0044,
                                   "mean_pearson": 0.1363, "elite_hit_rate": 0.105},
        },
        "q_gate_passed": {"within_r2": False, "mean_pearson": False, "elite_hit_rate": False},
    },
}


def _published_20ep_summary(artifact_path):
    """Read the full 20-episode (seeds 11..30) summary of a committed MPC artifact."""
    with open(os.path.join(ROOT, artifact_path), encoding="utf-8") as handle:
        doc = json.load(handle)
    eps = doc["per_episode"]
    all_ep = doc["all_episodes"]
    return {
        "artifact": artifact_path,
        "episodes": len(eps),
        "reach": sum(1 for e in eps if e["reached_target"]),
        "upright_arrivals": sum(1 for e in eps if e["reached_target_upright"]),
        "pct_steps_in_band": all_ep["mean_pct_steps_in_band"],
        "share_while_standing": all_ep["share_while_standing"],
        "closing_while_standing_m": all_ep["closing_while_standing_m"],
        "closing_while_down_m": all_ep["closing_while_down_m"],
        "max_longest_band_run_s": all_ep["max_longest_band_run_s"],
        "episodes_with_band_run_ge_2s": all_ep["episodes_with_band_run_ge_2s"],
        "episodes_never_in_band": all_ep["episodes_never_in_band"],
    }


STAGE_BC_STUDENT_AND_SAC_GAIT_20EP = {
    "bc_gait_raw": {
        "mean_return": 2098.76, "median_return": 1326.6, "reach_20ep": 0, "upright_20ep": 0,
        "episodes": 20, "max_longest_band_run_s": 0.39, "episodes_with_band_run_ge_2s": 0,
        "episodes_never_in_band": 18, "mean_pct_steps_in_band": 0.325,
        "closing_while_standing_m": 0.09, "closing_while_down_m": 2.21, "share_while_standing": 0.04,
    },
    "bc_gait_band": {
        "mean_return": 1473.76, "median_return": 1341.75, "reach_20ep": 0, "upright_20ep": 0,
        "episodes": 20, "max_longest_band_run_s": 0.49, "episodes_with_band_run_ge_2s": 0,
        "episodes_never_in_band": 18, "mean_pct_steps_in_band": 0.365,
        "closing_while_standing_m": 0.31, "closing_while_down_m": 2.16, "share_while_standing": 0.126,
    },
    "critic_gait_q": {
        "mean_return": -599.22, "median_return": -1685.23, "reach_20ep": 0, "upright_20ep": 0,
        "episodes": 20, "max_longest_band_run_s": 0.39, "episodes_with_band_run_ge_2s": 0,
        "episodes_never_in_band": 18, "mean_pct_steps_in_band": 0.24,
        "closing_while_standing_m": 0.26, "closing_while_down_m": 2.89, "share_while_standing": 0.083,
    },
    "critic_gait_q_bc": {
        "mean_return": -599.21, "median_return": -1685.23, "reach_20ep": 0, "upright_20ep": 0,
        "episodes": 20, "max_longest_band_run_s": 0.39, "episodes_with_band_run_ge_2s": 0,
        "episodes_never_in_band": 18, "mean_pct_steps_in_band": 0.24,
        "closing_while_standing_m": 0.26, "closing_while_down_m": 2.89, "share_while_standing": 0.083,
    },
    "sac_strict_band_200k": {
        "mean_return": 5649.88, "median_return": 5113.41, "reach_20ep": 0, "upright_20ep": 0,
        "episodes": 20, "max_longest_band_run_s": 0.41, "episodes_with_band_run_ge_2s": 0,
        "episodes_never_in_band": 13, "mean_pct_steps_in_band": 1.045,
        "closing_while_standing_m": 0.42, "closing_while_down_m": 15.79, "share_while_standing": 0.026,
    },
    "sac_warmstart_gait_100k": {
        "mean_return": 2373.13, "median_return": 2031.12, "reach_20ep": 1, "upright_20ep": 0,
        "episodes": 20, "max_longest_band_run_s": 0.74, "episodes_with_band_run_ge_2s": 0,
        "episodes_never_in_band": 18, "mean_pct_steps_in_band": 0.55,
        "closing_while_standing_m": 0.48, "closing_while_down_m": 9.17, "share_while_standing": 0.05,
    },
    "sac_warmstart_gait_200k": {
        "mean_return": 4355.17, "median_return": 5262.39, "reach_20ep": 0, "upright_20ep": 0,
        "episodes": 20, "max_longest_band_run_s": 0.38, "episodes_with_band_run_ge_2s": 0,
        "episodes_never_in_band": 14, "mean_pct_steps_in_band": 0.66,
        "closing_while_standing_m": 0.43, "closing_while_down_m": 16.80, "share_while_standing": 0.025,
    },
}


def build_artifact(run_all=False):
    with open(os.path.join(ROOT, "benchmarks", "bc_teacher_benchmark.json"),
              encoding="utf-8") as handle:
        bc_bench = json.load(handle)
    with open(os.path.join(ROOT, "benchmarks", "gait_bc_teacher_benchmark.json"),
              encoding="utf-8") as handle:
        gait_bc_bench = json.load(handle)
    with open(os.path.join(ROOT, "benchmarks", "gait_planner_q_probe.json"),
              encoding="utf-8") as handle:
        gait_q_probe = json.load(handle)
    if run_all:
        arms = {
            k: evaluate_gait_arm(cell["label"], **cell["config"])
            for k, cell in MEASURED_GAIT_ARMS.items()
        }
    else:
        arms = MEASURED_GAIT_ARMS

    return {
        "protocol": (
            "4-episode probe on seeds 11..14 and full 20-episode official benchmark on seeds 11..30 "
            "(reset_mode=mixed, task_phase=target, 1000 steps, published v9 world, scored on the "
            "unmodified published reward by bench_approach_mechanism.trace_episode) comparing the "
            "published CEM teachers against Gait-MPC configurations (strict z > 1.0 m locomotion gate, "
            "stability_weight=60, 5-step knot hold, slew-rate regularisation, and two-state settled-"
            "standing fall termination), plus 28-episode student imitation / offline-critic transfer "
            "(seeds 101..128), 200k-step online SAC warm-start, and within-plan centered Q(s, a) "
            "reconciliation on benchmarks/demos/mpc_candidates.npz"
        ),
        "published_comparators_4ep": {
            "mpc_baseline_24x2_h60": _published_4ep_summary("benchmarks/mpc_target_baseline.json"),
            "mpc_budget_48x3_h100": _published_4ep_summary("benchmarks/mpc_target_budget.json"),
            "mpc_elite_mean_48x3_h100": bc_bench["action_extraction_compare"],
            "mpc_argmin_28ep_jitter": bc_bench["teacher_jitter"],
        },
        "gait_mpc_arms": arms,
        "stage_a_teacher_20ep_official": {
            "published_mpc_baseline_20ep": _published_20ep_summary("benchmarks/mpc_target_baseline.json"),
            "published_mpc_budget_20ep": _published_20ep_summary("benchmarks/mpc_target_budget.json"),
            "gait_mpc_v11_20ep": _published_20ep_summary("benchmarks/mpc_gait_target_v11.json"),
            "gait_mpc_v08_20ep": _published_20ep_summary("benchmarks/mpc_gait_target_v08.json"),
        },
        "stage_b_demos_and_offline_students": {
            "teacher_28ep_demos": gait_bc_bench["teacher"],
            "teacher_jitter": gait_bc_bench["teacher_jitter"],
            "predictability_val_mse": gait_bc_bench["predictability_val_mse"],
            "beats_the_trivial_baseline": gait_bc_bench["beats_the_trivial_baseline"],
            "teacher_value_predictability": gait_bc_bench["teacher_value_predictability"],
            "gait_candidates_q_probe_28ep": {
                k: _clean_row(v) for k, v in gait_q_probe["cells"].items()
            },
            "students_20ep_gait": {
                k: STAGE_BC_STUDENT_AND_SAC_GAIT_20EP[k]
                for k in ("bc_gait_raw", "bc_gait_band", "critic_gait_q", "critic_gait_q_bc")
            },
        },
        "stage_c_online_sac_strict_band_and_warmstart": {
            "budget_steps": 200000,
            "num_envs": 32,
            "utd_ratio": 4,
            "seed": 7,
            "arms_20ep_gait": {
                k: STAGE_BC_STUDENT_AND_SAC_GAIT_20EP[k]
                for k in ("sac_strict_band_200k", "sac_warmstart_gait_100k", "sac_warmstart_gait_200k")
            },
        },
        "planner_q_reconciliation": reconcile_planner_q(),
        "findings": {
            "why_paying_below_band_is_inverted": (
                "envs/walker_ragdoll_env.py sets stand_height = clip((z - 0.65)/0.60, 0, 1), so "
                "locomotion terms already pay up to 58.3% below the z = 1.0 m standing band; "
                "paying more locomotion below the band rewards lunging and crawling, whereas "
                "zeroing locomotion outside (z > 1.0, upright > 0.7) forces the planner to stand "
                "before advancing."
            ),
            "two_state_termination_rule": (
                "Enabling terminate_when_unhealthy=True unconditionally inside _evaluate() arms "
                "_ever_healthy on the first frame z crosses 1.001 m during a dynamic stand-up from "
                "the floor and punishes the subsequent settling wobble by -500. Enabling it only "
                "when plan() starts already settled in standing posture (z0 > 1.12 m, upright0 > 0.80) "
                "prevents diving from a stand without booby-trapping recovery."
            ),
            "layer3_gait_verdict": (
                "Under the corrected gait objective, in-band time rises from 12.1%-18.2% to "
                "41.2%-54.2%, continuous standing runs reach 2.98 s-3.02 s (crossing the 2.0 s "
                "threshold no published MPC or learned arm had crossed), and the share of approach "
                "closed while standing rises from 0.279 to 0.612 (at v_ref=1.1 m/s) and 1.018 "
                "(at v_ref=0.8 m/s, 11.95 m closed standing vs -0.21 m while down)."
            ),
            "layer1_action_imitability_verdict": (
                "Holding control knots for 5 steps with slew-rate regularisation cuts consecutive-"
                "action MSE by 6x-11x (0.7567 -> 0.0703-0.1252, dropping from 1.99x action energy "
                "to 0.38x-0.39x) and allows a held-out ridge regressor to beat the constant-zero "
                "predictor by 10.85%-12.64% where both argmin and 1-step elite-mean previously lost."
            ),
            "layer2_q_probe_verdict": (
                "Fixing the missing prediction centering in bench_planner_q.within_state_metrics "
                "and centering training returns per plan puts state_only at 0.0000 by construction "
                "(instead of -2.095) and raises joint_interactions on mpc_candidates.npz from "
                "within_r2 = -0.3507 (pearson 0.0043) to within_r2_centered = +0.0130 (pearson "
                "+0.1152), matching the theoretical ~1/H = 1% variance share of a 1-step action in "
                "a 100-step random rollout. Even with 5-step knots (pearson +0.1500), the "
                "pre-registered Q(s, a) gate (>= 0.30 R2, >= 0.50 pearson, >= 0.50 elite hit) "
                "remains closed."
            ),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-all", action="store_true",
                    help="re-run the three 4-episode gait-MPC arms from scratch (~30 min on CPU)")
    ap.add_argument("--reconcile-only", action="store_true",
                    help="rebuild the artifact using the measured gait cells and live Q reconciliation")
    ap.add_argument("--out", default="benchmarks/mpc_gait_teacher_benchmark.json")
    args = ap.parse_args()

    payload = build_artifact(run_all=args.run_all)
    out = os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
