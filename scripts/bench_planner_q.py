"""Is the planner's action-value learnable where its action is not?

The measured situation (benchmarks/bc_teacher_benchmark.json, critic_teacher_benchmark.json): no predictor
of the executed MPC action beats a constant, because that action is the argmin of a stochastic search whose
consecutive-action MSE (0.7567) exceeds the action's own energy (0.3806). But the same search evaluates
samples*iterations candidate sequences per replan and assigns each a predicted return, and those pairs are
samples of Q(s, a) - the object a policy can be extracted from without ever imitating a noisy argmin. The
lookahead *value* (max over those candidates) was already shown learnable (ridge R2 0.7722 on held-out
episodes), which is why this is a question and not a hope: the max is learnable, the argmin is not; Q is the
object in between.

Nothing is trained and no environment is stepped here. `collect_mpc_demos.py --record-candidates` writes
the triples; this script fits them and says whether the direction is worth an arm.

Pre-registered gate (fixed before any candidate dataset existed):

    within-state R2 >= 0.30   AND   mean per-state Pearson >= 0.50   AND   elite hit rate >= 0.50

where "within-state" removes each plan's mean from both sides, so a predictor that only knows the state
scores 0 by construction. Elite hit rate is the share of held-out plans whose predicted argmax lands in the
true top 10% of that plan's candidates. Below any threshold the line closes without a training run; above
all three, the next step is to extract the argmax policy and compare it, on the 20 scored episodes, with
the teacher's own upright-arrival column.

    python scripts/bench_planner_q.py --self-check
    python scripts/bench_planner_q.py --candidates benchmarks/demos/mpc_candidates.npz \
        --out benchmarks/planner_q_probe.json
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WITHIN_R2_MIN = 0.30
PEARSON_MIN = 0.50
ELITE_HIT_MIN = 0.50
ELITE_FRACTION = 0.10


# --------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------
def load_candidates(path, holdout_episodes):
    """Rows grouped by planning step, split by episode - never by row.

    A held-out *row* from an episode the model trained on is not held out: consecutive plans in one
    rollout sit on nearly the same state. The last ``holdout_episodes`` episodes are the unit of
    separation, the same rule the teacher-predictability probes used.
    """
    doc = np.load(path)
    obs, cand = doc["obs"], doc["candidates"]
    ep = doc["plan_episode"]
    if obs.ndim != 2 or cand.ndim != 3:
        raise ValueError(f"unexpected shapes: obs {obs.shape}, candidates {cand.shape}")
    if cand.shape[0] != obs.shape[0] or ep.shape[0] != obs.shape[0]:
        raise ValueError("obs, candidates and plan_episode disagree on the number of plans")
    episodes = np.unique(ep)
    if len(episodes) <= holdout_episodes:
        raise ValueError(f"{len(episodes)} episodes cannot hold out {holdout_episodes}")
    holdout = episodes[-holdout_episodes:]
    train = ~np.isin(ep, holdout)
    return {"obs": obs, "candidates": cand, "episode": ep}, train, np.isin(ep, holdout), holdout


def flatten(mask_plans, data, mode="interactions"):
    """(features, returns, group ids) for the selected plans, under one fixed model class.

    ``mode`` is one of:
      * ``state``      - the state alone, the "no action information" control;
      * ``linear``     - state and action linearly;
      * ``interactions`` - state, action, action squared and every state x action product.
    The last one is the primary model, and the reason is mechanical rather than convenient: Q(s, a)
    enters through the mass matrix and the contact impulses, which are state-dependent coefficients
    multiplying the action, so a purely linear-in-action fit is not a fair test of learnability.
    """
    obs = data["obs"][mask_plans]
    cand = data["candidates"][mask_plans]
    n_plans, n_cand, width = cand.shape
    returns = cand[:, :, -1].reshape(-1)
    actions = cand[:, :, :-1].reshape(-1, width - 1)
    groups = np.repeat(np.arange(n_plans), n_cand)
    states = np.repeat(obs, n_cand, axis=0)
    if mode == "state":
        return states, returns, groups
    if mode == "linear":
        return np.concatenate([states, actions], axis=1), returns, groups
    if mode == "interactions":
        interactions = (states[:, :, None] * actions[:, None, :]).reshape(states.shape[0], -1)
        return (np.concatenate([states, actions, actions ** 2, interactions], axis=1),
                returns, groups)
    raise ValueError(f"unknown feature mode {mode!r}")


# --------------------------------------------------------------------------------------------
# ridge and the within-state metrics
# --------------------------------------------------------------------------------------------
def fit_ridge(features, target, alpha=1.0):
    mu = features.mean(axis=0)
    sd = features.std(axis=0)
    sd[sd < 1e-8] = 1.0
    x = (features - mu) / sd
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += alpha
    weights = np.linalg.solve(gram, x.T @ target)
    return {"mu": mu, "sd": sd, "weights": weights}


def predict_ridge(model, features):
    x = (features - model["mu"]) / model["sd"]
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return x @ model["weights"]


def center_by_group(values, groups):
    """Subtract each group's mean from ``values``."""
    _, inverse = np.unique(groups, return_inverse=True)
    means = (np.bincount(inverse, weights=values) / np.bincount(inverse))[inverse]
    return values - means


def within_state_metrics(target, prediction, groups):
    """Metrics that can only be earned by varying with the action, inside a state.

    The per-group mean is removed from *both* target and prediction before the variance is taken, so a
    predictor that is a function of the state alone scores exactly zero however well it fits the level.
    ``within_r2`` retains the historical uncentered residual ``sum((y - p)**2)`` for reproducibility of
    ``benchmarks/planner_q_probe.json``; ``within_r2_centered`` removes ``p.mean()`` as well as
    ``y.mean()`` so a state-only predictor scores ``0.0`` by construction.
    """
    order = np.argsort(groups, kind="stable")
    target, prediction, groups = target[order], prediction[order], groups[order]
    boundaries = np.flatnonzero(np.diff(groups)) + 1
    ss_res = ss_res_centered = ss_within = 0.0
    rhos, hits, regrets = [], [], []
    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(groups)]):
        y, p = target[start:stop], prediction[start:stop]
        if y.size < 4:
            continue
        y_centered = y - y.mean()
        p_centered = p - p.mean()
        ss_within += float(np.sum(y_centered ** 2))
        ss_res += float(np.sum((y - p) ** 2))
        ss_res_centered += float(np.sum((y_centered - p_centered) ** 2))
        if y.std() > 1e-9 and p.std() > 1e-9:
            rhos.append(float(np.corrcoef(y, p)[0, 1]))
        elite = max(1, int(round(ELITE_FRACTION * y.size)))
        top = np.argsort(-y)[:elite]
        pick = int(np.argmax(p))
        hits.append(float(pick in set(top.tolist())))
        span = float(y.max() - y.min())
        regrets.append(float((y.max() - y[pick]) / span) if span > 1e-9 else 0.0)
    return {
        "within_r2": 1.0 - ss_res / ss_within if ss_within > 0 else float("nan"),
        "within_r2_centered": 1.0 - ss_res_centered / ss_within if ss_within > 0 else float("nan"),
        "mean_pearson": float(np.mean(rhos)) if rhos else float("nan"),
        "elite_hit_rate": float(np.mean(hits)) if hits else float("nan"),
        "mean_regret_fraction": float(np.mean(regrets)) if regrets else float("nan"),
        "groups_scored": len(hits),
    }


def gate(metrics, centered=False):
    r2_key = "within_r2_centered" if centered else "within_r2"
    return {
        "within_r2": metrics[r2_key] >= WITHIN_R2_MIN,
        "mean_pearson": metrics["mean_pearson"] >= PEARSON_MIN,
        "elite_hit_rate": metrics["elite_hit_rate"] >= ELITE_HIT_MIN,
    }


def probe(data, train_mask, holdout_mask):
    """Fit the three baselines on train plans, score on held-out plans."""
    rows = {}
    for name, mode in (("state_only", "state"), ("joint_linear", "linear"),
                       ("joint_interactions", "interactions")):
        x_train, y_train, _ = flatten(train_mask, data, mode)
        x_hold, y_hold, g_hold = flatten(holdout_mask, data, mode)
        model = fit_ridge(x_train, y_train, alpha=1.0)
        rows[name] = {"features": int(x_train.shape[1]),
                      **within_state_metrics(y_hold, predict_ridge(model, x_hold), g_hold)}
    # The trivial reference: predict each plan's own mean. Within-state R2 is exactly 0 by construction,
    # and it is reported so the table cannot be read as if the level carried the signal.
    _, y_hold, g_hold = flatten(holdout_mask, data, "state")
    _, inverse = np.unique(g_hold, return_inverse=True)
    means = (np.bincount(inverse, weights=y_hold) / np.bincount(inverse))[inverse]
    rows["plan_mean"] = {"features": 0, **within_state_metrics(y_hold, means, g_hold)}
    won = gate(rows["joint_interactions"])
    return rows, won


def probe_centered(data, train_mask, holdout_mask, alpha=10.0):
    """Fit ridge on within-plan centered returns y - y_plan_mean and score with centered within-state R2."""
    rows = {}
    for name, mode in (("state_only", "state"), ("joint_linear", "linear"),
                       ("joint_interactions", "interactions")):
        x_train, y_train, g_train = flatten(train_mask, data, mode)
        x_hold, y_hold, g_hold = flatten(holdout_mask, data, mode)
        y_train_centered = center_by_group(y_train, g_train)
        model = fit_ridge(x_train, y_train_centered, alpha=alpha)
        rows[name] = {"features": int(x_train.shape[1]),
                      **within_state_metrics(y_hold, predict_ridge(model, x_hold), g_hold)}
    _, y_hold, g_hold = flatten(holdout_mask, data, "state")
    _, inverse = np.unique(g_hold, return_inverse=True)
    means = (np.bincount(inverse, weights=y_hold) / np.bincount(inverse))[inverse]
    rows["plan_mean"] = {"features": 0, **within_state_metrics(y_hold, means, g_hold)}
    won = gate(rows["joint_interactions"], centered=True)
    return rows, won


# --------------------------------------------------------------------------------------------
# self-check: the probe must PASS on a signal and FAIL on noise
# --------------------------------------------------------------------------------------------
def _synthetic(kind, rng, plans=60, candidates=24, obs_dim=6, act_dim=3):
    obs = rng.normal(size=(plans, obs_dim))
    actions = rng.uniform(-1, 1, size=(plans, candidates, act_dim))
    if kind == "signal":
        # A value that is a smooth function of state and action, plus a little noise: exactly the
        # structure the probe claims to detect.
        values = (2.0 * obs[:, None, 0] - 1.5 * obs[:, None, 1] * actions[:, :, 0]
                  + 0.8 * actions[:, :, 1] ** 2)
    else:
        values = np.zeros((plans, candidates))
    values = values + rng.normal(scale=0.1 if kind == "signal" else 1.0, size=values.shape)
    return {"obs": obs.astype(np.float32),
            "candidates": np.concatenate([actions, values[:, :, None]], axis=2).astype(np.float32),
            "episode": np.repeat(np.arange(plans // 3), 3)[:plans].astype(np.int32)}


def self_check():
    rng = np.random.default_rng(0)
    failures = []
    for kind, should_pass in (("signal", True), ("noise", False)):
        data = _synthetic(kind, rng)
        episodes = np.unique(data["episode"])
        holdout = episodes[-3:]
        train = ~np.isin(data["episode"], holdout)
        keep = np.isin(data["episode"], holdout)
        rows, won = probe(data, train, keep)
        passed = all(won.values())
        print(f"[self-check] {kind:6s} within_r2={rows['joint_interactions']['within_r2']:+.3f} "
              f"pearson={rows['joint_interactions']['mean_pearson']:+.3f} "
              f"elite_hit={rows['joint_interactions']['elite_hit_rate']:.2f} -> "
              f"{'PASS' if passed else 'FAIL'}")
        if passed != should_pass:
            failures.append(f"{kind}: probe said {'PASS' if passed else 'FAIL'}")
        if rows["plan_mean"]["within_r2"] > 1e-6:
            failures.append(f"{kind}: the plan-mean baseline earned within-state R2 "
                            f"({rows['plan_mean']['within_r2']:.3f}), so the metric is leaking")
    if failures:
        raise SystemExit("self-check failed: " + "; ".join(failures))
    print("[self-check] ok: the gate opens on signal and stays shut on noise")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=os.path.join("benchmarks", "demos", "mpc_candidates.npz"))
    ap.add_argument("--holdout-episodes", type=int, default=3)
    ap.add_argument("--out", default="benchmarks/planner_q_probe.json")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return

    path = os.path.join(ROOT, args.candidates)
    data, train_mask, holdout_mask, holdout = load_candidates(path, args.holdout_episodes)
    rows, won = probe(data, train_mask, holdout_mask)
    passed = all(won.values())
    result = {
        "protocol": (f"ridge probes on the CEM's own candidate evaluations: {int(train_mask.sum())} "
                     f"training plans, {int(holdout_mask.sum())} held-out plans from episodes "
                     f"{holdout.tolist()}; every predictor scored on the same held-out candidates, "
                     "within-state (each plan's mean removed from target and prediction)"),
        "gate": {"within_r2_min": WITHIN_R2_MIN, "mean_pearson_min": PEARSON_MIN,
                 "elite_hit_rate_min": ELITE_HIT_MIN,
                 "elite_fraction_of_candidates": ELITE_FRACTION},
        "cells": rows,
        "gate_passed": won,
        "verdict": "extract" if passed else "close",
        "decision_rule": ("above all three thresholds the next step is to extract the argmax policy and "
                          "score it on the 20 published episodes against the teacher's own upright "
                          "arrivals; below any of them no training run is spent"),
    }
    out = os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"cells": {k: {m: round(v, 3) for m, v in c.items()}
                                for k, c in rows.items()},
                      "gate_passed": won, "verdict": result["verdict"]}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
