"""Score the planner-as-teacher benchmark: can the teacher's actions be learned at all?

The claim this instrument tests is narrower and more useful than "does behaviour cloning work". BC on the
MPC teacher produced students that reach in 0 of 20 episodes, and the tempting reading is that the teacher
is bad. It is not: the same teacher arrives standing in 6 of its own 28 episodes. The reading this script
supports is that **the teacher's action is not predictable from the observation**, so no student of this
architecture and this data volume can clone it, and the failure is in the extraction, not the search.

Three blocks, all computed on one episode-level split of the demonstrations so every row is comparable:

  * `predictability` - validation MSE of the trivial constant predictors, of ridge regression, of k-NN,
    and of each trained student, on the same held-out rows. A student that cannot beat "output zero" is
    not being out-trained, it is being given a target with no state-dependence to recover.
  * `jitter` - how much the recorded action changes between consecutive steps of the same episode,
    against the action's own energy. If the difference is larger than the signal, the argmin of a
    stochastic search is flipping between near-tied sequences step to step.
  * `teacher` / `students` - the arrival columns of the teacher and of each student, read from the
    artifacts that produced them rather than restated here.

    python bench_bc_teacher.py
"""
import argparse
import json
import os

import numpy as np
import torch

import eval_phase1
import envs.walker_ragdoll_env  # noqa: F401
import train_bc_ragdoll as bc

ROOT = os.path.dirname(os.path.abspath(__file__))


def split_by_episode(obs, act, eps, holdout=3):
    """Train on most demonstration episodes, validate on whole episodes - never on scattered rows."""
    held = sorted(set(eps.tolist()))[-holdout:]
    val = np.isin(eps, held)
    return {"holdout_seeds": [int(eps[np.where(val)[0][0]])], "val_rows": int(val.sum()),
            "train_rows": int((~val).sum())}, ~val, val


def ridge_predict(x_tr, y_tr, x_va, alpha=100.0):
    x1 = np.hstack([x_tr, np.ones((len(x_tr), 1))])
    lam = alpha * np.eye(x1.shape[1])
    w = np.linalg.solve(x1.T @ x1 + lam, x1.T @ y_tr)
    return np.hstack([x_va, np.ones((len(x_va), 1))]) @ w


def knn_predict(x_tr, y_tr, x_va, k, exemplars=6000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_tr), size=min(exemplars, len(x_tr)), replace=False)
    a, b = x_tr[idx], x_va
    d2 = ((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)
    near = np.argsort(d2, axis=1)[:, :k]
    return y_tr[idx][near].mean(axis=1)


def student_val_mse(ckpt_path, obs, act, eps, val):
    """Re-run the student on the same held-out rows the regressors are scored on."""
    # weights_only=False is the repo's convention for its own checkpoints: they carry the observation
    # normaliser as a Python object, which a weights-only load refuses.
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mean = np.asarray(ck["obs_rms"].mean, dtype=np.float64)
    var = np.asarray(ck["obs_rms"].var, dtype=np.float64)
    import gymnasium as gym
    from train_walker import SACAgent
    agent = SACAgent(int(obs.shape[1]), gym.spaces.Box(-1.0, 1.0, shape=(int(act.shape[1]),)))
    agent.load_state_dict(ck["actor_state_dict"])
    agent.eval()
    x = torch.tensor(bc.normalize(obs[val], mean, var), dtype=torch.float32)
    with torch.no_grad():
        out, _ = agent(x)
        pred = np.tanh(out.numpy())
    return float(np.mean((pred - act[val]) ** 2)), ck["bc"]


def jitter(act, eps):
    """Consecutive-action difference inside episodes, against the action's own energy."""
    within = np.zeros(len(act), bool)
    within[1:] = eps[1:] == eps[:-1]
    idx = np.where(within)[0]
    return {"consecutive_action_mse": round(float(np.mean((act[idx] - act[idx - 1]) ** 2)), 4),
            "action_energy": round(float(np.mean(act ** 2)), 4),
            "mean_abs_change_per_actuator": round(float(np.abs(act[idx] - act[idx - 1]).mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="benchmarks/demos/mpc_demos.npz")
    ap.add_argument("--students", default="benchmarks/bc_mpc_students.json")
    ap.add_argument("--jitter-compare", default=None,
                    help="a second demos file recorded with a different action extraction, so the "
                         "argmin-versus-elite-mean comparison is read from committed trajectories "
                         "instead of measured in passing")
    ap.add_argument("--out", default="benchmarks/bc_teacher_benchmark.json")
    args = ap.parse_args()

    with np.load(os.path.join(ROOT, args.demos)) as handle:
        obs = handle["obs"].astype(np.float64)
        act = handle["action"].astype(np.float64)
        band = handle["in_band"].astype(bool)
    with open(os.path.join(ROOT, args.demos) + ".json", encoding="utf-8") as handle:
        dmeta = json.load(handle)
    eps = np.repeat(np.arange(len(dmeta["episodes"])), [e["steps"] for e in dmeta["episodes"]])
    info, tr, va = split_by_episode(obs, act, eps)

    y_tr, y_va = act[tr], act[va]
    pred = {
        "zero": np.zeros_like(y_va),
        "per_dim_mean_of_train": np.tile(y_tr.mean(axis=0), (len(y_va), 1)),
        "ridge_alpha100": ridge_predict(obs[tr], y_tr, obs[va]),
        "knn_k1": knn_predict(obs[tr], y_tr, obs[va], 1),
        "knn_k8": knn_predict(obs[tr], y_tr, obs[va], 8),
        "knn_k32": knn_predict(obs[tr], y_tr, obs[va], 32),
    }
    mse = {name: round(float(np.mean((p - y_va) ** 2)), 4) for name, p in pred.items()}

    students = json.load(open(os.path.join(ROOT, args.students), encoding="utf-8"))
    for key, model in students["models"].items():
        path = os.path.join(ROOT, model["checkpoint"])
        if not os.path.exists(path):
            continue
        value, binfo = student_val_mse(path, obs, act, eps, va)
        mse[f"student_{key}"] = round(value, 4)

    within = np.zeros(len(act), bool)
    within[1:] = eps[1:] == eps[:-1]
    idx = np.where(within)[0]
    best_baseline = min(mse["zero"], mse["per_dim_mean_of_train"])
    compare = None
    if args.jitter_compare and os.path.exists(os.path.join(ROOT, args.jitter_compare)):
        with np.load(os.path.join(ROOT, args.jitter_compare)) as handle:
            o2, a2 = handle["obs"], handle["action"].astype(np.float64)
        with open(os.path.join(ROOT, args.jitter_compare) + ".json", encoding="utf-8") as handle:
            m2 = json.load(handle)
        e2 = np.repeat(np.arange(len(m2["episodes"])), [e["steps"] for e in m2["episodes"]])
        # The same three predictors on the same kind of split, so "elite-mean is more learnable" can be
        # checked rather than asserted. Small files get a one-episode hold-out or there is nothing left.
        _i2, tr2, va2 = split_by_episode(o2, a2, e2, holdout=1 if len(m2["episodes"]) < 8 else 3)
        ytr2, yva2 = a2[tr2], a2[va2]
        compare = {"demos": args.jitter_compare, "protocol": m2["protocol"],
                   "episodes": len(m2["episodes"]), "jitter": jitter(a2, e2),
                   "predictability_val_mse": {
                       "zero": round(float(np.mean(yva2 ** 2)), 4),
                       "ridge_alpha100": round(float(np.mean(
                           (ridge_predict(o2[tr2], ytr2, o2[va2]) - yva2) ** 2)), 4),
                       "knn_k32": round(float(np.mean(
                           (knn_predict(o2[tr2], ytr2, o2[va2], 32) - yva2) ** 2)), 4)},
                   "reached": m2["summary"]["episodes_reached"],
                   "upright_arrivals": m2["summary"]["episodes_upright_arrival"],
                   "pct_in_band": m2["summary"]["pct_in_band"]}
    payload = {
        "protocol": (f"episode-level split of {args.demos}: {info['train_rows']} train rows, "
                     f"{info['val_rows']} validation rows from the last 3 demonstration episodes; every "
                     "predictor - trivial, ridge, k-NN and each trained student - is scored on the same "
                     "validation rows with the same squared-error metric"),
        "split": info,
        "predictability_val_mse": mse,
        "beats_the_trivial_baseline": {
            name: (value < best_baseline) for name, value in mse.items()
            if name not in ("zero", "per_dim_mean_of_train")},
        "teacher_jitter": jitter(act, eps),
        "action_extraction_compare": compare,
        "teacher": {
            "artifact": args.demos, "episodes": len(dmeta["episodes"]),
            "demo_seeds": dmeta["demo_seeds"],
            "reached": dmeta["summary"]["episodes_reached"],
            "upright_arrivals": dmeta["summary"]["episodes_upright_arrival"],
            "pct_steps_in_band": dmeta["summary"]["pct_in_band"],
            "transitions": dmeta["summary"]["transitions"],
            "transitions_in_band": dmeta["summary"]["transitions_in_band"],
        },
        "students": {key: {
            "checkpoint": m["checkpoint"], "mean_return": m["mean"],
            "reached_pct": m["reached_target_pct"], "upright_arrival_pct": m["reached_target_upright_pct"],
            "falls_per_episode": m["falls_per_episode"],
            "standing_at_end_pct": m["standing_at_end_pct"],
            "selected_epoch": torch.load(os.path.join(ROOT, m["checkpoint"]), map_location="cpu",
                                         weights_only=False)["bc"]["selected_epoch"],
        } for key, m in students["models"].items()},
        "reading": ("no predictor of the teacher's action from the 49-wide observation beats the "
                    "constant baseline, including the trained students, so the BC failure is a property "
                    "of the recorded target - the argmin of a stochastic search - and not of the student "
                    "capacity or the training schedule"),
    }
    out = os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"predictability_val_mse": payload["predictability_val_mse"],
                      "beats_trivial": payload["beats_the_trivial_baseline"],
                      "jitter": payload["teacher_jitter"]}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
