"""Fit the planner's lookahead value as a frozen potential over states: `Phi(s)`.

The two measurements that led here are in `bench_bc_teacher.py`. The planner's action cannot be predicted
from the observation - nothing beats a constant - but its lookahead value can: ridge reaches R2 0.77 on
held-out episodes, and -0.0013 when the regression is restricted to the target components of the
observation, so the value carries posture and contact information rather than goal geometry re-derived.
A policy trained offline by ascending that value diverged. What this script does is smaller and is the
precondition for using the value inside an online loop: fit a *state function* and freeze it, so SAC can
be given `gamma * Phi(s') - Phi(s)` as shaping without anyone having to trust a learned critic while it
is still learning.

The split and the metric are the ones the value probe already used, so the number published here is
comparable to the number published there: episodes, not steps, decide train and validation, the last
`--holdout` demonstration episodes are held out, and the checkpoint kept is the epoch with the best
held-out error rather than the last epoch - the failure mode the offline critic already showed, where
validation worsened from epoch 5 onwards while training loss kept falling.

Two outputs:

  * `checkpoints/planner_potential/potential_best<EPOCH>.npz` (+ `.json` sidecar) - the arrays the trainer
    loads. Gitignored, like every weight file here;
  * `benchmarks/planner_potential.json` - the committed evidence: held-out R2 against the ridge baseline
    on the same split, the clamp range, the holdout seeds, the whole epoch history.

    python fit_planner_potential.py
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import envs.walker_ragdoll_env  # noqa: F401  (registers WalkerRagdoll-v0)
from bench_bc_teacher import ridge_predict
from evaluate_merging import EPSILON
from train_bc_ragdoll import episode_ids, load_demos, normalize

ROOT = os.path.dirname(os.path.abspath(__file__))


def r2(y_tr, y_va, pred):
    """1 - MSE / the error of predicting the *training* mean, both on the validation rows.

    Same convention as `bench_bc_teacher.py`, which divides by `mean((v_va - v_tr.mean()) ** 2)`: the
    baseline a learner would fall back to is the mean of what it saw, not the mean of the exam. Using the
    validation mean instead would shrink the denominator and print a smaller R2 for the same predictions,
    which is how two numbers about one quantity stop being comparable.
    """
    y = np.asarray(y_va, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    ss = float(np.mean((y - float(np.mean(y_tr))) ** 2))
    return float(1.0 - float(np.mean((y - p) ** 2)) / ss)


def ridge_baseline(x_tr, y_tr, x_va, y_va, alpha=100.0):
    """The value probe's own ridge, on this script's split, so the two R2 figures are the same estimator."""
    pred = ridge_predict(x_tr, y_tr.reshape(-1, 1), x_va, alpha=alpha)
    return r2(y_tr, y_va, pred), float(np.mean((y_va - pred) ** 2))


def split_by_episode(eps, holdout):
    """Last `holdout` episodes are validation. Episode-level, because neighbouring steps of one rollouts
    are nearly identical: a step-level split would score the fit on a state it already trained on."""
    order = sorted(set(eps.tolist()))
    if holdout < 1 or holdout >= len(order):
        raise SystemExit(f"--holdout must be in 1..{len(order) - 1} for {len(order)} episodes; the "
                         "checkpoint is selected on held-out episodes and a fit with none is unmeasured")
    val_ids = set(order[-holdout:])
    val = np.isin(eps, sorted(val_ids))
    return ~val, val, sorted(val_ids)


def fit(args):
    arrays, meta = load_demos(os.path.join(ROOT, args.demos))
    if "plan_value" not in arrays:
        raise SystemExit(f"{args.demos} has no plan_value column; the potential is a *value* regressor, "
                         "collect with collect_mpc_demos.py (which records the planner's lookahead)")
    obs, value = arrays["obs"], arrays["plan_value"].astype(np.float64)
    mean = np.asarray(meta["obs_rms"]["mean"], dtype=np.float64)
    var = np.asarray(meta["obs_rms"]["var"], dtype=np.float64)
    x = normalize(obs, mean, var)
    eps = episode_ids(meta)
    tr, va, val_ids = split_by_episode(eps, args.holdout)
    scale = np.sqrt(var + EPSILON)
    value_mean, value_scale = float(value[tr].mean()), float(value[tr].std())
    if value_scale <= 0:
        raise SystemExit("the recorded value has no spread on the training episodes; there is nothing "
                         "to regress")
    y = (value - value_mean) / value_scale

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    xt = torch.tensor(x[tr], dtype=torch.float32, device=device)
    yt = torch.tensor(y[tr].reshape(-1, 1), dtype=torch.float32, device=device)
    xv = torch.tensor(x[va], dtype=torch.float32, device=device)
    yv = torch.tensor(y[va].reshape(-1, 1), dtype=torch.float32, device=device)
    if yt.shape[1] != 1 or yv.shape[1] != 1:
        raise SystemExit("the value target must be a column vector; a (batch,) target beside a (batch,1) "
                         "prediction broadcasts and the loss stops meaning anything")

    dims, layers = [int(x.shape[1])] + list(args.hidden) + [1], []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]).to(device))
    weights = [p for layer in layers for p in layer.parameters()]
    opt = torch.optim.Adam(weights, lr=args.lr)

    def predict(batch):
        out = batch
        for i, layer in enumerate(layers):
            out = layer(out)
            if i + 1 < len(layers):
                out = torch.tanh(out)
        return out

    history, best = [], None
    for epoch in range(args.epochs):
        perm = torch.randperm(len(xt), device=device)
        total = 0.0
        for start in range(0, len(perm), args.batch):
            i = perm[start:start + args.batch]
            loss = F.mse_loss(predict(xt[i]), yt[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(i)
        with torch.no_grad():
            val_mse = float(F.mse_loss(predict(xv), yv))
            # R2 in the planner's own units on the held-out episodes, with the training mean as the
            # baseline - the same convention the value probe used, so the two figures sit in one column.
            val_pred = predict(xv).cpu().numpy().reshape(-1)
            r2_val = r2(y[tr], y[va], val_pred)
        history.append({"epoch": epoch + 1, "train_mse": round(total / len(perm), 5),
                        "val_mse": round(val_mse, 5), "r2_holdout": round(r2_val, 4)})
        if best is None or val_mse < best["val_mse"]:
            best = {"epoch": epoch + 1, "val_mse": val_mse, "r2": r2_val,
                    "state": [{k: v.detach().cpu().numpy() for k, v in layer.state_dict().items()}
                              for layer in layers]}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:3d} train={history[-1]['train_mse']:.4f} "
                  f"val={val_mse:.4f} R2={r2_val:.4f}", flush=True)
    if best is None:
        raise SystemExit("no epoch completed; nothing to export")

    r2_ridge, _ridge_mse = ridge_baseline(x[tr], y[tr], x[va], y[va])
    # The published value probe fitted its ridge on raw observations, and ridge with a fixed penalty is
    # not invariant to input scaling, so that number is not comparable to the one above unless it is
    # re-derived here. Refitting it on the same raw inputs reproduces it and says which of the two figures
    # is a like-for-like comparison with the network.
    r2_ridge_raw, _ = ridge_baseline(np.asarray(arrays["obs"], dtype=np.float64)[tr],
                                     value[tr], np.asarray(arrays["obs"], dtype=np.float64)[va],
                                     value[va])
    os.makedirs(os.path.join(ROOT, args.out_dir), exist_ok=True)
    stem = os.path.join(ROOT, args.out_dir, f"potential_best{best['epoch']}")
    np.savez(stem + ".npz",
             **{f"W{i}": np.ascontiguousarray(sd["weight"].T) for i, sd in enumerate(best["state"])},
             **{f"B{i}": sd["bias"] for i, sd in enumerate(best["state"])},
             obs_mean=mean, obs_scale=scale,
             value_mean=np.array(value_mean), value_scale=np.array(value_scale))
    clamp = [float(value.min()), float(value.max())]
    sidecar = {
        "demos": args.demos, "arch": list(dims), "obs_dim": int(x.shape[1]),
        "hidden": list(args.hidden), "epochs": args.epochs, "selected_epoch": best["epoch"],
        # The seed and the device are what make this file reproducible rather than merely described: a
        # potential a shaped arm names has to be refittable, or the arm's provenance is a claim about a
        # weight file nobody can produce again.
        "seed": args.seed, "device": args.device, "lr": args.lr, "batch": args.batch,
        "clamp_value": [round(clamp[0], 3), round(clamp[1], 3)],
        "value_mean": round(value_mean, 3), "value_scale": round(value_scale, 3),
        "holdout_episodes": [int(i) for i in val_ids],
        "holdout_seeds": [int(meta["episodes"][i]["seed"]) for i in val_ids],
        "val_rows": int(va.sum()), "train_rows": int(tr.sum()),
        "r2_holdout": round(best["r2"], 4), "val_mse_holdout": round(best["val_mse"], 5),
        "r2_ridge_same_split": round(r2_ridge, 4),
        "r2_ridge_raw_inputs": round(r2_ridge_raw, 4),
        "history": history,
    }
    with open(stem + ".npz.json", "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=1)

    artifact = dict(sidecar)
    artifact.update({
        "protocol": (f"Phi is fitted on {args.demos}: the planner's own lookahead value per step, "
                     "standardised by the demonstration observation statistics, episode-level hold-out "
                     "(the last demonstrations are never trained on), checkpoint chosen by held-out MSE"),
        "eval_seeds_are_disjoint": sorted(set(meta["demo_seeds"]) &
                                          set(range(11, 31))) == [],
        "reading": ("the planner's value is fitted as a frozen state function so it can be handed to an "
                    "online learner as a potential; unlike a critic learned during the run it never has "
                    "to be trusted while it is still moving"),
    })
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=1)
    print(f"wrote {stem}.npz (+ .json) and {args.out} "
          f"(held-out R2 {best['r2']:.4f} vs ridge {r2_ridge:.4f})")
    return artifact, stem + ".npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="benchmarks/demos/mpc_preference_demos.npz")
    ap.add_argument("--out-dir", default=os.path.join("checkpoints", "planner_potential"))
    ap.add_argument("--out", default=os.path.join("benchmarks", "planner_potential.json"))
    ap.add_argument("--hidden", type=int, nargs="*", default=[128, 128])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    fit(args)


if __name__ == "__main__":
    main()
