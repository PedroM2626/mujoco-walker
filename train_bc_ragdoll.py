"""Behaviour-cloning student for the MPC teacher: same network as the RL arms, different signal.

Why this is a benchmark and not a demo: the student is `train_walker.SACAgent`, the exact architecture the
SAC and TD3 rows use, and it is scored by the same `eval_phase1` / `bench_approach_mechanism` path, so the
comparison holds the network and the observation normalisation fixed and changes only where the labels came
from. Two cells answer the question the teacher's own numbers raise:

  * `--filter raw`  - imitate every planner step, including the ones where it is below the standing band.
    This is the honest copy of the teacher, and the teacher closes only ~28% of its ground on its feet.
  * `--filter band` - imitate only the steps the planner took inside the band. If the useful part of the
    teacher is the part that is standing, this is where it shows.

The demonstration seeds are disjoint from the scored seeds by construction: `collect_mpc_demos.py` records
that fact in its metadata and this script refuses to train when the two ranges overlap, because a student
scored on episodes it memorised would be measuring the split rather than the policy.

    python train_bc_ragdoll.py --filter raw
    python train_bc_ragdoll.py --filter band
"""
import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch

import envs.walker_ragdoll_env  # noqa: F401  (registers WalkerRagdoll-v0)
from envs.normalize_compat import RunningMeanStd
from envs.reward_shaping import TRAINING_REWARD_KWARGS
from envs.walker_ragdoll_env import ENV_VERSION
from evaluate_merging import EPSILON, CLIP
from train_walker import SACAgent

ROOT = os.path.dirname(os.path.abspath(__file__))
# The protocol every published row in this repository is scored under. Demos must come from elsewhere.
EVAL_SEEDS = set(range(11, 31))


def load_demos(path):
    # Read the arrays eagerly: np.load returns a lazy NpzFile that keeps the handle open, and on Windows
    # that is enough to make a caller's temporary directory undeletable.
    with np.load(path) as handle:
        arrays = {key: handle[key] for key in handle.files}
    with open(path + ".json", encoding="utf-8") as handle:
        meta = json.load(handle)
    overlap = EVAL_SEEDS & set(meta["demo_seeds"])
    if overlap:
        raise SystemExit(f"demo seeds overlap the scored protocol on {sorted(overlap)}; a student "
                         "trained on the episodes it is graded on measures memorisation, not the "
                         "teacher")
    return arrays, meta


def normalize(obs, mean, var):
    """The transform `evaluate_merging._policy_input` applies at eval time, computed once here.

    Kept as its own function so the trainer and the scorer cannot drift into two different notions of the
    observation: a student trained on raw states and evaluated on normalised ones is a broken benchmark
    that still produces a number.
    """
    return np.clip((np.asarray(obs, dtype=np.float64) - mean) / np.sqrt(var + EPSILON), -CLIP, CLIP)


def episode_ids(meta):
    """Map every transition row back to the episode it came from, for an episode-level split.

    The collector concatenates episodes in order, so the boundaries are recoverable from the per-episode
    step counts; splitting by step instead would leak the same trajectory across train and validation.
    """
    ids = []
    for ep_index, ep in enumerate(meta["episodes"]):
        ids.extend([ep_index] * ep["steps"])
    return np.asarray(ids)


def train(args):
    arrays, meta = load_demos(os.path.join(ROOT, args.demos))
    obs, act, band = arrays["obs"], arrays["action"], arrays["in_band"]
    mean = np.asarray(meta["obs_rms"]["mean"], dtype=np.float64)
    var = np.asarray(meta["obs_rms"]["var"], dtype=np.float64)
    keep = np.ones(obs.shape[0], dtype=bool) if args.filter == "raw" else band.astype(bool)
    if keep.sum() < args.min_transitions:
        raise SystemExit(f"filter={args.filter} leaves {int(keep.sum())} transitions, below "
                         f"--min-transitions={args.min_transitions}")
    x = torch.tensor(normalize(obs[keep], mean, var), dtype=torch.float32)
    y = torch.tensor(act[keep], dtype=torch.float32)
    ids = episode_ids(meta)[keep]

    # Episode-level split: 10% of the demonstration episodes are held out, never 10% of the rows.
    unique = np.unique(ids)
    rng = np.random.default_rng(args.seed)
    val_eps = set(rng.choice(unique, size=max(1, int(0.1 * len(unique))), replace=False).tolist())
    is_val = np.isin(ids, list(val_eps))
    device = torch.device(args.device)
    model = SACAgent(int(x.shape[1]), gym.spaces.Box(-1.0, 1.0, shape=(int(y.shape[1]),))).to(device)
    # Only the mean head is a BC target; the log-std head is frozen so the saved state dict still loads
    # through the same path eval_phase1 uses for a SAC actor.
    for param in model.fc_logstd.parameters():
        param.requires_grad_(False)
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    xt, yt = x[~is_val].to(device), y[~is_val].to(device)
    xv, yv = x[is_val].to(device), y[is_val].to(device)

    history = []
    # Best-by-validation weights, not last-epoch weights. On 28k demonstration transitions a
    # 256x256 net overfits from the first epoch, so scoring the final weights would grade the training
    # schedule rather than what imitation actually transfers - and would report "the teacher is a bad
    # teacher" when the measurement says "we kept the worst checkpoint".
    best = {"val": float("inf"), "epoch": 0, "state": None}
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(xt.shape[0], device=device)
        total = 0.0
        for start in range(0, xt.shape[0], args.batch):
            idx = perm[start:start + args.batch]
            optim.zero_grad()
            mean_out, _ = model(xt[idx])
            loss = torch.nn.functional.mse_loss(torch.tanh(mean_out), yt[idx])
            loss.backward()
            optim.step()
            total += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            mv, _ = model(xv)
            val = float(torch.nn.functional.mse_loss(torch.tanh(mv), yv)) if xv.shape[0] else None
        history.append({"epoch": epoch + 1, "train_mse": round(total / max(xt.shape[0], 1), 6),
                        "val_mse": round(val, 6) if val is not None else None})
        if val is not None and val < best["val"]:
            best = {"val": val, "epoch": epoch + 1,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:3d} train_mse={history[-1]['train_mse']:.6f} "
                  f"val_mse={history[-1]['val_mse']}", flush=True)
    if best["state"] is None:
        raise SystemExit("no validation rows, so no checkpoint can be selected by validation")
    model.load_state_dict(best["state"])
    print(f"best epoch {best['epoch']} val_mse={best['val']:.6f} "
          f"(last epoch val_mse={history[-1]['val_mse']})")

    rms = RunningMeanStd(shape=(int(x.shape[1]),))
    rms.mean, rms.var, rms.count = mean, var, int(meta["obs_rms"]["count"])
    ckpt_dir = os.path.join(ROOT, "checkpoints", args.run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    # The number in the filename is the epoch selected by validation, not a training-step budget: a
    # reader comparing it against "sac_actor_5000000" would otherwise think this arm saw 1 step of data.
    path = os.path.join(ckpt_dir, f"bc_student_best{best['epoch']}.pt")
    torch.save({
        "algo": "bc",
        "env_version": ENV_VERSION,
        "task_phase": "target",
        "target_forward_velocity": 1.2,
        "reward_kwargs": dict(TRAINING_REWARD_KWARGS),
        "actor_state_dict": model.state_dict(),
        "obs_rms": rms,
        "global_step": best["epoch"],
        "bc": {
            "demos": args.demos, "filter": args.filter, "teacher": meta["protocol"],
            "transitions_used": int(x.shape[0]), "transitions_available": int(obs.shape[0]),
            "pct_of_demos_used": round(100.0 * x.shape[0] / obs.shape[0], 2),
            "demo_seeds": meta["demo_seeds"], "eval_seeds_disjoint": True,
            "val_episodes": sorted(int(v) for v in val_eps),
            "epochs": args.epochs, "lr": args.lr, "batch": args.batch, "seed": args.seed,
            "selected_epoch": best["epoch"], "selected_val_mse": round(best["val"], 6),
            "final_train_mse": history[-1]["train_mse"], "final_val_mse": history[-1]["val_mse"],
            "history": history,
        },
    }, path)
    print(f"wrote {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="benchmarks/demos/mpc_demos.npz")
    ap.add_argument("--filter", choices=["raw", "band"], required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-transitions", type=int, default=2000)
    args = ap.parse_args()
    args.run_id = args.run_id or f"bc_mpc_{args.filter}"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
