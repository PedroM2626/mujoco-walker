"""Offline actor-critic on the planner's transitions: learn from what it valued, not from what it did.

Behaviour cloning of the MPC teacher failed for a measured reason - no regressor of its action from the
observation beat a constant, because the executed action is the argmin of a stochastic search
(`bench_bc_teacher.py`). Its *value* is different: ridge predicts the lookahead value with R2 0.77 on the
28-episode preference set - 0.47 on the 8-episode probe that first showed it - and predicts nothing when
restricted to the target components of the observation. So the usable product of the planner is a
judgement about states, not a set of labels over actions.

This script consumes the preference dataset (`collect_mpc_demos.py` with the reward and plan_value columns)
as transitions (s, a, r, s') and fits a twin critic by TD on them. The actor is then trained against that
critic, never against a demonstrated action - except in `--actor q_bc`, which keeps a TD3+BC-style
in-distribution term, because pure Q-ascent on 28k transitions from one controller is exactly the regime
where extrapolation error wins. Both modes are reported, so the choice is measured rather than assumed.

The checkpoint loads through `eval_phase1.build_policy`'s generic actor branch, which is what lets the
student be scored on the same 20 seeded episodes and the same per-step trace as every published row.

    python scripts/train_offline_critic.py --actor q
    python scripts/train_offline_critic.py --actor q_bc
"""
import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs.walker_ragdoll_env  # noqa: F401
from envs.normalize_compat import RunningMeanStd
from envs.reward_shaping import TRAINING_REWARD_KWARGS
from envs.walker_ragdoll_env import ENV_VERSION
from train_bc_ragdoll import EVAL_SEEDS, load_demos, normalize
from train_walker import SACAgent, SoftQNetwork

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_transitions(arrays, meta, mean, var):
    """Turn the recorded trajectories into (s, a, r, s', not_done) with episode boundaries respected.

    A transition that crosses an episode end is marked terminal: without that, the TD backup would chain
    one demonstration episode into the next and the critic would learn a fake continuity the task does
    not have.
    """
    obs, act, rew = arrays["obs"], arrays["action"], arrays["reward"]
    bounds = np.cumsum([e["steps"] for e in meta["episodes"]])[:-1]
    not_done = np.ones(len(obs), dtype=np.float32)
    # The rows that end an episode are the last step *of* each one, plus the very last row of the file.
    # Marking the boundary index instead would make the first step of an episode terminal and let the
    # last step bootstrap from the next episode's opening state.
    last_rows = np.concatenate([bounds - 1, [len(obs) - 1]]).astype(int)
    not_done[last_rows] = 0.0
    x = normalize(obs, mean, var).astype(np.float32)
    x_next = np.roll(x, -1, axis=0)
    # Column vectors throughout: SoftQNetwork returns (batch, 1), and a (batch,) reward column next to a
    # (batch, 1) target broadcasts to (batch, batch) - a wrong loss that trains and prints numbers.
    return {"obs": x, "obs_next": x_next, "act": act.astype(np.float32),
            "rew": rew.astype(np.float32).reshape(-1, 1),
            "plan_value": arrays["plan_value"].astype(np.float32).reshape(-1, 1),
            "mc_return": compute_mc_return(rew, bounds, gamma=0.99).reshape(-1, 1),
            "not_done": not_done.astype(np.float32).reshape(-1, 1)}, bounds


def compute_mc_return(rew, bounds, gamma=0.99):
    """Exact discounted return G_t = sum_{k>=0} gamma^k r_{t+k} within each episode."""
    rew = np.asarray(rew, dtype=np.float64).reshape(-1)
    mc = np.zeros_like(rew, dtype=np.float64)
    starts = np.concatenate([[0], np.asarray(bounds, dtype=int)])
    ends = np.concatenate([np.asarray(bounds, dtype=int), [len(rew)]])
    for s, e in zip(starts, ends):
        g = 0.0
        for t in range(e - 1, s - 1, -1):
            g = rew[t] + gamma * g
            mc[t] = g
    return mc.astype(np.float32)


def episode_of(n_rows, bounds):
    return np.searchsorted(np.asarray(bounds), np.arange(n_rows), side="right")


def train(args):
    arrays, meta = load_demos(os.path.join(ROOT, args.demos))
    if "reward" not in arrays or "plan_value" not in arrays:
        raise SystemExit("the demos predate the reward and plan_value columns; re-run "
                         "collect_mpc_demos.py - offline critic learning needs the reward")
    mean = np.asarray(meta["obs_rms"]["mean"], dtype=np.float64)
    var = np.asarray(meta["obs_rms"]["var"], dtype=np.float64)
    tr, bounds = build_transitions(arrays, meta, mean, var)
    eps = episode_of(len(tr["obs"]), bounds)
    held = sorted(set(eps.tolist()))[-args.holdout:]
    val = np.isin(eps, held)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    def t(name):
        return torch.tensor(tr[name], device=device)

    cols = ("obs", "obs_next", "act", "rew", "plan_value", "mc_return", "not_done")
    data = {k: t(k)[~val] for k in cols}
    vdata = {k: t(k)[val] for k in cols}
    obs_dim, act_dim = int(tr["obs"].shape[1]), int(tr["act"].shape[1])
    q1, q2 = SoftQNetwork(obs_dim, act_dim).to(device), SoftQNetwork(obs_dim, act_dim).to(device)
    q1t, q2t = SoftQNetwork(obs_dim, act_dim).to(device), SoftQNetwork(obs_dim, act_dim).to(device)
    for src, dst in ((q1, q1t), (q2, q2t)):
        dst.load_state_dict(src.state_dict())
        for p in dst.parameters():
            p.requires_grad_(False)
    actor = SACAgent(obs_dim, gym.spaces.Box(-1.0, 1.0, shape=(act_dim,))).to(device)
    init_actor_path = getattr(args, "init_actor", None)
    if init_actor_path:
        full_init = init_actor_path if os.path.isabs(init_actor_path) else os.path.join(ROOT, init_actor_path)
        init_ckpt = torch.load(full_init, map_location=device, weights_only=False)
        actor.load_state_dict(init_ckpt["actor_state_dict"])
    for p in actor.fc_logstd.parameters():
        p.requires_grad_(False)

    mode = getattr(args, "mode", "td")
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr,
                             weight_decay=1e-4 if mode != "td" else 0.0)
    opt_pi = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=args.lr)
    lam = args.bc_lambda / float(data["rew"].abs().mean()) if args.actor == "q_bc" else 0.0

    def policy_action(x):
        mean_out, _ = actor(x)
        return torch.tanh(mean_out)

    history = []
    best = {"val": float("inf"), "val_r2": None, "epoch": 0, "actor": None, "q1": None, "q2": None}
    if mode in ("planner_value", "mc_return"):
        target_col = "plan_value" if mode == "planner_value" else "mc_return"
        v_std = float(data[target_col].std().clamp(min=1.0))
        anchor_scale = 0.25 * v_std
        y_val_true = vdata[target_col]
        y_val_var = float(((y_val_true - y_val_true.mean()) ** 2).mean().clamp(min=1e-8))
        for epoch in range(args.epochs):
            perm = torch.randperm(data["obs"].shape[0], device=device)
            q_sum = 0.0
            for start in range(0, len(perm), args.batch):
                i = perm[start:start + args.batch]
                b_obs = data["obs"][i]
                b_act = data["act"][i]
                b_val = data[target_col][i]
                # Supervised state-action value fit on demonstrated (s, a*) plus perturbed actions
                # a_pert = clip(a* + noise, -1, 1) with local quadratic penalty so dQ/da at step 0
                # points smoothly back toward the demonstrated action knot a*.
                noise = torch.randn_like(b_act) * 0.25
                b_act_pert = (b_act + noise).clamp(-1.0, 1.0)
                act_dist_sq = ((b_act_pert - b_act) ** 2).mean(dim=1, keepdim=True)
                b_val_pert = b_val - anchor_scale * act_dist_sq

                loss_on = F.mse_loss(q1(b_obs, b_act), b_val) + F.mse_loss(q2(b_obs, b_act), b_val)
                loss_off = (F.mse_loss(q1(b_obs, b_act_pert), b_val_pert)
                            + F.mse_loss(q2(b_obs, b_act_pert), b_val_pert))
                loss_q = loss_on + 0.5 * loss_off
                opt_q.zero_grad()
                loss_q.backward()
                opt_q.step()
                q_sum += float(loss_on) * len(i)
            with torch.no_grad():
                vq = 0.5 * (q1(vdata["obs"], vdata["act"]) + q2(vdata["obs"], vdata["act"]))
                val_mse = float(F.mse_loss(vq, y_val_true))
                val_r2 = float(1.0 - val_mse / y_val_var)
            history.append({
                "epoch": epoch + 1,
                "train_value_mse": round(q_sum / len(perm), 2),
                "val_td_mse": round(val_mse, 2),
                "val_value_r2": round(val_r2, 4),
            })
            if val_mse < best["val"]:
                best = {
                    "val": val_mse,
                    "val_r2": round(val_r2, 4),
                    "epoch": epoch + 1,
                    "actor": {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
                    "q1": {k: v.detach().cpu().clone() for k, v in q1.state_dict().items()},
                    "q2": {k: v.detach().cpu().clone() for k, v in q2.state_dict().items()},
                }
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"epoch {epoch + 1:3d} mode={mode} train_mse={history[-1]['train_value_mse']:.1f} "
                      f"val_mse={val_mse:.1f} val_r2={val_r2:.4f}", flush=True)
    else:
        for epoch in range(args.epochs):
            perm = torch.randperm(data["obs"].shape[0], device=device)
            q_sum = pi_sum = 0.0
            for start in range(0, len(perm), args.batch):
                i = perm[start:start + args.batch]
                b = {k: v[i] for k, v in data.items()}
                with torch.no_grad():
                    a_next = policy_action(b["obs_next"]).clamp(-1, 1)
                    tgt = torch.min(q1t(b["obs_next"], a_next), q2t(b["obs_next"], a_next))
                    target = b["rew"] + args.gamma * tgt * b["not_done"]
                if target.shape != b["rew"].shape or target.dim() != 2:
                    raise SystemExit(f"TD target has shape {tuple(target.shape)}, expected "
                                     f"{tuple(b['rew'].shape)} - a broadcast would train a wrong loss")
                q_pred = q1(b["obs"], b["act"])
                if q_pred.shape != target.shape:
                    raise SystemExit(f"Q output {tuple(q_pred.shape)} does not match the TD target "
                                     f"{tuple(target.shape)}")
                loss_q = F.mse_loss(q_pred, target) + F.mse_loss(q2(b["obs"], b["act"]), target)
                opt_q.zero_grad(); loss_q.backward(); opt_q.step()
                q_sum += float(loss_q) * len(i)
                for src, dst in ((q1, q1t), (q2, q2t)):
                    for ps, pd in zip(src.parameters(), dst.parameters()):
                        pd.data.mul_(1 - args.tau).add_(args.tau * ps.data)
                opt_pi.zero_grad()
                a = policy_action(b["obs"])
                loss_pi = -q1(b["obs"], a).mean()
                if lam:
                    loss_pi = loss_pi + lam * F.mse_loss(a, b["act"])
                loss_pi.backward(); opt_pi.step()
                pi_sum += float(loss_pi) * len(i)
            with torch.no_grad():
                a_v = policy_action(vdata["obs"]).clamp(-1, 1)
                vq = torch.min(q1(vdata["obs"], a_v), q2(vdata["obs"], a_v))
                vtgt = vdata["rew"] + args.gamma * torch.min(q1t(vdata["obs_next"], a_v),
                                                              q2t(vdata["obs_next"], a_v)) * vdata["not_done"]
                val_loss = float(F.mse_loss(vq, vtgt))
            history.append({"epoch": epoch + 1, "train_td_mse": round(q_sum / len(perm), 2),
                            "train_actor_loss": round(pi_sum / len(perm), 3),
                            "val_td_mse": round(val_loss, 2)})
            if val_loss < best["val"]:
                best = {"val": val_loss, "val_r2": None, "epoch": epoch + 1,
                        "actor": {k: v.detach().cpu().clone()
                                  for k, v in actor.state_dict().items()},
                        "q1": {k: v.detach().cpu().clone() for k, v in q1.state_dict().items()},
                        "q2": {k: v.detach().cpu().clone() for k, v in q2.state_dict().items()}}
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"epoch {epoch + 1:3d} td={history[-1]['train_td_mse']:.1f} "
                      f"actor={history[-1]['train_actor_loss']:.3f} val_td={val_loss:.1f}", flush=True)
    if best["actor"] is None:
        raise SystemExit("no validation rows; nothing to select the checkpoint by")
    actor.load_state_dict(best["actor"])
    rms = RunningMeanStd(shape=(obs_dim,))
    rms.mean, rms.var, rms.count = mean, var, int(meta["obs_rms"]["count"])
    ckpt_dir = os.path.join(ROOT, "checkpoints", args.run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"critic_student_best{best['epoch']}.pt")
    torch.save({
        "algo": "offline_critic", "env_version": ENV_VERSION, "task_phase": "target",
        "target_forward_velocity": 1.2, "reward_kwargs": dict(TRAINING_REWARD_KWARGS),
        "actor_state_dict": actor.state_dict(),
        "qf1_state_dict": best["q1"],
        "qf2_state_dict": best["q2"],
        "qf1_target_state_dict": best["q1"],
        "qf2_target_state_dict": best["q2"],
        "obs_rms": rms, "global_step": best["epoch"],
        "critic": {
            "demos": args.demos, "mode": mode, "actor_mode": args.actor,
            "init_actor": init_actor_path, "epochs": args.epochs,
            "selected_epoch": best["epoch"], "val_td_mse": round(best["val"], 2),
            "val_value_r2": best["val_r2"],
            "gamma": args.gamma, "tau": args.tau, "bc_lambda": args.bc_lambda if lam else 0.0,
            "transitions": int(len(perm)), "val_rows": int(val.sum()),
            "holdout_episodes": [int(meta["episodes"][e]["seed"]) for e in
                                 np.unique(eps[val])],
            "demo_seeds": meta["demo_seeds"],
            "eval_seeds_disjoint": not (EVAL_SEEDS & set(meta["demo_seeds"])),
            "history": history,
        },
    }, path)
    print(f"wrote {path} (best epoch {best['epoch']}, val_td_mse={best['val']:.1f}, val_r2={best['val_r2']})")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="benchmarks/demos/mpc_preference_demos.npz")
    ap.add_argument("--mode", choices=["td", "planner_value", "mc_return"], default="td",
                    help="'td' runs offline Bellman backups; 'planner_value' and 'mc_return' fit "
                         "twin critics by supervised regression onto the planner lookahead value "
                         "or discounted trajectory return without TD bootstrapping divergence")
    ap.add_argument("--init-actor", default=None,
                    help="optional BC actor checkpoint to carry in actor_state_dict for SAC warm-start")
    ap.add_argument("--actor", choices=["q", "q_bc"], default="q")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--bc-lambda", type=float, default=1.0,
                    help="weight of the in-distribution term for --actor q_bc, scaled by the mean "
                         "|reward| so it means the same thing at any reward scale")
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.run_id = args.run_id or f"critic_mpc_{args.actor}"
    if args.holdout < 1:
        raise SystemExit("--holdout must be >= 1: the checkpoint is selected on held-out episodes")
    train(args)


if __name__ == "__main__":
    main()
