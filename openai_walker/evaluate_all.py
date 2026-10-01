"""Avaliador headless da Fase 4 (Walker2d-v5, strictly offline + offline-to-online).

Compatibilidade histórica: este arquivo já foi um avaliador do ragdoll da
Fase 2 (obs 188 -> ação 17, arquivos `teacher_model.pt`/`scaler.pkl`) que não
existem mais. Foi reescrito como contraparte headless (sem GUI) do
`play_race.py` canônico da Fase 4 (obs 17 -> ação 6, `bc_model.pt`,
`iql_full_ckpt.pt`, ...).

Uso:
    cd openai_walker
    python evaluate_all.py --episodes 2
    python evaluate_all.py --episodes 1 --models bc iql cql
"""

import argparse
import os
import sys

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Reusa as arquiteturas canônicas do play_race para não divergir.
from play_race import (  # noqa: E402
    BCPolicy,
    SACActor,
    GAILActor,
    PolicyNetIQL,
)


def parse_args():
    p = argparse.ArgumentParser(description="Eval headless Fase 4 (Walker2d-v5).")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--out", default=os.path.join(HERE, "comparison.png"))
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subseto: bc iql cql bc_sac_naive bc_sac_reg bc_sac_con iql_sac cql_sac gail airl bcq dt maxent pqr teacher",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _load(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=True)


def build_candidates(state_dim, action_dim, max_action, device, only=None):
    """Retorna [(nome, modelo_torch_ou_wrapper, kwargs_eval)] só p/ artefatos existentes."""
    cands = []

    def want(key):
        return only is None or key in only

    def p(name):
        return os.path.join(HERE, name)

    # Teacher SB3 (lazy para não exigir SB3 se filtrado)
    if want("teacher") and os.path.exists(p("sac_walker2d_final.zip")):
        from stable_baselines3 import SAC

        teacher = SAC.load(p("sac_walker2d_final.zip"), device=device)

        class TeacherWrapper(nn.Module):
            def forward(self, state):
                action, _ = teacher.predict(state.cpu().numpy(), deterministic=True)
                return torch.FloatTensor(action).to(device)

        cands.append(("Teacher (Upper Bound)", TeacherWrapper(), {}))

    if want("bc") and os.path.exists(p("bc_model.pt")):
        m = BCPolicy(state_dim, action_dim).to(device)
        m.load_state_dict(_load(p("bc_model.pt")))
        m.eval()
        cands.append(("BC", m, {}))

    if want("iql") and os.path.exists(p("iql_full_ckpt.pt")):
        m = PolicyNetIQL(state_dim, action_dim, max_action).to(device)
        ckpt = _load(p("iql_full_ckpt.pt"))
        m.load_state_dict(ckpt["policy"] if isinstance(ckpt, dict) and "policy" in ckpt else ckpt)
        m.eval()
        cands.append(("IQL", m, {}))

    if want("cql") and os.path.exists(p("cql_full_ckpt.pt")):
        m = PolicyNetIQL(state_dim, action_dim, max_action).to(device)
        ckpt = _load(p("cql_full_ckpt.pt"))
        m.load_state_dict(ckpt["policy"] if isinstance(ckpt, dict) and "policy" in ckpt else ckpt)
        m.eval()
        cands.append(("CQL", m, {}))

    for key, label, fname in (
        ("bc_sac_naive", "BC+SAC (Naive)", "bc_sac_naive_model.pt"),
        ("bc_sac_reg", "BC+SAC (Regularized)", "bc_sac_regularized_model.pt"),
        ("bc_sac_con", "BC+SAC (Constrained)", "bc_sac_constrained_model.pt"),
    ):
        if want(key) and os.path.exists(p(fname)):
            m = SACActor(state_dim, action_dim, max_action).to(device)
            m.load_state_dict(_load(p(fname)))
            m.eval()
            cands.append((label, m, {}))

    for key, label, fname in (
        ("iql_sac", "IQL+SAC", "iql_sac_model.pt"),
        ("cql_sac", "CQL+SAC", "cql_sac_model.pt"),
    ):
        if want(key) and os.path.exists(p(fname)):
            m = PolicyNetIQL(state_dim, action_dim, max_action).to(device)
            m.load_state_dict(_load(p(fname)))
            m.eval()
            cands.append((label, m, {}))

    if want("gail") and os.path.exists(p("gail_model.pt")):
        m = GAILActor(state_dim, action_dim, max_action).to(device)
        m.load_state_dict(_load(p("gail_model.pt")))
        m.eval()
        cands.append(("GAIL", m, {"is_gail": True}))

    if want("airl") and os.path.exists(p("airl_model.pt")):
        from train_irl_airl import SACActor as AIRLActor

        m = AIRLActor(state_dim, action_dim, max_action).to(device)
        m.load_state_dict(_load(p("airl_model.pt")))
        m.eval()
        cands.append(("AIRL", m, {"is_airl": True}))

    if want("maxent") and os.path.exists(p("maxent_model.pt")):
        from train_irl_maxent import SACActor as MaxEntActor

        m = MaxEntActor(state_dim, action_dim, max_action).to(device)
        m.load_state_dict(_load(p("maxent_model.pt")))
        m.eval()
        cands.append(("MaxEnt", m, {"is_maxent": True}))

    if want("pqr") and os.path.exists(p("pqr_policy.pt")):
        from train_irl_pqr import PolicyNet as PQRPolicy

        m = PQRPolicy(state_dim, action_dim, max_action).to(device)
        m.load_state_dict(_load(p("pqr_policy.pt")))
        m.eval()
        cands.append(("PQR", m, {}))

    if want("bcq") and os.path.exists(p("bcq_vae.pt")) and os.path.exists(p("bcq_perturbation.pt")):
        from train_offline_bcq import VAE, PerturbationNetwork

        vae = VAE(state_dim, action_dim, action_dim * 2, max_action).to(device)
        vae.load_state_dict(_load(p("bcq_vae.pt")))
        pert = PerturbationNetwork(state_dim, action_dim, max_action).to(device)
        pert.load_state_dict(_load(p("bcq_perturbation.pt")))
        vae.eval()
        pert.eval()

        class BCQWrapper(nn.Module):
            def forward(self, state):
                return pert(state, vae.decode(state))

        cands.append(("BCQ", BCQWrapper().to(device), {}))

    if want("dt") and os.path.exists(p("dt_model.pt")):
        from train_offline_dt import DecisionTransformer

        m = DecisionTransformer(state_dim, action_dim, hidden_size=128, max_length=20).to(device)
        m.load_state_dict(_load(p("dt_model.pt")))
        m.eval()
        cands.append(("DT", m, {"is_dt": True}))

    return cands


def evaluate_headless(env, model, device, episodes, seed=0, is_gail=False, is_airl=False,
                      is_dt=False, is_maxent=False, dt_context=20):
    from play_race import evaluate_model as gui_evaluate  # noqa: F401  (referência canônica)

    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_r = 0.0
        if is_dt:
            state_seq = torch.zeros((1, dt_context, env.observation_space.shape[0]), dtype=torch.float32, device=device)
            action_seq = torch.zeros((1, dt_context, env.action_space.shape[0]), dtype=torch.float32, device=device)
            rtg_seq = torch.zeros((1, dt_context, 1), dtype=torch.float32, device=device)
            rtg = 4000.0
        while not done:
            with torch.no_grad():
                if is_dt:
                    state_seq[0, :-1] = state_seq[0, 1:].clone()
                    state_seq[0, -1] = torch.FloatTensor(obs).to(device)
                    rtg_seq[0, :-1] = rtg_seq[0, 1:].clone()
                    rtg_seq[0, -1] = torch.tensor([rtg], dtype=torch.float32).to(device)
                    timesteps = torch.arange(0, dt_context, device=device).unsqueeze(0)
                    action = model(state_seq, action_seq, rtg_seq, timesteps)[0, -1].cpu().numpy().flatten()
                    action_seq[0, :-1] = action_seq[0, 1:].clone()
                    action_seq[0, -1] = torch.FloatTensor(action).to(device)
                else:
                    s = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    if is_airl or is_maxent:
                        mean, _ = model(s)
                        action = (torch.tanh(mean) * model.max_action).cpu().numpy().flatten()
                    else:
                        action = model(s).squeeze(0).cpu().numpy()
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            if is_dt:
                rtg -= r
            done = term or trunc
        rewards.append(ep_r)
    return float(np.mean(rewards)), rewards


def main():
    args = parse_args()
    env = gym.make("Walker2d-v5")
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(env.action_space.high[0])
    device = torch.device("cpu")

    cands = build_candidates(state_dim, action_dim, max_action, device, only=args.models)
    if not cands:
        print("Nenhum artefato encontrado em openai_walker/.")
        print("Treine ao menos um modelo (ex: python train_bc.py) ou rode o pipeline via run_all.bat.")
        print("Referência canônica com GUI: python play_race.py")
        sys.exit(2)

    results = {}
    for name, model, kw in cands:
        avg, per_ep = evaluate_headless(env, model, device, args.episodes, seed=args.seed, **kw)
        results[name] = (avg, per_ep)
        arr = np.asarray(per_ep, dtype=float)
        print(f"{name}: {avg:.2f} avg over {args.episodes} ep "
              f"(std {arr.std():.2f}, min {arr.min():.2f}, max {arr.max():.2f})")

    env.close()

    print(f"\n=== FINAL RESULTS (headless, {args.episodes} episodes, seed {args.seed}) ===")
    lines = []
    for k, (avg, _) in sorted(results.items(), key=lambda kv: kv[1][0], reverse=True):
        arr = np.asarray(results[k][1], dtype=float)
        line = (f"{k}: {avg:.2f} Avg Reward | std {arr.std():.2f} "
                f"| min {arr.min():.2f} | max {arr.max():.2f}")
        lines.append(line)
        print(line)

    # A separate file: final_results.txt is the historical single-episode record that the
    # README table was corrected against, and overwriting it would destroy that evidence.
    report = os.path.join(HERE, f"final_results_{args.episodes}ep_seed{args.seed}.txt")
    with open(report, "w", encoding="utf-8") as handle:
        handle.write(f"Protocol: evaluate_all.py --episodes {args.episodes} --seed {args.seed}\n")
        handle.write("Model scores are per-episode return means with spread over episodes.\n\n")
        handle.write("\n".join(lines) + "\n")
    print(f"\n wrote {os.path.basename(report)}")

    plt.figure(figsize=(10, 5))
    labels = list(results.keys())
    avgs = [results[k][0] for k in labels]
    plt.bar(labels, avgs)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Avg Reward")
    plt.title(f"Walker2d-v5 headless eval ({args.episodes} ep/model)")
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
