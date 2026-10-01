"""Benchmark das estratégias de merging da Fase 3 (ragdoll target-phase).

Compara Hardcoded Supervisor vs MoE vs Weight Averaging vs Task Arithmetic.
O paper usou `--num-episodes 1000` (~4M steps); o default é 5 para smoke test.
 Fame: checkpoints em `checkpoints/walker_{recovery,target}_v1/` + `merged_*.pt`.
"""
import argparse
from envs import normalize_compat  # noqa: F401  pickle shim for checkpoint obs_rms; import before torch.load
import os
import torch
import numpy as np
import gymnasium as gym
import gymnasium.wrappers
import time
from train_walker import SACAgent
from train_moe_gate import MoEGate

EPSILON = 1e-8
CLIP = 10.0


def frozen_normalize(obs, obs_rms):
    """Aplicar a normalização salva no checkpoint, sem atualizar as estatísticas.

    Idêntico a `play.FrozenNormalizeObservation`. Sem isto o agente recebe estados
    crus enquanto foi treinado com estados normalizados — e os números medidos aqui
    passam a descrever uma política diferente da que `play.py` executa.
    """
    if obs_rms is None:
        return np.asarray(obs, dtype=np.float64)
    normalized = (np.asarray(obs, dtype=np.float64) - obs_rms.mean) / np.sqrt(
        obs_rms.var + EPSILON
    )
    return np.clip(normalized, -CLIP, CLIP)


def load_agent(checkpoint_path, device, input_dim=46):
    # Checkpoint local (pode conter RunningMeanStd em obs_rms).
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(17,))
    agent = SACAgent(input_dim, action_space).to(device)
    state_dict = checkpoint.get("actor_state_dict", checkpoint)
    agent.load_state_dict(state_dict)
    agent.eval()
    obs_rms = checkpoint.get("obs_rms", None)
    if obs_rms is None:
        print(f"[WARN] {checkpoint_path} não tem obs_rms; avaliando com observações cruas.")
    return agent, obs_rms


def _policy_input(agent, obs, obs_rms, dim, device):
    array = frozen_normalize(obs, obs_rms)[:dim]
    return torch.FloatTensor(array).unsqueeze(0).to(device)


def evaluate_paradigm(env_name, paradigm_name, device, rec_agent=None, tgt_agent=None,
                      single_agent=None, moe_gate=None, num_episodes=5, seed=1,
                      rec_rms=None, tgt_rms=None, single_rms=None):
    """Return (mean reward, survival rate, mean falls per episode).

    Survival is measured from the env's own health condition: `terminated` is always
    False here because these runs use terminate_when_unhealthy=False, so the old
    "did it terminate?" definition reported 100% for every strategy regardless of
    whether the robot was lying on the floor.
    """
    env = gym.make(env_name, reset_mode="mixed", task_phase="target")
    total_rewards = []
    survivals = []
    falls_per_episode = []

    print(f"\n--- Evaluating {paradigm_name} ---")
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        falls = 0
        was_healthy = env.unwrapped.is_healthy

        for step in range(1000):
            if isinstance(obs, dict):
                obs_array = obs["obs"]
            else:
                obs_array = obs

            with torch.no_grad():
                if paradigm_name == "Hardcoded Supervisor":
                    z = env.unwrapped.data.qpos[2]
                    upright = env.unwrapped.upright_factor
                    if z < 1.1 or upright < 0.8:
                        inp = _policy_input(rec_agent, obs_array, rec_rms, 46, device)
                        action = rec_agent.get_action(inp, deterministic=True)[0].cpu().numpy()
                    else:
                        inp = _policy_input(tgt_agent, obs_array, tgt_rms, 49, device)
                        action = tgt_agent.get_action(inp, deterministic=True)[0].cpu().numpy()
                elif paradigm_name == "Mixture of Experts":
                    g = moe_gate(_policy_input(rec_agent, obs_array, rec_rms, 46, device)).item()
                    a_rec = rec_agent.get_action(
                        _policy_input(rec_agent, obs_array, rec_rms, 46, device), deterministic=True
                    )[0].cpu().numpy()
                    a_tgt = tgt_agent.get_action(
                        _policy_input(tgt_agent, obs_array, tgt_rms, 49, device), deterministic=True
                    )[0].cpu().numpy()
                    action = g * a_rec + (1.0 - g) * a_tgt
                else:
                    inp = _policy_input(single_agent, obs_array, single_rms, 49, device)
                    action = single_agent.get_action(inp, deterministic=True)[0].cpu().numpy()

            action = action.flatten()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            is_healthy = env.unwrapped.is_healthy
            if was_healthy and not is_healthy:
                falls += 1
            was_healthy = is_healthy

            if terminated:
                break

        survived = env.unwrapped.is_healthy and env.unwrapped.upright_factor > 0.8
        total_rewards.append(ep_reward)
        survivals.append(1.0 if survived else 0.0)
        falls_per_episode.append(falls)
        print(f"Episode {ep+1} | Reward: {ep_reward:.2f} | Quedas: {falls} | "
              f"Terminou de pé: {survived}")

    env.close()
    return np.mean(total_rewards), np.mean(survivals), np.mean(falls_per_episode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark merging Fase 3.")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Episódios por paradigma (paper: 1000).")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed base; episode i usa seed+i para o reset ser reproduzível.")
    parser.add_argument("--rec-ckpt", default="checkpoints/walker_recovery_v1/sac_ckpt_20000000.pt")
    parser.add_argument("--tgt-ckpt", default="checkpoints/walker_target_v1/sac_ckpt_40000000.pt")
    parser.add_argument("--avg-ckpt", default="merged_avg_model.pt")
    parser.add_argument("--ta-ckpt", default="merged_ta_model.pt")
    parser.add_argument("--gate-ckpt", default="moe_gate.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    rec_ckpt = args.rec_ckpt
    tgt_ckpt = args.tgt_ckpt

    for p in (rec_ckpt, tgt_ckpt, args.avg_ckpt, args.ta_ckpt, args.gate_ckpt):
        if not os.path.exists(p):
            print(f"Checkpoint ausente: {p}")
            print("Treine a Fase 3 ou informe --rec-ckpt/--tgt-ckpt/--avg-ckpt/--ta-ckpt/--gate-ckpt.")
            raise SystemExit(2)

    rec_agent, rec_rms = load_agent(rec_ckpt, device, input_dim=46)
    tgt_agent, tgt_rms = load_agent(tgt_ckpt, device, input_dim=49)
    avg_agent, avg_rms = load_agent(args.avg_ckpt, device, input_dim=49)
    ta_agent, ta_rms = load_agent(args.ta_ckpt, device, input_dim=49)

    moe_gate = MoEGate().to(device)
    moe_gate.load_state_dict(torch.load(args.gate_ckpt, map_location=device, weights_only=True))
    moe_gate.eval()

    results = {}
    num_ep = args.num_episodes
    shared = dict(num_episodes=num_ep, seed=args.seed, rec_rms=rec_rms, tgt_rms=tgt_rms)

    paradigms = [
        ("Hardcoded Supervisor", dict(rec_agent=rec_agent, tgt_agent=tgt_agent)),
        ("Mixture of Experts", dict(rec_agent=rec_agent, tgt_agent=tgt_agent, moe_gate=moe_gate)),
        ("Weight Averaging", dict(single_agent=avg_agent, single_rms=avg_rms)),
        ("Task Arithmetic", dict(single_agent=ta_agent, single_rms=ta_rms)),
    ]

    for name, kwargs in paradigms:
        reward, survival, falls = evaluate_paradigm(
            "WalkerRagdoll-v0", name, device, **shared, **kwargs
        )
        results[name] = {"Reward": reward, "Survival Rate": survival, "Falls/ep": falls}

    print("\n=== FINAL RANKING ===")
    for k, v in results.items():
        print(f"{k}: {v['Reward']:.2f} Avg Reward | {v['Survival Rate']*100:.0f}% Terminou de pé "
              f"| {v['Falls/ep']:.2f} quedas/ep")
