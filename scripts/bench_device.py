"""Measure SAC update cost on CPU vs CUDA for this repo's actual network sizes.

The trainers use 256-256 MLPs with batch 512, which is small enough that CUDA kernel
launch overhead can dominate. This isolates the learner from the environment so the
device comparison is not confounded by rollout speed or by a crash at startup.
"""
import time

import numpy as np
import torch

OBS, ACT, BATCH, UPDATES = 46, 17, 512, 300


def build(device):
    import train_walker as tw

    actor = tw.SACAgent(OBS, gym_space()).to(device)
    qf1 = tw.SoftQNetwork(OBS, ACT).to(device)
    qf2 = tw.SoftQNetwork(OBS, ACT).to(device)
    obs = torch.randn(BATCH, OBS, device=device)
    acts = torch.randn(BATCH, ACT, device=device)
    return actor, qf1, qf2, obs, acts


def gym_space():
    import gymnasium as gym

    return gym.spaces.Box(-1.0, 1.0, shape=(ACT,), dtype=np.float32)


def timed(device, updates):
    actor, qf1, qf2, obs, acts = build(device)
    opt = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=3e-4)
    target = torch.randn(BATCH, device=device)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(updates):
        q1 = qf1(obs, acts)
        q2 = qf2(obs, acts)
        loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        a, _, _ = actor.get_action(obs)
        actor_loss = -qf1(obs, a).mean()
        opt.zero_grad()
        actor_loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / updates


if __name__ == "__main__":
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
    cpu = timed("cpu", UPDATES)
    print(f"CPU : {cpu*1000:6.2f} ms per SAC update pair (2 critics + 1 actor, batch {BATCH})")
    if torch.cuda.is_available():
        gpu = timed("cuda", UPDATES)
        print(f"GPU : {gpu*1000:6.2f} ms per SAC update pair")
        print(f"ratio CPU/GPU = {cpu/gpu:.2f}x  (>1 means GPU is faster)")
        # What the budget actually spends: 1M env steps at UTD 1 with 32 envs.
        print(f"1M steps / 32 envs = {1_000_000/32:,.0f} vec steps -> "
              f"CPU {cpu*1_000_000/32/60:.1f} min, GPU {gpu*1_000_000/32/60:.1f} min of pure updates")
