"""Avaliação headless da política de BC da Fase 2 (ragdoll, frame-stacking).

Gera `trajectory.png` com vista top-down e altura do torso. Os artefatos da
Fase 2 não são versionados; o script valida paths e sai com mensagem
acionável (exit 2) em vez de traceback.
"""

import argparse
import mujoco
import numpy as np
import os
import sys
import torch
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import deque

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "openai_walker"))

try:
    from train import WalkerTeacherNet  # noqa: F401  definição canônica da Fase 2
except Exception:  # fallback local
    class WalkerTeacherNet(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.Mish(),
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.Mish(),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.Mish(),
                nn.Linear(512, output_dim),
            )

        def forward(self, x):
            return self.net(x)


def parse_args():
    p = argparse.ArgumentParser(description="Eval headless da política BC Fase 2.")
    p.add_argument("--model", default=os.path.join(REPO_ROOT, "mujoco_mpc_walker", "teacher_model.pt"))
    p.add_argument("--scaler", default=os.path.join(REPO_ROOT, "mujoco_mpc_walker", "scaler.pkl"))
    p.add_argument("--xml", default=os.path.join(REPO_ROOT, "walker_ragdoll.xml"))
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "trajectory.png"))
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    missing = [n for n, v in (("model", args.model), ("scaler", args.scaler), ("xml", args.xml)) if not os.path.exists(v)]
    if missing:
        print(f"Artefatos ausentes para eval headless ({', '.join(missing)}).")
        print(f"  modelo esperado em: {args.model}")
        print(f"  scaler esperado em: {args.scaler}")
        print(f"  xml esperado em:    {args.xml}")
        print("Gere os artefatos da Fase 2 ou informe --model/--scaler/--xml alternativos.")
        sys.exit(2)

    # Load model (188 inputs due to Frame Stacking)
    net = WalkerTeacherNet(188, 17)
    net.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    net.eval()

    # Load scaler
    with open(args.scaler, 'rb') as f:
        scaler = pickle.load(f)
    scaler_mean = scaler.mean_
    scaler_scale = scaler.scale_

    # Load mujoco
    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)

    target_mocap_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
    mocapid = m.body_mocapid[target_mocap_id]
    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")

    # Initial pos
    d.mocap_pos[mocapid, 0] = 2.0
    d.mocap_pos[mocapid, 1] = 0.0

    mujoco.mj_forward(m, d)

    last_action = np.zeros(m.nu)
    history_len = 3
    state_history = deque(maxlen=history_len + 1)

    torso_x_hist = []
    torso_y_hist = []
    target_x_hist = []
    target_y_hist = []
    z_hist = []

    steps = args.steps

    for i in range(steps):
        tx = d.mocap_pos[mocapid, 0]
        ty = d.mocap_pos[mocapid, 1]

        torso_x = d.xpos[torso_id, 0]
        torso_y = d.xpos[torso_id, 1]
        torso_z = d.xpos[torso_id, 2]

        torso_x_hist.append(torso_x)
        torso_y_hist.append(torso_y)
        target_x_hist.append(tx)
        target_y_hist.append(ty)
        z_hist.append(torso_z)

        dist = np.sqrt((torso_x - tx)**2 + (torso_y - ty)**2)
        if dist < 0.5:
            d.mocap_pos[mocapid, 0] = torso_x + rng.uniform(-2.5, 2.5)
            d.mocap_pos[mocapid, 1] = torso_y + rng.uniform(-2.5, 2.5)

        qpos = d.qpos.copy()
        qvel = d.qvel.copy()

        rel_tx_global = tx - qpos[0]
        rel_ty_global = ty - qpos[1]

        qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        rel_tx = rel_tx_global * np.cos(yaw) + rel_ty_global * np.sin(yaw)
        rel_ty = -rel_tx_global * np.sin(yaw) + rel_ty_global * np.cos(yaw)

        x_np = np.concatenate(([rel_tx, rel_ty], qpos[2:], qvel))

        state_history.append(x_np)
        while len(state_history) < history_len + 1:
            state_history.append(x_np)

        stacked_state = np.concatenate(list(reversed(state_history)))
        # Normalize stacked frame (188 dims)
        stacked_state_scaled = (stacked_state - scaler_mean) / scaler_scale
        state_tensor = torch.tensor(stacked_state_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action = net(state_tensor).numpy()[0]
            # Injetar pequeno ruído Gaussiano para quebrar o congelamento multimodal
            action += rng.normal(0, 0.15, size=action.shape)

            # Action Smoothing (Removido o peso forte para evitar latência fatal)
            action = 0.2 * last_action + 0.8 * action
            action = np.clip(action, -1.0, 1.0)
            last_action = action.copy()

        d.ctrl[:] = action
        mujoco.mj_step(m, d)

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(torso_x_hist, torso_y_hist, label='Robot Trajectory', color='blue')
    plt.scatter(target_x_hist, target_y_hist, label='Target Positions', color='red', marker='x')
    plt.title('Top-Down View of Movement')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(z_hist, label='Torso Height (Z)', color='green')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Fall Threshold (0.5m)')
    plt.title('Robot Balance over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    out_path = args.out
    plt.savefig(out_path)
    print("Saved visualization to:", out_path)


if __name__ == "__main__":
    main()
