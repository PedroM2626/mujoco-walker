"""Verifica o pipeline de Behavioral Cloning da Fase 2 (legado/arquivado).

Compara um scaler recém-ajustado no dataset com o scaler salvo e mede o MSE
do modelo no dataset de treino. Os artefatos da Fase 2
(`mujoco_mpc_walker/build/dataset.csv`, `scaler.pkl`, `teacher_model.pt`)
não são mais versionados; o script agora falha com mensagem acionável em vez
de traceback com import/path hardcoded.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "openai_walker"))

try:
    from train import WalkerTeacherNet  # noqa: F401  (definição canônica da Fase 2)
except Exception:  # fallback local para não quebrar se o módulo mudar
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
    p = argparse.ArgumentParser(description="Verifica artefatos da Fase 2 (BC/MPC).")
    p.add_argument("--dataset", default=os.path.join(REPO_ROOT, "mujoco_mpc_walker", "build", "dataset.csv"))
    p.add_argument("--scaler", default=os.path.join(REPO_ROOT, "mujoco_mpc_walker", "scaler.pkl"))
    p.add_argument("--model", default=os.path.join(REPO_ROOT, "mujoco_mpc_walker", "teacher_model.pt"))
    p.add_argument("--input-dim", type=int, default=47)
    p.add_argument("--output-dim", type=int, default=17)
    return p.parse_args()


def main():
    args = parse_args()
    missing = [n for n, v in (("dataset", args.dataset), ("scaler", args.scaler), ("model", args.model)) if not os.path.exists(v)]
    if missing:
        print(f"Artefatos da Fase 2 ausentes ({', '.join(missing)}).")
        print(f"  dataset esperado em: {args.dataset}")
        print(f"  scaler esperado em:  {args.scaler}")
        print(f"  modelo esperado em:  {args.model}")
        print("A Fase 2 (MPC+C++) foi arquivada; gere os artefatos via mujoco_mpc_walker/build.* ou use a Fase 4 (openai_walker/dataset_openai.csv).")
        sys.exit(2)

    df = pd.read_csv(args.dataset)
    qpos_cols = [c for c in df.columns if c.startswith('qpos_') and c not in ('qpos_0', 'qpos_1')]
    qvel_cols = [c for c in df.columns if c.startswith('qvel_')]
    ctrl_cols = [c for c in df.columns if c.startswith('ctrl_')]

    df['rel_tx'] = df['target_x'] - df['qpos_0']
    df['rel_ty'] = df['target_y'] - df['qpos_1']
    X_cols = ['rel_tx', 'rel_ty'] + qpos_cols + qvel_cols
    X_np_full = df[X_cols].values

    scaler1 = StandardScaler()
    X_np_scaled = scaler1.fit_transform(X_np_full)

    with open(args.scaler, 'rb') as f:
        scaler2 = pickle.load(f)

    print("Scaler mean match:", np.allclose(scaler1.mean_, scaler2.mean_))
    print("Scaler scale match:", np.allclose(scaler1.scale_, scaler2.scale_))

    # Also evaluate the model on the full training set!
    net = WalkerTeacherNet(args.input_dim, args.output_dim)
    net.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    net.eval()

    with torch.no_grad():
        pred_all = net(torch.tensor(X_np_scaled, dtype=torch.float32)).numpy()
        actual_all = df[ctrl_cols].values
        mse_all = np.mean((pred_all - actual_all)**2)
        print("MSE on full dataset:", mse_all)

        pred_0 = pred_all[0]
        actual_0 = actual_all[0]
        print("Pred 0:", pred_0[:5])
        print("Actual 0:", actual_0[:5])


if __name__ == "__main__":
    main()
