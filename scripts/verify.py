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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


REPRO_HELP = """
O dataset C++ da Fase 2 (`mujoco_mpc_walker/build/dataset.csv`) nao esta versionado.

Os dois coletores C++ ja estao configurados no codigo-fonte:
  * Coletor headless (`mujoco_mpc_walker/generate_data.cc`, alvo `mjpc_dataset_tool` no
    `CMakeLists.txt`): instancia `mjpc::Agent` diretamente (`Initialize -> Allocate ->
    Reset -> state.Set -> PlanIteration -> ActionFromPolicy`) e grava 15.000 transicoes
    (`target_x,target_y,qpos_*,qvel_*,ctrl_*,reward,done`) em `dataset.csv`.
  * Coletor interativo (`mujoco_mpc_walker/main.cc`, alvo `walker_mpc`): aceita as flags
    `--dataset_path=dataset.csv --transitions=15000` (sem flags, abre apenas a GUI).

Como `build/walker_mpc.exe` foi compilado em junho antes das flags e `mjpc_dataset_tool`
precisa ser linkado, regerar o CSV exige toolchain C++ (Visual Studio Build Tools + CMake,
descobertos automaticamente via `vswhere` em `mujoco_mpc_walker/build.bat`).

Passos para reproduzir a Fase 2 C++:
  1. Instale Visual Studio Build Tools (Desktop development with C++) e CMake;
  2. Rode `mujoco_mpc_walker\\build.bat`;
  3. Rode `mujoco_mpc_walker\\build\\Release\\mjpc_dataset_tool.exe` (headless) ou
     `mujoco_mpc_walker\\build\\Release\\walker_mpc.exe --dataset_path=dataset.csv --transitions=15000`;
  4. `python openai_walker/train.py` (le `mujoco_mpc_walker/build/dataset.csv` e salva
     `scaler.pkl` e `teacher_model.pt` em `mujoco_mpc_walker/`);
  5. `python scripts/verify.py`.

Alternativas ja versionadas e reproduziveis sem compilador C++:
  * Pipeline Python Gait-MPC (`WalkerRagdoll-v0`, 49D): `python scripts/verify.py --gait-mpc`
    verifica `benchmarks/gait_bc_teacher_benchmark.json` e os checkpoints BC
    `checkpoints/bc_gait_mpc_raw/bc_student_best2.pt` e `checkpoints/bc_gait_mpc_band/bc_student_best3.pt`.
  * Fase 4 (`Walker2d-v4`): `openai_walker/dataset_openai.csv` (100k transicoes SAC).
"""


def parse_args():
    p = argparse.ArgumentParser(description="Verifica artefatos da Fase 2 (BC/MPC C++ ou Python Gait-MPC).")
    p.add_argument("--dataset", default=os.path.join(REPO_ROOT, "mujoco_mpc_walker", "build", "dataset.csv"))
    p.add_argument("--scaler", default=os.path.join(REPO_ROOT, "mujoco_mpc_walker", "scaler.pkl"))
    p.add_argument("--model", default=os.path.join(REPO_ROOT, "mujoco_mpc_walker", "teacher_model.pt"))
    p.add_argument("--input-dim", type=int, default=188,
                   help="Dimensao de entrada da WalkerTeacherNet (188 para 4 frames empilhados, 47 para 1 frame)")
    p.add_argument("--output-dim", type=int, default=17)
    p.add_argument("--gait-mpc", action="store_true",
                   help="Verifica os artefatos versionados do pipeline Python Gait-MPC + BC (Rota 4)")
    return p.parse_args()


def verify_gait_mpc():
    import json
    bench_path = os.path.join(REPO_ROOT, "benchmarks", "gait_bc_teacher_benchmark.json")
    raw_ckpt = os.path.join(REPO_ROOT, "checkpoints", "bc_gait_mpc_raw", "bc_student_best2.pt")
    band_ckpt = os.path.join(REPO_ROOT, "checkpoints", "bc_gait_mpc_band", "bc_student_best3.pt")
    missing = [p for p in (bench_path, raw_ckpt, band_ckpt) if not os.path.exists(p)]
    if missing:
        print(f"Artefatos Gait-MPC ausentes: {missing}")
        sys.exit(2)
    with open(bench_path, encoding="utf-8") as f:
        bench = json.load(f)
    for label, ckpt_path in (("bc_gait_raw", raw_ckpt), ("bc_gait_band", band_ckpt)):
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "actor_state_dict" in payload and "obs_rms" in payload, f"Checkpoint invalido: {ckpt_path}"
        val_mse = bench["predictability_val_mse"][f"student_{label}"]
        beats = bench["beats_the_trivial_baseline"][f"student_{label}"]
        print(f"[OK] {label}: checkpoint={os.path.relpath(ckpt_path, REPO_ROOT)} "
              f"val_mse={val_mse:.4f} beats_trivial_baseline={beats}")
    teacher = bench["teacher"]
    print(f"[OK] Teacher demos ({teacher['artifact']}): {teacher['episodes']} ep, "
          f"{teacher['transitions']} steps, in_band={teacher['transitions_in_band']} "
          f"({teacher['pct_steps_in_band']:.2f}%)")
    return 0


def main():
    args = parse_args()
    if args.gait_mpc:
        sys.exit(verify_gait_mpc())
    missing = [n for n, v in (("dataset", args.dataset), ("scaler", args.scaler), ("model", args.model)) if not os.path.exists(v)]
    if missing:
        print(f"Artefatos da Fase 2 ausentes ({', '.join(missing)}).")
        print(f"  dataset esperado em: {args.dataset}")
        print(f"  scaler esperado em:  {args.scaler}")
        print(f"  modelo esperado em:  {args.model}")
        print(REPRO_HELP)
        sys.exit(2)

    df = pd.read_csv(args.dataset).dropna()
    qpos_cols = [c for c in df.columns if c.startswith('qpos_') and c not in ('qpos_0', 'qpos_1')]
    qvel_cols = [c for c in df.columns if c.startswith('qvel_')]
    ctrl_cols = [c for c in df.columns if c.startswith('ctrl_')]

    with open(args.scaler, 'rb') as f:
        scaler2 = pickle.load(f)

    expected_dim = int(scaler2.mean_.shape[0])
    # Se o scaler salvo tiver > 47 colunas (ex.: 188 = 4 frames empilhados por train.py),
    # aplica a rotacao de yaw local e o frame-stacking por episodio de train.py.
    if expected_dim > 47 and expected_dim % 47 == 0:
        yaw = np.arctan2(
            2.0 * (df['qpos_3'] * df['qpos_6'] + df['qpos_4'] * df['qpos_5']),
            1.0 - 2.0 * (df['qpos_5'] ** 2 + df['qpos_6'] ** 2),
        )
        rel_tx_global = df['target_x'] - df['qpos_0']
        rel_ty_global = df['target_y'] - df['qpos_1']
        df['rel_tx'] = rel_tx_global * np.cos(yaw) + rel_ty_global * np.sin(yaw)
        df['rel_ty'] = -rel_tx_global * np.sin(yaw) + rel_ty_global * np.cos(yaw)
        X_cols = ['rel_tx', 'rel_ty'] + qpos_cols + qvel_cols
        episode_starts = (df['qpos_0'].diff().abs() > 1.0) | (df.index == 0)
        df['episode'] = episode_starts.cumsum()
        history_len = (expected_dim // 47) - 1
        stacked_cols = X_cols.copy()
        new_cols = {}
        for lag in range(1, history_len + 1):
            for col in X_cols:
                col_name = f"{col}_t-{lag}"
                new_cols[col_name] = df.groupby('episode')[col].shift(lag)
                stacked_cols.append(col_name)
        df = pd.concat([df, pd.DataFrame(new_cols)], axis=1).dropna()
        X_np_full = df[stacked_cols].values
    else:
        df['rel_tx'] = df['target_x'] - df['qpos_0']
        df['rel_ty'] = df['target_y'] - df['qpos_1']
        X_cols = ['rel_tx', 'rel_ty'] + qpos_cols + qvel_cols
        X_np_full = df[X_cols].values

    scaler1 = StandardScaler()
    X_np_scaled = scaler1.fit_transform(X_np_full)

    print("Scaler mean match:", np.allclose(scaler1.mean_, scaler2.mean_))
    print("Scaler scale match:", np.allclose(scaler1.scale_, scaler2.scale_))

    # Also evaluate the model on the full training set!
    net = WalkerTeacherNet(expected_dim, args.output_dim)
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
