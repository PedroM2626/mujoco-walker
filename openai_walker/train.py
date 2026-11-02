"""Legado da Fase 2: clonagem comportamental do MPC (nao faz parte do pipeline da Fase 4).

Gera `teacher_model.pt` e `scaler.pkl` em `mujoco_mpc_walker/`, lidos por
`mujoco_mpc_walker/play_mpc.py`, `scripts/eval_headless.py` e `scripts/verify.py`.
Nasceu como `mujoco_mpc_walker/train.py` e foi movido para `openai_walker/` em `f95bf90`;
o caminho padrao de `--dataset` agora aponta de volta para `mujoco_mpc_walker/build/dataset.csv`
(gerado por `mjpc_dataset_tool` de `mujoco_mpc_walker/generate_data.cc` ou por
`walker_mpc.exe --dataset_path=dataset.csv --transitions=15000`).
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class WalkerTeacherNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WalkerTeacherNet, self).__init__()
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
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)


def parse_args():
    default_csv = os.path.join(REPO_ROOT, "mujoco_mpc_walker", "build", "dataset.csv")
    fallback_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "dataset.csv")
    if not os.path.exists(default_csv) and os.path.exists(fallback_csv):
        default_csv = fallback_csv
    p = argparse.ArgumentParser(description="Treina WalkerTeacherNet (BC da Fase 2) sobre dataset.csv do MPC.")
    p.add_argument("--dataset", default=default_csv, help="Caminho para dataset.csv do MPC C++")
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "mujoco_mpc_walker"),
                   help="Diretorio onde salvar scaler.pkl e teacher_model.pt")
    p.add_argument("--history-len", type=int, default=3,
                   help="Numero de frames anteriores empilhados (3 = 4 frames / 188 features; 0 = 1 frame / 47 features)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-mlflow", action="store_true",
                   help="Do not write a new run to the root mlruns.db archive")
    return p.parse_args()


def main():
    from contextlib import nullcontext
    args = parse_args()
    csv_path = args.dataset
    if not os.path.exists(csv_path):
        print(f"Dataset nao encontrado em: {csv_path}")
        print("Gere primeiro com mujoco_mpc_walker/generate_data.cc (mjpc_dataset_tool) "
              "ou walker_mpc.exe --dataset_path=dataset.csv --transitions=15000")
        return

    if not args.no_mlflow:
        import mlflow
        from mlflow_backend import set_uri
        set_uri()
        mlflow.set_experiment("Walker_Behavioral_Cloning")
        run_ctx = mlflow.start_run()
    else:
        mlflow = None
        run_ctx = nullcontext()

    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    df = df.dropna()
    
    # Calculate Relative Target Position in Robot's LOCAL coordinate frame (First-Person View)
    def quaternion_to_yaw(qw, qx, qy, qz):
        return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
    yaw = quaternion_to_yaw(df['qpos_3'], df['qpos_4'], df['qpos_5'], df['qpos_6'])
    rel_tx_global = df['target_x'] - df['qpos_0']
    rel_ty_global = df['target_y'] - df['qpos_1']
    
    df['rel_tx'] = rel_tx_global * np.cos(yaw) + rel_ty_global * np.sin(yaw)
    df['rel_ty'] = -rel_tx_global * np.sin(yaw) + rel_ty_global * np.cos(yaw)
    
    # Extract columns
    qpos_cols = [col for col in df.columns if col.startswith('qpos_')]
    qvel_cols = [col for col in df.columns if col.startswith('qvel_')]
    ctrl_cols = [col for col in df.columns if col.startswith('ctrl_')]
    
    # Define inputs and outputs (Drop absolute target_x, target_y, qpos_0, qpos_1)
    qpos_cols_no_xy = [c for c in qpos_cols if c not in ['qpos_0', 'qpos_1']]
    X_cols = ['rel_tx', 'rel_ty'] + qpos_cols_no_xy + qvel_cols
    Y_cols = ctrl_cols
    
    # === FRAME STACKING SEGURO POR EPISÓDIO ===
    # Detecta fronteiras de teleporte procurando saltos no eixo X maiores que 1 metro
    episode_starts = (df['qpos_0'].diff().abs() > 1.0) | (df.index == 0)
    df['episode'] = episode_starts.cumsum()
    
    # Vamos empilhar os últimos N frames (por padrao history_len=3 -> 4 frames total)
    history_len = args.history_len
    stacked_cols = X_cols.copy()
    
    new_cols = {}
    for lag in range(1, history_len + 1):
        for col in X_cols:
            col_name = f"{col}_t-{lag}"
            new_cols[col_name] = df.groupby('episode')[col].shift(lag)
            stacked_cols.append(col_name)
            
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    
    # Remove as linhas que não tem histórico completo (início dos episódios)
    df = df.dropna()
    
    X_np = df[stacked_cols].values
    Y_np = df[Y_cols].values
    
    # Normalize inputs
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np)
    
    os.makedirs(args.out_dir, exist_ok=True)
    scaler_path = os.path.join(args.out_dir, "scaler.pkl")
    model_path = os.path.join(args.out_dir, "teacher_model.pt")

    # Save scaler for play_mpc.py, eval_headless.py and verify.py
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float32)
    
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    
    model = WalkerTeacherNet(input_dim=X_np.shape[1], output_dim=len(Y_cols))
    
    # Try loading existing weights to continue training
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path} to continue training...")
        model.load_state_dict(torch.load(model_path, weights_only=True))
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()
    
    with run_ctx:
        if mlflow is not None:
            mlflow.log_param("model_type", "Pure_Behavioral_Cloning")
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("dataset_size", len(df))
        
        dataset = TensorDataset(X_tensor, Y_tensor)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        print("Starting Pure Behavioral Cloning Training...")
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X_batch, Y_batch in train_loader:
                optimizer.zero_grad()
                # Optional: Add small noise for robustness
                noise = torch.randn_like(X_batch) * 0.05
                predictions = model(X_batch + noise)
                
                # Standard MSE Loss
                loss = criterion(predictions, Y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_train_loss = total_loss / len(train_loader)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    predictions = model(X_batch)
                    val_loss += criterion(predictions, Y_batch).item()
            avg_val_loss = val_loss / len(val_loader)
            
            scheduler.step()
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            if mlflow is not None:
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
                
        # Save model
        torch.save(model.state_dict(), model_path)
        if mlflow is not None:
            mlflow.log_artifact(model_path)
        print(f"Model saved to {model_path}.")

        # Plot and save loss curves
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss Curve (Pure BC)")
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(args.out_dir, "loss_curve.png")
            plt.savefig(plot_path)
            plt.close()
            if mlflow is not None:
                mlflow.log_artifact(plot_path)
            print(f"Loss curve saved to {plot_path}")
        except Exception as e:
            print(f"Could not generate or log loss curve: {e}")

        # Log scaler
        if mlflow is not None and os.path.exists(scaler_path):
            mlflow.log_artifact(scaler_path)

if __name__ == "__main__":
    main()
