import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import mlflow
import os
import pickle
from sklearn.preprocessing import StandardScaler

# Set MLflow tracking URI to a local directory
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Walker_Behavioral_Cloning")

class WalkerTeacherNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WalkerTeacherNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    # Dataset gerado pelo collect_data.py (via generate_dataset.exe)
    csv_path = os.path.join(os.path.dirname(__file__), "build", "dataset.csv")
    if not os.path.exists(csv_path):
        print(f"Dataset nao encontrado em: {csv_path}")
        print("Execute primeiro: python collect_data.py")
        return

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
    
    X_np = df[X_cols].values
    Y_np = df[Y_cols].values
    
    # Normalize inputs
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np)
    
    # Save scaler for play.py
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float32)
    batch_size = 1024
    epochs = 300
    learning_rate = 1e-3
    
    model = WalkerTeacherNet(input_dim=len(X_cols), output_dim=len(Y_cols))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    with mlflow.start_run():
        mlflow.log_param("model_type", "Advantage_Weighted_Regression")
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 1e-3)
        mlflow.log_param("dataset_size", len(df))
        
        # Pure Behavioral Cloning: We trust the MPC expert completely.
        # We learn from ALL states, including fallen states, so the network learns how to get up.
        dones = df['done'].values
        weights = np.ones_like(dones, dtype=np.float32)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        
        dataset = TensorDataset(X_tensor, Y_tensor, weights_tensor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        print("Starting Advantage-Weighted Training...")
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X_batch, Y_batch, W_batch in train_loader:
                optimizer.zero_grad()
                noise = torch.randn_like(X_batch) * 0.1
                predictions = model(X_batch + noise)
                
                # Advantage-Weighted MSE Loss
                unweighted_loss = nn.functional.mse_loss(predictions, Y_batch, reduction='none')
                # Average loss across the action dimensions, then multiply by the advantage weight
                loss = (unweighted_loss.mean(dim=1) * W_batch).mean()
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_train_loss = total_loss / len(train_loader)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, Y_batch, _ in val_loader:
                    predictions = model(X_batch)
                    val_loss += criterion(predictions, Y_batch).item()
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
                
        # Save model
        model_path = "teacher_model.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        print(f"Model saved to {model_path} and registered in MLflow.")

        # Plot and save loss curves as artifact
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss Curve")
            plt.legend()
            plt.grid(True)
            plot_path = "loss_curve.png"
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)
            print(f"Loss curve saved to {plot_path} and registered in MLflow.")
        except Exception as e:
            print(f"Could not generate or log loss curve: {e}")

        # Log scaler as artifact
        if os.path.exists("scaler.pkl"):
            mlflow.log_artifact("scaler.pkl")
            print("Scaler scaler.pkl registered in MLflow.")

if __name__ == "__main__":
    main()
