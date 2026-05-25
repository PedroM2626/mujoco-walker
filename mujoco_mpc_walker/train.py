import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import mlflow
import os

# Set MLflow tracking URI to a local directory
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Walker_Behavioral_Cloning")

class WalkerTeacherNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WalkerTeacherNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    csv_path = "build/dataset.csv"
    if not os.path.exists(csv_path):
        print(f"Dataset {csv_path} not found! Please generate it first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    df = df.dropna()
    
    # Extract features (target_x, target_y, qpos_*, qvel_*) and labels (ctrl_*)
    target_cols = ['target_x', 'target_y']
    qpos_cols = [c for c in df.columns if c.startswith('qpos_')]
    qvel_cols = [c for c in df.columns if c.startswith('qvel_')]
    ctrl_cols = [c for c in df.columns if c.startswith('ctrl_')]
    
    X_cols = target_cols + qpos_cols + qvel_cols
    Y_cols = ctrl_cols
    
    X_np = df[X_cols].values
    Y_np = df[Y_cols].values
    
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = WalkerTeacherNet(input_dim=len(X_cols), output_dim=len(Y_cols))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    epochs = 50
    
    with mlflow.start_run():
        mlflow.log_param("model_type", "FeedForward_MLP")
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 1e-3)
        mlflow.log_param("dataset_size", len(df))
        
        print("Starting training...")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X_batch, Y_batch in train_loader:
                optimizer.zero_grad()
                predictions = model(X_batch)
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
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
                
        # Save model
        model_path = "teacher_model.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        print(f"Model saved to {model_path} and registered in MLflow.")

if __name__ == "__main__":
    main()
