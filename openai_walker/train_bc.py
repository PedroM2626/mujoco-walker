import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import mlflow
import os

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Walker_OpenAI_BC")

class BCPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BCPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    csv_path = "dataset_openai.csv"
    if not os.path.exists(csv_path):
        print(f"Dataset nao encontrado em: {csv_path}")
        return

    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    df = df.dropna()
    
    obs_cols = [c for c in df.columns if c.startswith('obs_')]
    act_cols = [c for c in df.columns if c.startswith('action_')]
    
    states = df[obs_cols].values
    actions = df[act_cols].values
    
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    
    dataset = TensorDataset(states_tensor, actions_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCPolicy(states_tensor.shape[1], actions_tensor.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    with mlflow.start_run(run_name="BC_Walker2d"):
        epochs = 50
        print("Starting Pure Behavioral Cloning Training...")
        for epoch in range(epochs):
            total_loss = 0
            for s, a in loader:
                s, a = s.to(device), a.to(device)
                optimizer.zero_grad()
                pred_a = model(s)
                loss = loss_fn(pred_a, a)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
            
        torch.save(model.state_dict(), "bc_model.pt")
        mlflow.log_artifact("bc_model.pt")
        print("Model saved to bc_model.pt")

if __name__ == "__main__":
    main()
