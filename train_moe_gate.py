import mock_wrappers
import os
import gymnasium as gym
import gymnasium.wrappers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import envs.walker_ragdoll_env

class MoEGate(nn.Module):
    def __init__(self, input_dim=46):
        super(MoEGate, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def collect_data(num_samples=100000):
    print("Collecting MoE dataset by exploring the environment...")
    env = gym.make("WalkerRagdoll-v0", reset_mode="mixed")
    
    states = []
    labels = []
    
    obs, _ = env.reset()
    for i in range(num_samples):
        # We need raw states without the dictionary
        if isinstance(obs, dict):
            obs_array = obs["obs"]
        else:
            obs_array = obs
            
        z = env.unwrapped.data.qpos[2]
        upright = env.unwrapped.upright_factor
        
        # Label: 1 if recovering, 0 if walking
        label = 1.0 if (z < 1.1 or upright < 0.8) else 0.0
        
        states.append(obs_array[:46])
        labels.append(label)
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
            
        if (i+1) % 10000 == 0:
            print(f"Collected {i+1} samples...")
            
    env.close()
    return np.array(states, dtype=np.float32), np.array(labels, dtype=np.float32)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states, labels = collect_data()
    
    dataset = TensorDataset(torch.from_numpy(states), torch.from_numpy(labels).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = MoEGate().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    print("Training MoE Gating Network...")
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0
        for batch_states, batch_labels in loader:
            batch_states = batch_states.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_states)
            loss = criterion(preds, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), "moe_gate.pt")
    print("Saved moe_gate.pt")

if __name__ == "__main__":
    train()
