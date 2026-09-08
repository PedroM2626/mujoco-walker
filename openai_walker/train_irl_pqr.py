import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import mlflow

class ExpertDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        obs_cols = [c for c in df.columns if c.startswith('obs_') and not c.startswith('next_obs_')]
        next_cols = [c for c in df.columns if c.startswith('next_obs_')]
        act_cols = [c for c in df.columns if c.startswith('action_')]
        if obs_cols and next_cols and act_cols:
            # Layout canônico do generate_dataset.py (17 obs, 17 next, 6 ações)
            self.states = torch.FloatTensor(df[obs_cols].values)
            self.next_states = torch.FloatTensor(df[next_cols].values)
            self.actions = torch.FloatTensor(df[act_cols].values)
        else:
            # Fallback posicional legado (dataset 42 cols: 0-16 obs, 17-33 next, 34-39 act)
            self.states = torch.FloatTensor(df.iloc[:, :17].values)
            self.next_states = torch.FloatTensor(df.iloc[:, 17:34].values)
            self.actions = torch.FloatTensor(df.iloc[:, 34:40].values)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx]

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.max_action = max_action

    def forward(self, state):
        return torch.tanh(self.net(state)) * self.max_action

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.net(sa)

class RewardNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.net(sa)

def train_pqr():
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Walker2d_Offline_to_Online")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("dataset_openai.csv"):
        print("Dataset nao encontrado em: dataset_openai.csv")
        print("Execute primeiro: python train_teacher.py && python generate_dataset.py")
        return
    
    env = gym.make("Walker2d-v5")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    dataset = ExpertDataset("dataset_openai.csv")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    
    # PQR: 1. Policy
    pi = PolicyNet(state_dim, action_dim, max_action).to(device)
    opt_pi = optim.Adam(pi.parameters(), lr=1e-3)
    
    # PQR: 2. Q-function
    q_net = QNet(state_dim, action_dim).to(device)
    opt_q = optim.Adam(q_net.parameters(), lr=1e-3)
    
    # PQR: 3. Reward
    r_net = RewardNet(state_dim, action_dim).to(device)
    opt_r = optim.Adam(r_net.parameters(), lr=1e-3)
    
    gamma = 0.99
    
    with mlflow.start_run(run_name="Deep_PQR_IRL"):
        mlflow.log_param("gamma", gamma)
        epochs = 10
        print("Training PQR - Step 1: Policy")
        for epoch in range(epochs):
            total_pi = 0.0
            n_pi = 0
            for s, a, ns in dataloader:
                s, a = s.to(device), a.to(device)
                loss_pi = F.mse_loss(pi(s), a)
                opt_pi.zero_grad()
                loss_pi.backward()
                opt_pi.step()
                total_pi += loss_pi.item()
                n_pi += 1
            mlflow.log_metric("pi_loss", total_pi / max(n_pi, 1), step=epoch)

        print("Training PQR - Step 2: Q-Function with Anchor Action")
        # Anchor action = 0, Q(s, 0) = 0 assumed for identifiability
        for epoch in range(epochs):
            total_q = 0.0
            n_q = 0
            for s, a, ns in dataloader:
                s, a, ns = s.to(device), a.to(device), ns.to(device)
                anchor_action = torch.zeros(s.shape[0], action_dim).to(device)

                with torch.no_grad():
                    next_a = pi(ns)
                    target_q = q_net(ns, next_a)
                
                # Assume Q-learning target: Q(s,a) = r + gamma * Q(s',a')
                # But we don't have r! We just try to make Q consistent with anchor
                q_val = q_net(s, a)
                q_anchor = q_net(s, anchor_action)
                
                # Minimize Q variance and anchor penalty
                loss_q = F.mse_loss(q_anchor, torch.zeros_like(q_anchor)) + F.mse_loss(q_val, target_q)
                opt_q.zero_grad()
                loss_q.backward()
                opt_q.step()
                total_q += loss_q.item()
                n_q += 1
            mlflow.log_metric("q_loss", total_q / max(n_q, 1), step=epoch)

        print("Training PQR - Step 3: Reward Extraction")
        # Extract Reward: R(s,a) = Q(s,a) - gamma * Q(s', a')
        for epoch in range(epochs):
            total_r = 0.0
            n_r = 0
            for s, a, ns in dataloader:
                s, a, ns = s.to(device), a.to(device), ns.to(device)
                with torch.no_grad():
                    target_r = q_net(s, a) - gamma * q_net(ns, pi(ns))

                loss_r = F.mse_loss(r_net(s, a), target_r)
                opt_r.zero_grad()
                loss_r.backward()
                opt_r.step()
                total_r += loss_r.item()
                n_r += 1
            mlflow.log_metric("r_loss", total_r / max(n_r, 1), step=epoch)

        torch.save(pi.state_dict(), "pqr_policy.pt")
        mlflow.log_artifact("pqr_policy.pt")
        print("Model saved as pqr_policy.pt")

if __name__ == "__main__":
    train_pqr()
