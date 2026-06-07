import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import mlflow
import os
import pickle
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Walker_Behavioral_Cloning")

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, max_action=1.0):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.Mish(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Mish()
        )
        self.mean_layer = nn.Linear(512, output_dim)
        self.log_std_layer = nn.Parameter(torch.zeros(1, output_dim))
        self.max_action = max_action

    def forward(self, x):
        features = self.net(x)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer.expand_as(mean)
        std = torch.exp(torch.clamp(log_std, -5, 2))
        return mean, std
        
    def get_action(self, x, deterministic=False):
        mean, std = self.forward(x)
        if deterministic:
            return torch.tanh(mean) * self.max_action
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()
        return torch.tanh(u) * self.max_action
        
    def log_prob(self, x, a):
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        # Compute log prob of a
        # a = tanh(u) -> u = atanh(a)
        # log_prob(a) = log_prob(u) - log(1 - a^2)
        u = torch.atanh(torch.clamp(a / self.max_action, -0.999999, 0.999999))
        log_prob = dist.log_prob(u).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(self.max_action * (1 - torch.tanh(u).pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
        return log_prob

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 1)
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 1)
        )
    def forward(self, state):
        return self.net(state)

def expectile_loss(q, v, expectile=0.7):
    diff = q - v
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)

def main():
    csv_path = "dataset_openai.csv"
    if not os.path.exists(csv_path):
        print(f"Dataset nao encontrado em: {csv_path}")
        return

    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    df = df.dropna()
    
    obs_cols = [c for c in df.columns if c.startswith('obs_')]
    next_obs_cols = [c for c in df.columns if c.startswith('next_obs_')]
    act_cols = [c for c in df.columns if c.startswith('action_')]
    
    X_np = df[obs_cols].values
    X_next_np = df[next_obs_cols].values
    Y_np = df[act_cols].values
    R_np = df['reward'].values.astype(np.float32).reshape(-1, 1) / 100.0 # Normalize rewards
    D_np = df['done'].values.astype(np.float32).reshape(-1, 1)
    
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np)
    X_next_np = scaler.transform(X_next_np)
    
    with open("scaler_iql.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Tensors
    states = torch.tensor(X_np, dtype=torch.float32)
    next_states = torch.tensor(X_next_np, dtype=torch.float32)
    actions = torch.tensor(Y_np, dtype=torch.float32)
    rewards = torch.tensor(R_np, dtype=torch.float32)
    dones = torch.tensor(D_np, dtype=torch.float32)
    
    dataset = TensorDataset(states, actions, rewards, next_states, dones)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    
    # Models
    q1 = QNetwork(state_dim, action_dim)
    q2 = QNetwork(state_dim, action_dim)
    target_q1 = QNetwork(state_dim, action_dim)
    target_q2 = QNetwork(state_dim, action_dim)
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
    v_net = ValueNetwork(state_dim)
    policy = PolicyNet(state_dim, action_dim)
    
    q_optimizer = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
    v_optimizer = optim.Adam(v_net.parameters(), lr=3e-4)
    policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    
    epochs = 50
    gamma = 0.99
    tau = 0.005
    expectile = 0.8
    beta = 3.0 # Temperature for AWR
    
    with mlflow.start_run(run_name="IQL_Training"):
        mlflow.log_param("model_type", "IQL")
        mlflow.log_param("expectile", expectile)
        mlflow.log_param("beta", beta)
        
        for epoch in range(epochs):
            total_q_loss = 0
            total_v_loss = 0
            total_pi_loss = 0
            
            for s, a, r, s_prime, d in train_loader:
                with torch.no_grad():
                    target_q = torch.min(target_q1(s, a), target_q2(s, a))
                v = v_net(s)
                
                # V Loss (Expectile Regression)
                v_loss = expectile_loss(target_q, v, expectile).mean()
                v_optimizer.zero_grad()
                v_loss.backward()
                v_optimizer.step()
                
                # Q Loss
                with torch.no_grad():
                    next_v = v_net(s_prime)
                    target_q_val = r + gamma * (1 - d) * next_v
                
                q1_val = q1(s, a)
                q2_val = q2(s, a)
                q_loss = F.mse_loss(q1_val, target_q_val) + F.mse_loss(q2_val, target_q_val)
                
                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()
                
                # Policy Loss (AWR)
                with torch.no_grad():
                    adv = target_q - v
                    weight = torch.exp(beta * adv).clamp(max=100.0)
                
                log_prob = policy.log_prob(s, a)
                pi_loss = -(weight * log_prob).mean()
                
                policy_optimizer.zero_grad()
                pi_loss.backward()
                policy_optimizer.step()
                
                # Update targets
                for param, target_param in zip(q1.parameters(), target_q1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(q2.parameters(), target_q2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
                total_q_loss += q_loss.item()
                total_v_loss += v_loss.item()
                total_pi_loss += pi_loss.item()
                
            n = len(train_loader)
            mlflow.log_metric("q_loss", total_q_loss/n, step=epoch)
            mlflow.log_metric("v_loss", total_v_loss/n, step=epoch)
            mlflow.log_metric("pi_loss", total_pi_loss/n, step=epoch)
            print(f"Epoch {epoch+1}/{epochs} | Q Loss: {total_q_loss/n:.4f} | V Loss: {total_v_loss/n:.4f} | Pi Loss: {total_pi_loss/n:.4f}")
            
        torch.save(policy.state_dict(), "iql_model.pt")
        torch.save({
            "policy": policy.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "v": v_net.state_dict()
        }, "iql_full_ckpt.pt")
        mlflow.log_artifact("iql_model.pt")
        print("IQL Training Complete.")

if __name__ == "__main__":
    main()
