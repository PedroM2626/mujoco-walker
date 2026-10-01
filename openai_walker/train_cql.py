import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import mlflow
from mlflow_backend import set_uri
import os
import pickle
from sklearn.preprocessing import StandardScaler

set_uri()
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
        
    def sample(self, x):
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()
        a = torch.tanh(u) * self.max_action
        log_prob = dist.log_prob(u).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(self.max_action * (1 - torch.tanh(u).pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
        return a, log_prob

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
    R_np = df['reward'].values.astype(np.float32).reshape(-1, 1) / 100.0
    D_np = df['done'].values.astype(np.float32).reshape(-1, 1)
    
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np)
    X_next_np = scaler.transform(X_next_np)
    
    with open("scaler_cql.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    states = torch.tensor(X_np, dtype=torch.float32)
    next_states = torch.tensor(X_next_np, dtype=torch.float32)
    actions = torch.tensor(Y_np, dtype=torch.float32)
    rewards = torch.tensor(R_np, dtype=torch.float32)
    dones = torch.tensor(D_np, dtype=torch.float32)
    
    dataset = TensorDataset(states, actions, rewards, next_states, dones)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    
    q1 = QNetwork(state_dim, action_dim)
    q2 = QNetwork(state_dim, action_dim)
    target_q1 = QNetwork(state_dim, action_dim)
    target_q2 = QNetwork(state_dim, action_dim)
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
    policy = PolicyNet(state_dim, action_dim)
    
    q_optimizer = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    
    epochs = 50
    gamma = 0.99
    tau = 0.005
    cql_alpha = 1.0 # CQL penalty coefficient
    num_random = 10 # Number of random actions to sample for CQL loss
    
    with mlflow.start_run(run_name="CQL_Training"):
        mlflow.log_param("model_type", "CQL")
        mlflow.log_param("cql_alpha", cql_alpha)
        
        for epoch in range(epochs):
            total_q_loss = 0
            total_pi_loss = 0
            total_cql_penalty = 0
            
            for s, a, r, s_prime, d in train_loader:
                batch_size = s.size(0)
                
                # Q Loss (Bellman + CQL)
                with torch.no_grad():
                    next_a, next_log_prob = policy.sample(s_prime)
                    target_q = torch.min(target_q1(s_prime, next_a), target_q2(s_prime, next_a)) - 0.2 * next_log_prob
                    target_q_val = r + gamma * (1 - d) * target_q
                
                q1_val = q1(s, a)
                q2_val = q2(s, a)
                bellman_loss = F.mse_loss(q1_val, target_q_val) + F.mse_loss(q2_val, target_q_val)
                
                # CQL Penalty computation
                # Sample random actions
                random_actions = torch.empty(batch_size * num_random, action_dim).uniform_(-1, 1)
                repeated_s = s.repeat_interleave(num_random, dim=0)
                
                # Sample policy actions
                with torch.no_grad():
                    curr_pi_actions, _ = policy.sample(repeated_s)
                    
                q1_rand = q1(repeated_s, random_actions).view(batch_size, num_random)
                q2_rand = q2(repeated_s, random_actions).view(batch_size, num_random)
                
                q1_pi = q1(repeated_s, curr_pi_actions).view(batch_size, num_random)
                q2_pi = q2(repeated_s, curr_pi_actions).view(batch_size, num_random)
                
                # Logsumexp over samples
                cat_q1 = torch.cat([q1_rand, q1_pi, q1_val], dim=1)
                cat_q2 = torch.cat([q2_rand, q2_pi, q2_val], dim=1)
                
                cql1_penalty = torch.logsumexp(cat_q1, dim=1).mean() - q1_val.mean()
                cql2_penalty = torch.logsumexp(cat_q2, dim=1).mean() - q2_val.mean()
                cql_loss = cql_alpha * (cql1_penalty + cql2_penalty)
                
                q_loss = bellman_loss + cql_loss
                
                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()
                
                # Policy Loss (SAC-style)
                pi_a, log_prob = policy.sample(s)
                q_pi = torch.min(q1(s, pi_a), q2(s, pi_a))
                pi_loss = (0.2 * log_prob - q_pi).mean()
                
                policy_optimizer.zero_grad()
                pi_loss.backward()
                policy_optimizer.step()
                
                # Update targets
                for param, target_param in zip(q1.parameters(), target_q1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(q2.parameters(), target_q2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
                total_q_loss += bellman_loss.item()
                total_cql_penalty += (cql1_penalty + cql2_penalty).item()
                total_pi_loss += pi_loss.item()
                
            n = len(train_loader)
            mlflow.log_metric("q_loss", total_q_loss/n, step=epoch)
            mlflow.log_metric("cql_penalty", total_cql_penalty/n, step=epoch)
            mlflow.log_metric("pi_loss", total_pi_loss/n, step=epoch)
            print(f"Epoch {epoch+1}/{epochs} | Q Loss: {total_q_loss/n:.4f} | CQL Penalty: {total_cql_penalty/n:.4f} | Pi Loss: {total_pi_loss/n:.4f}")
            
        torch.save(policy.state_dict(), "cql_model.pt")
        torch.save({
            "policy": policy.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
        }, "cql_full_ckpt.pt")
        mlflow.log_artifact("cql_model.pt")
        print("CQL Training Complete.")

if __name__ == "__main__":
    main()
