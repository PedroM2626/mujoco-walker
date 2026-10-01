import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import mlflow
from mlflow_backend import set_uri
from train_irl_gail import SACActor, SACCritic, ReplayBuffer

class ExpertDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.states = torch.FloatTensor(df.iloc[:, :17].values)
        self.actions = torch.FloatTensor(df.iloc[:, 34:40].values)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

def train_maxent():
    set_uri()
    mlflow.set_experiment("Walker2d_Offline_to_Online")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make("Walker2d-v5")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    dataset = ExpertDataset("dataset_openai.csv")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    
    # MaxEnt IRL uses a linear reward function: r(s,a) = theta^T phi(s,a)
    # We use phi(s,a) = concat(s, a)
    theta = torch.zeros(state_dim + action_dim, requires_grad=True, device=device)
    optimizer_theta = optim.Adam([theta], lr=1e-3)
    
    # Calculate Expert Feature Expectations (mu_exp)
    mu_exp = torch.zeros(state_dim + action_dim, device=device)
    count = 0
    for s, a in dataloader:
        s, a = s.to(device), a.to(device)
        sa = torch.cat([s, a], dim=1)
        mu_exp += sa.sum(dim=0)
        count += sa.shape[0]
    mu_exp /= count
    
    # Agent
    actor = SACActor(state_dim, action_dim, max_action).to(device)
    critic = SACCritic(state_dim, action_dim).to(device)
    critic_target = SACCritic(state_dim, action_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    
    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=3e-4)
    
    agent_buffer = ReplayBuffer(state_dim, action_dim)
    
    max_steps = 100000
    start_steps = 10000
    batch_size = 256
    alpha = 0.2
    gamma = 0.99
    tau = 0.005
    
    state, _ = env.reset()
    episode_reward = 0
    episode_num = 0
    
    with mlflow.start_run(run_name="MaxEnt_IRL"):
        for t in range(int(max_steps)):
            if t < start_steps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action, _ = actor.sample(state_t)
                    action = action.cpu().data.numpy().flatten()
                    
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            dead = 1.0 if terminated else 0.0
            
            agent_buffer.add(state, action, reward, next_state, dead)
            state = next_state
            episode_reward += reward
            
            if t >= start_steps:
                s_ag, a_ag, _, ns_ag, d_ag = agent_buffer.sample(batch_size)
                s_ag, a_ag, ns_ag, d_ag = s_ag.to(device), a_ag.to(device), ns_ag.to(device), d_ag.to(device)
                
                # ---------------------------------------------
                # MaxEnt IRL Reward Update (Feature Matching)
                # ---------------------------------------------
                sa_ag = torch.cat([s_ag, a_ag], dim=1)
                mu_ag = sa_ag.mean(dim=0)
                
                # Maximize log likelihood: Grad L = mu_exp - mu_ag
                loss_theta = - (theta * (mu_exp - mu_ag).detach()).sum()
                optimizer_theta.zero_grad()
                loss_theta.backward()
                optimizer_theta.step()
                
                # ---------------------------------------------
                # Train SAC Agent
                # ---------------------------------------------
                with torch.no_grad():
                    # Reward is linear: r = theta^T phi
                    r_maxent = torch.matmul(sa_ag, theta).unsqueeze(1)
                    
                    next_action, next_log_prob = actor.sample(ns_ag)
                    target_q1, target_q2 = critic_target(ns_ag, next_action)
                    target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                    target_q = r_maxent + (1 - d_ag) * gamma * target_q

                current_q1, current_q2 = critic(s_ag, a_ag)
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

                opt_critic.zero_grad()
                critic_loss.backward()
                opt_critic.step()

                curr_action, curr_log_prob = actor.sample(s_ag)
                q1_pi, q2_pi = critic(s_ag, curr_action)
                min_q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (alpha * curr_log_prob - min_q_pi).mean()

                opt_actor.zero_grad()
                actor_loss.backward()
                opt_actor.step()

                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
                if t % 1000 == 0:
                    mlflow.log_metric("theta_loss", loss_theta.item(), step=t)
                    mlflow.log_metric("actor_loss", actor_loss.item(), step=t)
                    mlflow.log_metric("r_maxent_mean", r_maxent.mean().item(), step=t)

            if done:
                print(f"Step {t+1} | Env Reward: {episode_reward:.2f}")
                state, _ = env.reset()
                episode_reward = 0
                episode_num += 1

        torch.save(actor.state_dict(), "maxent_model.pt")
        print("Model saved as maxent_model.pt")

if __name__ == "__main__":
    train_maxent()
