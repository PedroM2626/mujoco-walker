import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import mlflow
from mlflow_backend import set_uri
import os

class ExpertDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.states = torch.FloatTensor(df.iloc[:, :17].values)
        self.actions = torch.FloatTensor(df.iloc[:, 17:23].values)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.dead = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, dead):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dead[self.ptr] = dead
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.dead[ind])
        )

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1) # Logits output
        )
        
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.net(sa)
        
    def get_reward(self, state, action):
        with torch.no_grad():
            logits = self.forward(state, action)
            # GAIL Reward formulation: r = -log(1 - D(s,a)) 
            # D(s,a) = sigmoid(logits)
            # -log(1 - sigmoid(logits)) = softplus(logits)
            reward = F.softplus(logits)
        return reward

class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

def train_gail():
    set_uri()
    mlflow.set_experiment("Walker2d_Offline_to_Online")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("Walker2d-v5")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Expert Dataset
    expert_dataset = ExpertDataset("dataset_openai.csv")
    expert_loader = DataLoader(expert_dataset, batch_size=256, shuffle=True, drop_last=True)
    expert_iter = iter(expert_loader)
    
    # Agent Buffer
    agent_buffer = ReplayBuffer(state_dim, action_dim)
    
    # Networks
    discriminator = Discriminator(state_dim, action_dim).to(device)
    actor = SACActor(state_dim, action_dim, max_action).to(device)
    critic = SACCritic(state_dim, action_dim).to(device)
    critic_target = SACCritic(state_dim, action_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    
    # Optimizers
    opt_d = optim.Adam(discriminator.parameters(), lr=3e-4)
    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=3e-4)
    
    alpha = 0.2
    gamma = 0.99
    tau = 0.005
    batch_size = 256
    
    max_steps = 1000000
    start_steps = 10000
    
    state, _ = env.reset()
    episode_reward = 0
    episode_irl_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    bce_loss = nn.BCEWithLogitsLoss()
    
    with mlflow.start_run(run_name="GAIL_IRL"):
        mlflow.log_params({"algorithm": "GAIL", "max_steps": max_steps, "batch_size": batch_size})
        
        for t in range(int(max_steps)):
            episode_timesteps += 1
            
            # Select action
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
            
            # Train when we have enough data
            if t >= start_steps:
                # Sample Agent Batch
                s_ag, a_ag, r_ag_env, ns_ag, d_ag = agent_buffer.sample(batch_size)
                s_ag, a_ag, ns_ag, d_ag = s_ag.to(device), a_ag.to(device), ns_ag.to(device), d_ag.to(device)
                
                # Sample Expert Batch
                try:
                    s_exp, a_exp = next(expert_iter)
                except StopIteration:
                    expert_iter = iter(expert_loader)
                    s_exp, a_exp = next(expert_iter)
                s_exp, a_exp = s_exp.to(device), a_exp.to(device)
                
                # -----------------------------
                # 1. Train Discriminator
                # -----------------------------
                # Expert labels = 1, Agent labels = 0
                logits_exp = discriminator(s_exp, a_exp)
                logits_ag = discriminator(s_ag, a_ag)
                
                loss_d_exp = bce_loss(logits_exp, torch.ones_like(logits_exp))
                loss_d_ag = bce_loss(logits_ag, torch.zeros_like(logits_ag))
                loss_d = loss_d_exp + loss_d_ag
                
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                
                # -----------------------------
                # 2. Compute IRL Reward
                # -----------------------------
                # R_irl = -log(1 - D(s,a)) = softplus(logits)
                with torch.no_grad():
                    r_irl = discriminator.get_reward(s_ag, a_ag)
                    
                # -----------------------------
                # 3. Train SAC using IRL Reward
                # -----------------------------
                with torch.no_grad():
                    next_action, next_log_prob = actor.sample(ns_ag)
                    target_q1, target_q2 = critic_target(ns_ag, next_action)
                    target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                    target_q = r_irl + (1 - d_ag) * gamma * target_q

                current_q1, current_q2 = critic(s_ag, a_ag)
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

                opt_critic.zero_grad()
                critic_loss.backward()
                opt_critic.step()

                # Train Actor
                curr_action, curr_log_prob = actor.sample(s_ag)
                q1_pi, q2_pi = critic(s_ag, curr_action)
                min_q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (alpha * curr_log_prob - min_q_pi).mean()

                opt_actor.zero_grad()
                actor_loss.backward()
                opt_actor.step()

                # Update target networks
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
                if t % 1000 == 0:
                    mlflow.log_metric("discriminator_loss", loss_d.item(), step=t)
                    mlflow.log_metric("actor_loss", actor_loss.item(), step=t)
                    mlflow.log_metric("critic_loss", critic_loss.item(), step=t)
                    mlflow.log_metric("irl_reward_mean", r_irl.mean().item(), step=t)

                # Durability, not a feature. This loop wrote exactly one file, after the last step,
                # so a long run that dies at 900k steps leaves nothing (docs/lab-notes.md, "Losing a
                # long run to a shared GPU"). The step is in the name so an interim policy can never
                # be mistaken for the final one.
                if t % 100000 == 0:
                    torch.save(actor.state_dict(), f"gail_model_ckpt_{t + 1}.pt")

            if done:
                print(f"Step {t+1} | Episode {episode_num+1} | True Env Reward: {episode_reward:.2f}")
                if t >= start_steps:
                    mlflow.log_metric("true_env_reward", episode_reward, step=t)
                state, _ = env.reset()
                episode_reward = 0
                episode_irl_reward = 0
                episode_timesteps = 0
                episode_num += 1

        torch.save(actor.state_dict(), "gail_model.pt")
        print("Model saved as gail_model.pt")

if __name__ == "__main__":
    train_gail()
