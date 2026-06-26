import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import mlflow

class ExpertDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.states = torch.FloatTensor(df.iloc[:, :17].values)
        self.next_states = torch.FloatTensor(df.iloc[:, 17:34].values)
        self.actions = torch.FloatTensor(df.iloc[:, 34:40].values)
        self.rewards = torch.FloatTensor(df.iloc[:, 40:41].values)
        self.dones = torch.FloatTensor(df.iloc[:, 41:42].values)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx]

# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        
        u = self.decode(state, z)
        return u, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(state.device).clamp(-0.5, 0.5)
            
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

class PerturbationNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super(PerturbationNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        self.phi = phi

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = self.phi * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)

class CriticBCQ(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticBCQ, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

def train_bcq():
    mlflow.set_tracking_uri("sqlite:///../mlruns.db")
    mlflow.set_experiment("Walker2d_Offline_to_Online")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make("Walker2d-v5")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    dataset = ExpertDataset("dataset_openai.csv")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    
    latent_dim = action_dim * 2
    
    vae = VAE(state_dim, action_dim, latent_dim, max_action).to(device)
    vae_optimizer = torch.optim.Adam(vae.parameters())
    
    critic = CriticBCQ(state_dim, action_dim).to(device)
    critic_target = CriticBCQ(state_dim, action_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_optimizer = torch.optim.Adam(critic.parameters())
    
    perturbation = PerturbationNetwork(state_dim, action_dim, max_action).to(device)
    perturbation_target = PerturbationNetwork(state_dim, action_dim, max_action).to(device)
    perturbation_target.load_state_dict(perturbation.state_dict())
    perturbation_optimizer = torch.optim.Adam(perturbation.parameters())

    max_steps = 100000
    epochs = 10
    tau = 0.005
    gamma = 0.99
    lmbda = 0.75
    
    steps = 0
    with mlflow.start_run(run_name="BCQ_Offline"):
        for epoch in range(epochs):
            for s, a, r, ns, d in dataloader:
                s, a, r, ns, d = s.to(device), a.to(device), r.to(device), ns.to(device), d.to(device)
                
                # Variational Auto-Encoder Training
                recon, mean, std = vae(s, a)
                recon_loss = F.mse_loss(recon, a)
                KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
                vae_loss = recon_loss + 0.5 * KL_loss

                vae_optimizer.zero_grad()
                vae_loss.backward()
                vae_optimizer.step()

                # Critic Training
                with torch.no_grad():
                    # Duplicate next state 10 times
                    ns_rep = torch.repeat_interleave(ns, 10, 0)
                    
                    # Compute value of perturbed actions sampled from the VAE
                    target_Q1, target_Q2 = critic_target(ns_rep, perturbation_target(ns_rep, vae.decode(ns_rep)))
                    
                    # Soft Clipped Double Q-learning 
                    target_Q = lmbda * torch.min(target_Q1, target_Q2) + (1. - lmbda) * torch.max(target_Q1, target_Q2)
                    
                    # Take max over each action sampled from the VAE
                    target_Q = target_Q.reshape(256, 10, 1).max(1)[0]
                    target_Q = r + (1. - d) * gamma * target_Q

                current_Q1, current_Q2 = critic(s, a)
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Perturbation Training
                sampled_actions = vae.decode(s)
                perturbed_actions = perturbation(s, sampled_actions)
                
                # Update through DPG
                perturbation_loss = -critic.q1(s, perturbed_actions).mean()

                perturbation_optimizer.zero_grad()
                perturbation_loss.backward()
                perturbation_optimizer.step()

                # Update Target Networks 
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(perturbation.parameters(), perturbation_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
                steps += 1
                if steps % 1000 == 0:
                    mlflow.log_metric("vae_loss", vae_loss.item(), step=steps)
                    mlflow.log_metric("critic_loss", critic_loss.item(), step=steps)
                    mlflow.log_metric("perturbation_loss", perturbation_loss.item(), step=steps)
                
            print(f"Epoch {epoch} finished.")
            
        torch.save(vae.state_dict(), "bcq_vae.pt")
        torch.save(perturbation.state_dict(), "bcq_perturbation.pt")
        torch.save(critic.state_dict(), "bcq_critic.pt")
        print("Model saved")

if __name__ == "__main__":
    train_bcq()
