import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import mlflow
from mlflow_backend import set_uri
import os

set_uri()
mlflow.set_experiment("Walker_Offline_To_Online")

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

class SACActor(nn.Module):
    def __init__(self, input_dim, output_dim, max_action=1.0):
        super(SACActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(256, output_dim)
        self.log_std_layer = nn.Parameter(torch.zeros(1, output_dim))
        self.max_action = max_action

    def forward(self, x):
        features = self.net(x)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer.expand_as(mean)
        std = torch.exp(torch.clamp(log_std, -20, 2))
        return mean, std
        
    def sample(self, x):
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()
        a = torch.tanh(u) * self.max_action
        log_prob = dist.log_prob(u).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(self.max_action * (1 - torch.tanh(u).pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
        return a, log_prob

class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SACCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1000000):
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.done[ind])
        )

def main():
    env = gym.make("Walker2d-v5")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    actor = SACActor(state_dim, action_dim, max_action).to(device)
    bc_expert = BCPolicy(state_dim, action_dim).to(device)
    
    bc_model_path = "bc_model.pt"
    if os.path.exists(bc_model_path):
        print("Loading pretrained BC weights into Expert Network and SAC Actor...")
        bc_state_dict = torch.load(bc_model_path, map_location=device)
        bc_expert.load_state_dict(bc_state_dict)
        
        # Initialize SAC Actor with the same weights
        actor_state_dict = actor.state_dict()
        actor_state_dict['net.0.weight'] = bc_state_dict['net.0.weight']
        actor_state_dict['net.0.bias'] = bc_state_dict['net.0.bias']
        actor_state_dict['net.2.weight'] = bc_state_dict['net.2.weight']
        actor_state_dict['net.2.bias'] = bc_state_dict['net.2.bias']
        actor_state_dict['mean_layer.weight'] = bc_state_dict['net.4.weight']
        actor_state_dict['mean_layer.bias'] = bc_state_dict['net.4.bias']
        actor.load_state_dict(actor_state_dict)
    else:
        print("Warning: bc_model.pt not found!")

    # Freeze expert
    for param in bc_expert.parameters():
        param.requires_grad = False

    q1 = SACCritic(state_dim, action_dim).to(device)
    q2 = SACCritic(state_dim, action_dim).to(device)
    target_q1 = SACCritic(state_dim, action_dim).to(device)
    target_q2 = SACCritic(state_dim, action_dim).to(device)
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
    
    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
    
    alpha = 0.2
    gamma = 0.99
    tau = 0.005
    batch_size = 256
    lambda_bc = 10.0 # BC Regularization Coefficient
    
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    max_timesteps = 100000
    episode_reward = 0
    
    state, _ = env.reset()
    
    with mlflow.start_run(run_name="BC_SAC_Regularized_Walker2d"):
        mlflow.log_param("variant", "Regularized")
        mlflow.log_param("lambda_bc", lambda_bc)
        print("Starting SAC Offline-to-Online Fine-Tuning (Regularized)...")
        
        for t in range(max_timesteps):
            if t < 1000:
                with torch.no_grad():
                    mean, _ = actor(torch.FloatTensor(state).unsqueeze(0).to(device))
                    action = torch.tanh(mean).squeeze(0).cpu().numpy() * max_action
            else:
                with torch.no_grad():
                    action, _ = actor.sample(torch.FloatTensor(state).unsqueeze(0).to(device))
                    action = action.squeeze(0).cpu().numpy()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            done_bool = float(terminated)
            
            replay_buffer.add(state, action, reward, next_state, done_bool)
            
            state = next_state
            episode_reward += reward
            
            if done:
                mlflow.log_metric("rollout/ep_rew_mean", episode_reward, step=t)
                print(f"Step {t+1} | Episode Reward: {episode_reward:.2f}")
                state, _ = env.reset()
                episode_reward = 0
                
            if t >= 1000:
                s, a, r, s_prime, d = replay_buffer.sample(batch_size)
                s, a, r, s_prime, d = s.to(device), a.to(device), r.to(device), s_prime.to(device), d.to(device)
                
                with torch.no_grad():
                    next_a, next_log_prob = actor.sample(s_prime)
                    target_q = torch.min(target_q1(s_prime, next_a), target_q2(s_prime, next_a)) - alpha * next_log_prob
                    target_q_val = r + gamma * (1 - d) * target_q
                    
                q1_val = q1(s, a)
                q2_val = q2(s, a)
                q_loss = F.mse_loss(q1_val, target_q_val) + F.mse_loss(q2_val, target_q_val)
                
                q_opt.zero_grad()
                q_loss.backward()
                q_opt.step()
                
                # ACTOR UPDATE WITH REGULARIZATION
                pi_a, log_prob = actor.sample(s)
                q_pi = torch.min(q1(s, pi_a), q2(s, pi_a))
                
                # Compute expert action
                with torch.no_grad():
                    expert_a_raw = bc_expert(s)
                    # Note: BCPolicy was trained to output actions directly (using MSE on action directly)
                    expert_a = expert_a_raw
                
                bc_penalty = F.mse_loss(pi_a, expert_a)
                
                # Total policy loss
                pi_loss = (alpha * log_prob - q_pi).mean() + lambda_bc * bc_penalty
                
                actor_opt.zero_grad()
                pi_loss.backward()
                actor_opt.step()
                
                for param, target_param in zip(q1.parameters(), target_q1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(q2.parameters(), target_q2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
                if t % 5000 == 0:
                    mlflow.log_metric("train/actor_loss", pi_loss.item(), step=t)
                    mlflow.log_metric("train/critic_loss", q_loss.item(), step=t)
                    mlflow.log_metric("train/bc_penalty", bc_penalty.item(), step=t)

        torch.save(actor.state_dict(), "bc_sac_regularized_model.pt")
        mlflow.log_artifact("bc_sac_regularized_model.pt")
        print("Training Complete.")

if __name__ == "__main__":
    main()
