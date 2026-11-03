import argparse
from contextlib import nullcontext
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

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

def parse_args():
    p = argparse.ArgumentParser(description="SAC Offline-to-Online Fine-Tuning from IQL (Walker2d-v5)")
    p.add_argument("--ckpt", default=os.path.join(HERE, "iql_full_ckpt.pt"), help="Path to pretrained IQL checkpoint")
    p.add_argument("--out", default=os.path.join(HERE, "iql_sac_model.pt"), help="Output model path")
    p.add_argument("--max-timesteps", type=int, default=100000, help="Total online interaction steps")
    p.add_argument("--seed", type=int, default=2026, help="Random seed")
    p.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if not args.no_mlflow:
        import mlflow
        from mlflow_backend import set_uri
        set_uri()
        mlflow.set_experiment("Walker_Offline_To_Online")
        run_ctx = mlflow.start_run(run_name="IQL_SAC_FineTuning_Walker2d")
    else:
        mlflow = None
        run_ctx = nullcontext()

    env = gym.make("Walker2d-v5")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    actor = PolicyNet(state_dim, action_dim, max_action).to(device)
    q1 = QNetwork(state_dim, action_dim).to(device)
    q2 = QNetwork(state_dim, action_dim).to(device)
    target_q1 = QNetwork(state_dim, action_dim).to(device)
    target_q2 = QNetwork(state_dim, action_dim).to(device)
    
    model_path = args.ckpt
    if os.path.exists(model_path):
        print(f"Loading pretrained IQL weights from {model_path} into SAC...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint['policy'])
        q1.load_state_dict(checkpoint['q1'])
        q2.load_state_dict(checkpoint['q2'])
    else:
        print("Warning: iql_full_ckpt.pt not found! Training from scratch.")

    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
    
    # We use a much smaller learning rate to not destroy the pre-trained weights
    actor_opt = optim.Adam(actor.parameters(), lr=args.lr)
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)
    
    alpha = 0.2
    gamma = 0.99
    tau = 0.005
    batch_size = args.batch_size
    max_timesteps = args.max_timesteps
    
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    episode_reward = 0
    episode_timesteps = 0
    
    state, _ = env.reset(seed=args.seed)
    
    with run_ctx:
        print("Starting SAC Offline-to-Online Fine-Tuning from IQL...")
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
            episode_timesteps += 1
            
            if done:
                if mlflow is not None:
                    mlflow.log_metric("rollout/ep_rew_mean", episode_reward, step=t)
                if (t + 1) % 5000 == 0 or t < 3000:
                    print(f"Step {t+1}/{max_timesteps} | Episode Reward: {episode_reward:.2f}")
                state, _ = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                
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
                
                pi_a, log_prob = actor.sample(s)
                q_pi = torch.min(q1(s, pi_a), q2(s, pi_a))
                pi_loss = (alpha * log_prob - q_pi).mean()
                
                actor_opt.zero_grad()
                pi_loss.backward()
                actor_opt.step()
                
                for param, target_param in zip(q1.parameters(), target_q1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(q2.parameters(), target_q2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
                if (t + 1) % 5000 == 0:
                    if mlflow is not None:
                        mlflow.log_metric("train/actor_loss", pi_loss.item(), step=t)
                        mlflow.log_metric("train/critic_loss", q_loss.item(), step=t)
                    print(f"Step {t+1}/{max_timesteps} | Q-Loss: {q_loss.item():.4f} | Pi-Loss: {pi_loss.item():.4f}")

        torch.save(actor.state_dict(), args.out)
        if mlflow is not None:
            mlflow.log_artifact(args.out)
        print(f"Training Complete. Model saved to {args.out}")

if __name__ == "__main__":
    main()
