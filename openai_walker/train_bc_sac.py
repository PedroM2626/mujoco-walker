import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import mlflow
import os
import pickle
import sys
from collections import deque

# Add root project path to import envs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs.walker_ragdoll_env

class FrameStackAndScaleWrapper(gym.ObservationWrapper):
    def __init__(self, env, history_len=3, scaler_path="scaler.pkl"):
        super().__init__(env)
        self.history_len = history_len
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        # The base observation length we will extract from the raw qpos/qvel
        self.base_obs_len = 47
        self.stacked_obs_len = self.base_obs_len * (self.history_len + 1)
        
        # We replace the gym space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.stacked_obs_len,), dtype=np.float32)
        self.obs_history = deque(maxlen=history_len + 1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs_history.clear()
        
        # Get raw base obs
        base_obs = self._extract_base_obs()
        for _ in range(self.history_len + 1):
            self.obs_history.append(base_obs)
            
        return self._get_stacked_and_scaled(), info

    def observation(self, obs):
        base_obs = self._extract_base_obs()
        self.obs_history.append(base_obs)
        return self._get_stacked_and_scaled()
        
    def _extract_base_obs(self):
        # Must match the columns layout of dataset.csv:
        # rel_tx, rel_ty, qpos_2 to qpos_23, qvel_0 to qvel_22
        
        target_pos = self.env.unwrapped.target_pos if hasattr(self.env.unwrapped, "target_pos") else np.zeros(2)
        qpos = self.env.unwrapped.data.qpos
        qvel = self.env.unwrapped.data.qvel
        
        # Quaternion to Yaw
        qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        rel_tx_global = target_pos[0] - qpos[0]
        rel_ty_global = target_pos[1] - qpos[1]
        
        rel_tx = rel_tx_global * np.cos(yaw) + rel_ty_global * np.sin(yaw)
        rel_ty = -rel_tx_global * np.sin(yaw) + rel_ty_global * np.cos(yaw)
        
        qpos_no_xy = qpos[2:].copy()
        
        base_obs = np.concatenate([[rel_tx, rel_ty], qpos_no_xy, qvel])
        return base_obs

    def _get_stacked_and_scaled(self):
        # Flatten history: t, t-1, t-2, t-3
        stacked = []
        # obs_history[-1] is t, obs_history[-2] is t-1, etc.
        for i in range(1, len(self.obs_history) + 1):
            stacked.append(self.obs_history[-i])
        stacked_obs = np.concatenate(stacked)
        
        # Scale
        scaled_obs = self.scaler.transform(stacked_obs.reshape(1, -1))[0]
        return scaled_obs

class BCActor(nn.Module):
    def __init__(self, input_dim, output_dim, max_action=1.0):
        super(BCActor, self).__init__()
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

    def load_bc_weights(self, path):
        # Load weights from the deterministic WalkerTeacherNet into the stochastic BCActor
        print(f"Loading pre-trained BC weights from {path} into Actor...")
        state_dict = torch.load(path, weights_only=True)
        my_state_dict = self.state_dict()
        
        for k, v in state_dict.items():
            if k == "net.9.weight":
                my_state_dict["mean_layer.weight"] = v
            elif k == "net.9.bias":
                my_state_dict["mean_layer.bias"] = v
            elif k in my_state_dict:
                my_state_dict[k] = v
        self.load_state_dict(my_state_dict)

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
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
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
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Walker_Behavioral_Cloning")

    env = gym.make("WalkerRagdoll-v0")
    env = FrameStackAndScaleWrapper(env, history_len=3, scaler_path="scaler.pkl")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = BCActor(state_dim, action_dim, max_action)
    if os.path.exists("teacher_model.pt"):
        actor.load_bc_weights("teacher_model.pt")

    q1 = QNetwork(state_dim, action_dim)
    q2 = QNetwork(state_dim, action_dim)
    target_q1 = QNetwork(state_dim, action_dim)
    target_q2 = QNetwork(state_dim, action_dim)
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())

    # We use a very low learning rate for the Actor to preserve BC weights
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-5)
    # Critics learn faster from scratch
    q_optimizer = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    max_timesteps = 100000
    batch_size = 256
    gamma = 0.99
    tau = 0.005
    alpha = 0.2

    state, _ = env.reset()
    episode_reward = 0
    episode_num = 0

    with mlflow.start_run(run_name="BC_plus_SAC_Finetuning"):
        mlflow.log_param("model_type", "BC+SAC")
        mlflow.log_param("max_timesteps", max_timesteps)

        for t in range(int(max_timesteps)):
            # Interaction
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state.reshape(1, -1))
                action, _ = actor.sample(state_tensor)
                action = action.numpy()[0]
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Custom reward for online fine-tuning: encourage walking towards target and staying upright
            # The environment already has these, but we can augment it slightly if needed.
            # We'll use the env's native reward.

            replay_buffer.add(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward

            if done:
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Reward: {episode_reward:.2f}")
                mlflow.log_metric("online_reward", episode_reward, step=t)
                state, _ = env.reset()
                episode_reward = 0
                episode_num += 1

            # Training
            if replay_buffer.size > batch_size:
                s, a, r, s_prime, d = replay_buffer.sample(batch_size)

                # Critic Update
                with torch.no_grad():
                    next_a, next_log_prob = actor.sample(s_prime)
                    target_q = torch.min(target_q1(s_prime, next_a), target_q2(s_prime, next_a)) - alpha * next_log_prob
                    target_q_val = r + gamma * (1 - d) * target_q

                q1_val = q1(s, a)
                q2_val = q2(s, a)
                q_loss = F.mse_loss(q1_val, target_q_val) + F.mse_loss(q2_val, target_q_val)

                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()

                # Actor Update
                pi_a, log_prob = actor.sample(s)
                q_pi = torch.min(q1(s, pi_a), q2(s, pi_a))
                pi_loss = (alpha * log_prob - q_pi).mean()

                actor_optimizer.zero_grad()
                pi_loss.backward()
                actor_optimizer.step()

                # Targets Update
                for param, target_param in zip(q1.parameters(), target_q1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(q2.parameters(), target_q2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        torch.save(actor.state_dict(), "bc_sac_model.pt")
        mlflow.log_artifact("bc_sac_model.pt")
        print("BC+SAC Training Complete.")

if __name__ == "__main__":
    main()
