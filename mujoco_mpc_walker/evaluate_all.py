import mujoco
import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

# Definitions of architectures
class WalkerTeacherNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WalkerTeacherNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.Mish(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.net(x)

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
        return mean, None # Ignoring std for deterministic evaluation

    def get_action(self, x, deterministic=True):
        mean, _ = self.forward(x)
        return torch.tanh(mean) * self.max_action

def evaluate_model(model, scaler, model_type, num_episodes=5, max_steps=1000):
    xml_path = "../walker_ragdoll.xml"
    model_mj = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model_mj)

    # Re-create observation feature selection logic
    qpos_cols_no_xy_len = model_mj.nq - 2
    qvel_cols_len = model_mj.nv
    base_obs_len = 2 + qpos_cols_no_xy_len + qvel_cols_len # rel_tx, rel_ty + qpos + qvel
    history_len = 3
    input_dim = base_obs_len * (history_len + 1)
    
    print(f"\n--- Evaluating {model_type} ---")
    
    episode_rewards = []
    episode_survivals = []
    
    # Store Z heights to plot a recovery chart
    all_z_heights = []

    for ep in range(num_episodes):
        mujoco.mj_resetData(model_mj, data)
        target_pos = np.random.uniform(-5.0, 5.0, size=2)
        data.mocap_pos[0, 0:2] = target_pos
        
        obs_history = []
        total_reward = 0
        
        for step in range(max_steps):
            def quaternion_to_yaw(qw, qx, qy, qz):
                return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                
            yaw = quaternion_to_yaw(data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6])
            rel_tx_global = target_pos[0] - data.qpos[0]
            rel_ty_global = target_pos[1] - data.qpos[1]
            
            rel_tx = rel_tx_global * np.cos(yaw) + rel_ty_global * np.sin(yaw)
            rel_ty = -rel_tx_global * np.sin(yaw) + rel_ty_global * np.cos(yaw)
            
            qpos_no_xy = data.qpos[2:].copy()
            qvel = data.qvel.copy()
            
            obs = np.concatenate([[rel_tx, rel_ty], qpos_no_xy, qvel])
            obs_history.append(obs)
            
            if len(obs_history) > history_len + 1:
                obs_history.pop(0)
                
            if len(obs_history) < history_len + 1:
                padded_history = [obs_history[0]] * (history_len + 1 - len(obs_history)) + obs_history
            else:
                padded_history = obs_history
                
            # Flatten history: t, t-1, t-2, t-3
            # The dataset was [t, t-1, t-2, t-3]
            stacked_obs = np.concatenate([padded_history[-1], padded_history[-2], padded_history[-3], padded_history[-4]])
            
            # Normalize
            stacked_obs_scaled = scaler.transform(stacked_obs.reshape(1, -1))
            
            with torch.no_grad():
                tensor_obs = torch.tensor(stacked_obs_scaled, dtype=torch.float32)
                if model_type == 'BC':
                    action = model(tensor_obs).numpy()[0]
                else:
                    action = model.get_action(tensor_obs, deterministic=True).numpy()[0]
                    
            data.ctrl[:] = action
            mujoco.mj_step(model_mj, data)
            
            # Record Z height
            if ep == 0:
                all_z_heights.append(data.qpos[2])
            
            # Very simple reward for evaluation: survive + move towards target
            dist_to_target = np.sqrt(rel_tx_global**2 + rel_ty_global**2)
            reward = -dist_to_target * 0.01 + (data.qpos[2] > 0.4) * 1.0
            total_reward += reward
            
            # Periodically throw the robot in the air to test recovery
            if step > 0 and step % 200 == 0:
                data.qpos[0:3] += np.random.uniform(-1, 1, size=3)
                data.qpos[2] = 2.0
            
            if data.qpos[2] < 0.2: # Hard fall
                # Give it a chance to recover, but if it stays down for 100 steps, terminate
                pass
                
        episode_rewards.append(total_reward)
        episode_survivals.append(step + 1)
        print(f"Episode {ep+1} | Reward: {total_reward:.2f} | Survival: {step+1}/{max_steps}")
        
    return np.mean(episode_rewards), all_z_heights

def main():
    results = {}
    z_heights_dict = {}
    
    # 1. Evaluate BC
    print("Loading BC...")
    with open("scaler.pkl", "rb") as f:
        scaler_bc = pickle.load(f)
    bc_model = WalkerTeacherNet(188, 17)
    bc_model.load_state_dict(torch.load("teacher_model.pt", weights_only=True))
    bc_model.eval()
    avg_r, z_bc = evaluate_model(bc_model, scaler_bc, "BC")
    results['BC'] = avg_r
    z_heights_dict['BC'] = z_bc
    
    # 2. Evaluate IQL
    print("Loading IQL...")
    if os.path.exists("scaler_iql.pkl") and os.path.exists("iql_model.pt"):
        with open("scaler_iql.pkl", "rb") as f:
            scaler_iql = pickle.load(f)
        iql_model = PolicyNet(188, 17)
        iql_model.load_state_dict(torch.load("iql_model.pt", weights_only=True))
        iql_model.eval()
        avg_r, z_iql = evaluate_model(iql_model, scaler_iql, "IQL")
        results['IQL'] = avg_r
        z_heights_dict['IQL'] = z_iql
    else:
        print("IQL model not found.")
        
    # 3. Evaluate CQL
    print("Loading CQL...")
    if os.path.exists("scaler_cql.pkl") and os.path.exists("cql_model.pt"):
        with open("scaler_cql.pkl", "rb") as f:
            scaler_cql = pickle.load(f)
        cql_model = PolicyNet(188, 17)
        cql_model.load_state_dict(torch.load("cql_model.pt", weights_only=True))
        cql_model.eval()
        avg_r, z_cql = evaluate_model(cql_model, scaler_cql, "CQL")
        results['CQL'] = avg_r
        z_heights_dict['CQL'] = z_cql
    else:
        print("CQL model not found.")
        
    print("\n=== FINAL RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v:.2f} Avg Reward")
        
    # Plot Z heights
    plt.figure(figsize=(12, 6))
    for k, z in z_heights_dict.items():
        plt.plot(z, label=k, alpha=0.7)
    plt.axhline(y=0.4, color='r', linestyle='--', label='Fall Threshold')
    plt.title('Agent Height (Recovery Comparison) - First Episode')
    plt.xlabel('Steps (Drop every 200 steps)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison.png')
    print("Saved comparison.png")

if __name__ == "__main__":
    main()
