import mock_wrappers
import os
import torch
import numpy as np
import gymnasium as gym
import gymnasium.wrappers
import time
from train_walker import SACAgent
from train_moe_gate import MoEGate
import matplotlib.pyplot as plt

def load_agent(checkpoint_path, device, input_dim=46):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(17,))
    agent = SACAgent(input_dim, action_space).to(device)
    state_dict = checkpoint.get("actor_state_dict", checkpoint)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent

def evaluate_paradigm(env_name, paradigm_name, device, rec_agent=None, tgt_agent=None, single_agent=None, moe_gate=None, num_episodes=5):
    env = gym.make(env_name, reset_mode="mixed", task_phase="target")
    total_rewards = []
    survivals = []
    
    print(f"\n--- Evaluating {paradigm_name} ---")
    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        survived = True
        
        for step in range(1000):
            if isinstance(obs, dict):
                obs_array = obs["obs"]
            else:
                obs_array = obs
                
            obs_46 = torch.FloatTensor(obs_array[:46]).unsqueeze(0).to(device)
            obs_49 = torch.FloatTensor(obs_array).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if paradigm_name == "Hardcoded Supervisor":
                    z = env.unwrapped.data.qpos[2]
                    upright = env.unwrapped.upright_factor
                    if z < 1.1 or upright < 0.8:
                        action = rec_agent.get_action(obs_46, deterministic=True)[0].cpu().numpy()
                    else:
                        action = tgt_agent.get_action(obs_49, deterministic=True)[0].cpu().numpy()
                elif paradigm_name == "Mixture of Experts":
                    g = moe_gate(obs_46).item()
                    a_rec = rec_agent.get_action(obs_46, deterministic=True)[0].cpu().numpy()
                    a_tgt = tgt_agent.get_action(obs_49, deterministic=True)[0].cpu().numpy()
                    action = g * a_rec + (1.0 - g) * a_tgt
                else:
                    action = single_agent.get_action(obs_49, deterministic=True)[0].cpu().numpy()
            
            action = action.flatten()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            
            if terminated:
                survived = False
                break
                
        total_rewards.append(ep_reward)
        survivals.append(1 if survived else 0)
        print(f"Episode {ep+1} | Reward: {ep_reward:.2f} | Survived: {survived}")
        
    env.close()
    return np.mean(total_rewards), np.mean(survivals)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading models...")
    rec_ckpt = "checkpoints/walker_recovery_v1/sac_ckpt_20000000.pt"
    tgt_ckpt = "checkpoints/walker_target_v1/sac_ckpt_40000000.pt"
    
    rec_agent = load_agent(rec_ckpt, device, input_dim=46)
    tgt_agent = load_agent(tgt_ckpt, device, input_dim=49)
    avg_agent = load_agent("merged_avg_model.pt", device, input_dim=49)
    ta_agent = load_agent("merged_ta_model.pt", device, input_dim=49)
    
    moe_gate = MoEGate().to(device)
    moe_gate.load_state_dict(torch.load("moe_gate.pt", weights_only=True))
    moe_gate.eval()
    
    results = {}
    
    num_ep = 1000
    r, s = evaluate_paradigm("WalkerRagdoll-v0", "Hardcoded Supervisor", device, rec_agent=rec_agent, tgt_agent=tgt_agent, num_episodes=num_ep)
    results["Hardcoded Supervisor"] = {"Reward": r, "Survival Rate": s}
    
    r, s = evaluate_paradigm("WalkerRagdoll-v0", "Mixture of Experts", device, rec_agent=rec_agent, tgt_agent=tgt_agent, moe_gate=moe_gate, num_episodes=num_ep)
    results["Mixture of Experts"] = {"Reward": r, "Survival Rate": s}
    
    r, s = evaluate_paradigm("WalkerRagdoll-v0", "Weight Averaging", device, single_agent=avg_agent, num_episodes=num_ep)
    results["Weight Averaging"] = {"Reward": r, "Survival Rate": s}
    
    r, s = evaluate_paradigm("WalkerRagdoll-v0", "Task Arithmetic", device, single_agent=ta_agent, num_episodes=num_ep)
    results["Task Arithmetic"] = {"Reward": r, "Survival Rate": s}
    
    print("\n=== FINAL RANKING ===")
    for k, v in results.items():
        print(f"{k}: {v['Reward']:.2f} Avg Reward | {v['Survival Rate']*100:.0f}% Survival")
