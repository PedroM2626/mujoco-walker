import mock_wrappers
import os
import torch
import numpy as np
import copy
from train_walker import SACAgent
import gymnasium as gym
import gymnasium.wrappers

def load_agent(checkpoint_path, device, input_dim=46):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(17,))
    agent = SACAgent(input_dim, action_space).to(device)
    state_dict = checkpoint.get("actor_state_dict", checkpoint)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent

def weight_averaging(agent_rec, agent_tgt, alpha=0.5):
    """
    Linearly interpolates weights: W_merged = alpha * W_tgt + (1 - alpha) * W_rec
    agent_rec has 46 input dims, agent_tgt has 49.
    """
    merged_agent = copy.deepcopy(agent_tgt)
    dict_rec = agent_rec.state_dict()
    dict_tgt = agent_tgt.state_dict()
    
    merged_dict = {}
    for key in dict_tgt.keys():
        if key == "actor.backbone.0.weight" or key == "backbone.0.weight":
            # Pad the recovery weights with 0 for the last 3 target dimensions
            w_rec = dict_rec[key]
            padded_rec = torch.zeros_like(dict_tgt[key])
            padded_rec[:, :46] = w_rec
            merged_dict[key] = (1.0 - alpha) * padded_rec + alpha * dict_tgt[key]
        else:
            merged_dict[key] = (1.0 - alpha) * dict_rec[key] + alpha * dict_tgt[key]
        
    merged_agent.load_state_dict(merged_dict)
    return merged_agent

def task_arithmetic(base_agent_46, agent_rec_46, agent_tgt_49, lambda_A=1.0, lambda_B=1.0):
    """
    Merges via task vectors. Base is padded to 49.
    """
    merged_agent = copy.deepcopy(agent_tgt_49)
    dict_base = base_agent_46.state_dict()
    dict_rec = agent_rec_46.state_dict()
    dict_tgt = agent_tgt_49.state_dict()
    
    merged_dict = {}
    for key in dict_tgt.keys():
        w_base = dict_base[key]
        w_rec = dict_rec[key]
        w_tgt = dict_tgt[key]
        
        if key == "actor.backbone.0.weight" or key == "backbone.0.weight":
            padded_base = torch.zeros_like(w_tgt)
            padded_base[:, :46] = w_base
            w_base = padded_base
            
            padded_rec = torch.zeros_like(w_tgt)
            padded_rec[:, :46] = w_rec
            w_rec = padded_rec
            
        tau_rec = w_rec - w_base
        tau_tgt = w_tgt - w_base
        merged_dict[key] = w_base + lambda_A * tau_rec + lambda_B * tau_tgt
        
    merged_agent.load_state_dict(merged_dict)
    return merged_agent

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading checkpoints...")
    
    base_ckpt = "checkpoints/walker_recovery_v1/sac_ckpt_1000000.pt"
    rec_ckpt = "checkpoints/walker_recovery_v1/sac_ckpt_20000000.pt"
    tgt_ckpt = "checkpoints/walker_target_v1/sac_ckpt_40000000.pt"
    
    if not os.path.exists(base_ckpt) or not os.path.exists(rec_ckpt) or not os.path.exists(tgt_ckpt):
        print("Error: Could not find required checkpoints.")
        exit(1)
        
    base_agent = load_agent(base_ckpt, device, input_dim=46)
    rec_agent = load_agent(rec_ckpt, device, input_dim=46)
    tgt_agent = load_agent(tgt_ckpt, device, input_dim=49)
    
    print("Performing Weight Averaging (alpha=0.5)...")
    avg_agent = weight_averaging(rec_agent, tgt_agent, alpha=0.5)
    
    print("Performing Task Arithmetic (lambda_rec=1.0, lambda_tgt=1.0)...")
    ta_agent = task_arithmetic(base_agent, rec_agent, tgt_agent, lambda_A=1.0, lambda_B=1.0)
    
    # Save checkpoints with obs_rms from the target checkpoint
    tgt_checkpoint = torch.load(tgt_ckpt, map_location=device)
    obs_rms = tgt_checkpoint.get("obs_rms", None)
    
    avg_ckpt = {
        "algo": "sac",
        "actor_state_dict": avg_agent.state_dict(),
        "obs_rms": obs_rms
    }
    ta_ckpt = {
        "algo": "sac",
        "actor_state_dict": ta_agent.state_dict(),
        "obs_rms": obs_rms
    }
    
    torch.save(avg_ckpt, "merged_avg_model.pt")
    torch.save(ta_ckpt, "merged_ta_model.pt")
    print("Saved merged_avg_model.pt and merged_ta_model.pt")
