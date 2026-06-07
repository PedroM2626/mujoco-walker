import gymnasium as gym
from stable_baselines3 import SAC
import pandas as pd
import numpy as np
import os

def main():
    print("Loading Walker2d-v5 environment...")
    env = gym.make("Walker2d-v5")
    
    # Try to load the best model from evaluations, fallback to final model
    model_path = "./logs/best_model.zip"
    if not os.path.exists(model_path):
        model_path = "sac_walker2d_final.zip"
        
    print(f"Loading trained SAC teacher from {model_path}...")
    model = SAC.load(model_path, env=env)
    
    print("Generating dataset...")
    num_steps = 100000
    
    data = []
    
    obs, info = env.reset()
    for i in range(num_steps):
        # SAC uses deterministic=True for evaluation to remove exploration noise
        action, _states = model.predict(obs, deterministic=True)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # We save the current observation, next observation, action, reward, and done flag
        # For Walker2d-v5: obs is 17-dim, action is 6-dim
        row = list(obs) + list(next_obs) + list(action) + [reward, int(done)]
        data.append(row)
        
        if done:
            obs, info = env.reset()
        else:
            obs = next_obs
            
        if (i + 1) % 10000 == 0:
            print(f"Generated {i + 1} / {num_steps} steps...")
            
    print("Saving dataset to dataset_openai.csv...")
    
    # Create column names
    obs_cols = [f"obs_{i}" for i in range(env.observation_space.shape[0])]
    next_obs_cols = [f"next_obs_{i}" for i in range(env.observation_space.shape[0])]
    act_cols = [f"action_{i}" for i in range(env.action_space.shape[0])]
    columns = obs_cols + next_obs_cols + act_cols + ["reward", "done"]
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("dataset_openai.csv", index=False)
    print("Dataset saved successfully!")

if __name__ == "__main__":
    main()
