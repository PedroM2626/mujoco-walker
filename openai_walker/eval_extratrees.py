import gymnasium as gym
import numpy as np
import joblib
import time

def evaluate_extratrees():
    print("Loading Extra Trees model...")
    model = joblib.load("extratrees_model.pkl")
    
    env = gym.make("Walker2d-v5")
    
    print("Preparing to evaluate Extra Trees in Walker2D...")
    time.sleep(2)
    
    episodes = 5
    total_rewards = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            # scikit-learn expects 2D array for inference
            obs_reshaped = obs.reshape(1, -1)
            
            # Predict the continuous actions
            action = model.predict(obs_reshaped)[0]
            
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            
            # env.render() # Uncomment if you want to see the GUI
            
        total_rewards.append(ep_reward)
        print(f"[Extra Trees] Episode {ep+1} - Score: {ep_reward:.2f}")
        
    avg_reward = sum(total_rewards)/len(total_rewards)
    print("==================================================")
    print(f"Final Average Score: {avg_reward:.2f}")
    print("==================================================")
    
if __name__ == "__main__":
    evaluate_extratrees()
