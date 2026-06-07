import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import os

def main():
    print("Initializing environment Walker2d-v5...")
    env = gym.make("Walker2d-v5")
    
    print("Creating SAC teacher model...")
    # Using standard SAC hyperparameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000,
        verbose=1,
        tensorboard_log="./sac_walker_tensorboard/"
    )
    
    # Eval callback to save the best model
    eval_env = gym.make("Walker2d-v5")
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./logs/',
        log_path='./logs/', 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )
    
    print("Starting teacher training. This will learn a perfect walking policy...")
    model.learn(total_timesteps=500000, callback=eval_callback, progress_bar=False)
    model.save("sac_walker2d_final")
    print("Model saved to sac_walker2d_final.zip")

if __name__ == "__main__":
    main()
