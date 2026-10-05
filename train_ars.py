import os
import random
import time
import argparse
import gymnasium as gym
import numpy as np
import torch
import mlflow
from torch.utils.tensorboard import SummaryWriter

# Import wrappers and utilities from train_walker
from train_walker import (
    make_env,
    force_delete_run,
    get_checkpoint_dir,
    get_rng_state,
    set_rng_state,
    ENV_VERSION,
    start_mlflow_run,
    log_mlflow_metrics,
    log_mlflow_artifact,
    end_mlflow_run,
)
from envs.reward_shaping import TRAINING_REWARD_KWARGS

class ObservationNormalizer:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)

def parse_ars_args():
    parser = argparse.ArgumentParser(description="ARS Walker Ragdoll Training")
    parser.add_argument("--run-id", type=str, default="walker_ars_1m")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--reset-mode", type=str, default="upright")
    parser.add_argument("--fixed-reset-probability", type=float, default=0.25)
    parser.add_argument("--upright-reset-probability", type=float, default=0.15)
    parser.add_argument("--fallen-velocity-scale", type=float, default=0.35)
    parser.add_argument("--task-phase", type=str, default="target")
    parser.add_argument("--target-forward-velocity", type=float, default=0.8)
    parser.add_argument("--checkpoint-interval", type=int, default=200000)
    
    # ARS specific hyperparameters
    parser.add_argument("--num-directions", type=int, default=8, help="Number of random directions per epoch (N)")
    parser.add_argument("--num-top-directions", type=int, default=4, help="Number of top-performing directions to keep (b)")
    parser.add_argument("--step-size", type=float, default=0.02, help="Learning rate (alpha)")
    parser.add_argument("--noise-std", type=float, default=0.03, help="Perturbation standard deviation (nu)")
    
    return parser.parse_args()

def run_episode(env, weights, bias, normalizer, max_steps=1000, update_normalizer=True):
    obs = env.reset()[0]
    total_reward = 0
    steps = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated) and steps < max_steps:
        # Normalize observation
        if update_normalizer:
            normalizer.update(np.array([obs]))
        obs_norm = normalizer.normalize(obs)
        
        # Linear policy action selection
        action = np.tanh(np.dot(weights, obs_norm) + bias)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
    return total_reward, steps

def save_ars_checkpoint(ckpt_path, global_step, weights, bias, normalizer, task_phase, target_forward_velocity):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    state = {
        "algo": "ars",
        "env_version": ENV_VERSION,
        "global_step": global_step,
        "weights": weights,
        "bias": bias,
        "obs_rms": {
            "mean": normalizer.mean,
            "var": normalizer.var,
            "count": normalizer.count,
        },
        "task_phase": task_phase,
        "target_forward_velocity": target_forward_velocity,
        # Recorded so an evaluator scores this policy under the reward it was trained with. Without
        # it the scorer has to guess, and it guessed wrong once already (see reward_kwargs_for).
        "reward_kwargs": dict(TRAINING_REWARD_KWARGS),
        "rng_state": get_rng_state(),
    }
    torch.save(state, ckpt_path)
    print(f"[CHECKPOINT] Saved at step {global_step} -> {ckpt_path}")

def train_ars():
    args = parse_ars_args()
    run_name = f"{args.run_id}__{args.seed}"

    if not args.resume or args.force:
        force_delete_run(args.run_id, run_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # We only need one environment instance for sequential rollouts in ARS
    env_id = "WalkerRagdoll-v0"
    env = make_env(
        env_id,
        0,
        False,
        run_name,
        reset_mode=args.reset_mode,
        fixed_reset_probability=args.fixed_reset_probability,
        upright_reset_probability=args.upright_reset_probability,
        fallen_velocity_scale=args.fallen_velocity_scale,
        task_phase=args.task_phase,
        target_forward_velocity=args.target_forward_velocity,
        terminate_when_unhealthy=(args.task_phase == "target"),
    )()

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    # Initialize policy weights and bias
    weights = np.zeros((action_dim, obs_dim))
    bias = np.zeros(action_dim)
    normalizer = ObservationNormalizer(obs_dim)
    
    global_step = 0
    epoch = 0
    ckpt_dir = get_checkpoint_dir(args.run_id)
    next_checkpoint_step = ((global_step // args.checkpoint_interval) + 1) * args.checkpoint_interval

    # Setup WandB (removido: tracking canônico é MLflow + TensorBoard; wandb não é dependência)
    # Setup MLflow
    mlf_run = start_mlflow_run(args, run_name, "ars")

    # Initialize from checkpoint
    if args.resume:
        files = [f for f in os.listdir(ckpt_dir) if f.startswith("ars_ckpt_") and f.endswith(".pt")] if os.path.exists(ckpt_dir) else []
        if files:
            files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            ckpt_path = os.path.join(ckpt_dir, files[-1])
            print(f"[CHECKPOINT] Loading from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, weights_only=False)
            weights = checkpoint["weights"]
            bias = checkpoint["bias"]
            normalizer.mean = checkpoint["obs_rms"]["mean"]
            normalizer.var = checkpoint["obs_rms"]["var"]
            normalizer.count = checkpoint["obs_rms"]["count"]
            if checkpoint.get("rng_state") is not None:
                set_rng_state(checkpoint["rng_state"])
            global_step = checkpoint["global_step"]

    writer = SummaryWriter(os.path.join("runs", run_name))

    try:
        while global_step < args.total_timesteps:
            # 1. Generate random noise directions
            delta_weights = [np.random.normal(size=weights.shape) for _ in range(args.num_directions)]
            delta_biases = [np.random.normal(size=bias.shape) for _ in range(args.num_directions)]
            
            returns_pos = []
            returns_neg = []
            steps_epoch = 0
            
            # 2. Rollouts with positive and negative perturbations
            for i in range(args.num_directions):
                # Positive perturbation
                w_pos = weights + args.noise_std * delta_weights[i]
                b_pos = bias + args.noise_std * delta_biases[i]
                r_pos, s_pos = run_episode(env, w_pos, b_pos, normalizer, update_normalizer=True)
                returns_pos.append(r_pos)
                steps_epoch += s_pos
                
                # Negative perturbation
                w_neg = weights - args.noise_std * delta_weights[i]
                b_neg = bias - args.noise_std * delta_biases[i]
                r_neg, s_neg = run_episode(env, w_neg, b_neg, normalizer, update_normalizer=True)
                returns_neg.append(r_neg)
                steps_epoch += s_neg

            global_step += steps_epoch
            epoch += 1

            # 3. Sort perturbations by performance
            all_returns = np.array(returns_pos + returns_neg)
            sigma_r = np.std(all_returns)
            if sigma_r < 1e-8:
                sigma_r = 1e-8 # Prevent divide by zero

            # Rank directions based on the max return of the pair
            scores = {i: max(returns_pos[i], returns_neg[i]) for i in range(args.num_directions)}
            sorted_directions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Keep only the top b directions
            top_directions = sorted_directions[:args.num_top_directions]
            
            # 4. Update policy weights and bias
            weight_update = np.zeros_like(weights)
            bias_update = np.zeros_like(bias)
            
            for idx, _ in top_directions:
                weight_update += (returns_pos[idx] - returns_neg[idx]) * delta_weights[idx]
                bias_update += (returns_pos[idx] - returns_neg[idx]) * delta_biases[idx]
                
            weights += (args.step_size / (args.num_top_directions * sigma_r)) * weight_update
            bias += (args.step_size / (args.num_top_directions * sigma_r)) * bias_update

            # Evaluate current unperturbed policy
            eval_r, eval_s = run_episode(env, weights, bias, normalizer, update_normalizer=False)
            
            # Logging progress
            print(f"global_step={global_step}, epoch={epoch}, eval_return={eval_r:.2f}, eval_length={eval_s}, sigma_r={sigma_r:.4f}")
            writer.add_scalar("charts/episodic_return", eval_r, global_step)
            writer.add_scalar("charts/episodic_length", eval_s, global_step)
            log_mlflow_metrics(mlf_run, {
                "episodic_return": eval_r,
                "episodic_length": eval_s,
                "sigma_r": sigma_r,
            }, global_step)

            # Checkpoint saving
            if global_step >= next_checkpoint_step:
                ckpt_path = os.path.join(ckpt_dir, f"ars_ckpt_{global_step}.pt")
                save_ars_checkpoint(ckpt_path, global_step, weights, bias, normalizer, args.task_phase, args.target_forward_velocity)
                log_mlflow_artifact(mlf_run, ckpt_path, "ars", global_step)
                next_checkpoint_step += args.checkpoint_interval

    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted by user. Saving checkpoint...")
    finally:
        final_checkpoint_path = os.path.join(ckpt_dir, f"ars_ckpt_{global_step}.pt")
        save_ars_checkpoint(final_checkpoint_path, global_step, weights, bias, normalizer, args.task_phase, args.target_forward_velocity)
        log_mlflow_artifact(mlf_run, final_checkpoint_path, "ars", global_step)
        end_mlflow_run(mlf_run)
        env.close()
        writer.close()
        print(f"Training completed. Total steps: {global_step}")

if __name__ == "__main__":
    train_ars()
