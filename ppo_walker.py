"""
PPO Walker Ragdoll Training with CleanRL-style implementation.
Features: checkpointing, resume, force restart, run_id, seed, etc.
"""

import os
import sys
import time
import argparse
import random
from datetime import datetime
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch

# Monkey-patch for TensorBoard/Protobuf compatibility (Fixes TypeError in MessageToJson)
try:
    import google.protobuf.json_format as json_format
    _original_MessageToJson = json_format.MessageToJson
    def _patched_MessageToJson(message, **kwargs):
        kwargs.pop('including_default_value_fields', None)
        return _original_MessageToJson(message, **kwargs)
    json_format.MessageToJson = _patched_MessageToJson
except ImportError:
    pass

import envs.walker_ragdoll_env
from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    force_delete_run,
    get_run_dir,
    get_checkpoint_dir,
)


# Parse environment variables with defaults
ENV_VARS = {
    "RUN_ID": "walker_ppo",
    "SEED": "1",
    "CHECKPOINT_INTERVAL": "1000000",
    "TOTAL_TIMESTEPS": "1500000",
    "LEARNING_RATE": "3e-4",
    "NUM_ENVS": "64",
    "NUM_STEPS": "2048",
    "GAMMA": "0.99",
    "GAE_LAMBDA": "0.95",
    "NUM_MINIBATCHES": "32",
    "UPDATE_EPOCHS": "10",
    "CLIP_COEF": "0.2",
    "ENT_COEF": "0.01",
    "VF_COEF": "0.5",
    "MAX_GRAD_NORM": "0.5",
    "TARGET_KL": "0.01",
}


def get_env_or_default(key, default):
    return os.environ.get(key, default)


# ------------------------
# Argument parser (like ML-Agents)
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="PPO Walker Ragdoll Training")
    parser.add_argument(
        "--run-id",
        type=str,
        default=get_env_or_default("RUN_ID", ENV_VARS["RUN_ID"]),
        help="Unique identifier for this training run (used for checkpoints and logs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(get_env_or_default("SEED", ENV_VARS["SEED"])),
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the latest checkpoint. If not provided, the run will be FORCED (existing checkpoints and logs for this run-id will be deleted).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Explicitly force delete existing run data (default behavior if --resume is not used).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=int(get_env_or_default("CHECKPOINT_INTERVAL", ENV_VARS["CHECKPOINT_INTERVAL"])),
        help="Save a checkpoint every N timesteps",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=int(get_env_or_default("TOTAL_TIMESTEPS", ENV_VARS["TOTAL_TIMESTEPS"])),
        help="Total number of timesteps to train",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(get_env_or_default("LEARNING_RATE", ENV_VARS["LEARNING_RATE"])),
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=int(get_env_or_default("NUM_ENVS", ENV_VARS["NUM_ENVS"])),
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=int(get_env_or_default("NUM_STEPS", ENV_VARS["NUM_STEPS"])),
        help="Number of steps per environment per policy rollout",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=float(get_env_or_default("GAMMA", ENV_VARS["GAMMA"])),
        help="Discount factor gamma",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=float(get_env_or_default("GAE_LAMBDA", ENV_VARS["GAE_LAMBDA"])),
        help="Lambda for Generalized Advantage Estimation",
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=int(get_env_or_default("NUM_MINIBATCHES", ENV_VARS["NUM_MINIBATCHES"])),
        help="Number of minibatches per update",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=int(get_env_or_default("UPDATE_EPOCHS", ENV_VARS["UPDATE_EPOCHS"])),
        help="Number of epochs to update the policy",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=float(get_env_or_default("CLIP_COEF", ENV_VARS["CLIP_COEF"])),
        help="Surrogate clipping coefficient",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=float(get_env_or_default("ENT_COEF", ENV_VARS["ENT_COEF"])),
        help="Entropy coefficient",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=float(get_env_or_default("VF_COEF", ENV_VARS["VF_COEF"])),
        help="Value function coefficient",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=float(get_env_or_default("MAX_GRAD_NORM", ENV_VARS["MAX_GRAD_NORM"])),
        help="Maximum norm for gradient clipping",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=float(get_env_or_default("TARGET_KL", ENV_VARS["TARGET_KL"])),
        help="Target KL divergence threshold for early stopping",
    )
    parser.add_argument(
        "--capture-video",
        action="store_true",
        default=False,
        help="Capture video of the agent during training",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        default=False,
        help="Track training with Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="walker-ragdoll-ppo",
        help="Wandb project name",
    )
    return parser.parse_args()


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # We will apply NormalizeObservation and NormalizeReward to the vector env instead
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_space = getattr(envs, "single_observation_space", envs.observation_space)
        act_space = getattr(envs, "single_action_space", envs.action_space)
        obs_shape = np.array(obs_space.shape).prod()
        action_shape = np.prod(act_space.shape)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_shape), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


def train(start_time=None):
    if start_time is None:
        start_time = time.time()
    args = parse_args()

    run_name = f"{args.run_id}__{args.seed}__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Default to FORCE unless --resume is explicitly provided
    if not args.resume:
        print(f"[INFO] --resume not provided. Defaulting to FORCE for run-id: {args.run_id}")
        force_delete_run(args.run_id)
    elif args.force:
        # If user explicitly asked for force even with resume (unlikely but handled)
        force_delete_run(args.run_id)

    # Setup directories
    run_dir = get_run_dir(run_name)
    ckpt_dir = get_checkpoint_dir(args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Enable cudnn autotuner for potentially faster kernels on GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Tensorboard
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # MLflow Setup (Satisfying user_global MLOps rules)
    try:
        mlflow.set_experiment(args.wandb_project)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(vars(args))
        print(f"[MLFLOW] tracking started in experiment: {args.wandb_project}")
    except Exception as e:
        print(f"[MLFLOW] warning: could not start tracking: {e}")

    # Wandb tracking
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=None,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Environment setup
    env_id = "WalkerRagdoll-v0"
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(env_id, i, args.capture_video, run_name, args.gamma)
            for i in range(args.num_envs)
        ]
    )
    # Apply vector wrappers for normalization
    envs = gym.wrappers.NormalizeObservation(envs)
    envs = gym.wrappers.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))
    envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        _, loaded_step = load_checkpoint(agent, optimizer, args.run_id, envs=envs)
        if loaded_step > 0:
            start_step = loaded_step
            print(f"[RESUME] Starting from step {start_step}")
        else:
            print("[RESUME] No checkpoint found. Starting from scratch.")

    # Storage setup (use float32 on device to avoid unnecessary casts)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.float32, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.float32, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

    # Initialize environment
    global_step = start_step
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    batch_size = args.num_steps * args.num_envs
    num_updates = args.total_timesteps // batch_size
    update_start = start_step // batch_size

    # Training loop with graceful KeyboardInterrupt handling to save checkpoints
    interrupted = False
    try:
        for update in range(update_start, num_updates):
            initial_global_step = global_step

            # Annealing learning rate
            if args.learning_rate > 0:
                frac = 1.0 - (update / num_updates)
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # Action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminations, truncations, infos = envs.step(
                    action.cpu().numpy()
                )
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
                next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                next_done = torch.as_tensor(next_done, dtype=torch.float32, device=device)

                # Log episode statistics
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            ep_r = info["episode"]["r"].item() if hasattr(info["episode"]["r"], "item") else float(info["episode"]["r"])
                            ep_l = info["episode"]["l"].item() if hasattr(info["episode"]["l"], "item") else float(info["episode"]["l"])
                            print(
                                f"global_step={global_step}, episodic_return={ep_r:.4f}, episodic_length={ep_l:.4f}"
                            )
                            writer.add_scalar("charts/episodic_return", ep_r, global_step)
                            writer.add_scalar("charts/episodic_length", ep_l, global_step)
                            mlflow.log_metric("episodic_return", ep_r, step=global_step)
                            mlflow.log_metric("episodic_length", ep_l, step=global_step)

            # Bootstrap value
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # Flatten batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing policy and value network
            batch_size = args.num_steps * args.num_envs
            minibatch_size = batch_size // args.num_minibatches
            b_inds = np.arange(batch_size)
            clipfracs = []

            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # Approximate KL divergence
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        print(f"Early stopping at epoch {epoch} due to reaching max KL divergence.")
                        break

            # Logging
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            
            # Log training metrics to MLflow
            mlflow.log_metrics({
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy": entropy_loss.item(),
                "approx_kl": approx_kl.item(),
                "explained_variance": explained_var,
            }, step=global_step)

            # Checkpointing
            if global_step - initial_global_step >= args.checkpoint_interval:
                save_checkpoint(agent, optimizer, global_step, args.run_id, envs=envs)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[TRAIN] Interrupted by user. Saving checkpoint...")
        try:
            save_checkpoint(agent, optimizer, global_step, args.run_id, envs=envs)
            mlflow.pytorch.log_model(agent, "model_interrupt")
            print(f"[TRAIN] Checkpoint saved at step {global_step}")
        except Exception as e:
            print(f"[TRAIN] Failed to save checkpoint on interrupt: {e}")

    # Final checkpoint and cleanup
    try:
        save_checkpoint(agent, optimizer, global_step, args.run_id, envs=envs)
        mlflow.pytorch.log_model(agent, "model_final")
    except Exception as e:
        print(f"[CHECKPOINT] Failed final save: {e}")

    envs.close()
    writer.close()
    mlflow.end_run()
    print(f"Training completed. Total steps: {global_step}")


if __name__ == "__main__":
    start_time = time.time()
    train(start_time=start_time)
