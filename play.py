"""
Visualize a trained PPO Walker Ragdoll agent in real time.

Loads the latest (or a specific) checkpoint and runs the agent with rendering.
"""

import os
import sys
import time
import argparse

import gymnasium as gym
import numpy as np
import torch

import mujoco.viewer

import envs.walker_ragdoll_env
from ppo_walker import Agent, make_env
from utils.checkpoint import load_checkpoint, find_latest_checkpoint, get_checkpoint_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Play a trained Walker Ragdoll agent")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID of the trained agent to load",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Specific checkpoint step to load (loads latest if omitted)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="Number of episodes to run (0 = infinite until Ctrl+C)",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
        help="Render mode: 'human' opens a window, 'rgb_array' returns frames",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="Number of random steps to warm up observation normalization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Use deterministic actions (mean only, no sampling)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=120,
        help="Playback FPS when rendering to a window",
    )
    return parser.parse_args()


def make_play_env(env_id):
    """Create a single evaluation environment with the same wrappers used during training."""
    # Always use rgb_array internally to avoid gymnasium / mujoco 3.x "human" renderer bugs.
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    # NormalizeReward is training-only; skip it for evaluation
    return env


def warm_up_env(env, steps=200):
    """Run random actions to populate observation normalization statistics."""
    obs, _ = env.reset()
    for _ in range(steps):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    return obs


def play():
    args = parse_args()

    use_window = args.render_mode == "human"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_id = "WalkerRagdoll-v0"

    # Build environment (always rgb_array internally)
    env = make_play_env(env_id)

    # Seed for reproducibility
    obs, _ = env.reset(seed=args.seed)

    # Warm-up normalization statistics
    if args.warmup_steps > 0:
        print(
            f"[WARMUP] Running {args.warmup_steps} random steps to warm up "
            "observation normalization..."
        )
        obs = warm_up_env(env, steps=args.warmup_steps)
        print("[WARMUP] Done.")

    # Build agent and load checkpoint
    agent = Agent(env).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

    ckpt_dir = get_checkpoint_dir(args.run_id)
    if args.checkpoint_step is not None:
        ckpt_path = os.path.join(ckpt_dir, f"ckpt_{args.checkpoint_step}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[ERROR] Checkpoint {ckpt_path} not found. Falling back to latest.")
            ckpt_path = find_latest_checkpoint(ckpt_dir)
    else:
        ckpt_path = find_latest_checkpoint(ckpt_dir)

    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[ERROR] No checkpoint found for run-id '{args.run_id}' in {ckpt_dir}")
        sys.exit(1)

    # Load the checkpoint and the environment statistics (obs_rms)
    _, _ = load_checkpoint(agent, None, args.run_id, envs=env, global_step=args.checkpoint_step)
    agent.eval()

    total_reward = 0.0
    episode_count = 0
    step_count = 0

    infinite = args.num_episodes <= 0
    if infinite:
        print(
            f"[PLAY] Running infinitely with render_mode='{args.render_mode}' "
            "(press Ctrl+C to stop)"
        )
    else:
        print(
            f"[PLAY] Running {args.num_episodes} episode(s) with "
            f"render_mode='{args.render_mode}'"
        )
    if args.deterministic:
        print("[PLAY] Deterministic mode enabled (using action mean only)")

    viewer = None
    if use_window:
        viewer = mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data)

    try:
        while True if infinite else episode_count < args.num_episodes:
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                action_mean = agent.actor_mean(obs_tensor)
                if args.deterministic:
                    action = action_mean
                else:
                    action_logstd = agent.actor_logstd.expand_as(action_mean)
                    action_std = torch.exp(action_logstd)
                    dist = torch.distributions.Normal(action_mean, action_std)
                    action = dist.sample()
                action = action.cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if not use_window:
                env.render()
            elif viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    print("[PLAY] Viewer closed — exiting.")
                    break
                time.sleep(1.0 / args.fps)

            if terminated or truncated:
                episode_count += 1
                ep_return = info.get("episode", {}).get("r", total_reward)
                ep_length = info.get("episode", {}).get("l", step_count)
                ep_return = (
                    ep_return.item()
                    if hasattr(ep_return, "item")
                    else float(ep_return)
                )
                ep_length = (
                    ep_length.item()
                    if hasattr(ep_length, "item")
                    else int(ep_length)
                )
                print(
                    f"[EPISODE {episode_count}/{args.num_episodes}] "
                    f"Return={ep_return:.2f}  Length={ep_length}"
                )
                total_reward = 0.0
                step_count = 0
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\n[PLAY] Interrupted by user.")
    finally:
        env.close()
        if 'viewer' in locals() and viewer is not None:
            viewer.close()
        print("[PLAY] Finished.")


if __name__ == "__main__":
    play()
