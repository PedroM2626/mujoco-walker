"""
Visualize a trained SAC or PPO Walker Ragdoll agent in real time.
"""

import argparse
import os
import sys
import time

import gymnasium as gym
import mujoco.viewer
import numpy as np
import torch

import envs.walker_ragdoll_env
from train_walker import (
    SACAgent, PPOAgent,
    latest_sac_checkpoint, latest_ppo_checkpoint, latest_checkpoint_any,
    load_torch_checkpoint,
    RecoverySupervisor,
)
from utils.checkpoint import get_checkpoint_dir


class FrozenNormalizeObservation(gym.ObservationWrapper):
    """Normalize observations with saved training statistics without updating them."""

    def __init__(self, env, obs_rms, epsilon=1e-8):
        super().__init__(env)
        self.obs_rms = obs_rms
        self.epsilon = epsilon

    def observation(self, observation):
        obs = (observation - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        return np.clip(obs, -10, 10)


def parse_args():
    parser = argparse.ArgumentParser(description="Play a trained SAC or PPO Walker Ragdoll agent")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=0, help="0 = infinite until Ctrl+C")
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "rgb_array"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true", default=False, help="Sample from the policy instead of using the mean action")
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument(
        "--reset-mode",
        type=str,
        default="mixed",
        choices=["fixed", "mixed", "fallen", "upright"],
        help="Initial-state distribution used while visualizing.",
    )
    parser.add_argument(
        "--task-phase",
        type=str,
        default=None,
        choices=["recovery", "balance", "walk", "target"],
        help="Reward phase used while visualizing. Defaults to the checkpoint phase when available.",
    )
    parser.add_argument("--use-supervisor", action="store_true", default=False, help="Use the Recovery Supervisor when fallen and show visualization")
    return parser.parse_args()


def make_base_env(env_id, reset_mode="mixed", task_phase="recovery"):
    env = gym.make(
        env_id,
        render_mode="rgb_array",
        reset_mode=reset_mode,
        task_phase=task_phase,
    )
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env


def resolve_checkpoint(run_id, checkpoint_step):
    ckpt_dir = get_checkpoint_dir(run_id)
    if checkpoint_step is not None:
        # Try SAC first, then PPO
        for prefix in ("sac_ckpt_", "ppo_ckpt_"):
            ckpt_path = os.path.join(ckpt_dir, f"{prefix}{checkpoint_step}.pt")
            if os.path.exists(ckpt_path):
                return ckpt_path
        print(f"[ERROR] Checkpoint for step {checkpoint_step} not found. Falling back to latest.")
    return latest_checkpoint_any(ckpt_dir)


def play():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_id = "WalkerRagdoll-v0"

    ckpt_path = resolve_checkpoint(args.run_id, args.checkpoint_step)
    if ckpt_path is None:
        print(f"[ERROR] No checkpoint found for run-id '{args.run_id}'.")
        sys.exit(1)

    checkpoint = load_torch_checkpoint(ckpt_path, device)
    algo = checkpoint.get("algo", "unknown")
    if algo not in {"sac", "sac_actor", "ppo"}:
        print(f"[ERROR] {ckpt_path} has unknown algorithm type '{algo}'.")
        sys.exit(1)
    if checkpoint.get("obs_rms") is None:
        print(f"[ERROR] {ckpt_path} does not contain observation normalization statistics.")
        sys.exit(1)

    task_phase = args.task_phase or checkpoint.get("task_phase") or "recovery"
    env = FrozenNormalizeObservation(
        make_base_env(env_id, reset_mode=args.reset_mode, task_phase=task_phase),
        checkpoint["obs_rms"],
    )
    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))

    if algo in {"sac", "sac_actor"}:
        agent = SACAgent(obs_dim, env.action_space).to(device)
        agent.load_state_dict(checkpoint["actor_state_dict"])
        agent.eval()
        algo_label = "SAC"
    else:
        action_dim = int(np.prod(env.action_space.shape))
        agent = PPOAgent(obs_dim, action_dim).to(device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        agent.eval()
        algo_label = "PPO"

    print(f"[CHECKPOINT] Loaded {ckpt_path}")
    print(f"[PLAY] algo={algo_label} task_phase={task_phase}")
    print(f"[PLAY] {'Stochastic' if args.stochastic else 'Deterministic'} policy")

    viewer = None
    if args.render_mode == "human":
        viewer = mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data)

    supervisor = None
    if args.use_supervisor:
        print("[PLAY] Initializing Recovery Supervisor...")
        supervisor = RecoverySupervisor(device)

    total_reward = 0.0
    episode_count = 0
    step_count = 0
    infinite = args.num_episodes <= 0

    try:
        while True if infinite else episode_count < args.num_episodes:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                if algo_label == "SAC":
                    action, _, _ = agent.get_action(obs_tensor, deterministic=not args.stochastic)
                    action = action.cpu().numpy()[0]
                else:
                    if args.stochastic:
                        action, _, _, _ = agent.get_action_and_value(obs_tensor)
                    else:
                        action = agent.get_deterministic_action(obs_tensor)
                    action = action.cpu().numpy()[0]

            if supervisor is not None:
                actions_array = np.array([action])
                actions_array, is_rec = supervisor.get_actions([env], actions_array)
                action = actions_array[0]
                
                # Visual feedback: Change walker color when recovery network is active
                if viewer is not None:
                    model = env.unwrapped.model
                    for geom_id in range(model.ngeom):
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                        if name not in ["floor", "target_marker_geom"]:
                            if is_rec[0]:
                                model.geom_rgba[geom_id] = [1.0, 0.0, 0.0, 1.0] # Red
                            else:
                                model.geom_rgba[geom_id] = [0.8, 0.6, 0.4, 1.0] # Original color approx

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if args.render_mode == "rgb_array":
                env.render()
            elif viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    print("[PLAY] Viewer closed, exiting.")
                    break
                time.sleep(1.0 / args.fps)

            if terminated or truncated:
                episode_count += 1
                ep_return = info.get("episode", {}).get("r", total_reward)
                ep_length = info.get("episode", {}).get("l", step_count)
                ep_return = ep_return.item() if hasattr(ep_return, "item") else float(ep_return)
                ep_length = ep_length.item() if hasattr(ep_length, "item") else int(ep_length)
                print(f"[EPISODE {episode_count}/{args.num_episodes}] Return={ep_return:.2f}  Length={ep_length}")
                total_reward = 0.0
                step_count = 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\n[PLAY] Interrupted by user.")
    finally:
        env.close()
        if viewer is not None:
            viewer.close()
        print("[PLAY] Finished.")


if __name__ == "__main__":
    play()
