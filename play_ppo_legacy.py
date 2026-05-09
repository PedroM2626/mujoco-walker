"""
Visualize old PPO Walker Ragdoll checkpoints with frozen observation normalization.

This script preserves the pre-SAC environment shape/reward used by existing PPO
checkpoints, so it is intentionally separate from play.py.
"""

import argparse
import os
import sys
import time

import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco
import mujoco.viewer
import numpy as np
import torch

from ppo_walker import Agent
from utils.checkpoint import find_latest_checkpoint, get_checkpoint_dir


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
}


class LegacyWalkerRagdollEnv(MujocoEnv, gym.utils.EzPickle):
    """Environment definition used by the old PPO checkpoints."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(
        self,
        xml_file=None,
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        progress_reward_weight: float = 50.0,
        distance_penalty_weight: float = 0.0,
        ctrl_cost_weight: float = 0.01,
        upright_reward_weight: float = 10.0,
        height_reward_weight: float = 20.0,
        healthy_reward: float = 5.0,
        target_reached_bonus: float = 100.0,
        target_reach_threshold: float = 0.5,
        target_min_distance: float = 2.0,
        target_max_distance: float = 6.0,
        terminate_when_unhealthy: bool = False,
        healthy_z_range: tuple = (0.8, 2.0),
        reset_noise_scale: float = 5e-2,
        **kwargs,
    ):
        if xml_file is None:
            xml_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "walker_ragdoll.xml"
            )

        self._progress_reward_weight = progress_reward_weight
        self._distance_penalty_weight = distance_penalty_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._upright_reward_weight = upright_reward_weight
        self._height_reward_weight = height_reward_weight
        self._healthy_reward = healthy_reward
        self._target_reached_bonus = target_reached_bonus
        self._target_reach_threshold = target_reach_threshold
        self._target_min_distance = target_min_distance
        self._target_max_distance = target_max_distance
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._target_xy = np.zeros(2, dtype=np.float64)
        self._total_steps_counter = 0

        observation_space = Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        gym.utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            progress_reward_weight,
            distance_penalty_weight,
            ctrl_cost_weight,
            upright_reward_weight,
            height_reward_weight,
            healthy_reward,
            target_reached_bonus,
            target_reach_threshold,
            target_min_distance,
            target_max_distance,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            **kwargs,
        )

        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker"
        )

    @property
    def upright_factor(self):
        return float(self.data.body("torso").xmat[8])

    @property
    def healthy_reward(self):
        is_upright = self.upright_factor > 0.8
        is_standing = self.data.qpos[2] > 1.1
        return float(is_upright and is_standing) * self._healthy_reward

    @property
    def is_healthy(self):
        z = self.data.qpos[2]
        min_z, max_z = self._healthy_z_range
        return min_z < z < max_z

    @property
    def terminated(self):
        return not self.is_healthy if self._terminate_when_unhealthy else False

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def _get_obs(self):
        position = self.data.qpos[2:].copy()
        velocity = self.data.qvel.copy()
        target_rel_xy = self._target_xy - self.data.qpos[:2]
        upright = np.array([self.upright_factor], dtype=np.float64)
        return np.concatenate([position, velocity, target_rel_xy, upright])

    def _sample_target(self, base_xy):
        angle = self.np_random.uniform(0.0, 2.0 * np.pi)
        radius = self.np_random.uniform(
            self._target_min_distance, self._target_max_distance
        )
        offset = np.array([np.cos(angle), np.sin(angle)]) * radius
        return base_xy + offset

    def _set_target(self, target_xy):
        self._target_xy = np.array(target_xy, dtype=np.float64)
        self.model.body_pos[self._target_body_id] = np.array(
            [self._target_xy[0], self._target_xy[1], 0.05],
            dtype=np.float64,
        )
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        xy_before = self.data.qpos[:2].copy()
        dist_before = np.linalg.norm(self._target_xy - xy_before)
        self.do_simulation(action, self.frame_skip)
        xy_after = self.data.qpos[:2].copy()
        dist_after = np.linalg.norm(self._target_xy - xy_after)

        reached_target = bool(dist_after <= self._target_reach_threshold)
        reached_bonus = self._target_reached_bonus if reached_target else 0.0
        curriculum_factor = min(
            1.0, 0.1 + 0.9 * (self._total_steps_counter / 10_000_000)
        )
        progress_reward = (
            self._progress_reward_weight
            * (dist_before - dist_after)
            * curriculum_factor
        )
        self._total_steps_counter += self.frame_skip

        distance_penalty = self._distance_penalty_weight * dist_after
        ctrl_cost = self.control_cost(action)
        upright_reward = self._upright_reward_weight * max(0.0, self.upright_factor)
        height_reward = self._height_reward_weight * min(self.data.qpos[2], 1.3)
        quat = self.data.qpos[3:7]
        tilt_penalty = 5.0 * (1.0 - quat[0] ** 2)

        reward = (
            progress_reward
            + reached_bonus
            + self.healthy_reward
            + upright_reward
            + height_reward
            - ctrl_cost
            - distance_penalty
            - tilt_penalty
        )

        info = {
            "reward_progress": progress_reward,
            "reward_upright": upright_reward,
            "reward_height": height_reward,
            "reward_distance": -distance_penalty,
            "reward_target_bonus": reached_bonus,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": self.healthy_reward,
            "x_position": xy_after[0],
            "y_position": xy_after[1],
            "distance_to_target": dist_after,
            "target_x": self._target_xy[0],
            "target_y": self._target_xy[1],
            "target_reached": reached_target,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, self.terminated, False, info

    def reset_model(self):
        if self.np_random.uniform() < 0.5:
            noise_scale = 0.5
            qpos = self.init_qpos + self.np_random.uniform(
                low=-noise_scale, high=noise_scale, size=self.model.nq
            )
            qpos[2] = 0.3
        else:
            qpos = self.init_qpos + self.np_random.uniform(
                low=-self._reset_noise_scale,
                high=self._reset_noise_scale,
                size=self.model.nq,
            )

        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        self._set_target(self._sample_target(self.data.qpos[:2]))
        return self._get_obs()

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_position": self.data.qpos[2],
            "target_x": self._target_xy[0],
            "target_y": self._target_xy[1],
        }


class FrozenNormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env, obs_rms, epsilon=1e-8):
        super().__init__(env)
        self.obs_rms = obs_rms
        self.epsilon = epsilon

    def observation(self, observation):
        obs = (observation - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        return np.clip(obs, -10, 10)


def parse_args():
    parser = argparse.ArgumentParser(description="Play an old PPO Walker Ragdoll checkpoint")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=0, help="0 = infinite until Ctrl+C")
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "rgb_array"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--fps", type=int, default=120)
    return parser.parse_args()


def make_base_env(render_mode):
    env = LegacyWalkerRagdollEnv(render_mode="rgb_array")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env


def resolve_checkpoint(run_id, checkpoint_step):
    ckpt_dir = get_checkpoint_dir(run_id)
    if checkpoint_step is not None:
        ckpt_path = os.path.join(ckpt_dir, f"ckpt_{checkpoint_step}.pt")
        if os.path.exists(ckpt_path):
            return ckpt_path
        print(f"[ERROR] Checkpoint {ckpt_path} not found. Falling back to latest.")
    return find_latest_checkpoint(ckpt_dir)


def play():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_checkpoint(args.run_id, args.checkpoint_step)
    if ckpt_path is None:
        print(f"[ERROR] No PPO checkpoint found for run-id '{args.run_id}'.")
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=device)
    if "agent_state_dict" not in checkpoint:
        print(f"[ERROR] {ckpt_path} is not a legacy PPO checkpoint.")
        sys.exit(1)
    if "obs_rms" not in checkpoint:
        print(f"[ERROR] {ckpt_path} has no saved obs_rms; cannot replay without normalization drift.")
        sys.exit(1)

    env = FrozenNormalizeObservation(make_base_env(args.render_mode), checkpoint["obs_rms"])
    obs, _ = env.reset(seed=args.seed)

    agent = Agent(env).to(device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()

    print(f"[CHECKPOINT] Loaded {ckpt_path}")
    print("[PLAY] Frozen observation normalization enabled.")
    if args.deterministic:
        print("[PLAY] Deterministic mode enabled.")

    viewer = None
    if args.render_mode == "human":
        viewer = mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data)

    total_reward = 0.0
    episode_count = 0
    step_count = 0
    infinite = args.num_episodes <= 0

    try:
        while True if infinite else episode_count < args.num_episodes:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_mean = agent.actor_mean(obs_tensor)
                if args.deterministic:
                    action = action_mean
                else:
                    action_std = torch.exp(agent.actor_logstd.expand_as(action_mean))
                    action = torch.distributions.Normal(action_mean, action_std).sample()
                action = action.cpu().numpy()[0]

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
