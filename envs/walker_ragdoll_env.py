import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
}


class WalkerRagdollEnv(MujocoEnv, gym.utils.EzPickle):
    """
    Full humanoid ragdoll environment.

    Body layout (17 actuators):
      abdomen_y, abdomen_z, abdomen_x            (3)
      right_hip_x/z/y, right_knee               (4)
      left_hip_x/z/y,  left_knee                (4)
      right_shoulder1/2, right_elbow            (3)
      left_shoulder1/2,  left_elbow             (3)

    Root joint: free (nq=24, nv=23)
    Observation: qpos[2:] (22) + qvel (23) + target_rel_xy (2) = 47
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(
        self,
        xml_file=None,
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        progress_reward_weight: float = 8.0,
        distance_penalty_weight: float = 0.0,
        ctrl_cost_weight: float = 0.1,
        healthy_reward: float = 1.0,
        target_reached_bonus: float = 10.0,
        target_reach_threshold: float = 0.5,
        target_min_distance: float = 2.0,
        target_max_distance: float = 6.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple = (1.0, 2.0),
        reset_noise_scale: float = 1e-2,
        **kwargs,
    ):
        if xml_file is None:
            xml_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "walker_ragdoll.xml"
            )

        self._progress_reward_weight = progress_reward_weight
        self._distance_penalty_weight = distance_penalty_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._target_reached_bonus = target_reached_bonus
        self._target_reach_threshold = target_reach_threshold
        self._target_min_distance = target_min_distance
        self._target_max_distance = target_max_distance
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._target_xy = np.zeros(2, dtype=np.float64)

        # nq=24 (free joint: 3 pos + 4 quat + 17 joint angles)
        # obs = qpos[2:] + qvel + target_rel_xy = 22 + 23 + 2 = 47
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(47,), dtype=np.float64
        )

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
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    @property
    def is_healthy(self):
        # z position of root (qpos[2] for free joint)
        z = self.data.qpos[2]
        min_z, max_z = self._healthy_z_range
        return min_z < z < max_z

    @property
    def terminated(self):
        return not self.is_healthy if self._terminate_when_unhealthy else False

    def _get_obs(self):
        # Free joint: qpos = [x, y, z, qw, qx, qy, qz, joint_angles...]
        # Skip x (index 0) and y (index 1) — keep z + quaternion + joints
        position = self.data.qpos[2:].copy()   # 22 elements
        velocity = self.data.qvel.copy()       # 23 elements
        target_rel_xy = self._target_xy - self.data.qpos[:2]
        return np.concatenate([position, velocity, target_rel_xy])

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
        # Reward progress to the target and penalize remaining distance.
        xy_before = self.data.qpos[:2].copy()
        dist_before = np.linalg.norm(self._target_xy - xy_before)
        self.do_simulation(action, self.frame_skip)
        xy_after = self.data.qpos[:2].copy()
        dist_after = np.linalg.norm(self._target_xy - xy_after)

        progress_reward = self._progress_reward_weight * (dist_before - dist_after)
        distance_penalty = self._distance_penalty_weight * dist_after
        reached_target = bool(dist_after <= self._target_reach_threshold)
        reached_bonus = self._target_reached_bonus if reached_target else 0.0

        ctrl_cost = self.control_cost(action)
        healthy_reward = self.healthy_reward

        reward = progress_reward + reached_bonus + healthy_reward - ctrl_cost - distance_penalty
        terminated = self.terminated
        observation = self._get_obs()

        info = {
            "reward_progress": progress_reward,
            "reward_distance": -distance_penalty,
            "reward_target_bonus": reached_bonus,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_after[0],
            "y_position": xy_after[1],
            "distance_to_target": dist_after,
            "target_x": self._target_xy[0],
            "target_y": self._target_xy[1],
            "target_reached": reached_target,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
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


gym.register(
    id="WalkerRagdoll-v0",
    entry_point="envs.walker_ragdoll_env:WalkerRagdollEnv",
    max_episode_steps=1000,
)
