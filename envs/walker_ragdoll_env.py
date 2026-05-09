import os

import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco
import numpy as np


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
}


class WalkerRagdollEnv(MujocoEnv, gym.utils.EzPickle):
    """
    Full humanoid stand-up environment.

    Body layout (17 actuators):
      abdomen_y, abdomen_z, abdomen_x            (3)
      right_hip_x/z/y, right_knee               (4)
      left_hip_x/z/y,  left_knee                (4)
      right_shoulder1/2, right_elbow            (3)
      left_shoulder1/2,  left_elbow             (3)

    Root joint: free (nq=24, nv=23)
    Observation: qpos[2:] (22) + qvel (23) + torso upright factor (1) = 46
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
        uph_cost_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        impact_cost_weight: float = 0.5e-6,
        upright_reward_weight: float = 2.0,
        standing_reward: float = 5.0,
        terminate_when_unhealthy: bool = False,
        healthy_z_range: tuple = (0.8, 2.0),
        reset_noise_scale: float = 1e-2,
        **kwargs,
    ):
        if xml_file is None:
            xml_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "walker_ragdoll.xml"
            )

        self._uph_cost_weight = uph_cost_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._impact_cost_weight = impact_cost_weight
        self._upright_reward_weight = upright_reward_weight
        self._standing_reward = standing_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(46,), dtype=np.float64
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
            uph_cost_weight,
            ctrl_cost_weight,
            impact_cost_weight,
            upright_reward_weight,
            standing_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            **kwargs,
        )

        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker"
        )
        if self._target_body_id >= 0:
            self.model.body_pos[self._target_body_id] = np.array(
                [0.0, 0.0, -10.0], dtype=np.float64
            )

    @property
    def upright_factor(self):
        """Z component of the torso local up-vector in world coordinates."""
        return float(self.data.body("torso").xmat[8])

    @property
    def healthy_reward(self):
        is_upright = self.upright_factor > 0.8
        is_standing = self.data.qpos[2] > 1.1
        return float(is_upright and is_standing) * self._standing_reward

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

    @property
    def impact_cost(self):
        return min(self._impact_cost_weight * np.sum(np.square(self.data.cfrc_ext)), 10.0)

    def _get_obs(self):
        position = self.data.qpos[2:].copy()
        velocity = self.data.qvel.copy()
        upright = np.array([self.upright_factor], dtype=np.float64)
        return np.concatenate([position, velocity, upright])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        z_after = float(self.data.qpos[2])
        raw_uph_cost = self._uph_cost_weight * z_after / self.dt
        uph_cost = raw_uph_cost * max(0.0, self.upright_factor)
        ctrl_cost = self.control_cost(action)
        impact_cost = self.impact_cost
        upright_reward = self._upright_reward_weight * max(0.0, self.upright_factor)
        standing_reward = self.healthy_reward

        reward = uph_cost + 1.0 + upright_reward + standing_reward - ctrl_cost - impact_cost
        terminated = self.terminated
        observation = self._get_obs()

        info = {
            "reward_linup": uph_cost,
            "reward_raw_linup": raw_uph_cost,
            "reward_upright": upright_reward,
            "reward_standing": standing_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_impact": -impact_cost,
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_position": z_after,
            "upright": self.upright_factor,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq,
        )
        qpos[2] = 0.35
        qpos[3:7] = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])
        qvel = self.init_qvel + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nv,
        )

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_position": self.data.qpos[2],
            "upright": self.upright_factor,
        }


gym.register(
    id="WalkerRagdoll-v0",
    entry_point="envs.walker_ragdoll_env:WalkerRagdollEnv",
    max_episode_steps=1000,
)
