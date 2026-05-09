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
                os.path.dirname(os.path.dirname(__file__)), "walker_ragdoll.xml"
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
        self._total_steps_counter = 0  # To track progress for curriculum

        # nq=24 (free joint: 3 pos + 4 quat + 17 joint angles)
        # obs = qpos[2:] (22) + qvel (23) + target_rel_xy (2) + upright (1) = 48
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(48,), dtype=np.float64
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
    def healthy_reward(self):
        # Survival bonus only if upright and at a decent height
        is_upright = self.upright_factor > 0.8
        is_standing = self.data.qpos[2] > 1.1
        return (
            float(is_upright and is_standing)
            * self._healthy_reward
        )

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    @property
    def upright_factor(self):
        """Returns the Z-component of the torso's up-vector (local Z in world coords).
        1.0 means perfectly upright, -1.0 means upside down.
        """
        # xmat is a 3x3 rotation matrix for the body
        # torso is at body index 1 (usually)
        return self.data.body("torso").xmat[8]

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
        # position: skip x,y -> keep z, quat, joints (22 elements)
        position = self.data.qpos[2:].copy()   
        # velocity: (23 elements)
        velocity = self.data.qvel.copy()       
        # target: (2 elements)
        target_rel_xy = self._target_xy - self.data.qpos[:2]
        # upright: (1 element)
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
        # Reward progress to the target and penalize remaining distance.
        xy_before = self.data.qpos[:2].copy()
        dist_before = np.linalg.norm(self._target_xy - xy_before)
        self.do_simulation(action, self.frame_skip)
        xy_after = self.data.qpos[:2].copy()
        dist_after = np.linalg.norm(self._target_xy - xy_after)

        reached_target = bool(dist_after <= self._target_reach_threshold)
        reached_bonus = self._target_reached_bonus if reached_target else 0.0

        # Curriculum: Slowly increase progress reward importance
        # Start at 10% importance, reach 100% at 10M steps
        curriculum_factor = min(1.0, 0.1 + 0.9 * (self._total_steps_counter / 10_000_000))
        progress_reward = self._progress_reward_weight * (dist_before - dist_after) * curriculum_factor
        
        self._total_steps_counter += self.frame_skip
        distance_penalty = self._distance_penalty_weight * dist_after

        ctrl_cost = self.control_cost(action)
        healthy_reward = self.healthy_reward
        
        # Uprightness and Height rewards
        upright_reward = self._upright_reward_weight * max(0, self.upright_factor)
        height_reward = self._height_reward_weight * min(self.data.qpos[2], 1.3)

        # Deviation penalty: penalize if torso is tilted too much sideways or forward
        # Using xquat: [qw, qx, qy, qz]. qw should be near 1 for upright.
        quat = self.data.qpos[3:7]
        tilt_penalty = 5.0 * (1.0 - quat[0]**2) 

        reward = progress_reward + reached_bonus + healthy_reward + upright_reward + height_reward - ctrl_cost - distance_penalty - tilt_penalty
        terminated = self.terminated
        observation = self._get_obs()

        info = {
            "reward_progress": progress_reward,
            "reward_upright": upright_reward,
            "reward_height": height_reward,
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
        # 50% chance of starting in a "fallen" or "random" pose to learn recovery
        if self.np_random.uniform() < 0.5:
            # Start fallen: large joint noise, low Z
            noise_scale = 0.5 
            qpos = self.init_qpos + self.np_random.uniform(low=-noise_scale, high=noise_scale, size=self.model.nq)
            qpos[2] = 0.3  # Drop it to the floor
        else:
            # Start near standing
            noise_scale = self._reset_noise_scale
            qpos = self.init_qpos + self.np_random.uniform(low=-noise_scale, high=noise_scale, size=self.model.nq)

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


gym.register(
    id="WalkerRagdoll-v0",
    entry_point="envs.walker_ragdoll_env:WalkerRagdollEnv",
    max_episode_steps=1000,
)
