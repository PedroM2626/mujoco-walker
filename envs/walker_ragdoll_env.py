import math
import os
from collections import deque

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

ENV_VERSION = "standup_balance_walk_curriculum_v8"
FOOT_BODIES = {"left_foot", "right_foot"}

RESET_MODES = {"fixed", "mixed", "fallen", "upright"}
TASK_PHASES = {"recovery", "balance", "walk", "target"}


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
    Target phase observation adds relative target dx, dy, and distance = 49
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
        stand_height_reward_weight: float = 100.0,
        recovery_reward_weight: float = 5.0,
        ctrl_cost_weight: float = 0.1,
        impact_cost_weight: float = 0.5e-6,
        upright_reward_weight: float = 2.0,
        standing_reward: float = 50.0,
        stability_reward_weight: float = 30.0,
        stillness_penalty_weight: float = 2.0,
        lateral_drift_penalty_weight: float = 2.0,
        walk_reward_weight: float = 60.0,
        forward_velocity_reward_weight: float = 30.0,
        target_forward_velocity: float = 0.8,
        target_progress_reward_weight: float = 200.0,
        target_direction_reward_weight: float = 80.0,
        target_success_reward: float = 200.0,
        target_radius: float = 0.45,
        target_distance_range: tuple = (2.0, 5.0),
        target_curriculum_streak: int = 10,
        bad_support_penalty_weight: float = 15.0,
        low_upright_penalty_weight: float = 20.0,
        terminate_when_unhealthy: bool = False,
        healthy_z_range: tuple = (1.0, 2.0),
        reset_noise_scale: float = 1e-2,
        reset_mode: str = "mixed",
        fixed_reset_probability: float = 0.25,
        upright_reset_probability: float = 0.15,
        fallen_velocity_scale: float = 0.35,
        task_phase: str = "recovery",
        **kwargs,
    ):
        if xml_file is None:
            xml_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "walker_ragdoll.xml"
            )

        self._stand_height_reward_weight = stand_height_reward_weight
        self._recovery_reward_weight = recovery_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._impact_cost_weight = impact_cost_weight
        self._upright_reward_weight = upright_reward_weight
        self._standing_reward = standing_reward
        self._stability_reward_weight = stability_reward_weight
        self._stillness_penalty_weight = stillness_penalty_weight
        self._lateral_drift_penalty_weight = lateral_drift_penalty_weight
        self._walk_reward_weight = walk_reward_weight
        self._forward_velocity_reward_weight = forward_velocity_reward_weight
        self._target_forward_velocity = target_forward_velocity
        self._target_progress_reward_weight = target_progress_reward_weight
        self._target_direction_reward_weight = target_direction_reward_weight
        self._target_success_reward = target_success_reward
        self._target_radius = target_radius
        self._target_distance_range = target_distance_range
        self._curriculum_level = 0  # Now used to track total targets reached
        self._bad_support_penalty_weight = bad_support_penalty_weight
        self._low_upright_penalty_weight = low_upright_penalty_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        if reset_mode not in RESET_MODES:
            raise ValueError(
                f"reset_mode must be one of {sorted(RESET_MODES)}, got {reset_mode!r}"
            )
        if not 0.0 <= fixed_reset_probability <= 1.0:
            raise ValueError("fixed_reset_probability must be in [0, 1]")
        if not 0.0 <= upright_reset_probability <= 1.0:
            raise ValueError("upright_reset_probability must be in [0, 1]")
        if fixed_reset_probability + upright_reset_probability > 1.0:
            raise ValueError(
                "fixed_reset_probability + upright_reset_probability must be <= 1"
            )
        self._reset_mode = reset_mode
        self._fixed_reset_probability = fixed_reset_probability
        self._upright_reset_probability = upright_reset_probability
        self._fallen_velocity_scale = fallen_velocity_scale
        if task_phase not in TASK_PHASES:
            raise ValueError(
                f"task_phase must be one of {sorted(TASK_PHASES)}, got {task_phase!r}"
            )
        self._task_phase = task_phase
        self._target_xy = np.array([3.0, 0.0], dtype=np.float64)
        self._target_fixed_until_curriculum = True

        observation_size = 49 if task_phase == "target" else 46
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float64
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
            stand_height_reward_weight,
            recovery_reward_weight,
            ctrl_cost_weight,
            impact_cost_weight,
            upright_reward_weight,
            standing_reward,
            stability_reward_weight,
            stillness_penalty_weight,
            lateral_drift_penalty_weight,
            walk_reward_weight,
            forward_velocity_reward_weight,
            target_forward_velocity,
            target_progress_reward_weight,
            target_direction_reward_weight,
            target_success_reward,
            target_radius,
            target_distance_range,
            target_curriculum_streak,
            bad_support_penalty_weight,
            low_upright_penalty_weight,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            reset_mode,
            fixed_reset_probability,
            upright_reset_probability,
            fallen_velocity_scale,
            task_phase,
            **kwargs,
        )

        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker"
        )
        if self._target_body_id >= 0:
            self.model.body_pos[self._target_body_id] = np.array(
                [0.0, 0.0, -10.0], dtype=np.float64
            )

        # --- hot-loop caches ---------------------------------------------------
        # Every name lookup below used to happen on each env step; resolving them
        # once turns string hashing into integer indexing in the reward path.
        body = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self._torso_body_id = body("torso")
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        foot_body_ids = {body(name) for name in FOOT_BODIES}
        geom_bodyid = self.model.geom_bodyid
        self._geom_is_foot = [geom_bodyid[g] in foot_body_ids for g in range(self.model.ngeom)]
        self._geom_is_floor = [g == self._floor_geom_id for g in range(self.model.ngeom)]
        self._compute_impact = self._impact_cost_weight != 0.0
        self._ever_healthy = False
        # Scratch buffers so _get_obs allocates once per step instead of four times.
        self._scalar_buf = np.empty(1, dtype=np.float64)
        self._target_obs_buf = np.empty(3, dtype=np.float64)

    @property
    def upright_factor(self):
        """Z component of the torso local up-vector in world coordinates."""
        return float(self.data.xmat[self._torso_body_id, 8])

    def _healthy_reward(self, upright, contacts):
        is_upright = upright > 0.8
        is_standing = self.data.qpos[2] > 1.1
        bad_floor_contacts, foot_floor_contacts = contacts
        return (
            float(is_upright and is_standing and bad_floor_contacts == 0 and foot_floor_contacts > 0)
            * self._standing_reward
        )

    @property
    def healthy_reward(self):
        return self._healthy_reward(self.upright_factor, self.floor_contact_counts)

    @property
    def is_healthy(self):
        z = self.data.qpos[2]
        min_z, max_z = self._healthy_z_range
        return min_z < z < max_z

    def _latch_health(self):
        """Record that this episode has reached a standing pose at least once.

        Called from step() rather than as a side effect of is_healthy: terminated()
        short-circuits on the latch, so a latch that only is_healthy could set would
        never be set.
        """
        if self.is_healthy:
            self._ever_healthy = True
        return self._ever_healthy

    @property
    def terminated(self):
        if not self._terminate_when_unhealthy:
            return False
        # Starting on the floor is the task, not a failure. `mixed`/`fallen` resets put
        # torso z at 0.25-0.65 while healthy_z_range starts at 1.0, so without this
        # latch every one of those episodes ended after a single step and collected the
        # -500 fall penalty. Only a robot that stood up and then lost it has fallen.
        return self._ever_healthy and not self.is_healthy

    def control_cost(self, action):
        return self._ctrl_cost_weight * float(np.dot(action, action))

    @property
    def impact_cost(self):
        if not self._compute_impact:
            return 0.0
        cfrc = self.data.cfrc_ext.ravel()
        return min(self._impact_cost_weight * float(np.dot(cfrc, cfrc)), 10.0)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        # Gymnasium's base class always runs mj_rnePostConstraint because cfrc_ext
        # (used by the impact cost) is not produced by mj_step. With the weight
        # zeroed there is no consumer, so the extra pass is skipped.
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=n_frames)
        if self._compute_impact:
            mujoco.mj_rnePostConstraint(self.model, self.data)

    def _count_floor_contacts(self):
        is_foot = self._geom_is_foot
        is_floor = self._geom_is_floor
        bad_floor_contacts = 0
        foot_floor_contacts = 0
        contact = self.data.contact
        for i in range(self.data.ncon):
            geom1 = contact[i].geom1
            geom2 = contact[i].geom2
            if is_floor[geom1]:
                if is_foot[geom2]:
                    foot_floor_contacts += 1
                else:
                    bad_floor_contacts += 1
            elif is_floor[geom2]:
                if is_foot[geom1]:
                    foot_floor_contacts += 1
                else:
                    bad_floor_contacts += 1
        return bad_floor_contacts, foot_floor_contacts

    @property
    def floor_contact_counts(self):
        return self._count_floor_contacts()

    def _get_obs(self, upright=None):
        if upright is None:
            upright = self.upright_factor
        self._scalar_buf[0] = upright
        position = self.data.qpos[2:]
        velocity = self.data.qvel
        if self._task_phase != "target":
            return np.concatenate((position, velocity, self._scalar_buf))
        rel_xy = self._target_xy - self.data.qpos[:2]
        distance = math.hypot(float(rel_xy[0]), float(rel_xy[1]))
        # Clip to maximum training distance (5.0m) to prevent OOD observations
        if distance > 5.0:
            rel_xy = rel_xy * (5.0 / distance)
            distance = 5.0
        target_buf = self._target_obs_buf
        target_buf[0] = rel_xy[0]
        target_buf[1] = rel_xy[1]
        target_buf[2] = distance
        return np.concatenate((position, velocity, self._scalar_buf, target_buf))

    def _get_target_obs(self):
        rel_xy = self._target_xy - self.data.qpos[:2]
        distance = np.linalg.norm(rel_xy)

        # Clip to maximum training distance (5.0m) to prevent OOD observations
        if distance > 5.0:
            rel_xy = rel_xy * (5.0 / distance)
            distance = 5.0

        return np.array([rel_xy[0], rel_xy[1], distance], dtype=np.float64)

    def _distance_to_target(self, xy=None):
        if xy is None:
            return math.hypot(float(self._target_xy[0] - self.data.qpos[0]), float(self._target_xy[1] - self.data.qpos[1]))
        return math.hypot(float(self._target_xy[0] - xy[0]), float(self._target_xy[1] - xy[1]))


    def _set_target_marker(self):
        if self._target_body_id >= 0:
            if self._task_phase != "target":
                self.model.body_pos[self._target_body_id] = np.array(
                    [0.0, 0.0, -10.0], dtype=np.float64
                )
                return
            self.model.body_pos[self._target_body_id] = np.array(
                [self._target_xy[0], self._target_xy[1], 0.05], dtype=np.float64
            )

    def _impact(self):
        if not self._compute_impact:
            return 0.0
        cfrc = self.data.cfrc_ext.ravel()
        return min(self._impact_cost_weight * float(np.dot(cfrc, cfrc)), 10.0)

    @property
    def impact_cost(self):
        return self._impact()

    def step(self, action):
        qpos = self.data.qpos
        x_before = float(qpos[0])
        y_before = float(qpos[1])
        is_target_phase = self._task_phase == "target"
        target_distance_before = self._distance_to_target() if is_target_phase else 0.0
        self.do_simulation(action, self.frame_skip)

        qvel = self.data.qvel
        x_after = float(qpos[0])
        y_after = float(qpos[1])
        z_after = float(qpos[2])
        upright_raw = self.upright_factor
        upright = max(0.0, upright_raw)
        stand_height = min(max((z_after - 0.65) / 0.60, 0.0), 1.0)
        standing_gate = stand_height * upright * upright
        dt = self.dt
        x_velocity = (x_after - x_before) / dt
        y_velocity = (y_after - y_before) / dt
        root_angular_speed = math.sqrt(qvel[3] * qvel[3] + qvel[4] * qvel[4] + qvel[5] * qvel[5])
        root_linear_speed = math.sqrt(qvel[0] * qvel[0] + qvel[1] * qvel[1] + qvel[2] * qvel[2])
        stand_height_reward = self._stand_height_reward_weight * stand_height * upright * upright
        recovery_reward = self._recovery_reward_weight * z_after * upright
        ctrl_cost = self.control_cost(action)
        impact_cost = self._impact()
        upright_reward = self._upright_reward_weight * upright
        contacts = self._count_floor_contacts()
        bad_floor_contacts, foot_floor_contacts = contacts
        standing_reward = self._healthy_reward(upright, contacts)
        support_penalty_scale = min(max((upright - 0.4) / 0.6, 0.0), 1.0)
        bad_support_penalty = (
            self._bad_support_penalty_weight * bad_floor_contacts * support_penalty_scale
        )
        low_upright_penalty = self._low_upright_penalty_weight * max(
            0.0, 0.85 - z_after
        )
        stability_reward = 0.0
        stillness_penalty = 0.0
        lateral_drift_penalty = 0.0
        walk_reward = 0.0
        forward_velocity_reward = 0.0
        target_progress_reward = 0.0
        target_direction_reward = 0.0
        target_success_reward = 0.0
        target_distance_after = 0.0

        task_phase = self._task_phase
        if task_phase in {"balance", "walk", "target"}:
            stable_pose = standing_gate * math.exp(-0.25 * root_angular_speed)
            stable_support = float(foot_floor_contacts > 0 and bad_floor_contacts == 0)
            stability_reward = (
                self._stability_reward_weight * stable_pose * stable_support
            )

        if task_phase == "balance":
            stillness_penalty = (
                self._stillness_penalty_weight
                * standing_gate
                * min(root_linear_speed + root_angular_speed, 10.0)
            )
            lateral_drift_penalty = (
                self._lateral_drift_penalty_weight
                * standing_gate
                * (abs(y_after) + abs(y_velocity))
            )

        if task_phase == "walk":
            lateral_drift_penalty = (
                self._lateral_drift_penalty_weight
                * standing_gate
                * (abs(y_after) + abs(y_velocity))
            )
            velocity_error = x_velocity - self._target_forward_velocity
            walk_reward = (
                self._walk_reward_weight
                * standing_gate
                * math.exp(-(velocity_error * velocity_error) / 0.5)
            )
            forward_velocity_reward = (
                self._forward_velocity_reward_weight
                * standing_gate
                * min(max(x_velocity, 0.0), self._target_forward_velocity)
            )
        elif is_target_phase:
            dx = self._target_xy[0] - x_after
            dy = self._target_xy[1] - y_after
            target_distance_after = math.hypot(dx, dy)
            inv_distance = 1.0 / max(target_distance_after, 1e-6)
            direction_x = dx * inv_distance
            direction_y = dy * inv_distance
            velocity_toward_target = x_velocity * direction_x + y_velocity * direction_y
            progress = target_distance_before - target_distance_after
            target_progress_reward = (
                self._target_progress_reward_weight
                * standing_gate
                * min(max(progress, 0.0), 0.15)
            )
            target_direction_reward = (
                self._target_direction_reward_weight
                * standing_gate
                * min(max(velocity_toward_target, 0.0), self._target_forward_velocity)
            )
            # Perpendicular velocity (lateral velocity relative to target direction)
            perpendicular_x = x_velocity - velocity_toward_target * direction_x
            perpendicular_y = y_velocity - velocity_toward_target * direction_y
            perpendicular_speed = math.hypot(perpendicular_x, perpendicular_y)
            lateral_drift_penalty = (
                self._lateral_drift_penalty_weight * standing_gate * perpendicular_speed
            )
            reached_target = (
                target_distance_after <= self._target_radius
                and z_after > 1.0
                and upright > 0.7
            )
            target_success_reward = (
                self._target_success_reward * float(reached_target)
            )
            # No distance penalty — the robot is incentivised to approach
            # purely through the progress and direction rewards above.

            if reached_target:
                self._curriculum_level += 1
                self._sample_target()
                self._set_target_marker()

        reward = (
            1.0
            + stand_height_reward
            + recovery_reward
            + upright_reward
            + standing_reward
            + stability_reward
            + walk_reward
            + forward_velocity_reward
            + target_progress_reward
            + target_direction_reward
            + target_success_reward
            - ctrl_cost
            - impact_cost
            - bad_support_penalty
            - low_upright_penalty
            - stillness_penalty
            - lateral_drift_penalty
        )
        # Update the standing latch, then ask the property: it is the property that
        # honours terminate_when_unhealthy, so the penalty must not be computed from
        # the latch directly or a fall would cost 500 even with termination disabled.
        self._latch_health()
        terminated = self.terminated
        if terminated:
            reward -= 500.0

        observation = self._get_obs(upright_raw)

        info = {
            "reward_linup": stand_height_reward,
            "reward_recovery": recovery_reward,
            "reward_upright": upright_reward,
            "reward_standing": standing_reward,
            "reward_stability": stability_reward,
            "reward_walk": walk_reward,
            "reward_forward_velocity": forward_velocity_reward,
            "reward_target_progress": target_progress_reward,
            "reward_target_direction": target_direction_reward,
            "reward_target_success": target_success_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_impact": -impact_cost,
            "reward_bad_support": -bad_support_penalty,
            "reward_low_upright": -low_upright_penalty,
            "reward_stillness": -stillness_penalty,
            "reward_lateral_drift": -lateral_drift_penalty,
            "x_position": x_after,
            "y_position": y_after,
            "z_position": z_after,
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "target_x": self._target_xy[0],
            "target_y": self._target_xy[1],
            "target_distance": target_distance_after if is_target_phase else self._distance_to_target(),
            "upright": upright_raw,
            "bad_floor_contacts": bad_floor_contacts,
            "foot_floor_contacts": foot_floor_contacts,
            "task_phase": task_phase,
            "curriculum_level": self._curriculum_level,
            "target_success_streak": 0,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info


    @staticmethod
    def _normalize_quat(quat):
        quat = np.asarray(quat, dtype=np.float64)
        return quat / np.linalg.norm(quat)

    @staticmethod
    def _axis_angle_quat(axis, angle):
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        half_angle = 0.5 * angle
        return np.array(
            [
                np.cos(half_angle),
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quat_mul(left, right):
        w1, x1, y1, z1 = left
        w2, x2, y2, z2 = right
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )

    def _jitter_quat(self, quat, max_angle):
        axis = self.np_random.normal(size=3)
        angle = self.np_random.uniform(-max_angle, max_angle)
        jitter = self._axis_angle_quat(axis, angle)
        return self._normalize_quat(self._quat_mul(jitter, quat))

    def _sample_reset_kind(self):
        if self._reset_mode != "mixed":
            return self._reset_mode

        sample = self.np_random.random()
        if sample < self._fixed_reset_probability:
            return "fixed"
        if sample < self._fixed_reset_probability + self._upright_reset_probability:
            return "upright"
        return "fallen"

    def _apply_fixed_reset(self, qpos):
        qpos[2] = 0.35
        qpos[3:7] = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])

    def _apply_upright_reset(self, qpos):
        # z in [1.2, 1.4] gives safe margin above healthy_z_range minimum (1.0)
        qpos[2] = self.np_random.uniform(1.2, 1.4)
        qpos[3:7] = self._jitter_quat(np.array([1.0, 0.0, 0.0, 0.0]), max_angle=0.25)

    def _apply_fallen_reset(self, qpos):
        fallen_quats = (
            np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0]),
            np.array([np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0]),
            np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]),
            np.array([np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
        )
        quat = fallen_quats[self.np_random.integers(len(fallen_quats))]
        qpos[2] = self.np_random.uniform(0.25, 0.65)
        qpos[3:7] = self._jitter_quat(quat, max_angle=0.45)

        joint_noise = self.np_random.uniform(-0.35, 0.35, size=self.model.nq - 7)
        qpos[7:] = np.clip(qpos[7:] + joint_noise, -1.2, 1.2)

    def reset_model(self):

        qpos = self.init_qpos + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq,
        )
        reset_kind = self._sample_reset_kind()
        if reset_kind == "fixed":
            self._apply_fixed_reset(qpos)
        elif reset_kind == "upright":
            self._apply_upright_reset(qpos)
        else:
            self._apply_fallen_reset(qpos)

        qvel = self.init_qvel + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nv,
        )
        if reset_kind == "fallen":
            qvel += self.np_random.normal(scale=self._fallen_velocity_scale, size=self.model.nv)
        elif reset_kind == "upright":
            qvel += self.np_random.normal(scale=0.08, size=self.model.nv)

        self.set_state(qpos, qvel)
        self._ever_healthy = False
        # Always resample a new target on episode reset
        self._sample_target()
        self._set_target_marker()
        return self._get_obs()

    def _sample_target(self):
        if self._task_phase != "target":
            self._target_xy = np.array([3.0, 0.0], dtype=np.float64)
            return

        min_distance, max_distance = self._target_distance_range
        distance = self.np_random.uniform(min_distance, max_distance)
        angle = self.np_random.uniform(-0.15, 0.15)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        self._target_xy = self.data.qpos[:2].copy() + distance * direction

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_position": self.data.qpos[2],
            "upright": self.upright_factor,
            "reset_mode": self._reset_mode,
            "task_phase": self._task_phase,
            "target_x": self._target_xy[0],
            "target_y": self._target_xy[1],
            "target_distance": self._distance_to_target(),
        }


gym.register(
    id="WalkerRagdoll-v0",
    entry_point="envs.walker_ragdoll_env:WalkerRagdollEnv",
    max_episode_steps=1000,
)
