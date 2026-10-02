"""Unit and integration tests for the Walker Ragdoll environment and SAC training."""

import os
import shutil
import sys
import unittest
import numpy as np
import gymnasium as gym
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import envs.walker_ragdoll_env
from train_walker import SACAgent, PPOAgent, get_obs_rms, load_actor_initialization, make_env, set_obs_rms, wrap_normalize_observation, wrap_transform_observation
from utils.checkpoint import save_checkpoint, load_checkpoint, force_delete_run, find_latest_checkpoint


class TestWalkerRagdollEnv(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("WalkerRagdoll-v0")

    def tearDown(self):
        self.env.close()

    def test_env_creation(self):
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.observation_space.shape, (46,))

    def test_target_phase_env_creation(self):
        env = gym.make("WalkerRagdoll-v0", task_phase="target")
        try:
            self.assertEqual(env.observation_space.shape, (49,))
            obs, info = env.reset(seed=42)
            self.assertEqual(obs.shape, env.observation_space.shape)
            self.assertIn("target_distance", info)
            self.assertGreater(info["target_distance"], 0.0)
        finally:
            env.close()

    def test_reset(self):
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertTrue(np.isfinite(obs).all())
        self.assertIn("z_position", info)
        self.assertIn("upright", info)

    def test_reset_modes(self):
        for mode in ("fixed", "upright", "fallen", "mixed"):
            env = gym.make("WalkerRagdoll-v0", reset_mode=mode)
            try:
                obs, info = env.reset(seed=42)
                self.assertEqual(obs.shape, env.observation_space.shape)
                self.assertTrue(np.isfinite(obs).all())
                self.assertEqual(info["reset_mode"], mode)
            finally:
                env.close()

    def test_step(self):
        obs, info = self.env.reset(seed=42)
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("reward_linup", info)
        self.assertIn("reward_ctrl", info)
        self.assertIn("reward_stability", info)
        self.assertIn("reward_walk", info)

    def test_task_phases(self):
        for phase in ("recovery", "balance", "walk", "target"):
            env = gym.make("WalkerRagdoll-v0", task_phase=phase)
            try:
                obs, info = env.reset(seed=42)
                self.assertEqual(obs.shape, env.observation_space.shape)
                self.assertEqual(info["task_phase"], phase)
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                self.assertEqual(info["task_phase"], phase)
                if phase == "target":
                    self.assertIn("reward_target_progress", info)
                    self.assertIn("target_distance", info)
            finally:
                env.close()

    def test_episode(self):
        obs, _ = self.env.reset(seed=42)
        for _ in range(100):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        self.assertTrue(True)


class TestAgent(unittest.TestCase):
    def setUp(self):
        env = gym.make("WalkerRagdoll-v0")
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.act_dim = int(np.prod(env.action_space.shape))
        self.agent = SACAgent(self.obs_dim, env.action_space)
        env.close()

    def test_forward(self):
        obs = torch.randn(1, self.obs_dim)
        mean, log_std = self.agent(obs)
        self.assertEqual(mean.shape, (1, self.act_dim))
        self.assertEqual(log_std.shape, (1, self.act_dim))
        action, logprob, _ = self.agent.get_action(obs)
        self.assertEqual(action.shape, (1, self.act_dim))
        self.assertEqual(logprob.shape, (1, 1))

    def test_actor_initialization_expands_target_observation(self):
        source_env = gym.make("WalkerRagdoll-v0")
        target_env = gym.make("WalkerRagdoll-v0", task_phase="target")
        try:
            source_agent = SACAgent(
                int(np.prod(source_env.observation_space.shape)), source_env.action_space
            )
            target_agent = SACAgent(
                int(np.prod(target_env.observation_space.shape)), target_env.action_space
            )
            checkpoint = {"actor_state_dict": source_agent.state_dict()}
            expanded = load_actor_initialization(target_agent, checkpoint)
            self.assertTrue(expanded)
            self.assertTrue(
                torch.allclose(
                    target_agent.backbone[0].weight[:, : self.obs_dim],
                    source_agent.backbone[0].weight,
                )
            )
        finally:
            source_env.close()
            target_env.close()


class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.run_id = "test_run_123"
        self.base_dir = "test_checkpoints"
        force_delete_run(self.run_id, ckpt_base=self.base_dir)

    def tearDown(self):
        force_delete_run(self.run_id, ckpt_base=self.base_dir)

    def test_save_and_load(self):
        env = gym.make("WalkerRagdoll-v0")
        agent = SACAgent(int(np.prod(env.observation_space.shape)), env.action_space)
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        env.close()

        save_checkpoint(agent, optimizer, 1000, self.run_id, base_dir=self.base_dir)

        env2 = gym.make("WalkerRagdoll-v0")
        agent2 = SACAgent(int(np.prod(env2.observation_space.shape)), env2.action_space)
        optimizer2 = torch.optim.Adam(agent2.parameters(), lr=3e-4)
        env2.close()

        _, loaded_step = load_checkpoint(agent2, optimizer2, self.run_id, base_dir=self.base_dir)
        self.assertEqual(loaded_step, 1000)

    def test_latest_checkpoint(self):
        env = gym.make("WalkerRagdoll-v0")
        agent = SACAgent(int(np.prod(env.observation_space.shape)), env.action_space)
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        env.close()

        save_checkpoint(agent, optimizer, 1000, self.run_id, base_dir=self.base_dir)
        save_checkpoint(agent, optimizer, 2000, self.run_id, base_dir=self.base_dir)

        latest = find_latest_checkpoint(os.path.join(self.base_dir, self.run_id))
        self.assertIn("ckpt_2000.pt", latest)


class TestForceDeleteRunScope(unittest.TestCase):
    """A fresh start must not destroy a sibling run's logs.

    force_delete_run() used to glob `runs/<run_id>__*` and delete every match, so a second
    launch with the same --run-id and a different seed removed the first one's tensorboard
    directory while it was still being written - the live run then died inside the writer
    thread with FileNotFoundError.
    """

    base = "test_run_logs"

    def _make(self, name):
        path = os.path.join(self.base, name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "events.out.tfevents.probe"), "wb") as handle:
            handle.write(b"x")
        return path

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)
        shutil.rmtree("test_checkpoints/shared_run", ignore_errors=True)

    def test_only_the_callers_log_directory_is_removed(self):
        mine = self._make("shared_run__7")
        sibling = self._make("shared_run__8")
        force_delete_run("shared_run", "shared_run__7",
                         ckpt_base="test_checkpoints", run_base=self.base)
        self.assertFalse(os.path.isdir(mine), "this run's own log dir should be cleared")
        self.assertTrue(os.path.isdir(sibling),
                        "another seed's log dir belongs to another run")

    def test_glob_mode_still_clears_every_directory_of_the_run(self):
        for name in ("shared_run__7", "shared_run__8", "shared_run__9"):
            self._make(name)
        force_delete_run("shared_run", ckpt_base="test_checkpoints", run_base=self.base)
        self.assertEqual([d for d in os.listdir(self.base)], [],
                         "without run_name the caller means the whole run id")


class TestPostureIsABonus(unittest.TestCase):
    """v9: posture is a bonus, and the locomotion gate that keeps it a prerequisite stays.

    Before v9 the floor was punished: below z=0.65 the height bonus paid 0 while
    low_upright_penalty took 20*(0.85-z) every step, so posture entered the return as a
    punishment (-11.15/step measured on the 40M SAC policy). Removing a fall cost invites a
    collapse-the-floor exploit, so these pin the two things that have to stay true for the bonus
    form to be safe: nothing charges for being low, standing still out-earn lying still, and
    velocity rewards remain unreachable from the floor.
    """

    def _run(self, reset_mode, action, steps=120, seed=3):
        env = gym.make("WalkerRagdoll-v0", reset_mode=reset_mode, task_phase="recovery")
        try:
            env.reset(seed=seed)
            linup, gates, low = [], [], []
            for _ in range(steps):
                _obs, _r, _t, _tr, info = env.step(np.asarray(action, dtype=np.float32))
                linup.append(float(info["reward_linup"]))
                gates.append(float(info.get("standing_gate", 0.0)))
                low.append(float(info["reward_low_upright"]))
            return np.mean(linup), np.mean(gates), np.mean(low)
        finally:
            env.close()

    def test_no_term_charges_for_low_posture(self):
        _linup, _gate, low = self._run("fallen", np.zeros(17))
        self.assertEqual(low, 0.0, "the posture penalty is back; v9 made it default 0")

    def test_height_term_is_never_negative_and_orders_by_posture(self):
        prone, prone_gate, _ = self._run("fallen", np.zeros(17))
        upright, upright_gate, _ = self._run("upright", np.zeros(17))
        self.assertGreaterEqual(prone, 0.0, "the height term paid out negative posture")
        self.assertGreater(upright, prone,
                           "standing must out-earn lying, or the get-up incentive is gone")
        self.assertLess(prone_gate, 0.05,
                        "a prone robot was inside the standing gate: locomotion terms would be "
                        "reachable without posture, which is the exploit v9 had to keep closed")

    def test_locomotion_still_requires_posture(self):
        """standing_gate is the multiplicative gate on every velocity term; v9 did not remove it."""
        _linup, gate, _low = self._run("fallen", np.zeros(17))
        _up_linup, up_gate, _ = self._run("upright", np.zeros(17))
        self.assertLess(gate, 0.05)
        self.assertGreater(up_gate, gate)


class TestMakeEnv(unittest.TestCase):
    def test_make_env(self):
        env = make_env("WalkerRagdoll-v0", 0, False, "test_run")()
        self.assertIsNotNone(env)
        obs, _ = env.reset(seed=42)
        self.assertEqual(obs.shape, env.observation_space.shape)
        env.close()

    def test_obs_rms_helpers_target_normalize_wrapper(self):
        envs = gym.vector.SyncVectorEnv(
            [make_env("WalkerRagdoll-v0", 0, False, "test_run")]
        )
        envs = wrap_normalize_observation(envs)
        envs = wrap_transform_observation(envs, lambda obs: np.clip(obs, -10, 10))

        try:
            obs_rms = get_obs_rms(envs)
            obs_rms.mean[...] = 123.0
            set_obs_rms(envs, obs_rms)

            self.assertNotIn("obs_rms", envs.__dict__)
            self.assertTrue(np.allclose(envs.env.obs_rms.mean, 123.0))
            self.assertIs(get_obs_rms(envs), envs.env.obs_rms)
        finally:
            envs.close()


class TestFallenStartIsNotAFall(unittest.TestCase):
    """terminate_when_unhealthy must not punish the curriculum's own starting poses."""

    def _run(self, reset_mode, steps=6):
        env = gym.make(
            "WalkerRagdoll-v0", reset_mode=reset_mode, task_phase="target",
            terminate_when_unhealthy=True,
        )
        try:
            env.reset(seed=3)
            z0 = float(env.unwrapped.data.qpos[2])
            rewards, dones = [], []
            for _ in range(steps):
                _, reward, terminated, _, _ = env.step(np.zeros(17, np.float32))
                rewards.append(reward)
                dones.append(terminated)
            return z0, rewards, dones, env.unwrapped._ever_healthy
        finally:
            env.close()

    def test_fallen_start_does_not_terminate_immediately(self):
        z0, rewards, dones, ever_healthy = self._run("fallen")
        self.assertLess(z0, 1.0, "fallen reset should start below the healthy z range")
        self.assertFalse(dones[0], "a fallen start terminated on the very first step")
        self.assertFalse(ever_healthy, "no fall can be reported before the robot has stood")
        self.assertGreater(rewards[0], -400.0, "the -500 fall penalty fired on a reset pose")

    def test_upright_reset_start_is_inside_healthy_range(self):
        env = gym.make("WalkerRagdoll-v0", reset_mode="upright", terminate_when_unhealthy=True)
        try:
            env.reset(seed=3)
            self.assertGreater(float(env.unwrapped.data.qpos[2]), 1.0)
            self.assertTrue(env.unwrapped.is_healthy)
        finally:
            env.close()

    def test_default_backend_never_terminates_on_health(self):
        env = gym.make("WalkerRagdoll-v0", reset_mode="fallen")
        try:
            env.reset(seed=3)
            for _ in range(20):
                _, _, terminated, truncated, _ = env.step(np.zeros(17, np.float32))
                self.assertFalse(terminated)
                self.assertFalse(truncated)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
