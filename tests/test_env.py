"""Unit and integration tests for the Walker Ragdoll environment and SAC training."""

import os
import sys
import unittest
import numpy as np
import gymnasium as gym
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import envs.walker_ragdoll_env
from sac_walker import SACAgent, make_env
from utils.checkpoint import save_checkpoint, load_checkpoint, force_delete_run, find_latest_checkpoint


class TestWalkerRagdollEnv(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("WalkerRagdoll-v0")

    def tearDown(self):
        self.env.close()

    def test_env_creation(self):
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.observation_space.shape, (46,))

    def test_reset(self):
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertTrue(np.isfinite(obs).all())
        self.assertIn("z_position", info)
        self.assertIn("upright", info)

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


class TestMakeEnv(unittest.TestCase):
    def test_make_env(self):
        env = make_env("WalkerRagdoll-v0", 0, False, "test_run")()
        self.assertIsNotNone(env)
        obs, _ = env.reset(seed=42)
        self.assertEqual(obs.shape, env.observation_space.shape)
        env.close()


if __name__ == "__main__":
    unittest.main()
