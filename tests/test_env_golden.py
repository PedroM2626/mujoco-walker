"""Golden-value regression guard for the WalkerRagdoll reward and observation math.

The throughput work rewrote the per-step reward path (name lookups -> id tables,
np.clip -> builtins, np.linalg.norm -> math.hypot). Those rewrites are equivalent up
to floating-point association order, which is exactly the kind of change that is easy
to get subtly wrong and hard to notice later, so the fingerprints below pin the
behaviour of a fixed rollout.

Values were recorded against the pre-rewrite implementation and agree to <1e-7
accumulated over 300 steps. If one of these moves, the MDP moved.
"""

import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import envs.walker_ragdoll_env  # noqa: F401,E402
import gymnasium as gym  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_PATH = os.path.join(ROOT, "benchmarks", "walker_ragdoll_golden.json")
N_STEPS = 300


def rollout_fingerprint(task_phase, reset_mode="upright", seed=1234, action_seed=99):
    env = gym.make("WalkerRagdoll-v0", task_phase=task_phase, reset_mode=reset_mode)
    try:
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(action_seed)
        reward_sum = 0.0
        obs_sum = 0.0
        first_rewards = []
        max_z = 0.0
        for i in range(N_STEPS):
            action = rng.normal(0, 0.25, 17).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            obs_sum += float(obs.sum())
            max_z = max(max_z, float(info["z_position"]))
            if i < 3:
                first_rewards.append(float(reward))
        return {
            "reward_sum": reward_sum,
            "obs_sum": obs_sum,
            "first_rewards": first_rewards,
            "obs_dim": int(obs.shape[0]),
            "max_z": max_z,
        }
    finally:
        env.close()


class TestRewardMathGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(GOLDEN_PATH) as handle:
            cls.golden = {
                key: value for key, value in json.load(handle).items() if not key.startswith("_")
            }

    def test_every_phase_matches_golden(self):
        for phase, expected in self.golden.items():
            with self.subTest(task_phase=phase):
                actual = rollout_fingerprint(phase)
                self.assertEqual(actual["obs_dim"], expected["obs_dim"])
                self.assertAlmostEqual(
                    actual["reward_sum"], expected["reward_sum"], delta=1e-4,
                    msg="cumulative reward drifted: the reward function changed",
                )
                self.assertAlmostEqual(
                    actual["obs_sum"], expected["obs_sum"], delta=1e-4,
                    msg="observation content drifted",
                )
                self.assertAlmostEqual(
                    actual["max_z"], expected["max_z"], delta=1e-5,
                    msg="simulated trajectory drifted",
                )
                for got, want in zip(actual["first_rewards"], expected["first_rewards"]):
                    self.assertAlmostEqual(got, want, delta=1e-6)

    def test_phases_are_distinguishable(self):
        """A guard on the guard: if every phase scored alike, the test proves nothing."""
        scores = {
            phase: rollout_fingerprint(phase)["reward_sum"]
            for phase in ("recovery", "balance", "walk", "target")
        }
        self.assertGreater(len(set(round(v, 3) for v in scores.values())), 2, scores)

    def test_observation_is_finite_and_correctly_sized(self):
        for phase, size in (("recovery", 46), ("balance", 46), ("walk", 46), ("target", 49)):
            with self.subTest(task_phase=phase):
                env = gym.make("WalkerRagdoll-v0", task_phase=phase)
                try:
                    obs, _ = env.reset(seed=1)
                    self.assertEqual(obs.shape, (size,))
                    self.assertTrue(np.isfinite(obs).all())
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
