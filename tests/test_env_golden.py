"""Golden-value regression guard for the WalkerRagdoll reward and observation math.

The throughput work rewrote the per-step reward path (name lookups -> id tables,
np.clip -> builtins, np.linalg.norm -> math.hypot). Those rewrites are equivalent up
to floating-point association order, which is exactly the kind of change that is easy to
get subtly wrong and hard to notice later, so the fingerprints below pin the behaviour of
a fixed rollout.

Tolerances are deliberately two-tiered. This is a contact-rich, chaotic simulation: a
1-ulp difference in exp()/tanh() between platforms can flip the timing of a ground
contact, after which the trajectory is simply a different trajectory. So the 50-step
horizon is compared strictly (it is the regression detector - the -500-penalty inversion
introduced and caught during this work moved the sum by ~20000%), while the 300-step
horizon only has to stay within a few percent, which still catches any semantic change
while not failing merely because CI runs on a different CPU.

Values were recorded against the pre-rewrite implementation and agree to <1e-7
accumulated over 300 steps on the recording machine.
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
PHASES = ("recovery", "balance", "walk", "target")

REL_TOL_REWARD = 1e-9      # first few rewards: same code path, same platform primitives
REL_TOL_SHORT = 1e-6       # 50-step sums: the actual regression guard
REL_TOL_LONG = 0.05        # 300-step sums: chaotic amplification allowance


def rollout(task_phase, steps, reset_mode="upright", seed=1234, action_seed=99):
    env = gym.make("WalkerRagdoll-v0", task_phase=task_phase, reset_mode=reset_mode)
    try:
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(action_seed)
        reward_sum = obs_sum = 0.0
        first_rewards = []
        for i in range(steps):
            action = rng.normal(0, 0.25, 17).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            obs_sum += float(obs.sum())
            if i < 3:
                first_rewards.append(float(reward))
        return {
            "reward_sum": reward_sum,
            "obs_sum": obs_sum,
            "first_rewards": first_rewards,
            "obs_dim": int(obs.shape[0]),
        }
    finally:
        env.close()


def _rel(actual, expected):
    scale = max(1e-12, abs(expected))
    return abs(actual - expected) / scale


class TestRewardMathGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(GOLDEN_PATH, encoding="utf-8") as handle:
            cls.golden = {k: v for k, v in json.load(handle).items() if not k.startswith("_")}

    def test_structure_is_exact(self):
        for phase, expected in self.golden.items():
            with self.subTest(task_phase=phase):
                actual = rollout(phase, steps=5)
                self.assertEqual(actual["obs_dim"], expected["obs_dim"])
                self.assertTrue(np.isfinite(list(actual["first_rewards"])).all())

    def test_short_horizon_matches_strictly(self):
        """The regression guard: any change to the reward math shows up here immediately."""
        for phase, expected in self.golden.items():
            with self.subTest(task_phase=phase):
                want = expected["short"]
                actual = rollout(phase, steps=want["steps"])
                for got, exp in zip(actual["first_rewards"], want["first_rewards"]):
                    self.assertLess(_rel(got, exp), REL_TOL_REWARD,
                                    f"{phase}: per-step reward drifted")
                self.assertLess(_rel(actual["reward_sum"], want["reward_sum"]), REL_TOL_SHORT,
                                f"{phase}: {want['steps']}-step reward sum drifted "
                                f"{_rel(actual['reward_sum'], want['reward_sum']):.2e} rel")
                self.assertLess(_rel(actual["obs_sum"], want["obs_sum"]), REL_TOL_SHORT,
                                f"{phase}: {want['steps']}-step observation sum drifted")

    def test_long_horizon_stays_in_shape(self):
        """Loose enough for cross-platform FP divergence, tight enough for real changes."""
        for phase, expected in self.golden.items():
            with self.subTest(task_phase=phase):
                want = expected["long"]
                actual = rollout(phase, steps=want["steps"])
                deviation = _rel(actual["reward_sum"], want["reward_sum"])
                self.assertLess(
                    deviation, REL_TOL_LONG,
                    f"{phase}: 300-step reward moved {deviation * 100:.1f}% "
                    f"(actual {actual['reward_sum']:.2f} vs recorded {want['reward_sum']:.2f})",
                )

    def test_phases_are_distinguishable(self):
        """A guard on the guard: if every phase scored alike, the test proves nothing."""
        scores = {p: rollout(p, steps=50)["reward_sum"] for p in PHASES}
        self.assertGreater(len({round(v, 3) for v in scores.values()}), 2, scores)

    def test_observation_is_finite_and_correctly_sized(self):
        for phase, size in zip(PHASES, (46, 46, 46, 49)):
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
