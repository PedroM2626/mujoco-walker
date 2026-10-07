"""The step splitter in `bench_approach_mechanism.py` has to distinguish "never stood" from "stood".

Every claim in the README's walking-or-falling paragraph is produced by one function,
`trace_episode`, which attributes each step's change in distance-to-target to the standing band or to
outside it. If that attribution were wrong, the correction would be wrong with it - so this tests the
instrument on two episodes whose behaviour is known without needing a trained policy: a ragdoll held at
zero action from a fallen reset never enters the band, and one released from an upright reset enters it
and leaves it. Both run on cpu, both are a single episode.
"""
import os
import sys
import unittest

import gymnasium as gym
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import envs.walker_ragdoll_env  # noqa: E402,F401  (registers WalkerRagdoll-v0)
import bench_approach_mechanism as bam  # noqa: E402

RADIUS = 0.45


def run(reset_mode, seed=11, steps=400):
    env = gym.make("WalkerRagdoll-v0", reset_mode=reset_mode, task_phase="target")
    try:
        return bam.trace_episode(env, lambda obs: np.zeros(17), lambda: None, seed, 0, steps,
                                 float(env.unwrapped._target_radius))
    finally:
        env.close()


class TestTraceAttributesStepsToTheBand(unittest.TestCase):
    def test_a_body_that_never_rises_contributes_nothing_to_the_standing_side(self):
        t = run("fallen")
        self.assertEqual(t["pct_steps_in_band"], 0.0, "a fallen ragdoll at zero action entered the band")
        self.assertEqual(t["closing_while_standing_m"], 0.0)
        self.assertEqual(t["band_entries"], 0)
        self.assertIsNone(t["heading_velocity_in_band_mps"], "no in-band steps means no mean")
        self.assertFalse(t["reached_target"], "it slid into the target, which is not what this tests")

    def test_a_body_released_from_upright_enters_the_band_once_and_leaves_it(self):
        t = run("upright")
        self.assertGreater(t["pct_steps_in_band"], 0.0)
        self.assertGreaterEqual(t["band_entries"], 1)
        self.assertLess(t["longest_band_run_s"], 2.0, "zero action does not hold a ragdoll up")

    def test_the_two_sides_of_the_split_add_up_to_the_approach(self):
        for mode in ("fallen", "upright", "mixed"):
            t = run(mode)
            total = t["closing_while_standing_m"] + t["closing_while_down_m"]
            self.assertAlmostEqual(total, t["approach_distance_m"], places=3,
                                   msg=f"{mode}: the split does not account for the approach")
            # d0 and min_target_distance are each rounded to mm in the trace, so their difference is
            # only consistent with approach_distance_m to about a centimetre.
            self.assertAlmostEqual(t["approach_distance_m"],
                                   t["d0"] - t["min_target_distance"], places=2)
            if t["reached_target_upright"]:
                self.assertTrue(t["reached_target"],
                                "an upright arrival that the loose rule does not count")
