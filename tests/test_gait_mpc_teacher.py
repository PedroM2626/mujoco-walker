"""Tests for the corrected Route-4 Gait-MPC teacher and the centered within-state Q(s, a) probe.

Four invariants are guarded here:
  1. `CrossEntropyPlanner(..., objective="gait")` modifies only its internal candidate rollouts and
     restores `_terminate_when_unhealthy`, `_stability_reward_weight`, `_ever_healthy`, `_target_xy`
     and `_curriculum_level` on the outer scored environment after every candidate evaluation.
  2. Outside the standing band (`z <= 1.0` or `upright <= 0.7`), the gait objective zeroes locomotion
     rewards (`reward_target_progress`, `reward_target_direction`, `reward_forward_velocity`) and does
     not arm `_terminate_when_unhealthy` when a plan starts from a fallen posture (`z0 <= 1.12`), so
     dynamic stand-up trajectories are not punished by a step-1 latch at `z = 1.0 m`.
  3. Holding control knots (`hold > 1`) with `action="elite-mean"` repeats the elite first knot across
     `hold` steps and reduces consecutive-action MSE compared with unconstrained `hold=1` sampling.
  4. `bench_planner_q.within_state_metrics` with centered predictions (`within_r2_centered`) scores any
     state-only predictor at `0.0` by construction (fixing the uncentered `within_r2` leak where
     `state_only` scored `-2.095`), and `benchmarks/mpc_gait_teacher_benchmark.json` matches the
     committed comparators and live Q reconciliation.
"""
import json
import os
import sys
import unittest

import gymnasium as gym
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import envs.walker_ragdoll_env  # noqa: E402,F401
from envs.reward_shaping import TRAINING_REWARD_KWARGS  # noqa: E402
from bench_mpc import CrossEntropyPlanner  # noqa: E402
import bench_planner_q as bpq  # noqa: E402
import bench_gait_mpc_teacher as bgm  # noqa: E402


class TestGaitPlannerRestoreAndTwoStateRule(unittest.TestCase):
    def test_candidate_rollouts_restore_outer_env_flags_and_state(self):
        env = gym.make("WalkerRagdoll-v0", reset_mode="mixed", task_phase="target",
                       **TRAINING_REWARD_KWARGS)
        try:
            obs, _ = env.reset(seed=11)
            unw = env.unwrapped
            self.assertFalse(unw._terminate_when_unhealthy)
            orig_stab = float(unw._stability_reward_weight)
            orig_qpos = unw.data.qpos.copy()
            orig_target = np.asarray(unw._target_xy).copy()

            planner = CrossEntropyPlanner(
                horizon=10, samples=4, iterations=1, elite=2, replan=5,
                hold=5, seed=11, action="elite-mean", objective="gait",
                settled_fall_termination=True
            )
            planner.bind(env)
            action = planner(obs)
            self.assertEqual(action.shape, (17,))
            self.assertFalse(unw._terminate_when_unhealthy,
                             "planner leaked _terminate_when_unhealthy=True onto the scored env")
            self.assertAlmostEqual(float(unw._stability_reward_weight), orig_stab, places=9,
                                   msg="planner leaked _stability_reward_weight onto the scored env")
            np.testing.assert_allclose(unw.data.qpos, orig_qpos, rtol=0, atol=1e-12)
            np.testing.assert_allclose(unw._target_xy, orig_target, rtol=0, atol=1e-12)
        finally:
            env.close()

    def test_fallen_start_does_not_arm_fall_termination_latch_while_upright_start_does(self):
        env = gym.make("WalkerRagdoll-v0", reset_mode="fallen", task_phase="target",
                       **TRAINING_REWARD_KWARGS)
        try:
            env.reset(seed=12)
            unw = env.unwrapped
            z0 = float(unw.data.qpos[2])
            self.assertLess(z0, 1.0, "fallen reset must start below the standing band")
            # Even if the outer episode previously latched _ever_healthy=True before falling,
            # a plan starting from the floor must not terminate on step 1 of _evaluate.
            unw._ever_healthy = True
            planner = CrossEntropyPlanner(
                horizon=6, samples=4, iterations=1, elite=2, replan=3,
                hold=3, seed=12, action="elite-mean", objective="gait",
                settled_fall_termination=True
            )
            planner.bind(env)
            plan = np.zeros((6, 17), dtype=np.float64)
            planner._evaluate(plan, settled_standing=False)
            self.assertEqual(planner.rollout_steps, 6,
                             "fallen-start rollout terminated early instead of evaluating all 6 steps")
        finally:
            env.close()

    def test_knot_hold_and_elite_mean_repeat_first_knot_across_hold_steps(self):
        env = gym.make("WalkerRagdoll-v0", reset_mode="upright", task_phase="target",
                       **TRAINING_REWARD_KWARGS)
        try:
            obs, _ = env.reset(seed=11)
            planner = CrossEntropyPlanner(
                horizon=10, samples=6, iterations=1, elite=3, replan=5,
                hold=5, seed=11, action="elite-mean", objective="gait"
            )
            planner.bind(env)
            actions = [planner(obs) for _ in range(5)]
            for k in range(1, 5):
                np.testing.assert_allclose(
                    actions[k], actions[0], rtol=0, atol=1e-12,
                    err_msg="elite-mean with hold=5 must hold the first knot constant across 5 steps"
                )
        finally:
            env.close()


class TestCenteredWithinStateMetrics(unittest.TestCase):
    def test_any_state_only_predictor_scores_zero_centered_within_state_r2(self):
        rng = np.random.default_rng(42)
        groups = np.repeat(np.arange(20), 12)
        true_state_level = np.repeat(rng.normal(scale=500.0, size=20), 12)
        within_signal = rng.normal(scale=5.0, size=len(groups))
        y = true_state_level + within_signal
        # A state-only predictor that misses the state level by an arbitrary offset per group:
        pred_state_only = np.repeat(rng.normal(scale=500.0, size=20), 12)
        m = bpq.within_state_metrics(y, pred_state_only, groups)
        self.assertLess(m["within_r2"], -1.0,
                        "uncentered within_r2 should be negative when between-state level error is large")
        self.assertAlmostEqual(m["within_r2_centered"], 0.0, places=12,
                               msg="within_r2_centered must be 0.0 for any predictor constant within state")


class TestGaitMpcBenchmarkArtifact(unittest.TestCase):
    ART = os.path.join(ROOT, "benchmarks", "mpc_gait_teacher_benchmark.json")

    def test_the_artifact_exists_and_matches_the_reconciled_numbers(self):
        self.assertTrue(os.path.exists(self.ART), f"missing {self.ART}")
        with open(self.ART, encoding="utf-8") as handle:
            art = json.load(handle)

        # 1. Re-derive the Q(s, a) reconciliation directly from mpc_candidates.npz and verify match
        live_q = bgm.reconcile_planner_q()
        self.assertEqual(art["planner_q_reconciliation"], live_q)
        self.assertAlmostEqual(
            live_q["centered_within_plan"]["state_only"]["within_r2_centered"], 0.0, places=6
        )
        self.assertGreater(
            live_q["centered_within_plan"]["joint_interactions"]["within_r2_centered"], 0.0
        )
        self.assertFalse(any(live_q["gate_passed_centered"].values()),
                         "the pre-registered Q(s, a) gate must remain closed on mpc_candidates.npz")

        # 2. Verify all three Gait-MPC arms improve jitter and in-band share/time over comparators
        comp = art["published_comparators_4ep"]
        arms = art["gait_mpc_arms"]
        base_jitter = comp["mpc_argmin_28ep_jitter"]["consecutive_action_mse"]
        for name, cell in arms.items():
            self.assertLess(cell["jitter"]["consecutive_action_mse"], base_jitter / 5.0,
                            f"{name}: consecutive action MSE did not drop by at least 5x")
            self.assertLess(cell["jitter"]["jitter_to_energy_ratio"], 0.45,
                            f"{name}: jitter-to-energy ratio is not below 0.45")
            self.assertGreater(cell["gait"]["pct_steps_in_band"],
                               2.0 * comp["mpc_budget_48x3_h100"]["pct_steps_in_band"],
                               f"{name}: in-band step share did not double the published budget MPC")

        # 3. Specific arm checks across the 3 layers
        self.assertEqual(arms["gait_smooth_v08"]["gait"]["upright_arrivals"], 3)
        self.assertGreater(arms["gait_settled_v11"]["gait"]["share_while_standing"], 0.60)
        self.assertGreaterEqual(arms["gait_settled_v11"]["gait"]["max_longest_band_run_s"], 2.9)
        self.assertTrue(arms["gait_settled_v11"]["action_predictability_val_mse"]["ridge_beats_zero"])
        self.assertGreater(arms["gait_settled_v08"]["gait"]["share_while_standing"], 1.0)
        self.assertGreaterEqual(arms["gait_settled_v08"]["gait"]["max_longest_band_run_s"], 3.0)
        self.assertEqual(arms["gait_settled_v08"]["gait"]["episodes_with_band_run_ge_2s"], 2)
        self.assertTrue(arms["gait_settled_v08"]["action_predictability_val_mse"]["ridge_beats_zero"])

        # 4. Stage A (20-episode official teacher benchmark), Stage B (28-ep demos + students),
        #    and Stage C (200k-step Online SAC with strict band gate and warm-start)
        stage_a = art["stage_a_teacher_20ep_official"]
        self.assertEqual(stage_a["gait_mpc_v08_20ep"]["episodes"], 20)
        self.assertEqual(stage_a["gait_mpc_v08_20ep"]["episodes_with_band_run_ge_2s"], 5)
        self.assertGreaterEqual(stage_a["gait_mpc_v08_20ep"]["max_longest_band_run_s"], 4.0)
        self.assertGreater(stage_a["gait_mpc_v08_20ep"]["share_while_standing"], 0.55)
        self.assertEqual(stage_a["gait_mpc_v11_20ep"]["episodes_with_band_run_ge_2s"], 2)
        self.assertEqual(stage_a["gait_mpc_v11_20ep"]["upright_arrivals"], 4)

        stage_b = art["stage_b_demos_and_offline_students"]
        self.assertEqual(stage_b["teacher_28ep_demos"]["transitions"], 28000)
        self.assertEqual(stage_b["teacher_28ep_demos"]["upright_arrivals"], 10)
        self.assertTrue(stage_b["beats_the_trivial_baseline"]["ridge_alpha100"])
        self.assertTrue(stage_b["beats_the_trivial_baseline"]["student_bc_gait_raw"])
        self.assertTrue(stage_b["beats_the_trivial_baseline"]["student_bc_gait_band"])

        stage_c = art["stage_c_online_sac_strict_band_and_warmstart"]
        self.assertEqual(stage_c["budget_steps"], 200000)
        self.assertEqual(stage_c["arms_20ep_gait"]["sac_warmstart_gait_100k"]["reach_20ep"], 1)
        self.assertLess(
            stage_c["arms_20ep_gait"]["sac_strict_band_200k"]["episodes_never_in_band"],
            stage_b["students_20ep_gait"]["bc_gait_raw"]["episodes_never_in_band"],
        )


if __name__ == "__main__":
    unittest.main()
