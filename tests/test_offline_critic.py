"""Tests for the offline critic trained on the planner's transitions.

The thing these exist to catch is the failure mode this script actually hit while being written: a reward
column shaped (batch,) beside a Q output shaped (batch,1) broadcasts the TD target to (batch,batch),
which trains, prints a plausible loss, and is completely wrong. Everything below is aimed at that class of
silent-shape or silent-leak problem rather than at re-testing PyTorch.
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import envs.walker_ragdoll_env  # noqa: E402,F401
import train_offline_critic as toc  # noqa: E402


def write_preference_demos(directory, episodes=5, steps=30, obs_dim=49, act_dim=17, seed=0):
    """A preference dataset with the same schema collect_mpc_demos.py writes."""
    rng = np.random.default_rng(seed)
    obs, act, rew, val, band = [], [], [], [], []
    meta_eps = []
    for ep in range(episodes):
        o = rng.standard_normal((steps, obs_dim)).astype(np.float32) * 2.0
        a = rng.uniform(-1, 1, size=(steps, act_dim)).astype(np.float32)
        r = (rng.random(steps) * 100.0).astype(np.float32)
        v = (rng.random(steps) * 5000.0).astype(np.float32)
        b = (rng.random(steps) < 0.2).astype(np.float32)
        obs.append(o); act.append(a); rew.append(r); val.append(v); band.append(b)
        meta_eps.append({"episode": ep, "seed": 300 + ep, "steps": steps,
                         "min_target_distance": 1.0, "reached_target": False,
                         "reached_target_upright": False, "falls": 0, "pct_steps_in_band": 20.0})
    npz = os.path.join(directory, "pref.npz")
    np.savez(npz, obs=np.concatenate(obs), action=np.concatenate(act), reward=np.concatenate(rew),
             plan_value=np.concatenate(val), in_band=np.concatenate(band),
             distance=np.full(len(np.concatenate(rew)), 2.0, dtype=np.float32))
    stacked = np.concatenate(obs)
    with open(npz + ".json", "w", encoding="utf-8") as handle:
        json.dump({"protocol": "synthetic", "demo_seeds": [e["seed"] for e in meta_eps],
                   "obs_rms": {"mean": stacked.mean(axis=0).tolist(),
                               "var": stacked.var(axis=0).tolist(),
                               "count": int(stacked.shape[0])},
                   "episodes": meta_eps,
                   "summary": {"transitions": int(stacked.shape[0]),
                               "episodes_reached": 0, "episodes_upright_arrival": 0}}, handle)
    return npz


class TestTransitionShapes(unittest.TestCase):
    def test_the_td_target_is_a_column_and_not_an_outer_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz = write_preference_demos(tmp)
            arrays, meta = toc.load_demos(npz)
            mean = np.asarray(meta["obs_rms"]["mean"])
            var = np.asarray(meta["obs_rms"]["var"])
            tr, bounds = toc.build_transitions(arrays, meta, mean, var)
            for key in ("rew", "not_done"):
                self.assertEqual(tr[key].shape, (len(tr["obs"]), 1),
                                 f"{key} must be a column: beside a (batch,1) Q output a (batch,) "
                                 "vector broadcasts the TD target to (batch,batch)")
            self.assertEqual(tr["act"].shape[1], 17)

    def test_only_the_last_step_of_each_episode_is_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz = write_preference_demos(tmp, episodes=4, steps=25)
            arrays, meta = toc.load_demos(npz)
            tr, bounds = toc.build_transitions(
                arrays, meta, np.asarray(meta["obs_rms"]["mean"]),
                np.asarray(meta["obs_rms"]["var"]))
            zeros = np.where(tr["not_done"].reshape(-1) == 0.0)[0]
            self.assertEqual(zeros.tolist(), [24, 49, 74, 99],
                             "the TD backup chained one demonstration episode into the next")


class TestTrainingRuns(unittest.TestCase):
    def args(self, npz, tmp, **over):
        base = {"demos": npz, "actor": "q", "epochs": 2, "batch": 64, "lr": 1e-3, "gamma": 0.99,
                "tau": 0.005, "bc_lambda": 1.0, "holdout": 1, "seed": 7, "device": "cpu",
                "run_id": "critic_test_run"}
        base.update(over)
        return type("A", (), base)()

    def tearDown(self):
        import shutil
        shutil.rmtree(os.path.join(ROOT, "checkpoints", "critic_test_run"), ignore_errors=True)

    def test_a_two_epoch_run_selects_by_held_out_td_and_writes_a_loadable_checkpoint(self):
        import eval_phase1
        with tempfile.TemporaryDirectory() as tmp:
            npz = write_preference_demos(tmp, episodes=5)
            path = toc.train(self.args(npz, tmp))
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.assertIn("actor_state_dict", ck)
            self.assertEqual(ck["task_phase"], "target")
            self.assertEqual(ck["critic"]["actor_mode"], "q")
            self.assertLessEqual(ck["critic"]["selected_epoch"], 2)
            self.assertTrue(ck["critic"]["eval_seeds_disjoint"])
            self.assertEqual(len(ck["critic"]["history"]), 2)
            label, phase, width, policy, _reset, _r = eval_phase1.build_policy(path, "cpu")
            self.assertEqual((label, phase, width), ("offline_critic", "target", 49))
            action = np.asarray(policy(np.zeros(width)))
            self.assertEqual(action.shape, (17,))
            self.assertTrue(np.all(np.abs(action) <= 1.0 + 1e-6))

    def test_the_bc_variant_adds_the_in_distribution_term(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz = write_preference_demos(tmp, episodes=5)
            path = toc.train(self.args(npz, tmp, actor="q_bc"))
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.assertGreater(ck["critic"]["bc_lambda"], 0.0)
            path_q = toc.train(self.args(npz, tmp, actor="q", run_id="critic_test_run"))
            ck_q = torch.load(path_q, map_location="cpu", weights_only=False)
            self.assertEqual(ck_q["critic"]["bc_lambda"], 0.0)

    def test_demos_without_the_preference_columns_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz = write_preference_demos(tmp, episodes=3)
            arrays = dict(np.load(npz))
            arrays.pop("plan_value")
            np.savez(npz, **arrays)
            with self.assertRaises(SystemExit) as ctx:
                toc.train(self.args(npz, tmp))
            self.assertIn("plan_value", str(ctx.exception))

    def test_a_run_with_no_held_out_episode_is_refused(self):
        # Selection by validation needs a validation episode; holdout=0 would silently score the train
        # rows and keep the best-on-training checkpoint. The guard lives in main(), so that is what the
        # test drives.
        with tempfile.TemporaryDirectory() as tmp:
            npz = write_preference_demos(tmp, episodes=3)
            argv = ["train_offline_critic.py", "--demos", npz, "--holdout", "0",
                    "--run-id", "critic_test_run"]
            original = sys.argv
            sys.argv = argv
            try:
                with self.assertRaises(SystemExit) as ctx:
                    toc.main()
            finally:
                sys.argv = original
            self.assertIn("--holdout", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
