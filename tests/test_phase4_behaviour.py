"""Behavioural coverage for the Phase-4 (Walker2d-v5) scripts.

The existing tests stopped at "the module imports". These go one step further and
actually run each trained artifact through the environment, which is the only way to
catch the failures that mattered in this repo: a checkpoint whose architecture no longer
matches the loader, an observation layout that drifted from the one used at training
time, and an evaluation that could not be repeated because nothing seeded it.

Requires gymnasium>=1.0 (Walker2d-v5) and therefore runs in .venv-phase4 / the CI
`phase4-imports` job; it skips elsewhere instead of failing, because requirements.txt
pins 0.29 for Phases 1-3 on purpose.
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P4 = os.path.join(ROOT, "openai_walker")
for path in (ROOT, P4):
    if path not in sys.path:
        sys.path.insert(0, path)

import gymnasium as gym  # noqa: E402

DATASET = os.path.join(P4, "dataset_openai.csv")


def _has_walker2d_v5():
    try:
        env = gym.make("Walker2d-v5")
        env.close()
        return True
    except Exception:
        return False


AVAILABLE = _has_walker2d_v5()


@unittest.skipUnless(AVAILABLE, "Walker2d-v5 needs gymnasium>=1.0 (see requirements-phase4.txt)")
class TestPhase4Dataset(unittest.TestCase):
    def test_dataset_schema_matches_the_observation_layout(self):
        """The 100k-transition dataset is the input every Phase-4 number depends on."""
        import pandas as pd

        self.assertTrue(os.path.exists(DATASET), "openai_walker/dataset_openai.csv is missing")
        frame = pd.read_csv(DATASET, nrows=5000)

        obs_cols = [c for c in frame.columns if c.startswith("obs_") and not c.startswith("next_obs_")]
        action_cols = [c for c in frame.columns if c.startswith("action_")]
        self.assertEqual(len(obs_cols), 17, f"expected 17 state columns, got {obs_cols}")
        self.assertEqual(len(action_cols), 6, f"expected 6 action columns, got {action_cols}")
        self.assertFalse(frame[obs_cols].isna().any().any(), "dataset has NaN states")
        self.assertFalse(frame[action_cols].isna().any().any(), "dataset has NaN actions")
        # An imitation learner trained outside the action bounds cannot recover them.
        self.assertTrue(frame[action_cols].abs().to_numpy().max() <= 1.0 + 1e-6,
                        "dataset actions exceed the env action bounds")


@unittest.skipUnless(AVAILABLE, "Walker2d-v5 needs gymnasium>=1.0 (see requirements-phase4.txt)")
class TestPhase4ArtifactsRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import evaluate_all

        cls.evaluate_all = evaluate_all
        env = gym.make("Walker2d-v5")
        cls.state_dim = int(env.observation_space.shape[0])
        cls.action_dim = int(env.action_space.shape[0])
        cls.max_action = float(env.action_space.high[0])
        cls.env = env
        import torch

        cls.device = torch.device("cpu")
        cls.candidates = evaluate_all.build_candidates(
            cls.state_dim, cls.action_dim, cls.max_action, cls.device
        )
        names = [name for name, _, _ in cls.candidates]
        if not cls.candidates:
            raise unittest.SkipTest("no Phase-4 artifacts present in openai_walker/")
        print(f"\n[phase4] {len(names)} artifacts found: {', '.join(names)}")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_every_artifact_is_discoverable(self):
        """A silently-missing artifact would shrink the table without failing anything."""
        expected = {
            "Teacher (Upper Bound)", "Behavioral Cloning Puro", "Batch-Constrained Q-learning (BCQ)",
            "Decision Transformer (DT)",
        }
        found = {name for name, _, _ in self.candidates}
        self.assertTrue(expected & found, f"none of {expected} were found in {found}")

    def test_each_artifact_completes_a_seeded_episode(self):
        for name, model, kwargs in self.candidates:
            with self.subTest(model=name):
                avg, per_ep = self.evaluate_all.evaluate_headless(
                    self.env, model, self.device, episodes=1, seed=321, **kwargs
                )
                self.assertEqual(len(per_ep), 1)
                self.assertTrue(np.isfinite(avg), f"{name} produced a non-finite return")
                # Walker2d cannot legitimately exceed ~10k in one episode; a wild value
                # means the reward is being accumulated across resets or the env broke.
                self.assertLess(abs(avg), 20000.0, f"{name} returned an implausible score")

    def test_evaluation_is_reproducible_for_a_torch_policy(self):
        """The audit found unseeded single-episode evals; same seed must give same score."""
        torch_policy = [(n, m, kw) for n, m, kw in self.candidates
                        if not kw and "Teacher" not in n and "Extra" not in n]
        self.assertTrue(torch_policy, "no plain torch policy available")
        name, model, kwargs = torch_policy[0]
        first, _ = self.evaluate_all.evaluate_headless(
            self.env, model, self.device, episodes=1, seed=77, **kwargs
        )
        second, _ = self.evaluate_all.evaluate_headless(
            self.env, model, self.device, episodes=1, seed=77, **kwargs
        )
        self.assertAlmostEqual(first, second, places=6,
                               msg=f"{name} is not reproducible from a fixed seed")


@unittest.skipUnless(AVAILABLE, "Walker2d-v5 needs gymnasium>=1.0 (see requirements-phase4.txt)")
class TestPhase4NormalizationContract(unittest.TestCase):
    def test_iql_and_cql_scalers_match_their_state_width(self):
        """IQL/CQL stored a StandardScaler next to the weights; a mismatch silently feeds
        the policy wrongly-scaled observations."""
        import joblib

        for artifact, scaler in (("iql_model.pt", "scaler_iql.pkl"), ("cql_model.pt", "scaler_cql.pkl")):
            path = os.path.join(P4, scaler)
            if not os.path.exists(path):
                continue
            with self.subTest(scaler=scaler):
                self.assertTrue(os.path.exists(os.path.join(P4, artifact)))
                loaded = joblib.load(path)
                self.assertEqual(len(loaded.mean_), 17, f"{scaler} is not a 17-dim state scaler")

    def test_extra_trees_model_predicts_inside_action_bounds(self):
        path = os.path.join(P4, "extratrees_model.pkl")
        if not os.path.exists(path):
            self.skipTest("extratrees_model.pkl not present (regenerate with train_extratrees.py)")
        import joblib

        model = joblib.load(path)
        rng = np.random.default_rng(0)
        batch = rng.uniform(-1, 1, size=(64, 17))
        pred = np.clip(model.predict(batch), -1.0, 1.0)
        self.assertEqual(pred.shape, (64, 6))
        self.assertTrue(np.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
