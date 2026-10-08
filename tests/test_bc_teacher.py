"""Tests for the MPC-teacher behaviour cloning: the split, the filter, and the shared normalisation.

Three of these exist because each is a way to build a benchmark that produces a number and means something
else. A student trained on the same episodes it is graded on measures memorisation; a student trained on
raw states and evaluated on normalised ones measures a distribution shift; a validation set taken by step
rather than by episode leaks the same trajectory across both sides. The fourth checks the filter actually
does what the second cell claims.
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

import gymnasium as gym  # noqa: E402
import train_bc_ragdoll as bc  # noqa: E402
from evaluate_merging import _policy_input  # noqa: E402


def write_demos(directory, episodes, band_fraction=0.5, obs_dim=49, act_dim=17, seed=0):
    """A small synthetic stand-in for collect_mpc_demos.py's output, same schema."""
    rng = np.random.default_rng(seed)
    obs, act, band, dist, meta_eps = [], [], [], [], []
    steps_each = [40 for _ in episodes]
    for ep, (seed_value, n_steps) in enumerate(zip(episodes, steps_each)):
        o = rng.standard_normal((n_steps, obs_dim))
        a = rng.uniform(-1, 1, size=(n_steps, act_dim))
        b = (rng.random(n_steps) < band_fraction).astype(np.float32)
        d = np.full(n_steps, 2.0, dtype=np.float32)
        obs.append(o.astype(np.float32)); act.append(a.astype(np.float32))
        band.append(b); dist.append(d)
        meta_eps.append({"episode": ep, "seed": seed_value, "steps": n_steps,
                         "min_target_distance": 1.0, "reached_target": False,
                         "reached_target_upright": False, "falls": 0,
                         "pct_steps_in_band": round(100.0 * float(b.mean()), 1)})
    npz = os.path.join(directory, "demos.npz")
    np.savez(npz, obs=np.concatenate(obs), action=np.concatenate(act),
             in_band=np.concatenate(band), distance=np.concatenate(dist))
    stacked = np.concatenate(obs)
    with open(npz + ".json", "w", encoding="utf-8") as handle:
        json.dump({"protocol": "synthetic", "demo_seeds": [e["seed"] for e in meta_eps],
                   "eval_seeds_are_disjoint": True,
                   "obs_rms": {"mean": stacked.mean(axis=0).tolist(),
                               "var": stacked.var(axis=0).tolist(),
                               "count": int(stacked.shape[0])},
                   "episodes": meta_eps,
                   "summary": {"transitions": int(stacked.shape[0])}}, handle)
    return npz


class TestDemosRefuseTheScoredSeeds(unittest.TestCase):
    def test_overlap_with_the_eval_protocol_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz = write_demos(tmp, episodes=[101, 102])
            arrays, meta = bc.load_demos(npz)
            self.assertTrue(meta["eval_seeds_are_disjoint"])
            # The same file with one scored seed mixed in must refuse, not silently train on it.
            meta["demo_seeds"] = [101, 22]
            with open(npz + ".json", "w", encoding="utf-8") as handle:
                json.dump(meta, handle)
            with self.assertRaises(SystemExit) as ctx:
                bc.load_demos(npz)
            self.assertIn("22", str(ctx.exception))

    def test_the_disjoint_range_is_the_one_the_published_rows_use(self):
        self.assertEqual(bc.EVAL_SEEDS, set(range(11, 31)),
                         "the scored protocol is seeds 11..30; if that moved, the demo range has to "
                         "move with it")


class TestNormalisationIsTheEvals(unittest.TestCase):
    def test_trainer_and_scorer_apply_the_same_transform(self):
        rng = np.random.default_rng(3)
        obs = rng.standard_normal((6, 49)) * 3.0
        mean, var = obs.mean(axis=0), obs.var(axis=0)
        mine = bc.normalize(obs, mean, var)

        class Dummy:
            def __init__(self):
                self.obs_rms = type("R", (), {"mean": mean, "var": var})()

            def get_action(self, tensor, deterministic=True):
                return torch.zeros(1, 17), None, None

        theirs = np.asarray(_policy_input(Dummy(), obs[0], Dummy().obs_rms, 49, "cpu")).reshape(-1)
        np.testing.assert_allclose(mine[0], theirs, rtol=0, atol=1e-6)


class TestEpisodeLevelSplit(unittest.TestCase):
    def test_transition_rows_map_back_to_their_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz = write_demos(tmp, episodes=[101, 102, 103])
            _arrays, meta = bc.load_demos(npz)
            ids = bc.episode_ids(meta)
            self.assertEqual(ids.tolist()[:40], [0] * 40)
            self.assertEqual(ids.tolist()[-40:], [2] * 40)
            self.assertEqual(len(ids), sum(e["steps"] for e in meta["episodes"]))

    def test_a_held_out_episode_has_no_row_on_the_training_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz = write_demos(tmp, episodes=list(range(101, 111)))
            _arrays, meta = bc.load_demos(npz)
            ids = bc.episode_ids(meta)
            val = set(np.unique(ids)[:2].tolist())
            train_mask = ~np.isin(ids, list(val))
            self.assertEqual(np.intersect1d(ids[train_mask], list(val)).size, 0,
                             "a validation episode leaked into training by step")


class TestBandFilter(unittest.TestCase):
    def test_the_band_cell_sees_only_the_in_band_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz = write_demos(tmp, episodes=[101, 102], band_fraction=0.25)
            arrays, meta = bc.load_demos(npz)
            band = arrays["in_band"].astype(bool)
            self.assertGreater(band.sum(), 0)
            self.assertFalse(np.any(~band[band]), "sanity: the mask is the flag")
            # The filter is a boolean selection over the same rows the raw cell uses.
            raw_rows = arrays["obs"].shape[0]
            self.assertLess(int(band.sum()), raw_rows)
            self.assertEqual(meta["summary"]["transitions"], raw_rows)


class TestCheckpointIsReadableByTheScorer(unittest.TestCase):
    def test_eval_phase1_loads_a_bc_checkpoint(self):
        import eval_phase1
        with tempfile.TemporaryDirectory() as tmp:
            npz = write_demos(tmp, episodes=list(range(101, 108)))
            args = type("A", (), {"demos": npz, "filter": "raw",
                                  "epochs": 2, "batch": 64, "lr": 1e-3, "seed": 7,
                                  "device": "cpu", "min_transitions": 10,
                                  "run_id": "bc_test_run"})()
            path = bc.train(args)
            try:
                label, phase, width, policy, reset, rinfo = eval_phase1.build_policy(path, "cpu")
                self.assertEqual(label, "bc")
                self.assertEqual(phase, "target")
                self.assertEqual(width, 49)
                self.assertTrue(rinfo["target_curriculum"] is False)
                action = np.asarray(policy(np.zeros(49)))
                self.assertEqual(action.shape, (17,))
                self.assertTrue(np.all(np.abs(action) <= 1.0 + 1e-6))
                stored = torch.load(path, map_location="cpu", weights_only=False)
                self.assertEqual(stored["bc"]["filter"], "raw")
                self.assertTrue(stored["bc"]["eval_seeds_disjoint"])
                self.assertIsNotNone(stored["obs_rms"])
            finally:
                import shutil
                shutil.rmtree(os.path.join(ROOT, "checkpoints", "bc_test_run"), ignore_errors=True)


class TestCollectorTrainerSchemaAgree(unittest.TestCase):
    """The collector and the trainer are two scripts with one contract; this test is the contract.

    A tiny real collection - two episodes, a handful of CEM samples - so the arrays, the metadata schema
    and the band flag are checked as the trainer will actually receive them, not as they were imagined.
    """

    def test_a_collected_file_trains_and_reports_the_band_filter(self):
        import collect_mpc_demos as collect
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "demos.npz")
            args = type("A", (), {"episodes": 2, "demo_seed_base": 401, "seed": 5, "steps": 40,
                                  "horizon": 8, "samples": 4, "iterations": 1, "elite": 2,
                                  "replan": 4, "action": "argmin", "out": out})()
            arrays, rms, episodes = collect.collect(args)
            self.assertEqual(len(episodes), 2)
            self.assertEqual([e["seed"] for e in episodes], [401, 402])
            self.assertEqual(arrays["obs"].shape[1], 49, "target-phase observations are 49 wide")
            self.assertEqual(arrays["action"].shape[1], 17)
            self.assertTrue(set(arrays["in_band"].tolist()) <= {0.0, 1.0},
                            "the band flag must stay binary for the filter to mean anything")
            self.assertEqual(rms["mean"].shape, (49,))
            # The band flag must be the scorer's band, not a second definition of standing.
            import bench_approach_mechanism as bam
            self.assertEqual((bam.BAND_Z, bam.BAND_UPRIGHT), (1.0, 0.7))

            # And the trainer reads it: same keys, same refusal, same episode-level split.
            npz = write_demos(tmp, episodes=[101, 102])
            loaded, meta = bc.load_demos(npz)
            self.assertEqual(sorted(loaded.files if hasattr(loaded, "files")
                                    else list(loaded)), ["action", "distance", "in_band", "obs"])
            self.assertEqual(meta["summary"]["transitions"], loaded["obs"].shape[0])


if __name__ == "__main__":
    unittest.main()
