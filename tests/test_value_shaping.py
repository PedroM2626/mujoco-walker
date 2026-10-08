"""The potential-based shaping path: the arithmetic of the signal, and who is allowed to see it.

Three claims have to survive here, and each is a way the experiment could be quietly wrong. (1) The
shaping term is exactly `weight * (gamma * Phi(s') - Phi(s))`, with the successor potential zeroed at a
terminal state - that is the condition under which it telescopes and leaves the optimal policy alone, so
a sign error or a missing zero turns "support signal" into "a different task". (2) Shaping must not move
the *measured* quantities: the trajectory under a fixed seed is identical, and the episode return the
instrument reports is still the published reward, because the wrapper sits outside
`RecordEpisodeStatistics`. (3) Scoring must never load the potential at all - a shaped checkpoint is
graded by the same function as an unshaped one, and the only honest way to test that is to make loading a
potential explode and show the scorer does not care.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A test that spawns a real trainer must not write into the repository's MLflow archive.
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///" + os.path.join(
    tempfile.gettempdir(), "test_mlruns_walker.db").replace(os.sep, "/"))

import gymnasium as gym  # noqa: E402
import envs.walker_ragdoll_env  # noqa: E402,F401  (registers WalkerRagdoll-v0)
import eval_phase1  # noqa: E402
import train_walker as tw  # noqa: E402
from envs.value_potential import PotentialShaping, ValuePotential  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS_DIM, ACT_DIM = 49, 17


def write_potential(directory, index=0, name="potential.npz", obs_dim=OBS_DIM):
    """A one-layer potential whose value is `obs[index]`, written in the format the fitter exports.

    Deliberately hand-built rather than fitted: the arithmetic tests need to know Phi(s) in closed form,
    and a fitted network makes a wrong sign look plausible.
    """
    path = os.path.join(directory, name)
    w = np.zeros((obs_dim, 1), dtype=np.float64)
    w[index, 0] = 1.0
    np.savez(path, W0=w, B0=np.zeros(1, dtype=np.float64),
             obs_mean=np.zeros(obs_dim), obs_scale=np.ones(obs_dim),
             value_mean=np.array(0.0), value_scale=np.array(1.0))
    with open(path + ".json", "w", encoding="utf-8") as handle:
        json.dump({"demos": "synthetic", "arch": [obs_dim, 1], "obs_dim": obs_dim, "hidden": [],
                   "epochs": 1, "selected_epoch": 1, "clamp_value": [-1e9, 1e9],
                   "value_mean": 0.0, "value_scale": 1.0, "holdout_episodes": [0],
                   "holdout_seeds": [999], "val_rows": 1, "train_rows": 1,
                   "r2_holdout": 1.0, "val_mse_holdout": 0.0, "r2_ridge_same_split": 1.0,
                   "r2_ridge_raw_inputs": 1.0, "history": []}, handle)
    return path


class ScriptedEnv(gym.Env):
    """An env that replays a script of (obs, reward, terminated, truncated) and resets to `start`."""

    def __init__(self, script, start):
        super().__init__()
        self.script, self.start = script, np.asarray(start, dtype=np.float64)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (len(self.start),), np.float64)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float64)
        self.i = 0

    def reset(self, *, seed=None, options=None):
        self.i = 0
        return self.start.copy(), {}

    def step(self, action):
        obs, reward, terminated, truncated = self.script[self.i]
        self.i += 1
        return np.asarray(obs, dtype=np.float64), float(reward), terminated, truncated, {}


class TestShapingArithmetic(unittest.TestCase):
    def setUp(self):
        self.phi = lambda obs: float(obs[0]) * 10.0
        self.start = np.array([1.0] + [0.0] * 6)
        self.script = [
            ([2.0, 0, 0, 0, 0, 0, 0, 0], 1.0, False, False),   # ordinary step
            ([3.0, 0, 0, 0, 0, 0, 0, 0], 1.0, False, True),    # TimeLimit
            ([4.0, 0, 0, 0, 0, 0, 0, 0], 1.0, True, False),    # a real terminal state
        ]

    def wrap(self, weight=2.0, gamma=0.5):
        return PotentialShaping(ScriptedEnv(self.script, self.start), self.phi, weight, gamma)

    def test_an_ordinary_step_is_the_discounted_difference(self):
        env = self.wrap()
        env.reset()
        obs, reward, _t, _tr, _i = env.step(0.0)
        # Phi(s) = 10 at the start state, Phi(s') = 20 at the first step.
        self.assertAlmostEqual(reward, 1.0 + 2.0 * (0.5 * 20.0 - 10.0), places=9)

    def test_a_terminal_state_scores_the_successor_potential_as_zero(self):
        env = self.wrap()
        env.reset()
        for _ in range(2):
            env.step(0.0)
        obs, reward, terminated, _trunc, _i = env.step(0.0)
        self.assertTrue(terminated)
        # The telescoping argument needs Phi(terminal) = 0: here F = weight * (0 - Phi(s)) = 2 * (0 - 30).
        self.assertAlmostEqual(reward, 1.0 + 2.0 * (0.0 - 30.0), places=9)

    def test_a_truncation_carries_the_potential_instead_of_punishing_it(self):
        env = self.wrap()
        env.reset()
        env.step(0.0)
        obs, reward, _t, truncated, _i = env.step(0.0)
        self.assertTrue(truncated)
        # The horizon running out is not the end of the MDP, so the successor potential survives.
        self.assertAlmostEqual(reward, 1.0 + 2.0 * (0.5 * 30.0 - 20.0), places=9)


class NamespaceLike:
    """A stand-in for the parsed arguments, with only the fields shaping_config reads."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class TestPotentialFile(unittest.TestCase):
    def test_the_exported_arrays_evaluate_to_what_was_fitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_potential(tmp, index=3)
            phi = ValuePotential.load(path)
            obs = np.arange(OBS_DIM, dtype=np.float64)
            self.assertAlmostEqual(phi(obs), 3.0, places=9)

    def test_a_file_without_its_provenance_is_refused_by_the_trainer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_potential(tmp)
            os.remove(path + ".json")
            # The arrays still evaluate, so a bare load cannot tell the difference. It is the trainer's
            # config check that refuses, because a result quoted without its held-out accuracy is not
            # evidence of anything.
            phi = ValuePotential.load(path)
            self.assertAlmostEqual(phi(np.ones(OBS_DIM)), 1.0, places=9)
            with self.assertRaises(SystemExit) as ctx:
                tw.shaping_config(NamespaceLike(value_potential=path, shaping_weight=1.0,
                                                shaping_gamma=None, gamma=0.99))
            self.assertIn("provenance", str(ctx.exception))


class TestShapingConfig(unittest.TestCase):
    def test_no_potential_means_no_shaping(self):
        self.assertIsNone(tw.shaping_config(NamespaceLike(value_potential="", shaping_weight=1.0,
                                                          shaping_gamma=None, gamma=0.99)))

    def test_the_gamma_defaults_to_the_run_s_own_discount(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_potential(tmp)
            cfg = tw.shaping_config(NamespaceLike(value_potential=path, shaping_weight=0.3,
                                                  shaping_gamma=None, gamma=0.987))
            self.assertAlmostEqual(cfg["gamma"], 0.987, places=9)
            self.assertAlmostEqual(cfg["weight"], 0.3, places=9)
            self.assertEqual(cfg["r2_holdout"], 1.0)
            self.assertEqual(cfg["checkpoint"], path)

    def test_a_weight_of_zero_is_refused_rather_than_run_as_a_disguised_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_potential(tmp)
            with self.assertRaises(SystemExit) as ctx:
                tw.shaping_config(NamespaceLike(value_potential=path, shaping_weight=0.0,
                                                shaping_gamma=None, gamma=0.99))
            self.assertIn("control arm", str(ctx.exception))

    def test_a_missing_potential_is_refused_with_the_command_that_makes_one(self):
        with self.assertRaises(SystemExit) as ctx:
            tw.shaping_config(NamespaceLike(value_potential="nope/potential.npz", shaping_weight=1.0,
                                            shaping_gamma=None, gamma=0.99))
        self.assertIn("fit_planner_potential.py", str(ctx.exception))


class TestScoringIsUnaffected(unittest.TestCase):
    """The claim the whole comparison rests on: shaping changes what the learner sees, not what we measure."""

    def test_the_environment_return_the_instrument_reports_is_the_unshaped_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_potential(tmp, index=0)
            shaped = tw.shaping_config(NamespaceLike(value_potential=path, shaping_weight=1.0,
                                                     shaping_gamma=None, gamma=0.99))
            envs = {}
            for name, value_shaping in (("plain", None), ("shaped", shaped)):
                envs[name] = tw.make_env("WalkerRagdoll-v0", 0, False, "shaping_test",
                                         reset_mode="mixed", task_phase="target",
                                         target_forward_velocity=1.2,
                                         reward_kwargs=dict(tw.TRAINING_REWARD_KWARGS),
                                         value_shaping=value_shaping)()
            out = {}
            for name, env in envs.items():
                _obs, _info = env.reset(seed=5)
                actions = np.random.default_rng(0).uniform(-1, 1, size=(60, ACT_DIM))
                rewards, episodes = [], []
                for a in actions:
                    _o, r, term, trunc, info = env.step(a)
                    rewards.append(r)
                    if "episode" in info:
                        episodes.append(float(np.asarray(info["episode"]["r"]).item()))
                out[name] = (rewards, episodes)
                env.close()
            plain_r, plain_ep = out["plain"]
            shaped_r, shaped_ep = out["shaped"]
            self.assertEqual(len(plain_r), len(shaped_r))
            self.assertNotEqual(plain_r, shaped_r,
                                 "the shaped env handed the learner the same reward as the plain one, "
                                 "so the signal is not reaching training")
            self.assertEqual(plain_ep, shaped_ep,
                             "the reported episode return moved, so the shaped arm is no longer scored "
                             "on the published reward")

    def test_scoring_a_checkpoint_never_loads_a_potential(self):
        """Make loading a potential explode and show the scorer still runs.

        The alternative - reading the scorer's source for the word "potential" - would pass while a future
        refactor quietly wired the shaping into evaluation, which is exactly the failure this guards.
        """
        original = ValuePotential.load
        ValuePotential.load = classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(
            AssertionError("the scorer loaded a potential; the scored reward is no longer the "
                           "published one")))
        try:
            rewards, _falls, _standing, tele = eval_phase1.score(
                lambda obs: np.zeros(ACT_DIM), 2, 11, "target", steps=40, reset_mode="mixed")
        finally:
            ValuePotential.load = original
        self.assertEqual(len(rewards), 2)
        self.assertEqual(len(tele), 2)
        self.assertTrue(all(row["steps"] <= 40 for row in tele))

    def test_a_checkpoint_s_shaping_record_survives_into_the_score_artifact(self):
        ck = {"algo": "sac_actor", "actor_state_dict": {}, "obs_rms": None}
        self.assertIsNone(eval_phase1.reward_info_for(ck)["value_shaping"])
        ck["value_shaping"] = {"checkpoint": "p.npz", "weight": 0.5, "gamma": 0.99,
                               "r2_holdout": 0.8}
        self.assertEqual(eval_phase1.reward_info_for(ck)["value_shaping"]["weight"], 0.5)


class TestTrainerProvenance(unittest.TestCase):
    """A short real run, because the fields above are the trainer's promise about its own checkpoints."""

    run_id = "vshape_test_run"

    def tearDown(self):
        shutil.rmtree(os.path.join(ROOT, "checkpoints", self.run_id), ignore_errors=True)
        runs = os.path.join(ROOT, "runs")
        if os.path.isdir(runs):
            for d in os.listdir(runs):
                if d.startswith(self.run_id):
                    shutil.rmtree(os.path.join(runs, d), ignore_errors=True)

    def command(self, tmp, path, extra=()):
        return [sys.executable, "train_walker.py", "--device", "cpu", "--run-id", self.run_id,
                "--seed", "42", "--total-timesteps", "1024", "--num-envs", "2",
                "--learning-starts", "128", "--batch-size", "64", "--buffer-size", "1024",
                # The target phase is what the potential was fitted on: 49 observations instead of 46.
                "--task-phase", "target", "--reset-mode", "mixed",
                "--value-potential", path, "--shaping-weight", "0.4", *extra]

    def run_trainer(self, tmp, path, extra=()):
        return subprocess.run(self.command(tmp, path, extra), cwd=ROOT, capture_output=True,
                              text=True, timeout=1200)

    def test_a_potential_fitted_on_another_phase_is_refused_before_any_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A recovery-phase run emits 46 observations; the potential above reads 49.
            path = write_potential(tmp)
            result = self.run_trainer(tmp, path, extra=("--task-phase", "recovery"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("was fitted on 49-wide observations", result.stderr + result.stdout)

    def test_a_shaped_run_announces_the_signal_and_records_it_in_both_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_potential(tmp)
            result = self.run_trainer(tmp, path)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            self.assertIn("[SHAPING]", result.stdout)
            self.assertIn("episode returns and scoring stay on the published reward", result.stdout)
            ckpt_dir = os.path.join(ROOT, "checkpoints", self.run_id)
            for name in os.listdir(ckpt_dir):
                ck = torch.load(os.path.join(ckpt_dir, name), map_location="cpu", weights_only=False)
                record = ck.get("value_shaping")
                self.assertIsNotNone(record, f"{name}: the checkpoint does not say the arm was shaped, "
                                   "so it could later be quoted as a control")
                self.assertAlmostEqual(record["weight"], 0.4, places=9)
                self.assertEqual(record["checkpoint"], path)
                self.assertEqual(record["r2_holdout"], 1.0)
                self.assertEqual(ck["env_version"], tw.ENV_VERSION,
                                  "shaping is not an environment change; if it were stamped into the "
                                  "version string, scoring a shaped arm would look like a different world")

    def test_resuming_a_shaped_run_with_different_shaping_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_potential(tmp)
            first = self.run_trainer(tmp, path)
            self.assertEqual(first.returncode, 0, first.stderr[-2000:])
            again = self.run_trainer(tmp, path, extra=("--resume", "--shaping-weight", "0.9"))
            self.assertNotEqual(again.returncode, 0,
                                 "the resume accepted a different shaping under a replay buffer full of "
                                 "the old one")
            self.assertIn("Value-shaping mismatch", again.stderr + again.stdout)


class TestPotentialEvidence(unittest.TestCase):
    """The fitted potential the arms will use has to be measured, not asserted."""

    FIT = os.path.join(ROOT, "benchmarks", "planner_potential.json")
    PROBE = os.path.join(ROOT, "benchmarks", "critic_teacher_benchmark.json")

    def test_the_fitted_potential_beats_the_linear_baseline_on_its_own_split(self):
        with open(self.FIT, encoding="utf-8") as handle:
            fit = json.load(handle)
        self.assertGreater(fit["r2_holdout"], 0.5)
        self.assertGreaterEqual(fit["r2_holdout"], fit["r2_ridge_same_split"],
                                 "the network no longer explains the planner's value at least as well "
                                 "as the ridge the route was justified with")

    def test_the_ridge_rederived_here_is_the_number_already_published(self):
        """Continuity with the value probe: same estimator, same split, same observations, same R2."""
        with open(self.FIT, encoding="utf-8") as handle:
            fit = json.load(handle)
        with open(self.PROBE, encoding="utf-8") as handle:
            probe = json.load(handle)["teacher_value_predictability"]
        self.assertAlmostEqual(fit["r2_ridge_raw_inputs"], probe["r2_ridge"], places=4,
                                msg="the split or the normalisation drifted from the probe that "
                                    "justified this route, so the two R2 figures are not comparable")

    def test_the_demonstration_seeds_stay_disjoint_from_the_scored_protocol(self):
        with open(self.FIT, encoding="utf-8") as handle:
            fit = json.load(handle)
        self.assertEqual(sorted(set(fit["holdout_seeds"]) & set(range(11, 31))), [],
                         "the potential is fitted on episodes the arms are scored on")


if __name__ == "__main__":
    unittest.main()
