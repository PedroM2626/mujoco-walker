"""The reward-provenance invariant: evaluators score a policy under the reward it was trained with.

`train_walker.make_env` passes `TRAINING_REWARD_KWARGS` into every sub-environment, and every
Phase-1 trainer builds its environments through it - SAC/TD3/PPO call it directly, and
`train_dreamer.py`, `train_redq.py` and `train_ars.py` import it. Until 2026-10-04 nothing wrote
that fact down, so `reward_kwargs_for` had to guess for a checkpoint that did not record its
reward, and it guessed "Dreamer/REDQ/ARS trained against the environment defaults". That was
false, and the cost was that every published return for those three algorithms was computed
against a reward nobody optimised.

The two halves of the invariant are pinned separately, because the original failure was in the
reasoning between them: (1) make_env really does apply the shaping, which is what makes the
legacy fallback in `reward_kwargs_for` correct rather than convenient; (2) every trainer records
it anyway, so no future evaluator has to reason at all.
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from envs.reward_shaping import TRAINING_REWARD_KWARGS, reward_kwargs_for
from train_walker import make_env
from train_ars import save_ars_checkpoint
from train_dreamer import save_dreamer_checkpoint
from train_redq import save_redq_checkpoint

ENV_ID = "WalkerRagdoll-v0"


def _tiny():
    return nn.Linear(2, 2)


def _opt():
    return torch.optim.Adam(_tiny().parameters(), lr=1e-3)


class TestMakeEnvAppliesTheShaping(unittest.TestCase):
    """Half 1: the trainers do not see the environment's reward defaults."""

    def test_every_shaping_kwarg_reaches_the_built_env(self):
        env = make_env(ENV_ID, 0, False, "test-run", task_phase="target")()
        self.addCleanup(env.close)
        for name, value in TRAINING_REWARD_KWARGS.items():
            attr = "_" + name
            self.assertTrue(
                hasattr(env.unwrapped, attr),
                f"WalkerRagdoll no longer stores `{name}` as `{attr}`, so this test would pass "
                f"vacuously against a renamed attribute - update it.",
            )
            self.assertEqual(getattr(env.unwrapped, attr), value, f"{name} was not applied")

    def test_the_shaping_is_not_the_default(self):
        """Guards against the test above going vacuous if the two ever coincide."""
        env = gym.make(ENV_ID, task_phase="target")
        self.addCleanup(env.close)
        differing = {
            name: (getattr(env.unwrapped, "_" + name), value)
            for name, value in TRAINING_REWARD_KWARGS.items()
            if getattr(env.unwrapped, "_" + name) != value
        }
        self.assertTrue(
            differing,
            "TRAINING_REWARD_KWARGS now equals the environment defaults, so `reward_kwargs_for` "
            "no longer has anything to resolve and this file's premise is stale.",
        )
        self.assertEqual(differing["standing_reward"], (50.0, 0.0))

    def test_a_legacy_checkpoint_falls_back_to_the_shaping(self):
        """Half of the invariant that was wrong: no recorded reward means trained shaped, not default."""
        kwargs, source = reward_kwargs_for({"algo": "dreamer"})
        self.assertEqual(kwargs, dict(TRAINING_REWARD_KWARGS))
        self.assertIn("make_env", source)
        self.assertIn("8d37846", source)

    def test_a_recorded_reward_wins_over_the_fallback(self):
        kwargs, source = reward_kwargs_for({"reward_kwargs": {"standing_reward": 7.0}})
        self.assertEqual(kwargs, {"standing_reward": 7.0})
        self.assertEqual(source, "checkpoint")

    def test_target_forward_velocity_is_carried_into_the_kwargs(self):
        kwargs, _ = reward_kwargs_for({"target_forward_velocity": 1.25})
        self.assertEqual(kwargs["target_forward_velocity"], 1.25)
        recorded, _ = reward_kwargs_for(
            {"reward_kwargs": {"target_forward_velocity": 0.5}, "target_forward_velocity": 1.25})
        self.assertEqual(recorded["target_forward_velocity"], 0.5)


class TestTrainersRecordTheShaping(unittest.TestCase):
    """Half 2: every Phase-1 trainer writes the reward down, so nobody has to infer it."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="reward-provenance-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _assert_records(self, path):
        # weights_only=False because a resume checkpoint carries RNG and obs-rms state that the
        # restricted unpickler rejects (the same reason eval_phase1.py loads this way). This file
        # was written by the line above it in the same test, so it is not an untrusted input.
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(state["reward_kwargs"], dict(TRAINING_REWARD_KWARGS))
        kwargs, source = reward_kwargs_for(state)
        self.assertEqual(source, "checkpoint", f"{path} did not resolve from its own record")
        self.assertEqual(kwargs["standing_reward"], TRAINING_REWARD_KWARGS["standing_reward"])

    def test_dreamer_records_it_in_both_files(self):
        path = os.path.join(self.tmp, "dreamer_ckpt_10.pt")
        save_dreamer_checkpoint(
            path, 10,
            SimpleNamespace(rssm=_tiny(), encoder=_tiny(), state_dict=lambda: {}),
            _tiny(), _tiny(), _opt(), _opt(), _opt(),
            SimpleNamespace(), "target", 0.8,
        )
        self._assert_records(path)
        self._assert_records(path.replace("dreamer_ckpt_", "dreamer_actor_"))

    def test_redq_records_it(self):
        path = os.path.join(self.tmp, "redq_ckpt_10.pt")
        save_redq_checkpoint(
            path, 10, _tiny(), _tiny(), _tiny(), _opt(), _opt(), _opt(),
            torch.zeros(1, requires_grad=True), SimpleNamespace(), None, "target", 0.8,
            False, 2, 2, 1,
        )
        self._assert_records(path)

    def test_ars_records_it(self):
        path = os.path.join(self.tmp, "ars_ckpt_10.pt")
        save_ars_checkpoint(
            path, 10, np.zeros((2, 2)), np.zeros(2),
            SimpleNamespace(mean=np.zeros(2), var=np.ones(2), count=1.0),
            "target", 0.8,
        )
        self._assert_records(path)


if __name__ == "__main__":
    unittest.main()
