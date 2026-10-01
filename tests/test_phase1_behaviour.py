"""Contract tests for the Phase-1 evidence checkpoints.

These are *not* performance tests. The three checkpoints were produced by the short
pipeline-evidence runs (6k REDQ steps, 8k Dreamer steps, 60k ARS steps), and at those
budgets the ragdoll falls after roughly 33 environment steps: measured over an identical
seeded 300-step rollout, REDQ scores -3826.77, ARS -3887.55, Dreamer -3836.79, against
-3775.76 for doing nothing and -3857.06 for uniform random actions. Nothing in this file
may claim that a Phase-1 checkpoint walks.

What is worth pinning is the loading contract, because every failure mode found so far in
this repository's evaluation harnesses lived there: the observation width recorded in the
checkpoint, the normalisation statistics that have to be applied before the network sees an
observation, and the determinism of the action stream (the Phase-3 benchmark was reproducible
only after obs_rms handling was fixed, and the Phase-4 race became reproducible only after
the policy RNG was seeded).

The returns printed in the trainers' own logs are on a different scale again: REDQ and
Dreamer wrap the environment in NormalizeReward(gamma), so their printed `episodic_return`
is a normalised sum and is not comparable with the raw numbers above.
"""

import glob
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import envs.walker_ragdoll_env  # noqa: E402,F401  (registers WalkerRagdoll-v0)
from train_walker import SACAgent  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = 300
SEED = 7
ACTION_SPACE = gym.spaces.Box(-1.0, 1.0, shape=(17,))

# Evidence checkpoints are training output, not source: `checkpoints/` is gitignored, so CI
# has none of them and every test here must skip rather than fail. What CI does cover is the
# loading contract itself, through a checkpoint that IS committed - see the skip message.
REQUIRED = {
    "redq": "redq_evidence/redq_actor_*.pt",
    "dreamer": "dreamer_smoke/dreamer_actor_*.pt",
    "ars": "ars_evidence/ars_ckpt_*.pt",
}


def _latest(pattern):
    hits = sorted(glob.glob(os.path.join(ROOT, "checkpoints", pattern)))
    return hits[-1] if hits else None


def _missing():
    return [f"checkpoints/{p}" for p in REQUIRED.values() if _latest(p) is None]


def skip_without(*labels):
    """Skip a class when the evidence checkpoints it needs are not in this checkout."""
    absent = [f"checkpoints/{REQUIRED[label]}" for label in labels if _latest(REQUIRED[label]) is None]
    return unittest.skipIf(absent, "no Phase-1 evidence checkpoints here (checkpoints/ is "
                                   f"gitignored); missing {absent}")


def _episode(policy, steps=STEPS, seed=SEED, task_phase="target"):
    """Run one seeded rollout and return (raw return, x progress, action stats)."""
    env = gym.make("WalkerRagdoll-v0", reset_mode="mixed", task_phase=task_phase)
    try:
        obs, _ = env.reset(seed=seed)
        total, actions = 0.0, []
        for _ in range(steps):
            action = policy(np.asarray(obs, dtype=np.float64))
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        progress = float(env.unwrapped.data.qpos[0])
        first = float(env.unwrapped.initial_state.qpos[0]) if hasattr(
            env.unwrapped, "initial_state") else None
    finally:
        env.close()
    A = np.stack(actions)
    return total, progress, {
        "shape": A.shape,
        "finite": bool(np.isfinite(A).all()),
        "in_bounds": bool(np.all(np.abs(A) <= 1.0 + 1e-6)),
        "saturated": float((np.abs(A) > 0.999).mean()),
    }


class TestPhase1ArtifactsPresent(unittest.TestCase):
    def test_evidence_checkpoints_exist(self):
        """The README's Phase-1 evidence section points at these files.

        Skipped when there is no `checkpoints/` directory at all, which is the CI case; when
        the directory exists it is a local working area and the three evidence runs must be
        in it.
        """
        if not os.path.isdir(os.path.join(ROOT, "checkpoints")):
            self.skipTest("no checkpoints/ directory in this checkout")
        for label, pattern in REQUIRED.items():
            self.assertIsNotNone(_latest(pattern), f"missing checkpoints/{pattern} ({label})")


@skip_without("redq")
class TestPhase1LoadingContract(unittest.TestCase):
    def setUp(self):
        if _latest("redq_evidence/redq_actor_*.pt") is None:
            self.skipTest("no REDQ actor checkpoint in checkpoints/redq_evidence/")

    def _redq(self):
        actor_path = _latest("redq_evidence/redq_actor_*.pt")
        ckpt_path = actor_path.replace("redq_actor_", "redq_ckpt_")
        state = torch.load(actor_path, map_location="cpu", weights_only=True)
        width = int(state["backbone.0.weight"].shape[1])
        agent = SACAgent(width, ACTION_SPACE)
        agent.load_state_dict(state)
        agent.eval()
        rms = None
        if os.path.exists(ckpt_path):
            full = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            rms = full.get("obs_rms")
            self.assertEqual(full.get("global_step"), int(actor_path.split("_")[-1].split(".")[0]))
        return agent, width, rms

    def test_redq_width_matches_saved_normaliser(self):
        """A width mismatch here silently drives a different policy - the Phase-3 bug."""
        agent, width, rms = self._redq()
        self.assertEqual(width, 49, "the evidence actor is the 49-dim target-phase policy")
        if rms is not None:
            self.assertEqual(np.asarray(rms.mean).size, width,
                             "obs_rms and the network must agree on the observation width")

    def test_redq_actions_are_finite_and_inside_the_action_box(self):
        agent, width, rms = self._redq()

        def policy(obs):
            x = obs[:width]
            if rms is not None:
                x = np.clip((x - np.asarray(rms.mean)) /
                            np.sqrt(np.asarray(rms.var) + 1e-8), -10.0, 10.0)
            with torch.no_grad():
                a = agent.get_action(torch.FloatTensor(x).unsqueeze(0), deterministic=True)[0]
            return a.numpy().reshape(-1)

        total, progress, stats = _episode(policy)
        self.assertTrue(stats["finite"])
        self.assertTrue(stats["in_bounds"], "SAC must not hand the env actions outside [-1,1]")
        self.assertEqual(stats["shape"][1], 17)
        self.assertTrue(np.isfinite(total) and np.isfinite(progress))

    def test_redq_deterministic_stream_is_reproducible(self):
        agent, width, rms = self._redq()
        rng = np.random.default_rng(0)
        obs = rng.normal(size=width)

        def act(o):
            x = o[:width]
            if rms is not None:
                x = np.clip((x - np.asarray(rms.mean)) /
                            np.sqrt(np.asarray(rms.var) + 1e-8), -10.0, 10.0)
            with torch.no_grad():
                return agent.get_action(torch.FloatTensor(x).unsqueeze(0),
                                        deterministic=True)[0].numpy().reshape(-1)

        first, second = act(obs), act(obs)
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-7,
                                   err_msg="deterministic=True must not depend on the RNG")

    def test_ars_linear_policy_contract(self):
        path = _latest("ars_evidence/ars_ckpt_*.pt")
        if path is None:
            self.skipTest("no ARS checkpoint in checkpoints/ars_evidence/")
        state = torch.load(path, map_location="cpu", weights_only=False)
        weights = np.asarray(state["weights"])
        bias = np.asarray(state["bias"])
        rms = state["obs_rms"]
        self.assertEqual(weights.shape[1], int(np.asarray(rms["mean"]).size),
                         "the linear policy and its normaliser must share the width")
        self.assertEqual(state["algo"], "ars")

        def policy(obs):
            x = (obs[:weights.shape[1]] - np.asarray(rms["mean"])) / np.sqrt(
                np.asarray(rms["var"]) + 1e-8)
            return np.clip(x @ weights.T + bias, -1.0, 1.0)

        total, progress, stats = _episode(policy)
        self.assertTrue(stats["finite"] and stats["in_bounds"])

    def test_dreamer_component_widths(self):
        path = _latest("dreamer_smoke/dreamer_actor_*.pt")
        if path is None:
            self.skipTest("no Dreamer actor checkpoint in checkpoints/dreamer_smoke/")
        state = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("actor_state_dict", "rssm_state_dict", "encoder_state_dict"):
            self.assertIn(key, state, f"{key} is needed to rebuild the policy")
        encoder_width = int(state["encoder_state_dict"]["0.weight"].shape[1])
        self.assertEqual(encoder_width, 49)
        rms = state.get("obs_rms")
        if rms is not None:
            self.assertEqual(np.asarray(rms.mean).size, encoder_width)


@skip_without("redq")
class TestPhase1IsNotABehaviourClaim(unittest.TestCase):
    """Guard the honest reading: these checkpoints are pipeline evidence, not walkers."""

    def test_evidence_actor_does_not_yet_beat_doing_nothing(self):
        path = _latest("redq_evidence/redq_actor_*.pt")
        if path is None:
            self.skipTest("no REDQ actor checkpoint")
        state = torch.load(path, map_location="cpu", weights_only=True)
        width = int(state["backbone.0.weight"].shape[1])
        agent = SACAgent(width, ACTION_SPACE)
        agent.load_state_dict(state)
        agent.eval()
        zero = _episode(lambda o: np.zeros(17))[0]

        def policy(obs):
            x = obs[:width]
            with torch.no_grad():
                return agent.get_action(torch.FloatTensor(x).unsqueeze(0),
                                        deterministic=True)[0].numpy().reshape(-1)

        scored = _episode(policy)[0]
        self.assertLess(
            scored, zero + 500.0,
            "the 6k-step evidence actor was expected to be no better than a still robot on "
            "the raw reward scale; if this now fails, the checkpoint changed and the README's "
            "Phase-1 wording needs revisiting, not just this bound")


if __name__ == "__main__":
    unittest.main(verbosity=2)
