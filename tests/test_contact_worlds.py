"""Contract tests for the selectable contact worlds, and for the stamp that makes them honest.

Three things are pinned here, in order of how bad each would be if it were wrong.

`env_version_of` is asked for the version of a *vector* env by every checkpoint save site, and a
`SyncVectorEnv` carries no `_env_version` - so before this module existed, a run in `euler`, `fast` or
a rewritten collision set stamped itself as the published v9 MDP. The first test class is that
regression: the stamp has to reach the sub-env, and a checkpoint has to be unable to lie about which
world produced it.

Second, the worlds themselves: `contact_world="explicit_full"` must load the model with the 126
declared pairs and nothing colliding automatically, `gait_reduced` the 20, and `published` the
committed XML unchanged. An option that silently trains in the default world while stamping itself as
something else is worse than no option, and that is exactly what the smoke run caught.

Third, the artifact the adoption rests on: the long-horizon equivalence cell. The short 6x300-step
proof said the explicit set is the same MDP; the policy-length replay says it harder, and the test
recomputes the claim from the committed JSON rather than from this file's memory of it.
"""

import json
import os
import sys
import unittest

import gymnasium as gym
import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import envs.walker_ragdoll_env as env_module  # noqa: E402
from envs.walker_ragdoll_env import (CONTACT_WORLDS, CONTACT_WORLD_SUFFIX, ENV_VERSION,  # noqa: E402
                                     env_version_of, resolve_contact_world)


def _make(world, **kwargs):
    return gym.make("WalkerRagdoll-v0", task_phase="target", contact_world=world, **kwargs)


class TestTheStampReachesTheSubEnv(unittest.TestCase):
    """The bug: a vector env has no version of its own, so every save site read the constant."""

    def _vec(self, world="published", n=2):
        # One world per sub-env, built from a captured default arg: `kwargs.pop` inside the lambda
        # would empty the shared dict on the first call and quietly build the second sub-env in the
        # published world, which is a test that passes for the wrong reason.
        return gym.vector.SyncVectorEnv([(lambda w=world: _make(w)) for _ in range(n)])

    def test_a_single_env_stamps_its_world(self):
        for world in CONTACT_WORLDS:
            env = _make(world)
            try:
                expected = (ENV_VERSION if world == "published"
                            else f"{ENV_VERSION}_{CONTACT_WORLD_SUFFIX[world]}")
                self.assertEqual(env_version_of(env), expected, world)
            finally:
                env.close()

    def test_a_vector_env_stamps_its_world_too(self):
        # This is the call shape the trainer's save sites use: env_version_of(envs) where envs is the
        # vector env. Returning the module constant here is what let a non-default world claim v9.
        vec = self._vec("explicit_full")
        try:
            self.assertEqual(env_version_of(vec), f"{ENV_VERSION}_pairs")
            self.assertEqual([int(e.unwrapped.model.npair) for e in vec.envs], [126, 126],
                             "the sub-envs are not all in the world the test claims to build")
        finally:
            vec.close()

    def test_a_preset_inside_a_vector_env_is_stamped_as_well(self):
        vec = gym.vector.SyncVectorEnv([
            (lambda: gym.make("WalkerRagdoll-v0", task_phase="target", physics_preset="euler"))
            for _ in range(2)])
        try:
            self.assertEqual(env_version_of(vec), f"{ENV_VERSION}_euler",
                             "the same hole that the contact world fell into is still open for presets")
        finally:
            vec.close()

    def test_the_parallel_backend_stamps_too(self):
        # The second instance of the same bug, and the one that matters: the published runs use
        # num_envs=32, which builds a ParallelVectorEnv whose sub-environments live in other
        # processes. There is nothing to walk into there, so the version has to be stamped by the
        # builder from the configuration it was handed.
        from types import SimpleNamespace

        from train_walker import build_vec_env
        args = SimpleNamespace(num_envs=2, vec_parallel_threshold=1, vec_dense_info=False,
                               reset_mode="mixed", fixed_reset_probability=0.25,
                               upright_reset_probability=0.15, fallen_velocity_scale=0.35,
                               task_phase="target", target_forward_velocity=1.2,
                               physics_preset="v9", contact_world="explicit_full",
                               capture_video=False)
        vec = build_vec_env(args, "stamp_parallel_test", num_envs=2)
        try:
            self.assertEqual(env_version_of(vec), f"{ENV_VERSION}_pairs")
        finally:
            vec.close()

    def test_the_published_world_still_stamps_the_constant(self):
        # The other direction: the fix must not start inventing suffixes for the default path, or
        # every committed checkpoint stops matching its own resume guard.
        vec = self._vec("published")
        try:
            self.assertEqual(env_version_of(vec), ENV_VERSION)
        finally:
            vec.close()

    def test_a_spec_without_an_env_falls_back_to_the_constant(self):
        self.assertEqual(env_version_of(None), ENV_VERSION)


class TestTheWorldsAreWhatTheySay(unittest.TestCase):
    def _model(self, world):
        env = _make(world)
        try:
            return env.unwrapped.model
        finally:
            env.close()

    def test_each_world_loads_the_pair_set_it_names(self):
        published = self._model("published")
        full = self._model("explicit_full")
        gait = self._model("gait_reduced")
        self.assertEqual(int(published.npair), 0)
        self.assertEqual(int(full.npair), 126)
        self.assertEqual(int(gait.npair), 20)
        collidable = lambda m: sum(1 for g in range(m.ngeom)
                                   if m.geom_contype[g] or m.geom_conaffinity[g])
        self.assertEqual(collidable(published), 18)
        self.assertEqual(collidable(full), 0, "the automatic filter is still on in the explicit world")
        self.assertEqual(collidable(gait), 0)

    def test_the_geoms_bodies_and_actuators_are_untouched(self):
        published = self._model("published")
        for world in ("explicit_full", "gait_reduced"):
            m = self._model(world)
            self.assertEqual((m.nbody, m.ngeom, m.nu, m.nq, m.nv),
                             (published.nbody, published.ngeom, published.nu,
                              published.nq, published.nv), world)
            np.testing.assert_allclose(m.geom_size, published.geom_size, atol=1e-12)
            self.assertEqual(float(m.opt.timestep), float(published.opt.timestep))
            self.assertEqual(int(m.opt.integrator), int(published.opt.integrator))

    def test_an_unknown_world_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            _make("nope").reset(seed=0)

    def test_resolve_returns_an_absolute_path_and_published_returns_none(self):
        self.assertIsNone(resolve_contact_world("published"))
        path = resolve_contact_world("gait_reduced")
        self.assertTrue(os.path.isabs(path) and os.path.exists(path))
        with self.assertRaises(ValueError):
            resolve_contact_world("nope")

    def test_an_explicit_xml_file_is_not_overwritten_by_the_option(self):
        # The clip tool passes a path it built; silently replacing it would make the picture and the
        # number disagree, so `xml_file` has to win.
        path = resolve_contact_world("gait_reduced")
        env = gym.make("WalkerRagdoll-v0", task_phase="target", xml_file=path,
                       contact_world="published")
        try:
            self.assertEqual(int(env.unwrapped.model.npair), 20)
        finally:
            env.close()


class TestTheTrainerAndEvaluatorCarryTheFlag(unittest.TestCase):
    def test_env_common_kwargs_passes_the_world_through(self):
        from types import SimpleNamespace
        from train_walker import current_env_version, env_common_kwargs

        args = SimpleNamespace(reset_mode="mixed", fixed_reset_probability=0.25,
                               upright_reset_probability=0.15, fallen_velocity_scale=0.35,
                               task_phase="target", target_forward_velocity=1.2,
                               physics_preset="v9", contact_world="explicit_full",
                               target_curriculum=False)
        self.assertEqual(env_common_kwargs(args)["contact_world"], "explicit_full")
        self.assertEqual(current_env_version(args), f"{ENV_VERSION}_pairs")

    def test_the_default_arguments_keep_the_published_world(self):
        # A caller with no opinion (an older harness, a test namespace) must get the committed world
        # and the committed version string, not an error and not a rewritten stamp.
        from types import SimpleNamespace
        from train_walker import current_env_version, env_common_kwargs

        args = SimpleNamespace(reset_mode="mixed", fixed_reset_probability=0.25,
                               upright_reset_probability=0.15, fallen_velocity_scale=0.35,
                               task_phase="target", target_forward_velocity=1.2)
        self.assertEqual(env_common_kwargs(args)["contact_world"], "published")
        self.assertEqual(current_env_version(args), ENV_VERSION)

    def test_both_scripts_expose_the_flag(self):
        for name in ("train_walker.py", "eval_phase1.py"):
            with open(os.path.join(ROOT, name), encoding="utf-8") as handle:
                src = handle.read()
            self.assertIn("--contact-world", src, f"{name} has no --contact-world")
        with open(os.path.join(ROOT, "eval_phase1.py"), encoding="utf-8") as handle:
            self.assertIn("CONTACT_WORLD_SUFFIX", handle.read(),
                          "the evaluator no longer builds the version string the checkpoint carries")


class TestTheLongEquivalenceArtifact(unittest.TestCase):
    """What the adoption rests on: a policy-length replay, not a 300-step spot check."""

    ART = os.path.join(ROOT, "benchmarks", "contact_equivalence_long.json")

    def setUp(self):
        with open(self.ART, encoding="utf-8") as handle:
            self.cells = json.load(handle)["long_verify"]["cells"]

    def test_the_explicit_set_is_bit_identical_over_the_whole_rollout(self):
        cell = self.cells["explicit_full"]
        self.assertEqual(cell["contact_set_mismatch_steps"], 0)
        self.assertEqual(cell["max_abs_qpos_diff"], 0.0,
                         "the explicit world now diverges from the published one, so it is no longer "
                         "the same MDP and the adoption argument is gone")
        self.assertEqual(cell["max_abs_qvel_diff"], 0.0)
        self.assertIsNone(cell["first_qpos_divergence_step"])
        self.assertGreaterEqual(cell["steps"], 2000)

    def test_the_reduced_set_diverges_immediately_and_hard(self):
        cell = self.cells["gait_reduced"]
        self.assertGreater(cell["contact_set_mismatch_steps"], cell["steps"] // 2)
        self.assertGreater(cell["max_abs_qpos_diff"], 1.0)
        self.assertIsNotNone(cell["first_qpos_divergence_step"])
        self.assertLess(cell["first_qpos_divergence_step"], 10)


if __name__ == "__main__":
    unittest.main()
