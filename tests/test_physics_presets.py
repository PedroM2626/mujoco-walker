"""Contract tests for the physics presets added beside the published v9 world.

These assert the *mechanism*, not any measurement: which model fields a preset touches, that the
default preset touches nothing, that the version string a checkpoint records follows the preset
(otherwise an `euler` run would be stamped as the published MDP), and that the flag survives the
two hops that actually matter - pickling into a process-parallel worker, and reaching the eval
script that scores the checkpoint afterwards.

The cost and the drift of the presets are measurements and live in `benchmarks/physics_presets.json`,
produced by `bench_physics_presets.py`; nothing here re-derives them.
"""

import os
import pickle
import re
import sys
import unittest
from types import SimpleNamespace

import gymnasium as gym
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import envs.walker_ragdoll_env as env_module
from envs.walker_ragdoll_env import ENV_VERSION, apply_physics_preset, env_version_of
from train_walker import current_env_version, env_common_kwargs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make(preset, **kwargs):
    return gym.make("WalkerRagdoll-v0", physics_preset=preset, **kwargs)


class TestDefaultPresetIsThePublishedWorld(unittest.TestCase):
    """Nothing that a committed number depends on may move because this feature exists."""

    def setUp(self):
        self.env = _make("v9")

    def tearDown(self):
        self.env.close()

    def reference_model(self):
        return mujoco.MjModel.from_xml_path(os.path.join(ROOT, "walker_ragdoll.xml"))

    def test_the_model_is_untouched(self):
        model, ref = self.env.unwrapped.model, self.reference_model()
        self.assertEqual(model.opt.integrator, ref.opt.integrator)
        self.assertEqual(model.opt.timestep, ref.opt.timestep)
        # Compared against a freshly compiled model rather than against assumed constants: some
        # geoms are inert in the XML by design (the target marker carries contype=0/conaffinity=0),
        # and a test that asserted them all at 1 would fail while nothing had actually changed.
        self.assertTrue((model.geom_contype == ref.geom_contype).all())
        self.assertTrue((model.geom_conaffinity == ref.geom_conaffinity).all())

    def test_the_recorded_version_is_the_module_constant(self):
        self.assertEqual(env_version_of(self.env), ENV_VERSION)
        self.assertEqual(current_env_version(SimpleNamespace(physics_preset="v9")), ENV_VERSION)

    def test_observation_contract_is_unchanged(self):
        self.assertEqual(self.env.observation_space.shape, (46,))
        with _make("v9", task_phase="target") as target_env:
            self.assertEqual(target_env.observation_space.shape, (49,))


class TestPresetsTouchWhatTheyClaim(unittest.TestCase):
    def reference_model(self):
        return mujoco.MjModel.from_xml_path(os.path.join(ROOT, "walker_ragdoll.xml"))

    def test_euler_changes_only_the_integrator(self):
        ref = self.reference_model()
        with _make("euler") as env:
            model = env.unwrapped.model
            self.assertEqual(model.opt.integrator, mujoco.mjtIntegrator.mjINT_EULER)
            self.assertEqual(model.opt.timestep, ref.opt.timestep)
            self.assertTrue((model.geom_contype == ref.geom_contype).all())
            self.assertTrue((model.geom_conaffinity == ref.geom_conaffinity).all())

    def test_fast_drops_self_collision_and_keeps_the_floor(self):
        ref = self.reference_model()
        with _make("fast") as env:
            model = env.unwrapped.model
            floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            marker = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_marker_geom")
            self.assertEqual(model.opt.integrator, mujoco.mjtIntegrator.mjINT_EULER)
            self.assertGreaterEqual(floor, 0, "the floor geom vanished")
            self.assertEqual(model.geom_contype[floor], 1)
            for gid in range(model.ngeom):
                if gid == floor:
                    continue
                inert_by_design = (ref.geom_contype[gid] == 0 and ref.geom_conaffinity[gid] == 0)
                # The pair rule is contype(a) & conaffinity(b) | contype(b) & conaffinity(a).
                # Body geoms initiate nothing, so body-vs-body never forms; the floor still
                # initiates against their affinity, so floor-vs-body still forms. A geom the
                # model already made inert must stay inert, or the preset has recruited a
                # decorative marker into the collision set.
                self.assertEqual(model.geom_contype[gid], 0, f"geom {gid} still initiates")
                self.assertEqual(model.geom_conaffinity[gid], 0 if inert_by_design else 1,
                                 f"geom {gid} affinity wrong")
            self.assertEqual(model.geom_conaffinity[marker], 0, "the target marker became solid")

    def test_floor_contacts_still_happen_in_fast(self):
        """The standing gate counts foot-vs-floor, so a preset that lost them would break it silently."""
        with _make("fast", task_phase="target") as env:
            env.reset(seed=3)
            feet_ever = False
            for _ in range(120):
                _, _, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float64))
                _, foot = env.unwrapped.floor_contact_counts
                feet_ever = feet_ever or foot > 0
            self.assertTrue(feet_ever, "no foot-floor contact in 120 steps: the fast preset would "
                                       "make the standing gate unreachable")

    def test_unknown_preset_is_rejected(self):
        with self.assertRaises(ValueError):
            _make("cheaper")

    def test_apply_helper_rejects_the_unknown_too(self):
        model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "walker_ragdoll.xml"))
        with self.assertRaises(ValueError):
            apply_physics_preset(model, "nope")


class TestVersionTravelsWithThePreset(unittest.TestCase):
    def test_each_preset_records_its_own_version(self):
        for preset in env_module.PHYSICS_PRESETS:
            with _make(preset) as env:
                want = (ENV_VERSION if preset == "v9" else f"{ENV_VERSION}_{preset}")
                self.assertEqual(env_version_of(env), want)

    def test_env_version_of_survives_wrappers_and_missing_attrs(self):
        self.assertEqual(env_version_of(None), ENV_VERSION)
        with _make("euler") as env:
            self.assertEqual(env_version_of(env.unwrapped), f"{ENV_VERSION}_euler")

    def test_the_preset_survives_pickling_into_a_worker(self):
        # The parallel backend pickles the spec, and EzPickle is what rebuilds the env in the
        # child: if the preset were not part of the recorded args, half the vector would silently
        # train in the published world.
        with _make("fast", task_phase="target") as env:
            clone = pickle.loads(pickle.dumps(env.unwrapped))
            try:
                self.assertEqual(clone._physics_preset, "fast")
                self.assertEqual(env_version_of(clone), f"{ENV_VERSION}_fast")
                self.assertEqual(clone.model.opt.integrator,
                                 mujoco.mjtIntegrator.mjINT_EULER)
            finally:
                clone.close()

    def test_the_trainers_pass_it_to_every_sub_env(self):
        args = SimpleNamespace(reset_mode="mixed", fixed_reset_probability=0.25,
                               upright_reset_probability=0.15, fallen_velocity_scale=0.35,
                               task_phase="target", target_forward_velocity=1.2,
                               physics_preset="fast")
        kwargs = env_common_kwargs(args)
        self.assertEqual(kwargs["physics_preset"], "fast",
                         "env_common_kwargs dropped the preset, so the trainers would build v9 "
                         "sub-envs while recording a fast version string")


class TestEveryTrainerCanActuallyAskForAPreset(unittest.TestCase):
    """The flag has to be declared by every trainer that builds sub-envs, not just by one.

    `env_common_kwargs` reads the attribute tolerantly (a harness that assembles its own args
    namespace should get the published world, not an AttributeError), and that tolerance is exactly
    what makes the failure silent: a trainer whose parser never declares `--physics-preset` accepts
    nothing, collects in v9, and records `..._v9` in its checkpoints while a user believes they asked
    for a cheaper world. Three of the four trainers were in that state when the preset landed.
    """

    TRAINERS = ("train_walker.py", "train_dreamer.py", "train_redq.py", "train_ars.py")

    def test_the_flag_is_declared(self):
        missing = []
        for name in self.TRAINERS:
            with open(os.path.join(ROOT, name), encoding="utf-8") as handle:
                src = handle.read()
            if '--physics-preset' not in src:
                missing.append(name)
        self.assertEqual(missing, [], f"these trainers accept no preset: {missing}")

    def test_the_ones_that_reuse_the_shared_builder_are_the_risky_ones(self):
        # train_dreamer/train_redq go through build_vec_env, so a missing declaration is invisible;
        # train_ars passes kwargs to make_env by hand, where a missing one is a TypeError instead.
        for name in ("train_dreamer.py", "train_redq.py"):
            with open(os.path.join(ROOT, name), encoding="utf-8") as handle:
                self.assertIn("build_vec_env(", handle.read(),
                              f"{name} no longer builds envs through the shared function; the "
                              "declaration check above may no longer cover it")
        with open(os.path.join(ROOT, "train_ars.py"), encoding="utf-8") as handle:
            self.assertIn("physics_preset=args.physics_preset", handle.read(),
                          "train_ars builds its env by hand and would silently drop the preset")


class TestEvalRefusesToMixWorlds(unittest.TestCase):
    """Source-level guards for eval_phase1.py, which has to stay loadable against old revisions."""

    def setUp(self):
        with open(os.path.join(ROOT, "eval_phase1.py"), encoding="utf-8") as handle:
            self.src = handle.read()

    def test_the_preset_is_only_passed_when_it_is_not_the_default(self):
        # `--env-commit` aliases an older env revision in as the module, and that revision has no
        # physics_preset parameter: an unconditional kwarg would break every re-score to date.
        self.assertIsNotNone(re.search(r'preset_kwargs = \{\} if physics_preset == "v9"', self.src),
                             "eval_phase1.py no longer withholds the preset kwarg for v9, which "
                             "breaks --env-commit re-scores")

    def test_env_commit_and_a_preset_cannot_be_combined(self):
        self.assertIn("aliases an older env revision that has no --physics-preset", self.src,
                      "the guard against scoring a preset run against an aliased revision is gone")

    def test_the_recorded_artifact_names_the_world_it_scored_in(self):
        self.assertIn('"scored_in_version"', self.src,
                      "the artifact stopped recording which world the episodes ran in, which is "
                      "the mistake the device field was added to fix")


if __name__ == "__main__":
    unittest.main()
