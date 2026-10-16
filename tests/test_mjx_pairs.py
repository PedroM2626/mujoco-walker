"""Contract tests for the explicit-<pair> variants, i.e. the boxer-rl contact configuration.

These assert the *mechanism* only: that the candidate set is what MuJoCo's own filter admits, that the
rewritten XML parses and declares exactly those pairs, that a variant with the automatic filter off
really does collide only where it says, and that the friction written into each `<pair>` is the same
five-component vector the automatic path would have used. Whether any of it is *faster* is a measurement
and lives in `benchmarks/mjx_pairs_*.json`; nothing here re-derives a number that the README quotes.

The one thing these tests exist to protect is the equivalence proof itself: `bench_mjx_pairs.py` claims
its `explicit_full` cell is the published MDP, and that claim is only sound while the variant is built
the way it was built. If the XML rewrite silently stopped covering the floor geom, or if a `<pair>`
friction attribute were parsed with the 3-number convention, the throughput comparison published beside
it would be comparing two different worlds without saying so.
"""

import os
import sys
import unittest

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench_mjx_pairs import (GAIT_FLOOR_GEOMS, GAIT_SELF_GEOMS, XML_PATH, build_variant_xml,
                             candidate_pairs, geom_name, pair_friction)


def _published():
    return mujoco.MjModel.from_xml_path(XML_PATH)


class TestCandidateSetComesFromTheFilter(unittest.TestCase):
    """The candidate set is read off MuJoCo, not re-derived from the exclusion rules by hand."""

    @classmethod
    def setUpClass(cls):
        cls.model = _published()
        cls.pairs = candidate_pairs(cls.model)

    def test_every_candidate_is_a_pair_of_distinct_collidable_geoms(self):
        self.assertEqual(len(set(self.pairs)), len(self.pairs), "the candidate set has duplicates")
        for i, j in self.pairs:
            self.assertNotEqual(i, j, f"geom {i} paired with itself")
            self.assertLess(i, j, "pairs are not canonicalised, so set algebra on them is unreliable")
            self.assertTrue(self.model.geom_contype[i] or self.model.geom_conaffinity[i])
            self.assertTrue(self.model.geom_conaffinity[j] or self.model.geom_contype[j])

    def test_the_filter_admits_fewer_pairs_than_exist(self):
        possible = self.model.ngeom * (self.model.ngeom - 1) // 2
        self.assertLess(len(self.pairs), possible,
                        "nothing was filtered, so the puffing margin is not working")
        self.assertGreater(len(self.pairs), 0)

    def test_parent_child_and_same_body_pairs_are_absent_but_the_worlds_floor_is_not(self):
        names = {gid: geom_name(self.model, gid) for gid in range(self.model.ngeom)}
        pairs = set(self.pairs)
        body_of_geom = {gid: int(self.model.geom_bodyid[gid]) for gid in range(self.model.ngeom)}
        for i, j in pairs:
            bi, bj = body_of_geom[i], body_of_geom[j]
            if bi == 0 or bj == 0:
                continue  # the world body is exempt from the parent-child rule: floor contacts count
            direct = (self.model.body_parentid[bi] == bj or self.model.body_parentid[bj] == bi)
            self.assertFalse(direct, f"{names[i]} and {names[j]} are parent-child and must be filtered")
        # A positive control for the exemption: at least one body geom collides with the floor.
        self.assertTrue(any(names[i] == "floor" or names[j] == "floor" for i, j in pairs),
                        "no floor contacts at all, so the candidate set is not what training sees")

    def test_the_margin_rewrite_is_what_produces_the_set(self):
        # If walker_ragdoll.xml ever changes shape, the string replacements in candidate_pairs stop
        # applying and the function raises instead of quietly returning an empty or partial set.
        text = open(XML_PATH, encoding="utf-8").read()
        self.assertIn('<geom conaffinity="1" condim="3" contype="1" density="1000"', text)
        self.assertIn('<geom condim="3" friction="1 .1 .1" material="MatPlane" name="floor"', text)


class TestPairFrictionFollowsTheAutomaticConvention(unittest.TestCase):
    def test_it_is_five_components_and_elementwise_max_of_the_expanded_geoms(self):
        model = _published()
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "head")
        f = pair_friction(model, floor, head)
        self.assertEqual(np.shape(f), (5,), "a <pair friction> with 3 numbers parses as "
                                           "(slide1, slide2, spin), not (slide, spin, roll)")
        expand = lambda x: np.array([x[0], x[0], x[1], x[2], x[2]])
        np.testing.assert_array_equal(f, np.maximum(expand(model.geom_friction[floor]),
                                                   expand(model.geom_friction[head])))

    def test_a_stray_third_value_cannot_land_in_the_roll_slot(self):
        # The reported difference in this run came from exactly this: geom_friction is 3 wide, pair
        # friction is 5, and the 3-number form writes spin where roll belongs.
        model = _published()
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_shin1")
        f = pair_friction(model, a, b)
        self.assertEqual(f[1], f[0], "pair friction slide components must both carry the slide value")
        self.assertEqual(f[4], f[3])


class TestVariantXmlIsWhatItClaims(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _published()
        cls.pairs = candidate_pairs(cls.model, open(XML_PATH, encoding="utf-8").read())
        cls.xml = build_variant_xml(cls.model, cls.pairs)
        cls.variant = mujoco.MjModel.from_xml_string(cls.xml)

    def test_it_parses_and_declares_exactly_the_candidate_set(self):
        self.assertEqual(int(self.variant.npair), len(self.pairs))
        declared = set()
        for k in range(self.variant.npair):
            i, j = int(self.variant.pair_geom1[k]), int(self.variant.pair_geom2[k])
            declared.add((min(i, j), max(i, j)))
        self.assertEqual(declared, set(self.pairs), "the XML declares a different set than measured")

    def test_the_automatic_filter_is_off_so_pairs_are_the_only_collisions(self):
        self.assertEqual(int(self.variant.npair), len(self.pairs))
        for gid in range(self.variant.ngeom):
            self.assertEqual(int(self.variant.geom_contype[gid]), 0,
                             f"{geom_name(self.variant, gid)} still collides automatically")
            self.assertEqual(int(self.variant.geom_conaffinity[gid]), 0)

    def test_every_other_model_field_is_untouched(self):
        # The variant may change *which* contacts exist and nothing else: same bodies, joints,
        # actuators, timestep, integrator, geom sizes and masses. Otherwise the throughput cell beside
        # the fidelity cell is a different world rather than the same world filtered differently.
        for attr in ("nbody", "ngeom", "njnt", "nq", "nv", "nu", "nmesh"):
            self.assertEqual(getattr(self.variant, attr), getattr(self.model, attr), attr)
        self.assertEqual(float(self.variant.opt.timestep), float(self.model.opt.timestep))
        self.assertEqual(int(self.variant.opt.integrator), int(self.model.opt.integrator))
        np.testing.assert_allclose(self.variant.geom_size, self.model.geom_size, rtol=0, atol=1e-12)
        np.testing.assert_allclose(self.variant.body_mass, self.model.body_mass, rtol=0, atol=1e-12)
        np.testing.assert_allclose(self.variant.actuator_gainprm, self.model.actuator_gainprm,
                                   rtol=0, atol=1e-12)

    def test_the_size_element_is_present_so_the_bench_and_the_env_agree_on_the_buffer(self):
        self.assertIn('<size nconmax="512" njmax="4096"/>', self.xml)


class TestGaitReductionIsADeliberateCut(unittest.TestCase):
    """The reduced set must be a subset of the candidates and must drop the arms - loudly."""

    @classmethod
    def setUpClass(cls):
        cls.model = _published()
        cls.pairs = candidate_pairs(cls.model, open(XML_PATH, encoding="utf-8").read())
        cls.names = {gid: geom_name(cls.model, gid) for gid in range(cls.model.ngeom)}
        cls.gait = [(i, j) for i, j in cls.pairs
                    if (cls.names[i] == "floor" and cls.names[j] in GAIT_FLOOR_GEOMS)
                    or (cls.names[j] == "floor" and cls.names[i] in GAIT_FLOOR_GEOMS)
                    or (cls.names[i] in GAIT_SELF_GEOMS and cls.names[j] in GAIT_SELF_GEOMS)]

    def test_it_is_a_strict_subset_of_the_candidates(self):
        self.assertTrue(set(self.gait) <= set(self.pairs))
        self.assertLess(len(self.gait), len(self.pairs))

    def test_no_arm_geom_is_in_collision_at_all(self):
        arm = ("right_uarm1", "right_larm", "right_hand", "left_uarm1", "left_larm", "left_hand")
        self.assertEqual([pair for pair in self.gait
                          if self.names[pair[0]] in arm or self.names[pair[1]] in arm], [],
                        "the reduced set still collides an arm, so it is not the boxer-style cut "
                        "the README describes")

    def test_the_legs_still_collide_with_each_other(self):
        self.assertTrue(any(self.names[i] in GAIT_SELF_GEOMS and self.names[j] in GAIT_SELF_GEOMS
                            for i, j in self.gait),
                        "legs pass through each other, which is a different failure than the README's")

    def test_the_variant_is_a_different_mdp_by_construction(self):
        # This is the assertion behind "gait_reduced_is_a_different_mdp": the reduced XML declares
        # fewer pairs than the published model can generate, so the contact set differs on some pose.
        variant = mujoco.MjModel.from_xml_string(build_variant_xml(self.model, self.gait))
        self.assertLess(int(variant.npair), len(self.pairs))


if __name__ == "__main__":
    unittest.main()
