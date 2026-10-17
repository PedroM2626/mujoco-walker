"""Tests for the instrument that prices a model in MJX: `bench_mjx.xml_with_sizes`/`build_mjx`.

These exist because two harnesses in this repo measure the same quantity and disagreed by ~40x. The
published table (`benchmarks/mjx_gpu_wsl.json`) compiles the model through `xml_with_sizes`, which
gives MJX concrete `<size nconmax njmax>` capacities; the G1 harness called `mjx.put_model` on a model
compiled straight from the file, which MuJoCo reports as `nconmax = njmax = -1`. Both numbers were
"MJX on the same GPU", and only one of them was the same simulator state. So the rule these tests hold:
every MJX arm in this repository goes through this function, whatever XML it is given.
"""

import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mujoco  # noqa: E402

from bench_mjx import XML_PATH, xml_with_sizes  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HERE_TMP = tempfile.mkdtemp(prefix="bench_mjx_sizes_")
SIZED = '<size nconmax="512" njmax="4096"/>'


def compile_text(text, directory):
    """Compile rewritten text the way the instrument does: by path, from a sibling of the original."""
    sibling = os.path.join(directory, "_mjx_sized_test.xml")
    with open(sibling, "w", encoding="utf-8") as handle:
        handle.write(text)
    return mujoco.MjModel.from_xml_path(sibling)


class TestSizesOnThisRepoModel(unittest.TestCase):
    def test_the_published_model_gets_capacities_and_keeps_its_own_option(self):
        model = compile_text(xml_with_sizes(512, 4096, xml=XML_PATH), HERE_TMP)
        self.assertEqual(int(model.nconmax), 512)
        self.assertEqual(int(model.njmax), 4096)
        # The environment is RK4 at 0.002 and every headline number depends on that staying true.
        self.assertEqual(mujoco.mjtIntegrator(model.opt.integrator).name, "mjINT_RK4")
        self.assertAlmostEqual(float(model.opt.timestep), 0.002)

    def test_the_injected_element_is_the_only_size_the_parser_can_see(self):
        text = xml_with_sizes(512, 4096, xml=XML_PATH)
        self.assertEqual(len(re.findall(r"<size\b", text)), 1)
        self.assertIn(SIZED, text)


class TestForeignModels(unittest.TestCase):
    """The G1 case: a scene whose <option> lives in an include and whose file may carry its own size."""

    def setUp(self):
        # A scene with no <option> of its own - exactly the shape of Menagerie's scene_mjx.xml.
        self.part = os.path.join(HERE_TMP, "part.xml")
        with open(self.part, "w", encoding="utf-8") as handle:
            handle.write('<mujoco><worldbody><body name="b" pos="0 0 1">'
                         '<joint type="free"/>'
                         '<geom name="g" type="sphere" size="0.05"/></body></worldbody></mujoco>')
        self.scene = os.path.join(HERE_TMP, "scene.xml")
        with open(self.scene, "w", encoding="utf-8") as handle:
            handle.write('<mujoco model="scene">\n  <include file="part.xml"/>\n'
                         '  <worldbody><geom name="floor" type="plane" size="0 0 0.1"/>'
                         '</worldbody>\n</mujoco>\n')

    def test_a_scene_without_its_own_option_still_gets_the_element_inside_the_root(self):
        text = xml_with_sizes(512, 4096, xml=self.scene)
        self.assertIn(SIZED, text)
        self.assertLess(text.index(SIZED), text.index("<include"))
        model = compile_text(text, HERE_TMP)
        self.assertEqual(int(model.nconmax), 512)
        self.assertEqual(int(model.njmax), 4096)
        # The include still resolves, which is the whole reason a foreign scene is compiled by path.
        self.assertEqual([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                          for i in range(model.ngeom)], ["floor", "g"])

    def test_a_model_that_ships_its_own_capacities_gets_them_replaced_not_duplicated(self):
        with open(self.scene, "w", encoding="utf-8") as handle:
            handle.write('<mujoco model="scene">\n  <size nconmax="7" njmax="9"/>\n'
                         '  <option timestep="0.004"/>\n'
                         '  <worldbody><geom name="floor" type="plane" size="0 0 0.1"/>'
                         '</worldbody>\n</mujoco>\n')
        text = xml_with_sizes(512, 4096, xml=self.scene)
        self.assertEqual(len(re.findall(r"<size\b", text)), 1,
                         "a second <size> makes which capacity wins depend on document order")
        self.assertNotIn('nconmax="7"', text)
        model = mujoco.MjModel.from_xml_string(text)
        self.assertEqual(int(model.nconmax), 512)
        self.assertEqual(int(model.njmax), 4096)
        # Rewriting the capacities must not be allowed to touch the thing being priced.
        self.assertAlmostEqual(float(model.opt.timestep), 0.004)

    def test_an_integrator_rewrite_reaches_a_model_that_does_not_spell_it_like_ours(self):
        # Menagerie's G1 declares implicitfast, not RK4; the old code replaced only 'integrator="RK4"'
        # and raised otherwise, so the flag silently could not be used on any borrowed model.
        with open(self.scene, "w", encoding="utf-8") as handle:
            handle.write('<mujoco model="scene">\n'
                         '  <option timestep="0.004" integrator="implicitfast"/>\n'
                         '  <worldbody><geom name="floor" type="plane" size="0 0 0.1"/>'
                         '</worldbody>\n</mujoco>\n')
        text = xml_with_sizes(512, 4096, "EULER", xml=self.scene)
        self.assertIn('integrator="Euler"', text)
        self.assertEqual(mujoco.mjtIntegrator(
            mujoco.MjModel.from_xml_string(text).opt.integrator).name, "mjINT_EULER")

    def test_an_integrator_rewrite_fails_loudly_when_the_model_names_none(self):
        with self.assertRaises(RuntimeError):
            xml_with_sizes(512, 4096, "EULER", xml=self.part)


class TestTheG1HarnessUsesThisInstrument(unittest.TestCase):
    def test_bench_g1_cost_no_longer_calls_put_model_on_its_own(self):
        # The bug was structural, not numerical: two harnesses, one quantity, two simulator states.
        # A shared call site is the cheapest way to make the split impossible to reintroduce.
        with open(os.path.join(ROOT, "bench_g1_cost.py"), encoding="utf-8") as handle:
            src = handle.read()
        self.assertIn("from bench_mjx import build_mjx", src)
        self.assertNotIn("mjx.put_model(", src,
                         "bench_g1_cost compiles for MJX again instead of borrowing the harness")

    def test_the_harness_compiles_by_path_so_a_borrowed_models_assets_resolve(self):
        # The first --xml run died on `Error opening file 'assets/right_hip_roll_link.STL'`:
        # from_xml_string resolves relative assets against the process's cwd, which is where this
        # repo's ragdoll lives and where Menagerie's robot does not.
        with open(os.path.join(ROOT, "bench_mjx.py"), encoding="utf-8") as handle:
            src = handle.read()
        start = src.index("def build_mjx(")
        body = src[start:src.index("def make_batched_data(", start)]
        self.assertIn("from_xml_path(sibling)", body)
        self.assertNotIn("from_xml_string", body,
                         "build_mjx compiles a model file from a string again")


if __name__ == "__main__":
    unittest.main()
