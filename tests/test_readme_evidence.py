"""Every number the README quotes about the REDQ ensemble must come from a committed artifact.

This is a transcription test, not a benchmark: it re-reads `benchmarks/redq_ensemble_ab.json` and
`benchmarks/throughput_baseline_vs_now.json`, recomputes each table cell and each prose multiplier of
the "Shipped: the REDQ critic ensemble as one batched pass" section, and reports *all* disagreements
at once. It exists because a hand-typed cell can survive CI while being wrong, and because these
figures are re-quoted in the prose too (the projected hours, the physics share of wall clock), which
is where drift is easiest to miss.

CI runs it against the committed artifacts, so it costs nothing and fails the moment someone edits a
README number without re-measuring it.
"""

import json
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
AB = os.path.join(ROOT, "benchmarks", "redq_ensemble_ab.json")
BASELINE = os.path.join(ROOT, "benchmarks", "throughput_baseline_vs_now.json")

SECTION = "**Shipped: the REDQ critic ensemble as one batched pass.**"
NEXT_SECTION = "**Measured and rejected"

# The two A/B arms that produced the table ran 10k env steps at n=16 parallel.
STEPS = 10000


def nums(text):
    """Floats in a README cell, tolerating thousands separators and ** bold."""
    return [float(t.replace(",", "")) for t in re.findall(r"\d[\d,]*\.?\d*", text.replace("*", ""))]


class TestReadmeRedqEnsembleCells(unittest.TestCase):
    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            self.readme = handle.read()
        start = self.readme.index(SECTION)
        self.block = self.readme[start:self.readme.index(NEXT_SECTION, start)]
        for path in (AB, BASELINE):
            self.assertTrue(os.path.exists(path), f"the README quotes {path}, which is missing")
        with open(AB, encoding="utf-8") as handle:
            self.ab = json.load(handle)
        with open(BASELINE, encoding="utf-8") as handle:
            self.sps_single_env = json.load(handle)["phases"]["after_env_python_rewrite"][
                "single_env_steps_per_s"]
        self.bad = []

    def tearDown(self):
        if self.bad:
            self.fail("README REDQ ensemble numbers drifted from the artifacts:\n  "
                      + "\n  ".join(self.bad))

    def cell(self, label, col):
        for line in self.block.splitlines():
            if line.startswith(f"| {label} |"):
                return line.strip("|").split("|")[col].strip().replace("*", "")
        raise AssertionError(f"README table row {label!r} is gone")

    def check(self, what, typed, measured, places=1, slack=0.0):
        """Compare one or many typed figures against recomputed ones, at the README's precision.

        `slack` is for figures the artifact cannot carry exactly: runs_s is stored in tenths of a
        second, so a rate recomputed from it can sit up to 0.05 s / dt away from the rate the
        harness printed, plus the printed figure's own rounding - ~0.12 env-steps/s at these
        budgets. Anything wider than that is drift, not rounding.
        """
        a = [float(str(x).replace(",", "")) for x in (typed if isinstance(typed, (list, tuple)) else [typed])]
        b = [float(str(x).replace(",", "")) for x in
             (measured if isinstance(measured, (list, tuple)) else [measured])]
        if len(a) != len(b):
            self.bad.append(f"{what}: README has {a}, artifact gives {b}")
            return
        for x, y in zip(a, b):
            if abs(round(x, places) - round(y, places)) > slack:
                self.bad.append(f"{what}: README says {x}, artifact gives {y:.6g}")

    def sentence(self, pattern, what):
        m = re.search(pattern, self.block)
        self.assertIsNotNone(m, f"the {what} sentence was reworded; re-point this test at it")
        return m

    def rate(self, tag):
        """env-steps/s for an arm, with the slack its own artifact can support.

        runs_s stores tenths of a second, and the harness printed the rate from the unrounded
        elapsed time, so a recomputed rate can sit up to (rate * 0.05 / dt) away from the quoted
        one - plus the printed figure's own tenth. That bound is ~0.13 here; a typo of one tick
        on the fastest row is above it and still fails.
        """
        dt = self.ab["runs_s"][tag]
        sps = STEPS / dt
        return sps, 0.05 + sps * 0.05 / dt

    def test_table_cells_match_the_runs(self):
        runs, grad = self.ab["runs_s"], self.ab["gradient_s"]
        for label, arm in (("loop ensemble", "l16"), ("batched ensemble", "b16")):
            rep = arm + "r" if runs.get(arm + "r") else None
            totals = [runs[arm]] + ([runs[rep]] if rep else [])
            self.check(f"{label} total s", nums(self.cell(label, 1)), totals)
            self.check(f"{label} gradient phase s", self.cell(label, 2), grad[arm])
            sps, tol = self.rate(arm)
            self.check(f"{label} env-steps/s", self.cell(label, 3), sps, slack=tol)
        self.check("collection floors", nums(self.cell("collection floors, loop / batched", 1)),
                   [runs["f16l"], runs["f16b"]])
        typed_rates = nums(self.cell("collection floors, loop / batched", 3))
        for i, tag in enumerate(("f16l", "f16b")):
            sps, tol = self.rate(tag)
            self.check(f"collection floor rate ({tag})", typed_rates[i], sps, slack=tol)

    def test_prose_multipliers_match_the_summary(self):
        s = self.ab["summary"]["n=16 parallel"]
        m = self.sentence(r"\*\*([\d.]+)x on the update phase and ([\d.]+)x end to end\*\*", "ratio")
        self.check("update-phase ratio", m.group(1), s["update_phase_speedup"], places=2)
        self.check("end-to-end ratio", m.group(2), s["whole_run_ratio"], places=2)

        fl, fb = self.ab["runs_s"]["f16l"], self.ab["runs_s"]["f16b"]
        m = self.sentence(r"the two floors agreeing to (\d+)%", "floor-agreement")
        self.check("floor disagreement", m.group(1), (max(fl, fb) / min(fl, fb) - 1) * 100, places=0)

        m = self.sentence(r"([\d.]+) h becoming ~?([\d.]+) h", "projected-hours")
        loop_rate = STEPS / self.ab["runs_s"]["l16"]
        batched_rate = STEPS / self.ab["runs_s"]["b16"]
        self.check("loop-path projection (h)", m.group(1), 1e6 / loop_rate / 3600)
        self.check("batched-path projection (h)", m.group(2), 1e6 / batched_rate / 3600)

    def test_the_throughput_section_requotes_the_same_two_ratios(self):
        """The summary section names these numbers a second time, outside the gated block."""
        pairs = re.findall(r"\*\*([\d.]+)x end to end, ([\d.]+)x on the update phase\*\*", self.readme)
        self.assertTrue(pairs, "no 'Nx end to end, Nx on the update phase' pair is left in the README")
        got = {(round(float(a), 2), round(float(b), 2)) for a, b in pairs}
        want = {(round(s["whole_run_ratio"], 2), round(s["update_phase_speedup"], 2))
                for s in self.ab["summary"].values()}
        self.assertEqual(sorted(got), sorted(want),
                         "the README quotes a ratio pair that no A/B arm measured")

    def test_the_trainer_table_row_points_at_the_same_projection(self):
        """The learner-side summary table gained a batched REDQ row; it quotes the A/B's hours."""
        m = re.search(r"^\| `train_redq\.py`, batched ensemble \|.*\|$", self.readme, re.M)
        self.assertIsNotNone(m, "the batched REDQ row of the trainer table is gone")
        cells = [c.strip() for c in m.group(0).strip("|").split("|")]
        self.assertEqual(4, len(cells), "the batched REDQ row no longer has four cells")
        self.assertIn("batched", cells[1], "the row must say which path it measured")
        batched_rate = STEPS / self.ab["runs_s"]["b16"]
        self.check("trainer-table projection (h)", nums(cells[3])[0], 1e6 / batched_rate / 3600)

    def test_the_n4_sync_arm_sentence_matches_its_arms(self):
        """The unobstructed arm: 4k env steps at n=4 on the sync backend, whole-run times."""
        s = self.ab["summary"]["n=4 sync"]
        m = self.sentence(r"(\d[\d.]*) s\s*\n?on the loop path against ([\d.]*) s batched", "n=4 sync")
        self.check("n=4 loop total s", m.group(1), self.ab["runs_s"]["l4"])
        self.check("n=4 batched total s", m.group(2), self.ab["runs_s"]["b4"])
        m = self.sentence(r"collection floors \(([\d.]*) s loop, ([\d.]*) s batched\) "
                          r"disagree by (\d+)%", "n=4 floors")
        self.check("n=4 loop floor s", m.group(1), self.ab["runs_s"]["f4l"])
        self.check("n=4 batched floor s", m.group(2), self.ab["runs_s"]["f4b"])
        self.check("n=4 floor disagreement", m.group(3),
                   (max(s["floors_s"]) / min(s["floors_s"]) - 1) * 100, places=0)

    def test_physics_share_derives_from_two_artifacts(self):
        m = self.sentence(r"runs at ~([\d,]+) env-steps/s while one environment steps physics at "
                          r"([\d,]+)/s", "physics-share")
        loop_rate = STEPS / self.ab["runs_s"]["l16"]
        self.check("loop-path rate quoted in prose", m.group(1), loop_rate, places=0)
        self.check("single-env physics rate", m.group(2), self.sps_single_env, places=0)
        m = self.sentence(r"MuJoCo inside a REDQ run is about \*\*(\d+)%", "physics share of the wall clock")
        self.check("physics share of wall clock", m.group(1), loop_rate / self.sps_single_env * 100,
                   places=0)

    def test_isolated_microbench_figures_match_the_artifact(self):
        iso = self.ab.get("isolated")
        self.assertIsNotNone(iso, "bench_redq_ensemble.py --isolated-only has not been run yet")
        gpu, cpu = iso["cuda"], iso["cpu"]
        m = self.sentence(r"\*\*([\d.]+) ms loop against ([\d.]+) ms batched on the GPU\*\* "
                          r"\(([\d.]+) ms against ([\d.]+) ms on CPU\)", "isolated-step")
        self.check("loop critic step ms (GPU)", m.group(1), gpu["critic_step_loop_ms"], places=2)
        self.check("batched critic step ms (GPU)", m.group(2), gpu["critic_step_batched_ms"], places=2)
        self.check("loop critic step ms (CPU)", m.group(3), cpu["critic_step_loop_ms"], places=1)
        self.check("batched critic step ms (CPU)", m.group(4), cpu["critic_step_batched_ms"], places=1)
        m = self.sentence(r"[Tt]he arithmetic (?:is )?equal to ([\d.e-]+) relative on the\s+forward"
                          r" and ([\d.e-]+) across all (\d+) weight gradients", "arithmetic-drift")
        self.check("forward drift", float(m.group(1)), gpu["forward_rel"], places=12)
        self.check("weight-gradient drift", float(m.group(2)), gpu["weight_grad_rel_max"], places=12)
        self.check("weight tensors compared", m.group(3), self.ab["config"]["ensemble_size"] * 3,
                   places=0)

        m = self.sentence(r"target update goes ([\d.]+) ms -> ([\d.]+) ms", "soft-update")
        self.check("loop soft update ms", m.group(1), gpu["soft_update_loop_ms"], places=2)
        self.check("batched soft update ms", m.group(2), gpu["soft_update_batched_ms"], places=2)

        m = self.sentence(r"the ([\d.]+)x and the ([\d.]+)x are the parts", "isolated speedups")
        self.check("critic-step speedup", m.group(1), gpu["critic_step_speedup"], places=2)
        self.check("soft-update speedup", m.group(2), gpu["soft_update_speedup"], places=2)

    def test_the_harness_measured_what_the_table_claims(self):
        """The table says 'two reps, floors per implementation': the artifact must have all of it."""
        for tag in ("b16", "l16", "b16r", "l16r", "f16b", "f16l"):
            self.assertIsNotNone(self.ab["runs_s"].get(tag), f"arm {tag} is missing from the artifact")
        self.assertEqual(20, self.ab["config"]["utd_ratio"])
        self.assertEqual(10, self.ab["config"]["ensemble_size"])
        self.assertEqual(256, self.ab["config"]["batch_size"])


if __name__ == "__main__":
    unittest.main()
