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

import glob
import hashlib
import datetime
import json
import os
import re
import sqlite3
import statistics
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
AB = os.path.join(ROOT, "benchmarks", "redq_ensemble_ab.json")
BASELINE = os.path.join(ROOT, "benchmarks", "throughput_baseline_vs_now.json")
PROFILE = os.path.join(ROOT, "benchmarks", "dreamer_update_profile.json")
DREAMER_AB = os.path.join(ROOT, "benchmarks", "dreamer_update_graph_ab.json")
REPRO = os.path.join(ROOT, "benchmarks", "dreamer_reproducibility.json")

SECTION = "**Shipped: the REDQ critic ensemble as one batched pass.**"
DREAMER_SECTION = "**Shipped: the Dreamer update as one graph replay.**"
NEXT_SECTION = "**Measured and rejected"

# The two A/B arms that produced the table ran 10k env steps at n=16 parallel.
STEPS = 10000

PHASE4_SECTION = "## \U0001f3c6 Final Benchmark Results (Offline-to-Online Race)"
PHASE4_HISTORY = "### Historical record: one unseeded episode per model"
PHASE4_ENV_NOTE = "\u26a0\ufe0f **The environment this table needs"

RACE_50EP = os.path.join(ROOT, "benchmarks", "phase4_race_50ep.json")
N1_ART = os.path.join(ROOT, "benchmarks", "phase4_n1_vs_50ep.json")
PAIRED_50EP = os.path.join(ROOT, "benchmarks", "phase4_paired_50ep_seed2026.json")
CURVE = os.path.join(ROOT, "benchmarks", "teacher_eval_curve.json")
PAIR_BCQ_TEACHER = "BCQ - Teacher (Upper Bound)"

# README row label (as it sits in the Markdown, bold included) -> artifact model key. Explicit, so
# renaming a row fails the gate instead of quietly un-gating it.
RACE_ROWS = {
    "**Behavioral Cloning (BC)**": "BC",
    "**Teacher (Online SAC)**": "Teacher (Upper Bound)",
    "**Extra Trees Cloner (sklearn)**": "Extra Trees Cloner (sklearn)",
    "**BC+SAC (Regularized)**": "BC+SAC (Regularized)",
    "**Batch-Constrained Q-learning (BCQ)**": "BCQ",
    "**Decision Transformer (DT)**": "DT",
    "**BC+SAC (Naive)**": "BC+SAC (Naive)",
    "**Inverse RL (GAIL)**": "GAIL",
    "**CQL+SAC**": "CQL+SAC",
    "**CQL Offline**": "CQL",
    "**MaxEnt IRL**": "MaxEnt",
    "**BC+SAC (Constrained)**": "BC+SAC (Constrained)",
    "**IQL Offline**": "IQL",
    "**Inverse RL (AIRL)**": "AIRL",
}
# The retired table carries the same rows except Extra Trees, which was never in that race and so
# has no single-episode score in any artifact.
N1_ROWS = {k: v for k, v in RACE_ROWS.items() if "Extra Trees" not in k}

MINUS = "\u2212"  # the README writes these gaps with a real minus sign

# The Phase-4 policies the README lists as archived in the MLflow file store, and the paths that
# hold both copies.
P4 = os.path.join(ROOT, "openai_walker")
MLRUNS = os.path.join(P4, "mlruns")
ARCHIVED_WEIGHTS = ["bc_model.pt", "iql_model.pt", "cql_model.pt", "bc_sac_naive_model.pt",
                    "bc_sac_regularized_model.pt", "bc_sac_constrained_model.pt",
                    "cql_sac_model.pt"]


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# Each retired draw as item 7 states it, and the artifact fields those figures must equal.
N1_PROSE = [
    ("BC", r"BC's ([\d.]+) is \*\*([-\d.]+) sigma\*\* and (\d+) of 50 episodes beat it",
     ("n1_score", "n1_in_sigmas", "episodes_at_or_above_n1")),
    ("BCQ", r"BCQ's ([\d.]+) is \*\*\+([-\d.]+) sigma\*\*, reached or exceeded in (\d+) of 50, "
            r"with a max of ([\d.]+)",
     ("n1_score", "n1_in_sigmas", "episodes_at_or_above_n1", "max")),
    ("Teacher (Upper Bound)", r"the teacher's ([\d.]+) is \+([-\d.]+) sigma",
     ("n1_score", "n1_in_sigmas")),
    ("BC+SAC (Constrained)", r"BC\+SAC \(Constrained\) at \+([-\d.]+) sigma \(only (\d+) of 50",
     ("n1_in_sigmas", "episodes_at_or_above_n1")),
    ("BC+SAC (Regularized)", r"BC\+SAC \(Regularized\) at \+([-\d.]+) sigma",
     ("n1_in_sigmas",)),
]


def nums(text):
    """Floats in a README cell, tolerating thousands separators and ** bold."""
    return [float(t.replace(",", "")) for t in re.findall(r"\d[\d,]*\.?\d*", text.replace("*", ""))]


def snums(text):
    """Same, but sign-preserving: three Phase-4 rows carry negative minimum scores."""
    return [float(t.replace(",", "")) for t in
            re.findall(r"-?\d[\d,]*\.?\d*", text.replace("*", ""))]


class ReadmeGate:
    """Cell-by-cell comparison plumbing: collect every disagreement, report them together.

    Mixed into one class per README section. Subclasses set `self.readme`, `self.block`, the
    artifacts they gate against, and `self.what` for the failure message.
    """

    def tearDown(self):
        if self.bad:
            self.fail(f"README {self.what} numbers drifted from the artifacts:\n  "
                      + "\n  ".join(self.bad))

    def cell(self, label, col):
        for line in self.block.splitlines():
            if line.startswith(f"| {label} |"):
                return line.strip("|").split("|")[col].strip().replace("*", "")
        raise AssertionError(f"README table row {label!r} is gone")

    def check(self, what, typed, measured, places=1, slack=0.0):
        """Compare one or many typed figures against recomputed ones, at the README's precision.

        `slack` is for figures the artifact cannot carry exactly: a benchmark stores elapsed seconds
        in tenths, so a rate recomputed from it can sit up to 0.05 s / dt away from the rate the
        harness printed, plus the printed figure's own rounding. Anything wider is drift, not
        rounding, and the bound is derived from those two roundings rather than tuned to pass.
        """
        a = [float(str(x).replace(",", "")) for x in
             (typed if isinstance(typed, (list, tuple)) else [typed])]
        b = [float(str(x).replace(",", "")) for x in
             (measured if isinstance(measured, (list, tuple)) else [measured])]
        if len(a) != len(b):
            self.bad.append(f"{what}: README has {a}, artifact gives {b}")
            return
        for x, y in zip(a, b):
            if abs(round(x, places) - round(y, places)) > slack:
                self.bad.append(f"{what}: README says {x}, artifact gives {y:.6g}")

    def sentence(self, pattern, what):
        """Search the section with its Markdown hard-wraps collapsed.

        The README wraps at 90 columns, so a prose figure is never guaranteed to sit on one line;
        matching against the reflowed text keeps the patterns about the numbers rather than about
        where the author broke a line.
        """
        m = re.search(pattern, re.sub(r"\s+", " ", self.block))
        self.assertIsNotNone(m, f"the {what} sentence was reworded; re-point this test at it")
        return m

    def read_artifacts(self, *paths):
        out = []
        for path in paths:
            self.assertTrue(os.path.exists(path), f"the README quotes {path}, which is missing")
            with open(path, encoding="utf-8") as handle:
                out.append(json.load(handle))
        return out


class TestReadmeRedqEnsembleCells(ReadmeGate, unittest.TestCase):
    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            self.readme = handle.read()
        start = self.readme.index(SECTION)
        self.block = self.readme[start:self.readme.index(DREAMER_SECTION, start)]
        self.ab, self.baseline = self.read_artifacts(AB, BASELINE)
        self.sps_single_env = self.baseline["phases"]["after_env_python_rewrite"][
            "single_env_steps_per_s"]
        self.what = "REDQ ensemble"
        self.bad = []

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


class TestReadmeDreamerGraphCells(ReadmeGate, unittest.TestCase):
    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            self.readme = handle.read()
        start = self.readme.index(DREAMER_SECTION)
        self.block = self.readme[start:self.readme.index(NEXT_SECTION, start)]
        self.profile, self.ab, self.repro = self.read_artifacts(PROFILE, DREAMER_AB, REPRO)
        self.what = "Dreamer graph"
        self.bad = []

    def test_the_table_matches_both_reps(self):
        s = self.ab["summary"]
        rows = {"eager update": ("eager_s", "ms_per_update_eager", "eager_h"),
                "captured update": ("graph_s", "ms_per_update_graph", "graph_h")}
        for label, (total_key, ms_key, hour_key) in rows.items():
            self.check(f"{label} total s", nums(self.cell(label, 1)),
                       [s["rep 1"][total_key], s["rep 2"][total_key]])
            self.check(f"{label} ms per update", nums(self.cell(label, 2)),
                       [s["rep 1"][ms_key], s["rep 2"][ms_key]])
            self.check(f"{label} projected 1M h", nums(self.cell(label, 3)),
                       [s["rep 1"]["projected_1m"][hour_key], s["rep 2"]["projected_1m"][hour_key]])
        floors = self.cell("collection floors, eager / captured", 1)
        self.check("collection floors", nums(floors),
                   [self.ab["runs_s"]["fe1"], self.ab["runs_s"]["fg1"],
                    self.ab["runs_s"]["fe2"], self.ab["runs_s"]["fg2"]])

    def test_the_profile_counts_in_the_prose(self):
        n = self.ab["config"]["updates_per_run"]
        m = self.sentence(r"\*\*([\d,]+) aten calls per update\*\* \(([\d,]+) over (\d+) updates: "
                          r"([\d,]+) `aten::linear`, ([\d,]+) `aten::t`, ([\d,]+) "
                          r"`rssm_transition` calls\)", "aten-count")
        total = self.profile["aten_call_count"]
        per_update = total / n
        self.check("aten calls per update", nums(m.group(1))[0], round(per_update, -2), places=0)
        self.check("aten calls total", nums(m.group(2))[0], total, places=0)
        self.check("profiled updates", m.group(3), n, places=0)
        ops = self.profile["top_ops_by_call_count"]
        self.check("aten::linear calls", nums(m.group(4))[0], ops["aten::linear"], places=0)
        self.check("aten::t calls", nums(m.group(5))[0], ops["aten::t"], places=0)
        self.check("rssm_transition calls", nums(m.group(6))[0],
                   self.profile["frames"]["rssm_transition"]["count"], places=0)

        m = self.sentence(r"self CPU time lands at ([\d.]+)x self CUDA time", "cpu-over-cuda")
        self.check("cpu/cuda ratio", m.group(1), self.profile["cpu_over_cuda_ratio"], places=2)

    def test_the_ratios_and_the_projection(self):
        s = self.ab["summary"]
        m = self.sentence(r"\*\*([\d.]+)x and ([\d.]+)x on the update phase, ([\d.]+)x and "
                          r"([\d.]+)x end to end", "measured ratios")
        for i, key in enumerate(("update_phase_ratio", "end_to_end_ratio")):
            self.check(f"{key} rep 1", m.group(2 * i + 1), s["rep 1"][key], places=2)
            self.check(f"{key} rep 2", m.group(2 * i + 2), s["rep 2"][key], places=2)
        m = self.sentence(r"the same rates buy \*\*([\d.]+)x and ([\d.]+)x\*\* \(([\d,]+) updates",
                          "projected ratio")
        self.check("projected ratio rep 1", m.group(1), s["rep 1"]["projected_1m"]["ratio"], places=2)
        self.check("projected ratio rep 2", m.group(2), s["rep 2"]["projected_1m"]["ratio"], places=2)
        self.check("updates at 1M", nums(m.group(3))[0], s["rep 1"]["projected_1m"]["updates"],
                   places=0)

    def test_the_spread_sentence(self):
        s = self.ab["summary"]
        eager = [s["rep 1"]["ms_per_update_eager"], s["rep 2"]["ms_per_update_eager"]]
        graph = [s["rep 1"]["ms_per_update_graph"], s["rep 2"]["ms_per_update_graph"]]
        m = self.sentence(r"per-update cost moved ([\d.]+)% between reps[\s\S]*?captured arm's moved "
                          r"([\d.]+)%", "spread")
        self.check("eager spread", m.group(1), (max(eager) / min(eager) - 1) * 100, places=1)
        self.check("captured spread", m.group(2), (max(graph) / min(graph) - 1) * 100, places=1)

    def test_the_reproducibility_numbers(self):
        m = self.sentence(r"\*\*([\d,]+) of (\d+) tensors bit-identical, worst\s+\|[^|]*\| "
                          r"([\d.e-]+) after (\d+) updates\*\*, and the two runs' ([\d,]+) "
                          r"episodic returns", "reproducibility")
        self.check("bit-identical tensors", m.group(1), self.repro["bit_identical_tensors"], places=0)
        self.check("tensors compared", m.group(2), self.repro["tensors_compared"], places=0)
        self.check("worst weight drift", float(m.group(3)), self.repro["worst_abs_weight_diff"],
                   places=3)
        self.check("updates in the reproducibility run", m.group(4),
                   (self.repro["steps"] - 5000) // 4, places=0)
        self.check("episodic returns per run", nums(m.group(5))[0],
                   len(self.repro["episodic_returns"]["a"]), places=0)
        self.assertFalse(self.repro["returns_sequence_equal"],
                         "the README calls Dreamer runs unreproducible; the artifact says they matched")

    def test_the_equivalence_tolerances_are_the_ones_the_test_asserts(self):
        """The README quotes the numbers tests/test_dreamer_graph.py holds the graph to."""
        m = self.sentence(r"holds the captured update to ([\d.e-]+) on the losses and ([\d.e-]+) on "
                          r"the weights", "equivalence tolerances")
        with open(os.path.join(ROOT, "tests", "test_dreamer_graph.py"), encoding="utf-8") as handle:
            src = handle.read()
        self.assertIn(f"loss_gap, {m.group(1)}", src, "the loss tolerance no longer matches the test")
        self.assertIn(f"param_gap, {m.group(2)}", src, "the weight tolerance no longer matches the test")

    def test_the_trainer_table_row(self):
        s = self.ab["summary"]["rep 1"]
        m = re.search(r"^\| `train_dreamer\.py`, captured update \|.*\|$", self.readme, re.M)
        self.assertIsNotNone(m, "the captured-update row of the trainer table is gone")
        cells = [c.strip() for c in m.group(0).strip("|").split("|")]
        self.check("trainer-row numbers", nums(cells[2]),
                   [self.ab["config"]["updates_per_run"], s["update_phase_graph_s"],
                    s["update_phase_eager_s"]])
        self.assertIn(str(round(s["projected_1m"]["graph_h"], 1)), cells[3],
                      "the row's 1M figure is not the rep-1 projection")


class TestReadmePhase4RaceCells(ReadmeGate, unittest.TestCase):
    """The Phase-4 tables and the retired-draw prose, cell by cell, against the artifacts.

    These are the numbers this repository has already had to retract once, and until now nothing
    in CI compared the typed table against `phase4_race_50ep.json`, `phase4_n1_vs_50ep.json` or the
    paired-test file. A transposed digit here is exactly the defect class that cost the phase its
    headline.
    """

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            self.readme = handle.read()
        start = self.readme.index(PHASE4_SECTION)
        end = self.readme.index(PHASE4_HISTORY, start)
        self.block = self.readme[start:end]
        hist_end = self.readme.index(PHASE4_ENV_NOTE, end)
        self.hist = self.readme[end:hist_end]
        self.race, self.n1art, self.paired = self.read_artifacts(RACE_50EP, N1_ART, PAIRED_50EP)
        self.what = "Phase-4 race"
        self.bad = []

    def row(self, label, cols):
        """The row's cells, or None with the miss recorded - one rename must not hide the rest."""
        try:
            return [self.cell(label, c) for c in range(cols)]
        except AssertionError as exc:
            self.bad.append(str(exc))
            return None

    def test_the_50_episode_table_matches_the_race_artifact(self):
        for label, key in RACE_ROWS.items():
            cells = self.row(label, 5)
            if cells is None:
                continue
            s = self.race["models"].get(key)
            self.assertIsNotNone(s, f"{key} is not in phase4_race_50ep.json")
            self.check(label, [snums(cells[c])[0] for c in (1, 2, 3, 4)],
                       [s["mean"], s["std"], s["min"], s["max"]], places=2)

    def test_the_historical_table_matches_the_recorded_draws(self):
        self.block = self.hist  # cell() reads the block under test
        for label, key in N1_ROWS.items():
            cells = self.row(label, 2)
            if cells is None:
                continue
            self.check(f"{label} (n=1)", snums(cells[1])[0],
                       self.n1art["models"][key]["n1_score"], places=2)

    def test_every_retired_draw_is_reported_where_its_distribution_puts_it(self):
        """Item 7 quotes each retired draw next to its sigma and its win count: check all three."""
        for name, pattern, fields in N1_PROSE:
            m = self.sentence(pattern, f"the {name} retired-draw sentence")
            art = self.n1art["models"][name]
            self.check(f"{name} retired draw against its own 50-episode distribution",
                       [float(g) for g in m.groups()], [float(art[f]) for f in fields], places=2)

    def test_the_bcq_teacher_verdict_is_the_paired_test_that_ran(self):
        p = next(r for r in self.paired["results"] if r["pair"] == PAIR_BCQ_TEACHER)
        m = self.sentence(r"BCQ is \*\*([\d.]+) below\*\* the teacher, 95% interval "
                          r"\[(-?[\d.]+), (-?[\d.]+)\], paired-t p=([\d.e-]+), .*?dz=(-?[\d.]+) "
                          r"- it wins \*\*(\d+) of the 50\*\*", "BCQ against the teacher")
        self.check("BCQ - Teacher", [float(m.group(1)), float(m.group(2)), float(m.group(3)),
                                     float(m.group(5)), int(m.group(6))],
                   [abs(p["mean_difference"]), p["bootstrap_95ci"][0], p["bootstrap_95ci"][1],
                    p["cohens_dz"], p["wins_a"]], places=2)
        self.check("BCQ - Teacher paired-t p", float(m.group(4)), p["paired_t_p"], places=4)

    def test_the_second_protocol_paragraph_is_the_curve_that_was_mined(self):
        """Item 9: the teacher's 100-episode evaluations, checked against both artifacts."""
        curve = self.read_artifacts(CURVE)[0]
        late, want = curve["after_400k_steps"], self.race["models"]["Teacher (Upper Bound)"]
        m = self.sentence(r"give min \*\*([\d.]+)\*\*, max \*\*([\d.]+)\*\*, mean \*\*([\d.]+)\*\*,"
                          r" against the seeded\s*50-episode table's \*\*([\d.]+)\*\*", "second protocol")
        self.check("after-400k late-checkpoint spread",
                   [float(m.group(1)), float(m.group(2)), float(m.group(3))],
                   [late["min"], late["max"], late["mean"]], places=2)
        self.check("seeded teacher mean", float(m.group(4)), want["mean"], places=2)

        n1 = self.n1art["models"]["Teacher (Upper Bound)"]["n1_score"]
        m = self.sentence(r"final evaluation, \*\*([\d.]+)\*\*, is within ([\d.]+) of\s*"
                          r"the single-episode record's (\d+(?:\.\d+)?)", "final evaluation")
        self.check("final evaluation", [float(m.group(1)), float(m.group(2))],
                   [curve["final_eval"], round(abs(n1 - curve["final_eval"]), 2)], places=2)
        self.check("recorded single draw", float(m.group(3)), n1, places=2)

    def test_the_paired_rows_the_table_quotes_are_in_the_artifact(self):
        # BCQ against the teacher has its own test; it is stated in prose, not as an "X - Y = n"
        # gap, because the number the retired table published for it was a single episode.
        for pair, label in (("BC - Teacher (Upper Bound)", "BC " + MINUS + " Teacher"),
                            ("BCQ - BC+SAC (Regularized)",
                             "BCQ " + MINUS + " BC+SAC (Regularized)"),
                            ("Extra Trees - BCQ", "Extra Trees " + MINUS + " BCQ")):
            r = next(x for x in self.paired["results"] if x["pair"] == pair)
            m = self.sentence(re.escape(label) + r" = [\[*\-+]*([\d.]+)", f"the {pair} gap")
            self.check(pair, abs(float(m.group(1))), abs(r["mean_difference"]), places=2)

    def test_the_count_of_pairs_and_of_ties_is_what_the_artifact_says(self):
        """The lead-in's "two are ties, one won on rank, one clear gap" is read off every pair."""
        by_set = {}
        for r in self.paired["results"]:
            by_set.setdefault(frozenset(r["pair"].split(" - ")), r)
        self.assertEqual(4, len(by_set),
                         f"the README counts four distinct pairs, the artifact holds {len(by_set)}")
        # A "tie" here is nothing significant on either test; Extra Trees against BCQ wins the rank
        # test while the mean difference stays inside the interval, which is a different sentence.
        ties = [p for p, r in by_set.items()
                if not r["significant_at_95"] and r["wilcoxon_p"] >= 0.05]
        rank_only = [p for p, r in by_set.items()
                     if not r["significant_at_95"] and r["wilcoxon_p"] < 0.05]
        clear = [p for p, r in by_set.items() if r["significant_at_95"]]
        self.assertEqual(2, len(ties), f"the README calls two pairs ties, these are: {ties}")
        self.assertEqual(1, len(rank_only),
                         f"the README names one rank-only win, these are: {rank_only}")
        self.assertEqual([frozenset(PAIR_BCQ_TEACHER.split(" - "))], clear,
                         "the README says BCQ against the teacher is the only significant pair")



class TestReadmePhase4WeightProvenance(unittest.TestCase):
    """The historical-table note claims seven retired policies can still be replayed byte for byte.

    That is a statement about files, not about numbers, and it is what tells a reader whether a
    retired score is re-measurable at all. It costs one hash per model and fails the moment a
    checkpoint on disk is retrained away from its archived copy.
    """

    def test_the_archived_weights_are_the_weights_being_scored(self):
        with open(N1_ART, encoding="utf-8") as handle:
            scored = json.load(handle)["models"]
        self.assertEqual(13, len(scored),
                         "the README says thirteen policies were scored in that race")
        self.assertEqual(7, len(ARCHIVED_WEIGHTS),
                         "the README says seven of them are archived; the list changed")
        for name in ARCHIVED_WEIGHTS:
            live = os.path.join(P4, name)
            if not os.path.exists(live):
                self.skipTest(f"{name} is not in this checkout (*.pt files are not committed)")
            archived = sorted(glob.glob(os.path.join(MLRUNS, "**", name), recursive=True))
            self.assertTrue(archived,
                            f"the README calls {name} archived, but no copy is under mlruns/")
            self.assertEqual(_digest(live), _digest(archived[0]),
                             f"{name}: the copy on disk is no longer the copy MLflow holds, so "
                             "the retired scores quoted next to it are not replayable")


class TestReadmeTestCount(unittest.TestCase):
    """The README states how many tests the suite holds, three times over, and that number is free
    to go stale the moment anyone adds a test - including in the same edit that fixes it.

    It was wrong twice in one file (32 in one section, 41 in another) before being measured. The
    count here is discovered live rather than remembered, so the sentence cannot drift again.
    """

    COUNT_PATTERNS = [r"\*\*(\d+) tests",                    # the "Running the tests" claim
                      r"`Ran (\d+) tests in [\d.]+s \.\.\. OK",  # the citation of the run it came from
                      r"# (\d+) tests in \.venv"]             # the reproduce-block comment

    def test_every_published_test_count_is_the_suite_that_exists(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        n = unittest.TestLoader().discover(os.path.join(ROOT, "tests"),
                                           top_level_dir=ROOT).countTestCases()
        typed = [int(m.group(1)) for p in self.COUNT_PATTERNS for m in re.finditer(p, readme)]
        self.assertTrue(typed, "no published test count was found to check")
        for value in typed:
            self.assertEqual(n, value,
                             f"the README says {value} tests, `discover` reports {n}")


class TestReadmeDreamerScalingCells(ReadmeGate, unittest.TestCase):
    """The "what is left after capture" block against benchmarks/dreamer_update_scaling.json.

    That section quotes ten figures and two derived products, and it is the evidence behind
    flipping the Dreamer default, so it is pinned the same way the throughput cells are.
    """

    SCALING = os.path.join(ROOT, "benchmarks", "dreamer_update_scaling.json")
    BASE = "L50_B16_H15"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            self.readme = handle.read()
        start = self.readme.index("**What is left after capture")
        self.block = self.readme[start:self.readme.index("Three obligations come with capture",
                                                         start)]
        self.art, = self.read_artifacts(self.SCALING)
        self.what = "Dreamer scaling"
        self.bad = []

    def test_the_marginal_costs_are_the_slopes_the_sweep_measured(self):
        # Only the captured arm is quoted in prose: the eager arm measured 132.63, 254.94 and
        # 293.56 ms for identical code in three windows, so it is in the artifact and not in a claim.
        marg = self.art["marginal_ms_per_unit"]
        m = self.sentence(r"one imagination step\s*costs \*\*([\d.]+) ms\*\* and one "
                          r"world-model timestep \*\*([\d.]+) ms\*\*", "captured marginals")
        self.check("captured marginals", [float(m.group(1)), float(m.group(2))],
                   [marg["imag_horizon_captured"], marg["seq_len_captured"]], places=3)

    def test_the_baseline_and_batch_rows_are_the_measured_medians(self):
        base = self.art["configs"][self.BASE]
        m = self.sentence(r"so of a \*\*([\d.]+) ms\*\* update", "baseline")
        self.check("update at the shipped sizes", float(m.group(1)),
                   base["captured"]["median_ms"], places=2)
        m = self.sentence(r"batch 16 -> 64 -> 256 costs\s*\*\*([\d.]+) -> ([\d.]+) -> "
                          r"([\d.]+) ms\*\*", "the captured batch row")
        want = self.art["batch_scaling_captured"]
        self.check("captured batch scaling", [float(m.group(i)) for i in (1, 2, 3)],
                   [want["16"], want["64"], want["256"]], places=2)

    def test_the_two_loop_shares_are_multiplied_from_the_sweep_not_remembered(self):
        """`about 3.9 ms` and `about 7.1 ms` are the marginals times the shipped loop lengths."""
        marg = self.art["marginal_ms_per_unit"]
        seq = self.art["configs"][self.BASE]["seq_len"]
        imag = self.art["configs"][self.BASE]["imag_horizon"]
        m = self.sentence(r"the\s*imagination loop is about ([\d.]+) ms and the world model "
                          r"about ([\d.]+) ms", "loop shares")
        self.check("loop shares", [float(m.group(1)), float(m.group(2))],
                   [round(marg["imag_horizon_captured"] * imag, 1),
                    round(marg["seq_len_captured"] * seq, 1)], places=1)


class TestReadmePpoTenMillionCells(ReadmeGate, unittest.TestCase):
    """PPO at 10M drawn twice: the behaviour replicates and the clock does not, so both are gated.

    The paragraph was rewritten by its own second draw - the sentence published from draw 1, that the
    whole PPO curve costs less clock than a SAC 5M arm, is false in draw 2's window - so this class
    pins the two-draw prose against `benchmarks/ppo_10m_draws.json`, keeps the draw-1 curve figures
    from `ppo_10m_vs_sac.json`, and fails if the cross-window caveat that now travels with the rate
    ratios is dropped.
    """

    DRAWS = os.path.join(ROOT, "benchmarks", "ppo_10m_draws.json")
    PPO = os.path.join(ROOT, "benchmarks", "ppo_10m_vs_sac.json")
    PAIR = os.path.join(ROOT, "benchmarks", "trainer_pair_ppo_sac.json")
    DOSE = os.path.join(ROOT, "benchmarks", "utd_dose_draws.json")
    START = "**The re-run happened, and the objection was the budget"
    END = "The Dreamer row was the whole story of the eager build"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            self.readme = handle.read()
        start = self.readme.index(self.START)
        self.block = self.readme[start:self.readme.index(self.END, start)]
        self.draws, self.art, self.pair, self.dose = self.read_artifacts(
            self.DRAWS, self.PPO, self.PAIR, self.DOSE)
        self.bad = []
        self.what = "PPO 10M"

    def test_both_table_rows_are_the_two_windows_of_one_recipe(self):
        d7 = self.draws["draws"]["seed7"]
        d8 = self.draws["draws"]["seed8"]
        rows = re.findall(r"\| `train_walker\.py --algo ppo`, 10M(?: again)? \|[^|]*\| "
                          r"([\d,]+) steps in ([\d,]+) s = ([\d,.]+)/s \| ([\d,.]+) s \|", self.readme)
        self.assertEqual(len(rows), 2, "the trainer table should carry both PPO 10M windows")
        for got, want in zip(rows, (d7, d8)):
            self.check("table row", [got[0], got[1], got[2], got[3]],
                       [want["collected_steps"], want["wall_clock_seconds"],
                        want["env_steps_per_second"], want["seconds_per_1m"]], places=1)
        for want in (d7, d8):
            self.assertEqual(want["collected_steps"] % (2048 * 32), 0,
                             "PPO no longer checkpoints at 2048x32 rollout boundaries")
            self.assertLess(want["collected_steps"], 10000000)
            self.assertEqual(want["env_steps_per_second"],
                             round(want["collected_steps"] / want["wall_clock_seconds"], 1))

    def test_the_curve_figures_in_the_prose_are_the_two_draws_curves(self):
        d7, d8 = self.draws["draws"]["seed7"], self.draws["draws"]["seed8"]
        m = self.sentence(r"the mean climbs from \*\*(-?[\d,.]+)\*\* at the\s*first rollout boundary to "
                          r"\*\*([\d,.]+)\*\* at ([\d,]+) in draw 1, and from\s*\*\*([\d,.]+)\*\* to "
                          r"\*\*([\d,.]+)\*\* in\s*draw 2", "the two climbs")
        self.check("draw 1 first mean", m.group(1),
                   d7["curve"]["s1048576"]["mean"], places=2)
        self.check("draw 1 peak and its step", [m.group(2), m.group(3)],
                   [d7["peak_mean"]["mean"], d7["peak_mean"]["step"]], places=2)
        self.check("draw 2 first and final mean", [m.group(4), m.group(5)],
                   [d8["curve"]["s1048576"]["mean"], d8["final"]["mean"]], places=2)
        m = self.sentence(r"best reach column is \*\*(\d+)% \((\d+) of (\d+)\)\*\* at ([\d,]+) in one "
                          r"draw and \*\*(\d+)% \((\d+) of (\d+)\)\*\* at\s*([\d,]+) in the other, with "
                          r"closest approach near \*\*([\d.]+) m\*\* and\s*\*\*([\d.]+) m\*\*",
                          "the two best-reach cells")
        self.check("draw 1 best reach", [m.group(1), m.group(2), m.group(4), m.group(9)],
                   [d7["best_reach"]["reached_pct"], int(d7["best_reach"]["reached_pct"]) * 20 // 100,
                    d7["best_reach"]["step"], d7["best_reach"]["mean_min_target_distance"]], places=2)
        self.check("draw 2 best reach", [m.group(5), m.group(6), m.group(8), m.group(10)],
                   [d8["best_reach"]["reached_pct"], int(d8["best_reach"]["reached_pct"]) * 20 // 100,
                    d8["best_reach"]["step"], d8["best_reach"]["mean_min_target_distance"]], places=2)
        self.assertGreater(d7["best_reach"]["reached_pct"], 0)
        self.assertGreater(d8["best_reach"]["reached_pct"], 0,
                           "draw 2 no longer reaches, so the replication claim has to be rewritten")

    def test_the_clock_spread_refutes_the_draw_one_sentence(self):
        s = self.draws["same_recipe_two_windows"]
        d7, d8 = self.draws["draws"]["seed7"], self.draws["draws"]["seed8"]
        m = self.sentence(r"cost \*\*([\d,]+) s\*\* in one window \(9,961,472 steps at\s*\*\*([\d,.]+) "
                          r"env-steps/s\*\*\) and \*\*([\d,]+) s\*\* in the next\s*\(\*\*([\d,.]+)/"
                          r"s\*\*\) - a\s*\*\*([\d.]+)x\*\* spread", "the two clocks")
        self.check("draw clocks and rates", [m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)],
                   [d7["wall_clock_seconds"], d7["env_steps_per_second"], d8["wall_clock_seconds"],
                    d8["env_steps_per_second"], s["spread_ratio"]], places=2)
        self.assertEqual(s["spread_ratio"], round(d8["wall_clock_seconds"]
                                                  / d7["wall_clock_seconds"], 3))
        cross = self.draws["against_sac_5m_arms_other_windows"]
        m = self.sentence(r"([\d,]+) s is \*more\* than SAC's ratio-1 arms \(([\d,]+) s and ([\d,]+) s\) "
                          r"and\s*less than its ratio-4 arms\s*\(([\d,]+) s and ([\d,]+) s\)",
                          "the SAC rows")
        self.check("ppo clock", m.group(1), d8["wall_clock_seconds"], places=0)
        self.check("sac ratio 1 clocks", [m.group(2), m.group(3)], cross["sac_ratio1_seconds"],
                   places=0)
        self.check("sac ratio 4 clocks", [m.group(4), m.group(5)], cross["sac_ratio4_seconds"],
                   places=0)
        ppo_seconds = int(m.group(1).replace(",", ""))
        self.assertTrue(all(ppo_seconds > s1 for s1 in cross["sac_ratio1_seconds"]),
                        "draw 2's PPO is no longer dearer than both ratio-1 SAC arms")
        self.assertTrue(all(ppo_seconds < s4 for s4 in cross["sac_ratio4_seconds"]),
                        "draw 2's PPO is no longer cheaper than both ratio-4 SAC arms")

    def test_the_rate_ratios_survive_labelled_as_cross_window(self):
        for phrase in ("cross-window numbers wearing a within-window look",
                       "no SAC arm ran in either PPO window"):
            self.assertIn(phrase, re.sub(r"\s+", " ", self.block),
                          f"the paragraph lost the caveat that says {phrase!r}")
        ratios = self.art["rate_ratios"]
        m = self.sentence(r"draw 1 \*\*([\d.]+)x\*\*|\*\*([\d.]+)x\*\* against SAC\s*"
                          r"at ratio 1, \*\*([\d.]+)x\*\* against the 8-env screens", "the old ratios")
        quoted = [g for g in m.groups() if g]
        self.assertIn(round(ratios["ppo_over_sac_n32_r1"], 1), [float(q) for q in quoted])
        self.assertIn(ratios["ppo_over_sac_n8_r1"], [float(q) for q in quoted])

    def test_the_behavioural_limits_quoted_are_in_the_curves(self):
        d7, d8 = self.draws["draws"]["seed7"], self.draws["draws"]["seed8"]
        r7 = [v["reached_target_pct"] for v in d7["curve"].values()]
        r8 = [v["reached_target_pct"] for v in d8["curve"].values()]
        m = self.sentence(r"inside draw 1 the reach column\s*swings \*\*(\d+) to (\d+)%\*\* across ten "
                          r"checkpoints, draw 2's \*\*(\d+) to (\d+)%\*\*", "the two swings")
        self.check("swings", [m.group(1), m.group(2), m.group(3), m.group(4)],
                   [min(r7), max(r7), min(r8), max(r8)], places=0)
        self.assertEqual(len(d7["curve"]), 10)
        self.assertEqual(len(d8["curve"]), 10)
        m = self.sentence(r"\*\*([\d.]+) to ([\d.]+) per episode\*\* at the top of draw 1 against the "
                          r"1M arm's (?:\*\*)?([\d.]+)", "the falls")
        f7 = [v["falls_per_episode"] for k, v in d7["curve"].items() if int(k[1:]) >= 6000000]
        self.check("falls band and the 1M arm", [m.group(1), m.group(2), m.group(3)],
                   [min(f7), max(f7), self.pair["arms"]["ppo_v9"]["falls_per_episode"]], places=2)

    def test_both_draws_are_the_same_instrument(self):
        for tag, draw in self.draws["draws"].items():
            self.assertEqual(draw["device"], "cuda", tag)
            self.assertEqual(draw["torch_threads"], 24, tag)
            self.assertEqual(draw["scored_in_version"], "standup_balance_walk_curriculum_v9", tag)
            self.assertEqual(draw["physics_preset"], "v9", tag)
            self.assertEqual(draw["seconds_per_1m"],
                             round(draw["wall_clock_seconds"] / (draw["collected_steps"] / 1e6), 1),
                             tag)
            self.assertGreater(draw["scoring_seconds"], 0, tag)
        self.assertEqual(self.draws["draws"]["seed7"]["seed"], 7)
        self.assertEqual(self.draws["draws"]["seed8"]["seed"], 8)
        self.assertIn("num_envs=32", self.draws["protocol"])
        self.assertIn("target_forward_velocity=1.2", self.draws["protocol"])

class TestReadmeDreamerLoopSplitCells(ReadmeGate, unittest.TestCase):
    """The iteration-attribution table and its prose against benchmarks/dreamer_loop_split.json.

    This is the section that replaces an inherited claim ("~96% of the wall clock is the learner")
    with a measurement, so every cell is recomputed from the artifact - including the percentages,
    which are the shares of the attributed iteration rather than stored numbers.
    """

    SPLIT = os.path.join(ROOT, "benchmarks", "dreamer_loop_split.json")
    START = "**Where a whole iteration goes"
    END = "**Cross-checked against the trainer itself"
    ROWS = {
        "policy pass (`actor.get_action(...).cpu()`)": "policy pass",
        "`envs.step` (4 envs, sync)": "envs.step",
        "recurrent-state pass (encoder + RSSM + posterior + reset loop)": "state pass",
        "replay buffer write": "rb.add",
        "replay sample + starting latents": "rb.sample",
        "captured update": "captured update",
    }

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.art, = self.read_artifacts(self.SPLIT)
        self.what = "Dreamer loop-split"
        self.bad = []

    def test_every_table_cell_is_a_measured_segment_and_its_share(self):
        seg, attributed = self.art["segments_ms"], self.art["attributed_iteration_ms"]
        for label, key in self.ROWS.items():
            self.check(f"{key} ms", self.cell(label, 1), seg[key], places=2)
            self.check(f"{key} share", self.cell(label, 2).rstrip("%"),
                       round(100.0 * seg[key] / attributed, 1), places=1)
        self.check("attributed iteration", self.cell("attributed iteration", 1), attributed,
                   places=2)
        # The attributed total has to be the sum of the rows printed above it, not a fourth number.
        self.check("segments sum to the attributed total", sum(seg.values()), attributed, places=1,
                   slack=0.05)

    def test_the_prose_figures_divide_out_of_the_same_artifact(self):
        m = self.sentence(r"with no boundary syncs is \*\*([\d.]+) ms\*\*, which is "
                          r"([\d.]+) env-steps/s", "the un-synced loop")
        self.check("whole loop ms", float(m.group(1)), self.art["whole_loop_ms"], places=2)
        self.check("env-steps/s at that loop", float(m.group(2)),
                   self.art["env_steps_per_s_at_this_loop"], places=1)
        m = self.sentence(r"the collection side is now only \*\*([\d.]+)%\*\*", "collection share")
        self.check("collection share", float(m.group(1)), self.art["collection_share_pct"], places=1)
        m = self.sentence(r"the recurrent-state pass is ([\d.]+) ms of that ([\d.]+) ms",
                          "the state pass inside the collection total")
        self.check("state pass and collection total", [float(m.group(1)), float(m.group(2))],
                   [self.art["segments_ms"]["state pass"], self.art["collection_ms"]], places=2)

    def test_the_window_and_rep_count_the_run_recorded_are_the_ones_quoted(self):
        m = self.sentence(r"repeats (\d+) iterations in one process and one window "
                          r"\(`benchmarks/dreamer_loop_split\.json`, (\d+) other CUDA contexts\)",
                          "the run's own provenance")
        self.check("reps and GPU contexts", [int(m.group(1)), int(m.group(2))],
                   [self.art["reps"], self.art["gpu_other_contexts"]], places=0)
        self.assertEqual(self.art["diverged_at_rep"], None,
                         "the loop diverged during the timed reps; the segments are not a "
                         "measurement of a live update any more")


class TestReadmeDreamerRealRateCells(ReadmeGate, unittest.TestCase):
    """The measured 1M-hour figure and the two-budget run behind it.

    Three instruments have to stay in the sentence: the isolated update sweep, the in-process loop
    split and this real-trainer subtraction. The claim being gated is not just that the numbers are
    the artifacts', but that they still agree - "three instruments agreeing within 3%" is a property
    of the measurements, and if one of them moves the prose has to say so instead of asserting it.
    """

    RATE = os.path.join(ROOT, "benchmarks", "dreamer_real_rate.json")
    SPLIT = os.path.join(ROOT, "benchmarks", "dreamer_loop_split.json")
    SCALING = os.path.join(ROOT, "benchmarks", "dreamer_update_scaling.json")
    START = "**Cross-checked against the trainer itself"
    END = "The batch axis is the second version"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = re.sub(r"\s+", " ", readme[start:readme.index(self.END, start)])
        self.readme_full = readme
        self.rate, self.split, self.scaling = self.read_artifacts(self.RATE, self.SPLIT, self.SCALING)
        self.what = "Dreamer real-rate"
        self.bad = []

    def test_the_two_budget_slopes_are_the_ones_the_run_recorded(self):
        m = self.sentence(r"\*\*([\d.]+) ms\*\* per iteration with the update against "
                          r"\*\*([\d.]+) ms\*\* without it, so \*\*([\d.]+) ms\*\* is the\s*captured "
                          r"update", "the two budgets")
        self.check("with-update ms/iter", float(m.group(1)),
                   self.rate["captured"]["ms_per_iteration"], places=2)
        self.check("collect-only ms/iter", float(m.group(2)),
                   self.rate["collect"]["ms_per_iteration"], places=2)
        self.check("update ms/iter", float(m.group(3)),
                   self.rate["update_ms_per_iteration"], places=2)
        m = self.sentence(r"At \*\*([\d.]+) env-steps/s\*\* that is\s*\*\*([\d.]+) h per 1M",
                          "the measured rate and the hours")
        self.check("env-steps/s", float(m.group(1)), self.rate["captured"]["env_steps_per_s"],
                   places=1)
        self.check("hours per 1M", float(m.group(2)), self.rate["hours_per_1m_env_steps"], places=2)

    def test_the_three_instruments_still_agree_on_the_update(self):
        """Prose names three independent measurements of the same 12-ms object."""
        isolated = self.scaling["configs"]["L50_B16_H15"]["captured"]["median_ms"]
        in_loop = self.split["segments_ms"]["captured update"]
        subtraction = self.rate["update_ms_per_iteration"]
        m = self.sentence(r"next to the ([\d.]+) ms the in-process split attributed and the "
                          r"([\d.]+) ms the isolated\s*sweep recorded", "the other two instruments")
        self.check("in-process split", float(m.group(1)), in_loop, places=2)
        self.check("isolated sweep", float(m.group(2)), isolated, places=2)
        widest = max(isolated, in_loop, subtraction) / min(isolated, in_loop, subtraction)
        m = self.sentence(r"three instruments agreeing within (\d+)%", "the agreement bound")
        self.check("agreement bound", float(m.group(1)), round((widest - 1) * 100), places=0,
                   slack=0.5)
        self.assertLess(widest, 1.1, "the three measurements of the update no longer agree to within "
                                     "10%; the prose has to stop claiming they do")

    def test_the_provenance_the_paragraph_claims_is_the_run_that_ran(self):
        budgets = self.rate["config"]["budgets"]
        self.assertEqual([20000, 40000], budgets, "the prose says 20,000 and 40,000 env steps")
        m = self.sentence(r"(`benchmarks/dreamer_real_rate\.json`, (\d+) other CUDA\s+contexts)",
                          "the window the run recorded")
        self.check("GPU contexts", int(m.group(2)), self.rate["gpu_other_contexts"], places=0)
        m = self.sentence(r"the two with-update slopes came out ([\d.]+) and ([\d.]+) ms, "
                          r"the two collect-only ones ([\d.]+) and ([\d.]+) ms", "the rep spread")
        self.check("with-update reps", [float(m.group(1)), float(m.group(2))],
                   self.rate["captured"]["per_rep_ms_per_iteration"], places=2)
        self.check("collect reps", [float(m.group(3)), float(m.group(4))],
                   self.rate["collect"]["per_rep_ms_per_iteration"], places=2)

    def test_the_trainer_table_row_carries_the_measured_hours(self):
        """The summary table now quotes both the old projection and this measurement."""
        m = re.search(r"^\| `train_dreamer\.py`, captured update \|.*\|$", self.readme_full, re.M)
        self.assertIsNotNone(m, "the captured-update row of the trainer table is gone")
        cell = m.group(0).strip("|").split("|")[-1]
        self.assertIn("3.6", cell, "the row no longer says what the short A/B projected")
        self.check("the row's measured hours", nums(cell)[-1],
                   self.rate["hours_per_1m_env_steps"], places=2)

    def test_the_startups_the_gate_aware_model_predicts(self):
        """The stored startup fields are re-derived from the raw totals, and the prose quotes them.

        The naive intercept was negative, because the first `learning_starts / num_envs` iterations of
        every run carry no update; the artifact's fields are the gate-aware version, so this recomputes
        them from the raw seconds rather than trusting the bench's own arithmetic.
        """
        rate, budgets = self.rate, self.rate["config"]["budgets"]
        it1 = budgets[0] / rate["config"]["num_envs"]
        gate_iters = rate["config"]["learning_starts"] / rate["config"]["num_envs"]
        want = {}
        for label, per_iter in (("collect", rate["collect"]["ms_per_iteration"]),
                                ("captured", rate["captured"]["ms_per_iteration"])):
            t1 = statistics.median(rate["raw_totals_s"][f"{label}_{budgets[0]}"])
            extra = gate_iters * rate["update_ms_per_iteration"] / 1e3 if label == "captured" else 0.0
            want[label] = round(t1 - per_iter * it1 / 1e3 + extra, 1)
        self.check("collect arm startup recomputed", rate["collect"]["startup_s"], want["collect"],
                   places=1)
        self.check("captured arm startup recomputed", rate["captured"]["startup_s"],
                   want["captured"], places=1)
        m = self.sentence(r"the process startup comes back as \*\*([\d.]+) s\*\* without the "
                          r"update and \*\*([\d.]+) s\*\*\s*with it", "the two startups")
        self.check("startups quoted", [float(m.group(1)), float(m.group(2))],
                   [want["collect"], want["captured"]], places=1)


class TestReadmeGpuWindowCells(unittest.TestCase):
    """The GPU-guard paragraph: its thresholds are the code's, and its costs are the artifact's.

    Two independent sources, because the paragraph claims both a policy (refuse above X, at budgets
    of Y) and measured numbers (checkpoint bytes, milliseconds, per-1M cost). A drift in either half
    would make the README tell a reader to do something the code does not do.
    """

    COST = os.path.join(ROOT, "benchmarks", "dreamer_checkpoint_cost.json")
    BLOCK_START = "**What was done about it.**"
    BLOCK_END = "**The dial that is not free"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.BLOCK_START)
        self.block = re.sub(r"\s+", " ", readme[start:readme.index(self.BLOCK_END, start)])
        with open(self.COST, encoding="utf-8") as handle:
            self.cost = json.load(handle)

    def find(self, pattern, what):
        m = re.search(pattern, self.block)
        self.assertIsNotNone(m, f"the {what} sentence was reworded; re-point this test at it")
        return m

    def test_the_thresholds_in_prose_are_the_ones_the_module_uses(self):
        import utils.gpu_window as gw
        m = self.find(r"A CUDA run of \*\*(\d+) steps\*\* or more is refused above a "
                      r"\*\*(\d+) MiB\*\* ceiling", "guard threshold")
        self.assertEqual(gw.LONG_RUN_STEPS, int(m.group(1)))
        self.assertEqual(gw.DEFAULT_CEILING_MIB, int(m.group(2)))

    def test_the_checkpoint_default_matches_the_documented_interval(self):
        with open(os.path.join(ROOT, "train_dreamer.py"), encoding="utf-8") as handle:
            src = handle.read()
        self.assertIn('parser.add_argument("--checkpoint-interval", type=int, default=50000)', src,
                      "the README says Dreamer now defaults to a 50k interval; the code decides")

    def test_the_cost_figures_come_from_the_artifact(self):
        mib = self.cost["checkpoint_bytes"] / 2 ** 20
        m = self.find(r"\*\*([\d.]+) MiB\*\* taking \*\*([\d.]+) ms\*\* to write, so (\d+) saves "
                      r"per 1M steps cost \*\*([\d.]+) s\*\* and ([\d.]+) MiB",
                      "checkpoint cost")
        for typed, want in ((m.group(1), mib), (m.group(2), self.cost["save_ms_median"]),
                            (m.group(4), self.cost["per_1m_steps"]["50000"]["write_s"]),
                            (m.group(5), self.cost["per_1m_steps"]["50000"]["disk_mib"])):
            self.assertAlmostEqual(float(typed), round(want, 2), places=2,
                                   msg=f"README says {typed}, artifact gives {want}")
        self.assertEqual(self.cost["per_1m_steps"]["50000"]["saves"], int(m.group(3)))

    def test_the_driver_and_card_quoted_as_wsl_are_this_box(self):
        """The WSL claim is that it is the *same* GPU, so its numbers must be the ones Windows sees."""
        import utils.gpu_window as gw
        m = self.find(r"\(`([\d.]+)`, `(\d+)` MiB total\)", "WSL card identity")
        window = gw.read_gpu_window()
        if not window:
            self.skipTest("no NVIDIA GPU visible to nvidia-smi in this checkout")
        self.assertEqual(window["gpu"]["driver_version"], m.group(1))
        self.assertEqual(window["gpu"]["memory_total_mib"], int(m.group(2)))


class TestReadmeJaxProbeCells(unittest.TestCase):
    """The learner-port paragraph: every figure comes from one of the two artifacts, checked."""

    JAX = os.path.join(ROOT, "benchmarks", "jax_rssm_imagination.json")
    SCALING = os.path.join(ROOT, "benchmarks", "dreamer_update_scaling.json")

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index("**The learner was the other half")
        self.block = re.sub(r"\s+", " ", readme[start:readme.index("`Dockerfile.mjx`", start)])
        with open(self.JAX, encoding="utf-8") as handle:
            self.jax = json.load(handle)
        with open(self.SCALING, encoding="utf-8") as handle:
            self.marg = json.load(handle)["marginal_ms_per_unit"]

    def find(self, pattern, what):
        m = re.search(pattern, self.block)
        self.assertIsNotNone(m, f"the {what} sentence was reworded; re-point this test at it")
        return m

    def test_the_two_loop_costs_are_the_ones_measured(self):
        m = self.find(r"Forward only: \*\*([\d.]+) ms\*\* for the 15 steps against the "
                      r"\*\*([\d.]+) ms\*\* the captured torch path", "loop costs")
        horizon = int(self.jax["protocol"].split("horizon ")[1].split(",")[0])
        self.assertEqual(float(m.group(1)), self.jax["jax_loop_ms"])
        self.assertEqual(float(m.group(2)),
                         round(self.marg["imag_horizon_captured"] * horizon, 2))

    def test_the_second_probe_is_gated_against_its_own_artifact(self):
        rerun_path = os.path.join(ROOT, "benchmarks", "jax_rssm_imagination_wsl_jax0112.json")
        with open(rerun_path, encoding="utf-8") as handle:
            rerun = json.load(handle)
        m = self.find(r"the rollout costs \*\*([\d.]+) ms\*\* forward-only and \*\*([\d.]+) ms\*\* "
                      r"with `value_and_grad`", "the second probe")
        self.assertEqual(float(m.group(1)), rerun["jax_loop_ms"],
                         "the forward-only figure is not the re-run's")
        self.assertEqual(float(m.group(2)), rerun["jax_forward_backward_ms"],
                         "the gradient figure is not the re-run's")
        ratio = rerun["jax_forward_backward_ms"] / round(
            self.marg["imag_horizon_captured"] * 15, 2)
        m = self.find(r"the gradient path is now\s*\*\*([\d.]+)x\*\* the captured torch loop",
                      "the re-run's ratio")
        self.assertAlmostEqual(float(m.group(1)), ratio, places=1,
                               msg="the multiple over the torch loop no longer divides out")
        m = self.find(r"\*\*([\d.]+) TFLOP/s\*\* against 15\.2, putting the\s*"
                      r"arithmetic floor at \*\*([\d.]+) ms\*\* and ([\d.]+)% of the loop",
                      "the re-run's floor")
        self.assertAlmostEqual(float(m.group(1)), rerun["gpu_fp32_tflops_measured"], places=1)
        self.assertEqual(float(m.group(2)), rerun["arithmetic_floor_ms"])
        self.assertAlmostEqual(float(m.group(3)),
                               100.0 * rerun["fraction_of_time_that_is_arithmetic"], places=1)
        self.assertGreater(rerun["gpu_fp32_tflops_measured"], self.jax["gpu_fp32_tflops_measured"],
                           "the paragraph's point is that the card was faster while the loop was "
                           "slower; if the floor no longer exceeds the first probe's, say so")
        self.assertGreater(rerun["jax_loop_ms"], self.jax["jax_loop_ms"],
                           "the same claim is made about the loop; it no longer holds")
        self.assertNotEqual(rerun["protocol"], self.jax["protocol"],
                            "the two captures have to be distinguishable as different instruments")

    def test_the_gradient_number_is_the_one_that_changed_the_conclusion(self):
        m = self.find(r"with `value_and_grad` the same loop costs \*\*([\d.]+) ms\*\*",
                      "forward+backward")
        self.assertEqual(float(m.group(1)), self.jax["jax_forward_backward_ms"])

    def test_the_arithmetic_context_is_derived_from_the_artifact(self):
        m = self.find(r"the loop is (\d+) MFLOP and the card measured \*\*([\d.]+) TFLOP/s\*\* "
                      r"on a square matmul, a floor of \*\*([\d.]+) ms\*\*", "arithmetic floor")
        self.assertEqual(int(m.group(1)), round(self.jax["loop_flops"] / 1e6))
        self.assertEqual(float(m.group(2)), self.jax["gpu_fp32_tflops_measured"])
        self.assertEqual(float(m.group(3)), self.jax["arithmetic_floor_ms"])


class TestReadmeLinks(unittest.TestCase):
    """Relative links in the README resolve to files in the repository.

    Added when three sections moved to docs/lab-notes.md: a pointer to a file that is not there is
    how documentation quietly loses its evidence.
    """

    def test_every_relative_link_points_at_a_committed_file(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        targets = [t for t in re.findall(r"\]\(([^)#][^)]*)\)", readme)
                   if not t.startswith(("http://", "https://", "mailto:"))]
        self.assertTrue(targets, "no relative links to check")
        for target in targets:
            self.assertTrue(os.path.exists(os.path.join(ROOT, target.replace("/", os.sep))),
                            f"README links to {target}, which is not in the repository")


REPLICATION = os.path.join(ROOT, "benchmarks", "throughput_replication.json")


class TestReadmeThroughputReplication(unittest.TestCase):
    """The re-measurement paragraph against the JSON that one `bench_env.py --mode all --reps 3` writes.

    These cells exist to pin a *correction*, not a headline: the section used to publish percentages
    of a window-dependent denominator, and the paragraph below re-derives them from the artifact.
    """

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index("**Re-measured in one process, three reps each**")
        # The block runs to the end of the section, because the sparse_info gain that the
        # replication prices is stated down in the numbered list, not beside the table.
        self.block = re.sub(r"\s+", " ", readme[start:readme.index(
            "**Physics was left alone on purpose.**", start)])
        with open(REPLICATION, encoding="utf-8") as handle:
            rows = json.load(handle)
        self.rows = {r["label"]: r for r in rows}
        self.physics = next(r for r in rows if r["label"].startswith("physics only"))
        self.env1 = next(r for r in rows if r["label"].startswith("gym env.step"))

    def row(self, prefix):
        return next(r for label, r in self.rows.items() if label.startswith(prefix))

    def find(self, pattern, what):
        m = re.search(pattern, self.block)
        self.assertIsNotNone(m, f"the {what} sentence was reworded; re-point this test at it")
        return m

    def test_the_microseconds_and_the_share_they_imply(self):
        skip = int(re.search(r"frame_skip=(\d+)", self.physics["note"]).group(1))
        physics_us = self.physics["us_per_step"] * skip
        python_us = self.env1["us_per_step"] - physics_us
        m = self.find(r"the Python side came out at\s*\*\*(\d+) µs\*\* again.*"
                      r"the physics side took (\d+) µs instead of \d+"
                      r".*Python share as (\d+)% rather than \d+%", "microseconds and share")
        # The README quotes what the harness prints, so the check formats the same way.
        self.assertEqual(m.group(1), f"{python_us:.0f}")
        self.assertEqual(int(m.group(2)), round(physics_us))
        self.assertEqual(int(m.group(3)), round(100 * python_us / self.env1["us_per_step"]))

    def test_the_spreads_and_ratios_the_paragraph_quotes(self):
        par = self.row("ParallelVectorEnv (n=32, per_worker=auto, sparse=True)")
        dense = self.row("ParallelVectorEnv (n=32, per_worker=auto, sparse=False)")
        sync = self.row("SyncVectorEnv (n=32, copy=True)")
        m = self.find(r"reached \*\*(\d+)%\*\*\s*on the physics row and (\d+)%\s*"
                      r"on the sparse-parallel row", "spreads")
        self.assertEqual(int(m.group(1)), round(self.physics["spread_pct"]))
        self.assertEqual(int(m.group(2)), round(par["spread_pct"]))
        m = self.find(r"sync-to-parallel ratio came out ([\d.]+)x", "sync to parallel")
        self.assertAlmostEqual(float(m.group(1)), round(par["steps_per_s"] / sync["steps_per_s"], 1),
                               places=1)
        m = self.find(r"(\d+\.\d+)x in the replication above", "sparse gain")
        self.assertAlmostEqual(float(m.group(1)),
                               round(par["steps_per_s"] / dense["steps_per_s"], 2), places=2)


class TestReadmeMlflowHistoryCells(unittest.TestCase):
    """The paragraph that quantifies the MLflow archive, against the sqlite file it describes.

    "One database, N runs, M metric rows" is published as the reason a reader should trust the
    archive, and it is the only figure in the README that grows by itself: every training run adds
    rows. So it is read back from `mlruns.db` in read-only mode here - total active runs, the split
    per experiment, the metric-row count and the schema revision - rather than remembered.
    """

    DB = os.path.join(ROOT, "mlruns.db")

    def setUp(self):
        if not os.path.exists(self.DB):
            self.skipTest("mlruns.db is not in this checkout")
        uri = "file:" + self.DB.replace(os.sep, "/") + "?mode=ro"
        self.con = sqlite3.connect(uri, uri=True, timeout=10)
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        self.whole = readme
        start = readme.index("The history lives in **one** database")
        # To the next heading, not a fixed character window: a 900-character one silently dropped the
        # last sentence of this paragraph the first time somebody added a clause to it, and the gate
        # for that sentence then failed with "reworded" when nothing had been reworded.
        end = readme.find("\n## ", start)
        self.block = re.sub(r"\s+", " ", readme[start:end if end > start else len(readme)])

    def tearDown(self):
        self.con.close()

    def find(self, pattern, what):
        m = re.search(pattern, self.block)
        self.assertIsNotNone(m, f"the {what} sentence was reworded; re-point this test at it")
        return m

    # The archive only grows, so an exact cell here can only ever be true while nothing is running -
    # the same reason the metric-row count below is a bound. Staleness is allowed; lying is not.
    STALENESS_LIMIT = 30

    def test_the_run_and_experiment_counts(self):
        active = self.con.execute("SELECT COUNT(*) FROM runs WHERE lifecycle_stage='active'").fetchone()[0]
        exps = self.con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        m = self.find(r"\*\*(\d+) active runs\*\* in (\d+) experiments", "the run and experiment counts")
        typed = int(m.group(1))
        self.assertLessEqual(typed, active,
                             f"the README claims {typed} active runs and the database holds {active}: "
                             f"over-claiming the archive is the dangerous direction")
        self.assertLessEqual(active - typed, self.STALENESS_LIMIT,
                             f"the published run count is {active - typed} runs behind the database; "
                             f"restate it (and its date) rather than let the bound go slack")
        self.assertEqual(int(m.group(2)), exps, "the README's experiment count is not the database's")

    def test_every_experiment_split_figure_is_its_own_run_count(self):
        want = dict(self.con.execute(
            "SELECT e.name, COUNT(r.run_uuid) FROM experiments e "
            "LEFT JOIN runs r ON r.experiment_id = e.experiment_id AND r.lifecycle_stage='active' "
            "GROUP BY e.name").fetchall())
        m = self.find(r"experiments \((.+?)\) and", "the per-experiment split")
        typed = {name: int(n) for n, name in re.findall(r"(\d+) `([A-Za-z0-9_.-]+)`", m.group(1))}
        self.assertEqual(set(typed), set(want),
                         "the README lists a different set of experiments than the database holds")
        for name, value in typed.items():
            self.assertLessEqual(value, want[name],
                                 f"{name}: README claims {value} runs, the database holds {want[name]}")
            self.assertLessEqual(want[name] - value, self.STALENESS_LIMIT,
                                 f"{name} is {want[name] - value} runs behind its published figure")

    def test_the_metric_row_count_and_the_schema_revision(self):
        rows = self.con.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        m = self.find(r"and \*\*([\d,]+) metric rows\*\* as of (\d{4}-\d{2}-\d{2})",
                      "the metric-row count and its date")
        # Deliberately a lower bound, not an equality: the trainers write metric rows continuously,
        # so a cell that must match the database exactly can only ever be true while nothing is
        # running - which is a gate that gets weakened the first time it fails. The dangerous
        # direction is claiming more history than exists, and that one is still caught.
        typed = int(m.group(1).replace(",", ""))
        self.assertLessEqual(typed, rows,
                             f"the README claims {typed:,} metric rows and the database holds {rows:,}")
        self.assertGreater(rows - typed, -1)
        self.assertLess(rows - typed, 50_000,
                        "the gap between the published row count and the database is larger than one "
                        "training run - re-measure rather than carry the old figure")
        m = re.search(r"`mlruns\.db` is at schema revision\s*`([0-9a-f]+)`", self.whole)
        self.assertIsNotNone(m, "the schema-revision sentence moved")
        rev = self.con.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        self.assertEqual(m.group(1), rev, "the published schema revision is not the file's")

    def test_the_stale_runs_it_says_were_closed_are_closed_and_tagged(self):
        closed = self.con.execute(
            "SELECT COUNT(*) FROM tags WHERE key='closed_as_stale'").fetchone()[0]
        m = self.find(r"(\w[\w-]*) runs killed mid-training were closed", "the closed-stale-run count")
        n = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5, "Six": 6, "Seven": 7,
             "Eight": 8, "Nine": 9, "Ten": 10}.get(m.group(1), None)
        if n is None:
            n = int(m.group(1)) if m.group(1).isdigit() else None
        self.assertIsNotNone(n, f"spell the count as a word the test understands, got {m.group(1)!r}")
        self.assertEqual(closed, n,
                         f"the README says {n} runs carry the closed_as_stale tag, the database has {closed}")

    def test_the_noise_figure_is_how_many_active_runs_a_test_left(self):
        """The archive's own admission: most of its run count is not research history."""
        noise = self.con.execute(
            "SELECT COUNT(*) FROM runs WHERE lifecycle_stage='active' AND ("
            "name LIKE '%integration\\_test%' ESCAPE '\\' OR name LIKE '%smoke%' "
            "OR name LIKE 'bench%' OR name LIKE '%\\_probe%' ESCAPE '\\')").fetchone()[0]
        m = self.find(r"and (\d+) of them carry an `integration_test`/`smoke`/`bench` name",
                      "the test-noise count")
        self.assertEqual(int(m.group(1)), noise,
                         "the README's share-of-noise figure is not the database's")


class TestReadmeGailAnomalyCells(unittest.TestCase):
    """The GAIL anomaly paragraph: its band arithmetic against the artifact, its code claims against source.

    The paragraph is a negative result, and negative results rot the same way positive ones do - if
    `evaluate_all.py` stopped loading the on-disk weights, or the trainer started printing a learned
    reward, the eliminations would stop being true while the prose kept asserting them.
    """

    N1_ART = os.path.join(ROOT, "benchmarks", "phase4_n1_vs_50ep.json")
    RETRAIN_ART = os.path.join(ROOT, "benchmarks", "phase4_gail_retrain.json")
    RETIRED_EARLIER = 1016.41  # the figure the README quoted before the retired table's 997.57
    START = "That distinction is what makes one retired figure stand out."
    END = "| Model Architecture | Final Score |"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = re.sub(r"\s+", " ", readme[start:readme.index(self.END, start)])
        with open(self.N1_ART, encoding="utf-8") as handle:
            self.gail = json.load(handle)["models"]["GAIL"]
        with open(self.RETRAIN_ART, encoding="utf-8") as handle:
            art = json.load(handle)
        self.art = art
        self.arms = art["arms"]
        self.cmp = art["comparison"]

    def find(self, pattern, what):
        m = re.search(pattern, self.block)
        self.assertIsNotNone(m, f"the {what} sentence was reworded; re-point this test at it")
        return m

    def test_the_band_and_the_gap_are_the_artifacts_numbers(self):
        m = self.find(r"seeded resets span ([\d.]+) to ([\d.]+), a band ([\d.]+) points wide",
                      "the band")
        self.assertEqual([float(m.group(1)), float(m.group(2))],
                         [self.gail["min"], self.gail["max"]], "the min/max are not the artifact's")
        self.assertAlmostEqual(float(m.group(3)),
                               round(self.gail["max"] - self.gail["min"], 2), places=2,
                               msg="the band width no longer subtracts out")
        m = self.find(r"quoted 1016\.41 for it: \*\*([\d.]+) above the best of the 50",
                      "the gap above the best episode")
        self.assertAlmostEqual(float(m.group(1)),
                               round(self.RETIRED_EARLIER - self.gail["max"], 2), places=2,
                               msg="1016.41 minus the best seeded episode is no longer this")

    def test_the_eliminations_are_what_the_source_still_says(self):
        with open(os.path.join(ROOT, "openai_walker", "evaluate_all.py"), encoding="utf-8") as h:
            self.assertIn("gail_model.pt", h.read(),
                          "evaluate_all.py no longer scores the on-disk GAIL weights, so the "
                          "paragraph's 'same policy' claim is void")
        with open(os.path.join(ROOT, "openai_walker", "train_irl_gail.py"), encoding="utf-8") as h:
            src = h.read()
        for needle in ("env.step(action)", "episode_reward += reward", "True Env Reward:",
                       "true_env_reward"):
            self.assertIn(needle, src, f"the trainer no longer contains {needle!r}; the reward"
                                       "-bookkeeping elimination has to be restated")

    def test_the_retrain_table_is_the_artifact_s_two_arms(self):
        found = re.findall(
            r"\| (June 2026|Retrain), [^|]*\| ([\d.]+) \| ([\d.]+) \| ([\d.]+) / ([\d.]+) \| "
            r"([\d.]+) \|", self.block)
        rows = {label: cells for label, *cells in found}
        self.assertEqual(set(rows), {"June 2026", "Retrain"},
                         "the two-arm GAIL table lost or renamed a row")
        want = {"June 2026": "june_control", "Retrain": "fresh_retrain"}
        for label, key in want.items():
            mean, std, mn, mx, train_max = rows[label]
            ev, tr = self.arms[key]["evaluation"], self.arms[key]["training_episodes"]
            self.assertEqual((float(mean), float(std), float(mn), float(mx), float(train_max)),
                             (ev["mean"], ev["std"], ev["min"], ev["max"], tr["max"]),
                             f"the {label} row is not what {key}'s artifact says")

    def test_the_retrain_arithmetic_sentences_subtract_out(self):
        m = self.find(r"mean ([\d.]+), std ([\d.]+), min ([\d.]+), max ([\d.]+) - so nothing in "
                      r"the harness moved", "the control reproduction")
        published = self.art["control_reproduces_published"]
        self.assertEqual([float(x) for x in m.groups()],
                         [published["measured"][k] for k in ("mean", "std", "min", "max")],
                         "the control row quoted in prose is not the control's measured row")
        self.assertTrue(published["exact"], "the control no longer reproduces the published row, "
                                            "so the sentence claiming it reproduced it exactly "
                                            "has to be rewritten")
        m = self.find(r"1016\.41 is \*\*([\d.]+) above the best episode the retrain",
                      "the gap above the retrain")
        self.assertAlmostEqual(float(m.group(1)), self.cmp["retired_above_fresh_max"], places=2)
        m = self.find(r"the mean moves ([\d.]+) points and the band goes from ([\d.]+) to "
                      r"([\d.]+) - ([\d.]+) times wider", "the run-to-run spread")
        self.assertAlmostEqual(float(m.group(1)), self.cmp["mean_delta_june_minus_fresh"], places=2)
        self.assertEqual([float(m.group(2)), float(m.group(3))],
                         [self.arms["june_control"]["evaluation"]["band"],
                          self.arms["fresh_retrain"]["evaluation"]["band"]])
        self.assertAlmostEqual(float(m.group(4)), self.cmp["band_ratio_fresh_over_june"], places=2)
        m = self.find(r"([\d]+ h [\d]+ min)", "the retrain's wall clock")
        seconds = self.arms["fresh_retrain"]["training_run"]["wall_clock_seconds"]
        self.assertEqual(m.group(1), f"{seconds // 3600} h {seconds % 3600 // 60} min",
                         "the quoted training time is not the recorded wall clock")

    def test_the_training_print_candidate_is_the_june_run_s_own_series(self):
        m = self.find(r"\*\*([\d]+) of its logged episodes sit at or above 1016\.41\*\*, the "
                      r"nearest being ([\d.]+) at step ([\d,]+), which is ([\d.]+) away",
                      "the candidate cause")
        nearest = self.cmp["nearest_june_training_episode"]
        self.assertEqual(int(m.group(1)), self.cmp["june_training_episodes_at_or_above_retired"])
        self.assertAlmostEqual(float(m.group(2)), nearest["value"], places=2)
        self.assertEqual(int(m.group(3).replace(",", "")), nearest["step"])
        self.assertAlmostEqual(float(m.group(4)), nearest["delta"], places=2)
        self.assertEqual(self.cmp["fresh_training_episodes_at_or_above_retired"], 0,
                         "the retrain now has episodes above the retired figure too, so the "
                         "asymmetry the paragraph rests on has to be restated")
        m = self.find(r"the retrain's best episode \(([\d.]+)\) and its last-20-episode mean "
                      r"\(([\d.]+), against June's ([\d.]+)\)", "the training-side comparison")
        fresh, june = (self.arms[k]["training_episodes"] for k in ("fresh_retrain", "june_control"))
        self.assertEqual([float(m.group(1)), float(m.group(2)), float(m.group(3))],
                         [fresh["max"], fresh["mean_of_last_20"], june["mean_of_last_20"]])

    def test_the_weights_behind_both_arms_are_the_ones_recorded(self):
        # The .pt files are gitignored, so a machine without them cannot check the bytes; it can
        # still check that the artifact was written against the pair the prose names.
        for key, named in (("june_control", "gail_model.pt"),
                           ("fresh_retrain", "gail_model_retrain_2026-10-05.pt")):
            spec = self.arms[key]["weights"]
            self.assertTrue(spec["file"].endswith(named),
                            f"{key}'s artifact names {spec['file']}, the prose says {named}")
            if spec["present"]:
                self.assertEqual(spec["sha256_on_disk"], spec["sha256"],
                                 f"{named} on disk is not the policy this artifact scored")


class TestReadmeStackedReplicationCells(ReadmeGate, unittest.TestCase):
    """The replicated n32 ladder and the two derived sentences that hang off it.

    The ratios this section published for a year came from `reps: 1` in a window whose own artifact
    says a training run shared the machine. Every figure here - medians, per-rep agreement, the
    1.17x / 1.40x, the startup subtraction and the 68% window gap - is recomputed from the two
    artifacts, because the point of the correction is arithmetic, not vibes.
    """

    REPS = os.path.join(ROOT, "benchmarks", "stacked_sac_n32_reps2.json")
    LADDER = os.path.join(ROOT, "benchmarks", "throughput_stacked_ladder.json")
    SPLIT = os.path.join(ROOT, "benchmarks", "throughput_replication.json")
    START = "\u26a0\ufe0f **That table was a single draw on a contended machine"
    END = "As the contended artifact's own note reads it"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.reps, self.ladder = self.read_artifacts(self.REPS, self.LADDER)
        with open(self.SPLIT, encoding="utf-8") as handle:
            self.par = next(r for r in json.load(handle)
                            if r["label"].startswith("ParallelVectorEnv (n=32, per_worker=auto, sparse=True)"))
        self.what = "stacked ladder replication"
        self.bad = []
        self.arms = self.reps["n32"]["arms"]
        self.med = {k: statistics.median(v) for k, v in self.arms.items()}

    def test_the_replicated_rows_are_the_two_reps_and_their_median(self):
        rows = {"original env + sync": "D32_orig_env_sync",
                "rewritten env + sync": "A32_new_env_sync",
                "rewritten env + parallel": "B32_new_env_parallel"}
        for label, key in rows.items():
            self.check(f"{label} reps", nums(self.cell(label, 1)), self.arms[key], places=1)
            shown = float(self.cell(label, 2))
            self.check(f"{label} median", shown, self.med[key], places=1)
            # The rate is what a reader re-divides from the median printed beside it, so it is gated
            # on that printed value; the artifact's unrounded median would give 1 env-step/s less.
            self.check(f"{label} env-steps/s", self.cell(label, 3), round(600000.0 / shown),
                       places=0)
        self.assertEqual(self.reps["n32"]["config"]["reps"], 2, "the prose says two reps per arm")
        old = self.ladder["n32_collection_only"]["seconds"]
        self.check("the contended draw", [old["D32_orig_env_sync"], old["A32_new_env_sync"],
                                          old["B32_new_env_parallel"]],
                   [540.9, 444.8, 290.9], places=1)

    def test_the_ratios_and_the_contention_lesson_subtract_out(self):
        d, a, b = (self.med[k] for k in ("D32_orig_env_sync", "A32_new_env_sync",
                                         "B32_new_env_parallel"))
        m = self.sentence(r"rewritten env \+ sync \| \d+(?:\.\d+)? / \d+(?:\.\d+)? \| \d+(?:\.\d+)? "
                          r"\| [\d,]+ \| \*\*([\d.]+)x\*\*", "the 1.17x cell")
        self.check("sync over committed", float(m.group(1)), round(d / a, 2), places=2)
        m = self.sentence(r"rewritten env \+ parallel \| \d+(?:\.\d+)? / \d+(?:\.\d+)? \| "
                          r"\d+(?:\.\d+)? \| [\d,]+ \| \*\*([\d.]+)x\*\*", "the 1.40x cell")
        self.check("parallel over committed", float(m.group(1)), round(d / b, 2), places=2)
        m = self.sentence(r"The two reps agree to ([\d.]+)%, ([\d.]+)% and ([\d.]+)%", "rep spreads")
        for i, key in enumerate(("D32_orig_env_sync", "A32_new_env_sync", "B32_new_env_parallel")):
            lo, hi = min(self.arms[key]), max(self.arms[key])
            self.check(f"rep spread {key}", float(m.group(i + 1)), round(100.0 * (hi - lo) / lo, 1),
                       places=1)
        m = self.sentence(r"the \*committed\* arm was\s*(?:\*\*)?([\d.]+)x faster",
                          "the contention factor")
        old = self.ladder["n32_collection_only"]["seconds"]["D32_orig_env_sync"]
        self.check("committed arm slowdown", float(m.group(1)), round(old / d, 1), places=1)

    def test_the_startup_subtraction_uses_its_own_two_artifacts(self):
        """~95 s of the 148.2 s: the median minus 600k steps at the replication's steady rate."""
        m = self.sentence(r"the boot cost is ~(\d+) s of the ([\d.]+) s", "the startup sentence")
        steady = self.par["steps_per_s"]
        want = round(self.med["B32_new_env_parallel"] - 600000.0 / steady)
        self.check("startup seconds", [float(m.group(1)), float(m.group(2))],
                   [want, self.med["B32_new_env_parallel"]], places=1)

    def test_the_summary_lines_requote_the_replicated_ratios(self):
        d, b = self.med["D32_orig_env_sync"], self.med["B32_new_env_parallel"]
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        m = re.search(r"come out at\s*\*\*([\d.]+)x and ([\d.]+)x\*\* on an idle machine", readme)
        self.assertIsNotNone(m, "the steady-state warning no longer quotes the replication")
        self.check("warning ratios", [float(m.group(1)), float(m.group(2))],
                   [round(d / self.med['A32_new_env_sync'], 2), round(d / b, 2)], places=2)
        m = re.search(r"and ([\d.]+)x when collection dominates, measured twice on an idle machine "
                      r"\(([\d.]+)x in the single contended", readme)
        self.assertIsNotNone(m, "the section summary no longer carries both readings")
        self.check("summary ratio", float(m.group(1)), round(d / b, 2), places=2)
        old = self.ladder["n32_collection_only"]["speedup_vs_D32"]["B32_new_env_parallel"]
        self.check("contended ratio in the summary", float(m.group(2)), old, places=1)


class TestReadmeSweepCells(ReadmeGate, unittest.TestCase):
    """The envs-per-worker curve as a three-rep measurement, against its own artifact."""

    SWEEP = os.path.join(ROOT, "benchmarks", "vec_backend_scaling_reps3.json")
    SPLIT = os.path.join(ROOT, "benchmarks", "throughput_replication.json")
    START = "**The envs-per-worker curve, with the same treatment**"
    END = "Three things mattered, and one deliberate non-change"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.prose = re.sub(r"\s+", " ", self.block)
        with open(self.SWEEP, encoding="utf-8") as handle:
            self.rows = {int(re.search(r"per_worker=(\d+)", r["label"]).group(1)): r
                         for r in json.load(handle)}
        with open(self.SPLIT, encoding="utf-8") as handle:
            self.par = next(r for r in json.load(handle)
                            if r["label"].startswith("ParallelVectorEnv (n=32, per_worker=auto, sparse=True)"))
        self.what = "envs-per-worker sweep"
        self.bad = []

    def test_every_cell_of_the_curve_is_the_median_its_run_recorded(self):
        for size, row in sorted(self.rows.items()):
            label = str(size)
            self.check(f"per_worker={size} rate", self.cell(label, 1), round(row["steps_per_s"]),
                       places=0)
            self.check(f"per_worker={size} spread", self.cell(label, 2).rstrip("%"),
                       row["spread_pct"], places=1)
        self.assertEqual({r["reps"] for r in self.rows.values()}, {3}, "the prose says three reps")
        self.assertEqual(sorted(self.rows), [1, 2, 3, 4, 6, 8],
                         "the curve cells a different set of groupings than the README lists")

    def test_the_flat_and_monotone_claims_hold_on_the_medians(self):
        rates = {k: round(v["steps_per_s"]) for k, v in self.rows.items()}
        self.assertEqual(max(rates.values()), rates[1],
                         "one env per worker is quoted as the fastest cell")
        after = [rates[k] for k in (4, 6, 8)]
        self.assertEqual(after, sorted(after, reverse=True),
                         "the prose calls the curve monotone from 4 up and it is not")
        # "inside each other's spread": the 3-vs-4 gap has to be smaller than the wider of the two
        # cells' own rep spread, which is what the sentence claims.
        lo, hi = min(rates[3], rates[4]), max(rates[3], rates[4])
        spread = max(self.rows[3]["spread_pct"], self.rows[4]["spread_pct"])
        self.assertLess(hi - lo, lo * spread / 100.0,
                        "3 and 4 no longer overlap within their own rep spread")
        self.assertIn("flat from 3 up, and\nmonotone after that", self.block,
                      "the reading sentence was reworded; re-point this test at it")

    def test_the_sixty_eight_percent_window_gap_is_two_artifacts_apart(self):
        m = re.search(r"measured ([\d,]+)\s*\n?env-steps/s here and ([\d,]+) in the morning "
                      r"replication[^-]*-\s*\*\*([\d]+)% apart\*\*", self.block)
        if m is None:
            m = re.search(r"measured ([\d,]+) env-steps/s here and ([\d,]+) in the morning "
                          r"replication of the same command - \*\*([\d]+)% apart\*\*",
                          re.sub(r"\s+", " ", self.block))
        self.assertIsNotNone(m, "the window-gap sentence was reworded; re-point this test at it")
        here, there = nums(m.group(1))[0], nums(m.group(2))[0]
        self.check("the two rates", [here, there],
                   [round(self.rows[1]["steps_per_s"]), round(self.par["steps_per_s"])], places=0)
        self.check("percent apart", float(m.group(3)), round(100.0 * (here / there - 1.0)), places=0,
                   slack=1.0)


class TestReadmeDreamerV3RunCells(ReadmeGate, unittest.TestCase):
    """The completed 1M-step Dreamer run: its eval statistics, and the wall clock its MLflow row holds.

    This is the first Phase-1 run to finish its full budget on env v9, and the paragraph's job is to
    keep it from reading like a score - so the same fields that make it a negative result (0 of 50
    inside the radius, closest approach, x-velocity) are gated as tightly as the mean.
    """

    P1 = os.path.join(ROOT, "benchmarks", "phase1_dreamer_v3_1m.json")
    P9 = os.path.join(ROOT, "benchmarks", "phase1_dreamer_v9_269k.json")
    ARS = os.path.join(ROOT, "benchmarks", "phase1_ars_v2_1m_v9.json")
    RATE = os.path.join(ROOT, "benchmarks", "dreamer_real_rate.json")
    DB = os.path.join(ROOT, "mlruns.db")
    START = "The second Phase-1 run to reach its full budget on v9"
    END = "What the task geometrically requires"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            self.readme = handle.read()
        start = self.readme.index(self.START)
        self.block = re.sub(r"\s+", " ", self.readme[start:self.readme.index(self.END, start)])
        load = lambda p: json.load(open(p, encoding="utf-8"))
        self.p1_art, self.p9_art = load(self.P1), load(self.P9)
        self.run = self.p1_art["models"]["dreamer_v3_1m"]
        self.partial = self.p9_art["models"]["dreamer_v9_269k"]
        self.ars = load(self.ARS)["models"]["ars_v9"]
        self.rate = load(self.RATE)
        self.what = "Dreamer v3 run"
        self.bad = []

    def find(self, pattern, what):
        m = re.search(pattern, self.block)
        self.assertIsNotNone(m, f"the {what} sentence was reworded; re-point this test at it")
        return m

    def test_the_eval_statistics_are_the_artifact_s(self):
        m = self.find(r"mean ([\d.-]+), median ([\d.-]+), std ([\d.-]+), min (-[\d.]+),\s*max "
                      r"([\d.]+), ([\d.]+) falls per episode", "the eval band")
        want = [self.run["mean"], self.run["median"], self.run["std"], self.run["min"],
                self.run["max"], self.run["falls_per_episode"]]
        for i, (typed, measured) in enumerate(zip(m.groups(), want)):
            self.assertAlmostEqual(float(typed), measured, places=2, msg=f"field {i}")
        m = self.find(r"\*\*0 of (\d+) inside the radius\*\*, mean closest approach\s*([\d.]+) m, "
                      r"mean x-velocity ([\d.-]+) m/s, standing at the end of the episode (\d+)%",
                      "the not-walking evidence")
        self.assertEqual([int(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))],
                         [self.run["episodes"], self.run["mean_min_target_distance"],
                          self.run["mean_x_velocity"], int(self.run["standing_at_end_pct"])])
        self.assertEqual(self.run["reached_target_pct"], 0.0, "the prose claims 0 of 50 reached it")
        self.assertEqual(self.run["global_step"], 1000000, "the run the paragraph describes")

    def test_the_two_comparisons_are_read_off_the_other_artifacts(self):
        m = self.find(r"the full budget moved\s*the mean from ([\d.]+) down to ([\d.]+) \(-([\d.]+)%\) "
                      r"and the std from ([\d.]+) up to ([\d.]+), on (\d+)\s*episodes rather than the "
                      r"(\d+) the partial one was scored on", "the partial-versus-full comparison")
        self.check("partial versus full mean and std",
                   [float(m.group(i)) for i in (1, 2, 4, 5)],
                   [self.partial["mean"], self.run["mean"], self.partial["std"], self.run["std"]],
                   places=2)
        self.check("the derived percentage", float(m.group(3)),
                   round(100.0 * (1.0 - self.run["mean"] / self.partial["mean"]), 1), places=1)
        self.check("episode counts", [int(m.group(6)), int(m.group(7))],
                   [self.run["episodes"], self.partial["episodes"]], places=0)
        self.assertEqual(self.partial["global_step"], 269404, "the partial checkpoint's step count")
        for art, name in ((self.p9_art, "partial"), (self.p1_art, "full")):
            self.assertEqual(art["device"], "cuda",
                             f"the {name} row must be scored on cuda: the retired comparison had one "
                             f"arm on cpu and the other on cuda, which is not a budget comparison")
        m = self.find(r"\(2\.893 m against ([\d.]+) m\) with a lower return", "the ARS comparison")
        self.assertAlmostEqual(float(m.group(1)), self.ars["mean_min_target_distance"], places=3)
        self.assertLess(self.run["mean"], self.ars["mean"],
                        "the paragraph says Dreamer's return is lower than ARS at the same budget")

    def test_the_retired_cpu_row_is_reproduced_by_a_committed_artifact(self):
        """The retired 6486.27 was a cpu score; this is the run that identifies it as one.

        Quoting the retired figure without this artifact would leave a reader with two different
        numbers for one checkpoint and no way to tell which command produced which.
        """
        path = os.path.join(ROOT, "benchmarks", "phase1_dreamer_v9_269k_cpu_reproduction.json")
        if not os.path.exists(path):
            self.skipTest("the cpu reproduction arm has not been generated")
        with open(path, encoding="utf-8") as handle:
            cpu = json.load(handle)
        self.assertEqual(cpu["device"], "cpu")
        row = cpu["models"]["dreamer_v9_269k"]
        m = self.find(r"6486\.27 was scored on\s*\*\*cpu\*\*", "the retired-figure sentence")
        self.check("the retired cpu mean", 6486.27, row["mean"], places=2)
        self.check("the retired cpu median and falls", [6176.57, 0.10],
                   [row["median"], row["falls_per_episode"]], places=2)
        self.assertEqual(row["reward_kwargs"], {},
                         "`--reward-weights env-default` is what reproduces the retired row, because "
                         "the retired fallback gave Dreamer the environment defaults")

    def test_the_wall_clock_is_the_run_s_own_mlflow_row(self):
        if not os.path.exists(self.DB):
            self.skipTest("mlruns.db is not in this checkout")
        con = sqlite3.connect("file:" + self.DB.replace(os.sep, "/") + "?mode=ro", uri=True, timeout=10)
        try:
            hours, status = con.execute(
                "SELECT (end_time-start_time)/3600000.0, status FROM runs "
                "WHERE name='dreamer_dreamer_v3_1m__7' AND status='FINISHED'").fetchone()
        finally:
            con.close()
        self.assertEqual("FINISHED", status)
        m = re.search(r"the run above took \*\*([\d.]+) h\*\* of machine time for its 1M steps",
                      self.readme)
        self.assertIsNotNone(m, "the run's wall clock sentence moved out of the Phase-1 section")
        self.assertAlmostEqual(float(m.group(1)), round(hours, 2), places=2,
                               msg="the README's hours are not the run's own start/end times")
        # And the prediction it is compared against: the two-budget measurement, in hours per 1M.
        m = re.search(r"measured ([\d.]+) h before it ran, and the run itself came out (\d+)% slower",
                      self.readme)
        self.assertIsNotNone(m, "the measured-versus-actual sentence moved")
        self.assertAlmostEqual(float(m.group(1)), self.rate["hours_per_1m_env_steps"], places=2)
        self.assertEqual(int(m.group(2)), round(100.0 * (hours / self.rate["hours_per_1m_env_steps"]
                                                         - 1.0)), "the 8% is derived, so recompute it")


class TestReadmeRewardProvenanceCells(ReadmeGate, unittest.TestCase):
    """The scoring defects, the corrected rows, and the artifacts that still reproduce the retired ones.

    Every figure here is gated as a pair: what the README publishes now, and a committed artifact that
    regenerates what it published before. The retired numbers stay measurable on purpose - both
    mechanisms that produced them (a reward fallback that split by algorithm, and an evaluator that did
    not record its device) are still reachable as flags, so a reader can reproduce the mistake and see
    why it was one.
    """

    FIX_START = "**That fallback was wrong for three of the four algorithms"
    FIX_END = "Re-scoring the same 200 episodes"
    ARS_START = "The first Phase-1 run trained on v9 is ARS"
    ARS_END = "The second Phase-1 run to reach its full budget"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        self.readme = readme
        self.block = self.section(self.FIX_START, self.FIX_END)
        self.ars_block = self.section(self.ARS_START, self.ARS_END)
        self.what = "reward-provenance"
        self.bad = []

    def section(self, start_marker, end_marker):
        start = self.readme.index(start_marker)
        return re.sub(r"\s+", " ", self.readme[start:self.readme.index(end_marker, start)])

    def artifact(self, name):
        path = os.path.join(ROOT, "benchmarks", name)
        self.assertTrue(os.path.exists(path), f"the README cites benchmarks/{name}, which is missing")
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_the_rescored_1m_row_moves_the_return_and_nothing_else(self):
        current = self.artifact("phase1_dreamer_v3_1m.json")
        retired = self.artifact("phase1_dreamer_v3_1m_env_default.json")
        a, b = current["models"]["dreamer_v3_1m"], retired["models"]["dreamer_v3_1m"]
        self.assertEqual(current["device"], retired["device"],
                         "the pair only isolates the reward if the device is the same on both arms")
        m = re.search(r"the mean moved (\d+(?:\.\d+)?) → \*\*(\d+(?:\.\d+)?)\*\* \(\+([\d.]+)%\)",
                      self.block)
        self.assertIsNotNone(m, "the re-scoring sentence was reworded")
        self.check("1M mean, retired and current", [m.group(1), m.group(2)], [b["mean"], a["mean"]],
                   places=2)
        self.check("the derived percentage", float(m.group(3)),
                   round(100.0 * (a["mean"] / b["mean"] - 1.0), 1), places=1)
        self.assertEqual(current["per_episode"]["dreamer_v3_1m__telemetry"],
                         retired["per_episode"]["dreamer_v3_1m__telemetry"],
                         "the telemetry is claimed identical to the digit and it is not")
        changed = sum(1 for x, y in zip(current["per_episode"]["dreamer_v3_1m"],
                                        retired["per_episode"]["dreamer_v3_1m"])
                      if abs(x - y) > 1e-9)
        m = re.search(r"(\d+) of the (\d+) returns changed", self.block)
        self.assertIsNotNone(m, "the changed-return count sentence was reworded")
        self.assertEqual([int(m.group(1)), int(m.group(2))], [changed, a["episodes"]])
        m = re.search(r"The extremes did not move at all \((-[\d.]+) and ([\d.]+) both before and "
                      r"after\)", self.block)
        self.assertIsNotNone(m, "the extremes sentence was reworded")
        self.check("the extremes, in both arms", [m.group(1), m.group(2)], [a["min"], a["max"]],
                   places=2)
        self.assertEqual([b["min"], b["max"]], [a["min"], a["max"]],
                         "the retired arm's extremes are claimed to be the same two numbers")
        self.assertEqual(b["reward_kwargs"], {},
                         "`--reward-weights env-default` is what reproduces the retired row")

    def test_the_episodes_that_moved_are_separated_by_the_standing_gate_floor(self):
        """Exact separation, not a tendency: changed episodes peak above 0.65 m, unchanged below.

        The first version of this sentence explained the unmoved extremes by saying those episodes
        "never get inside the standing gate", which is wrong - the gate is graded from z=0.65 and the
        healthy band starts at 1.0, so 12 episodes moved without ever reaching the band. The height
        that actually separates them is the gate's own floor, and it does so with no overlap.
        """
        current = self.artifact("phase1_dreamer_v3_1m.json")
        retired = self.artifact("phase1_dreamer_v3_1m_env_default.json")
        probe = self.artifact("phase1_posture_probe.json")["models"]["dreamer_v3_1m"]["per_episode"]
        a = current["per_episode"]["dreamer_v3_1m"]
        b = retired["per_episode"]["dreamer_v3_1m"]
        self.assertEqual(len(a), len(b))
        self.assertTrue(all(abs(p["return"] - x) < 0.01 for p, x in zip(probe, a)),
                        "the probe and the scorer disagree episode by episode, so the heights below "
                        "describe a different rollout than the returns")
        changed = {i for i, (x, y) in enumerate(zip(a, b)) if abs(x - y) > 1e-9}
        moved_z = [probe[i]["max_z"] for i in sorted(changed)]
        still_z = [probe[i]["max_z"] for i in range(len(a)) if i not in changed]
        m = re.search(r"all (\d+) that changed peak at or above \*\*([\d.]+) m\*\* of torso height "
                      r"and all (\d+)\s*that did not peak at or below \*\*([\d.]+) m\*\*", self.block)
        self.assertIsNotNone(m, "the separation sentence was reworded")
        self.assertEqual(int(m.group(1)), len(moved_z))
        self.assertEqual(int(m.group(3)), len(still_z))
        self.check("the floor of the moved set", m.group(2), min(moved_z), places=3)
        self.check("the ceiling of the unmoved set", m.group(4), max(still_z), places=3)
        self.assertGreater(min(moved_z), max(still_z),
                           "the two sets overlap, so the separation is not the exact one claimed")
        with open(os.path.join(ROOT, "envs", "walker_ragdoll_env.py"), encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("min(max((z_after - 0.65) / 0.60, 0.0), 1.0)", source,
                      "the standing-gate height factor the sentence quotes has moved or changed")
        m = re.search(r"`clip\(\(z - ([\d.]+)\) / ([\d.]+), 0, 1\)`", self.block)
        self.assertIsNotNone(m, "the clip expression was reworded")
        self.check("the clip the prose quotes", [m.group(1), m.group(2)], [0.65, 0.60], places=2)

    def test_the_ars_row_and_the_version_it_replaces(self):
        current = self.artifact("phase1_ars_v2_1m_v9.json")["models"]["ars_v9"]
        retired = self.artifact("phase1_ars_v2_1m_v9_env_default.json")["models"]["ars_v9"]
        m = re.search(r"mean ([\d.]+), median ([\d.]+), std ([\d.]+), ([\d.]+)\s*falls per episode, "
                      r"\*\*0 of (\d+) inside the radius\*\*, mean closest\s*approach ([\d.]+) m, mean "
                      r"x-velocity (-?[\d.]+) m/s", self.ars_block)
        self.assertIsNotNone(m, "the ARS statistics sentence was reworded")
        self.check("ARS row", [float(m.group(i)) for i in range(1, 8)],
                   [current["mean"], current["median"], current["std"],
                    current["falls_per_episode"], current["episodes"],
                    current["mean_min_target_distance"], current["mean_x_velocity"]], places=2)
        m = re.search(r"published once already as ([\d.]+) under the environment defaults",
                      self.ars_block)
        self.assertIsNotNone(m, "the retired-ARS-figure sentence was reworded")
        self.check("the retired ARS mean", m.group(1), retired["mean"], places=2)
        for field in ("falls_per_episode", "mean_min_target_distance", "mean_x_velocity"):
            self.assertEqual(current[field], retired[field],
                             f"ARS {field} is claimed identical in both versions")
        retired_weights = {k: v for k, v in retired["reward_kwargs"].items()
                           if k != "target_forward_velocity"}
        self.assertEqual(retired_weights, {},
                         "`--reward-weights env-default` is what reproduces the retired ARS row; the "
                         "checkpoint's own target_forward_velocity is carried through by design")
        self.assertNotEqual(current["mean"], retired["mean"],
                            "the paragraph says the return moved; both arms now agree")

    def test_the_two_controls_that_pin_the_resolution(self):
        """Auto resolution must equal the explicit flag, including for a checkpoint that records tfv.

        The first control is the 1M Dreamer, whose reward the fallback has to infer. The second is the
        40M SAC checkpoint, which records `target_forward_velocity=10.0`: before that was carried
        through, the forced arm silently changed the task and the two arms disagreed by 2.5%.
        """
        auto = self.artifact("phase1_dreamer_v3_1m.json")
        forced = self.artifact("phase1_dreamer_v3_1m_training_reward.json")
        self.assertEqual(auto["per_episode"]["dreamer_v3_1m"],
                         forced["per_episode"]["dreamer_v3_1m_training_reward"],
                         "default resolution and --reward-weights training disagree on the 1M run")
        self.assertEqual(auto["per_episode"]["dreamer_v3_1m__telemetry"],
                         forced["per_episode"]["dreamer_v3_1m_training_reward__telemetry"])
        evidence = self.artifact("phase1_evidence_eval.json")["models"]["sac_target_40M"]
        control = self.artifact("phase1_sac_40m_forced_training_control.json")["models"][
            "sac_target_40M"]
        self.assertEqual(evidence["mean"], control["mean"],
                         "the forced arm drops or alters target_forward_velocity again")
        self.assertEqual(control["reward_kwargs"].get("target_forward_velocity"), 10.0,
                         "the SAC checkpoint's tfv is 10.0 and the control must carry it")
        m = re.search(r"on cuda they both give\s*([\d.]+)\.", self.block)
        self.assertIsNotNone(m, "the control sentence was reworded")
        self.check("the SAC cuda row the control must equal", m.group(1), evidence["mean"], places=2)
        auto = self.artifact("phase1_evidence_eval_cpu_auto.json")
        buggy = self.artifact("phase1_evidence_eval_cpu_training.json")
        a, b = auto["models"]["sac_target_40M"], buggy["models"]["sac_target_40M"]
        m = re.search(r"dropping it moved the score from ([\d.]+) to\s*([\d.]+) on the same device",
                      self.block)
        self.assertIsNotNone(m, "the tfv sentence was reworded")
        self.check("the tfv pair", [m.group(1), m.group(2)], [a["mean"], b["mean"]], places=2)
        self.assertEqual(auto["device"], buggy["device"],
                         "the tfv pair only isolates the kwarg if the device is held")
        self.assertEqual(a["reward_kwargs"].get("target_forward_velocity"), 10.0)
        self.assertNotIn("target_forward_velocity", b["reward_kwargs"],
                         "the fixture is supposed to be the arm that dropped tfv")
        self.assertEqual(auto["per_episode"]["sac_target_40M__telemetry"],
                         buggy["per_episode"]["sac_target_40M__telemetry"],
                         "the prose says the telemetry is identical across the tfv pair")
        ra = auto["per_episode"]["sac_target_40M"]
        rb = buggy["per_episode"]["sac_target_40M"]
        differing = [i for i, (x, y) in enumerate(zip(ra, rb)) if abs(x - y) > 0.01]
        m = re.search(r"exactly \*\*(\d+) of the (\d+)\*\*\s*episodes change return \(([^)]*)\)",
                      self.block)
        self.assertIsNotNone(m, "the episode-count sentence was reworded")
        self.assertEqual([int(m.group(1)), int(m.group(2))], [len(differing), len(ra)])
        self.assertEqual([int(t) for t in re.findall(r"\d+", m.group(3))], differing,
                         "the prose names the episodes that moved")

    def test_the_retired_cpu_rows_reproduce_exactly(self):
        defaults = self.artifact("phase1_evidence_eval_cpu_defaults.json")
        auto = self.artifact("phase1_evidence_eval_cpu_auto.json")
        self.assertEqual(defaults["device"], "cpu")
        self.assertEqual(auto["device"], "cpu")
        m = re.search(r"returns \*\*([\d.]+) / ([\d.]+) / (-?[\d.]+)\*\* for the\s*REDQ, Dreamer and "
                      r"ARS smoke checkpoints", self.block)
        self.assertIsNotNone(m, "the retired evidence-table sentence was reworded")
        self.check("the three retired cpu rows", [float(m.group(i)) for i in (1, 2, 3)],
                   [defaults["models"][k]["mean"] for k in
                    ("redq_6k", "dreamer_8k", "ars_60k")], places=2)
        m = re.search(r"returns\s*\*\*([\d.]+)\*\* for the 40M SAC row", self.block)
        self.assertIsNotNone(m, "the retired SAC cpu sentence was reworded")
        self.check("the retired SAC cpu row", m.group(1), auto["models"]["sac_target_40M"]["mean"],
                   places=2)
        # The count in the sentence: five rows were cpu-scored, and each of them has to actually
        # differ from its cuda counterpart, or "the device changed the answer" is decoration.
        cuda = self.artifact("phase1_evidence_eval.json")
        moved = [k for k in auto["models"]
                 if auto["models"][k]["mean"] != cuda["models"][k]["mean"]]
        moved += [k for k in defaults["models"]
                  if defaults["models"][k]["mean"] != cuda["models"][k]["mean"]]
        moved = sorted(set(moved))
        partial = self.artifact("phase1_dreamer_v9_269k.json")
        cpu_partial = self.artifact("phase1_dreamer_v9_269k_cpu_reproduction.json")
        if cpu_partial["models"]["dreamer_v9_269k"]["mean"] != partial["models"][
                "dreamer_v9_269k"]["mean"]:
            moved.append("dreamer_v9_269k")
        m = re.search(r"\b(\w+) rows turn out to have been scored on \*\*cpu\*\*", self.block)
        self.assertIsNotNone(m, "the five-rows sentence was reworded")
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
        self.assertIn(m.group(1).lower(), words,
                      f"the row count is spelled {m.group(1)!r}, which this gate cannot read")
        self.assertEqual(words[m.group(1).lower()], len(moved),
                         f"the rows whose score depends on the device are {moved}")


class TestReadmePostureCells(ReadmeGate, unittest.TestCase):
    """The standing-band table and the three readings under it, against `bench_posture.py`'s artifact.

    This is the section that says the honest thing about Phase 1 - that the trained policies reach the
    env's standing band no more often than commanding zero does - so the comparison has to be
    recomputable, including the two prose figures that carry it (the band-episode counts and the
    identical fall counts).
    """

    POSTURE = os.path.join(ROOT, "benchmarks", "phase1_posture_probe.json")
    DEFAULTS = os.path.join(ROOT, "benchmarks", "phase1_posture_probe_env_default.json")
    SENS = os.path.join(ROOT, "benchmarks", "phase1_eval_device_sensitivity.json")
    START = "**And the rung below walking is the one nobody clears.**"
    END = "**Would one run that does everything beat the phase split?"

    ROWS = {"SAC at 40M steps": "sac_40m", "SAC at 9M steps": "sac_9m",
            "ARS at 1M": "ars_v9_1m", "Dreamer at 269,404": "dreamer_v9_269k",
            "**Dreamer at 1M**": "dreamer_v3_1m", "commanding zero": "commanding_zero",
            "uniform random": "uniform_random"}

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.flat = re.sub(r"\s+", " ", self.block)
        with open(self.POSTURE, encoding="utf-8") as handle:
            self.posture = json.load(handle)["models"]
        self.what = "standing-band table"
        self.bad = []

    def test_every_row_is_its_artifact_row(self):
        for label, key in self.ROWS.items():
            art = self.posture[key]
            n, of = re.match(r"(\d+) of (\d+)", self.cell(label, 1)).groups()
            self.check(f"{key} band episodes", [int(n), int(of)],
                       [art["episodes_ever_in_band"], art["episodes"]], places=0)
            self.check(f"{key} peak z", self.cell(label, 2), art["mean_max_z"], places=3)
            self.check(f"{key} band share", self.cell(label, 3).rstrip("%"),
                       art["mean_pct_steps_in_band"], places=2)
            self.check(f"{key} falls", self.cell(label, 4), art["mean_falls_per_episode"], places=2)
            self.check(f"{key} mean return", self.cell(label, 5), art["mean_return"], places=2)

    def test_no_row_is_labelled_with_a_checkpoint_it_was_not_scored_on(self):
        """The bug this guards: a roster entry keyed `sac_40m` pointing at `sac_ckpt_9000000.pt`.

        The row printed, the table looked fine, and the README described a 9M policy as a 40M one for
        a day. Every row now records the path it was scored from, so the step count in the label can
        be compared against the step count in the file rather than against the roster's key. "ARS at
        1M" is 1,000,917 steps, so the comparison is at 0.1% rather than exact.
        """
        for key, art in self.posture.items():
            self.assertIn("checkpoint", art, f"{key} does not record what it was scored on")
        paths = [a["checkpoint"] for a in self.posture.values() if a["checkpoint"] not in
                 ("none", "random")]
        self.assertEqual(len(paths), len(set(paths)), "two rows were scored on the same checkpoint")
        for label, key in self.ROWS.items():
            text = label.replace("*", "")
            m = re.search(r"at ([\d,]+)(M?)", text)
            if not m:
                continue
            want = float(m.group(1).replace(",", "")) * (1e6 if m.group(2) else 1.0)
            got = int(re.search(r"_(\d+)\.pt$", self.posture[key]["checkpoint"]).group(1))
            self.assertLessEqual(abs(got - want) / want, 0.001,
                                 f"the row labelled {text!r} was scored on step {got}")

    def test_the_whole_table_is_one_reward_function(self):
        """The passive references used to be scored under the environment defaults while every
        trained row used the shaping, which put two reward functions in one column - and the
        defaults pay `standing_reward=50/step`, so the yardstick was paid for lying still.

        The weights are now one set across the table. A row also carries the `target_forward_velocity`
        its checkpoint records, which is a task parameter and not a weight, so the gate pins the
        distinct speeds in play and which rows carry them rather than pretending the column away.
        """
        import inspect
        from envs.reward_shaping import TRAINING_REWARD_KWARGS
        from envs.walker_ragdoll_env import WalkerRagdollEnv
        default_tfv = inspect.signature(WalkerRagdollEnv.__init__).parameters[
            "target_forward_velocity"].default
        speeds = {}
        for key, art in self.posture.items():
            applied = dict(art["reward_kwargs_applied"])
            tfv = applied.pop("target_forward_velocity", None)
            self.assertEqual(applied, dict(TRAINING_REWARD_KWARGS),
                             f"{key} was scored under different reward weights from the rest")
            speeds[key] = default_tfv if tfv is None else tfv
        self.assertEqual(sorted(set(speeds.values())), sorted({default_tfv, 10.0, 1.2}),
                         f"the table spans {sorted(set(speeds.values()))}, not the three speeds "
                         f"the prose names")
        self.assertEqual(sorted(k for k, v in speeds.items() if v != default_tfv),
                         ["sac_40m", "sac_9m"],
                         "the prose says only the two SAC rows are off the environment's speed")
        self.assertEqual(speeds["dreamer_v3_1m"], speeds["commanding_zero"],
                         "reading 3 compares these two rows and they must be on the same task")
        m = re.search(r"so the table spans three task\s*speeds - the environment's ([\d.]+) for the "
                      r"two Dreamer rows, ARS and both passive references, and the\s*([\d.]+) and "
                      r"([\d.]+) that the 40M and 9M SAC runs were trained at", self.flat)
        self.assertIsNotNone(m, "the three-task-speeds sentence was reworded")
        self.check("the three speeds", [m.group(1), m.group(2), m.group(3)],
                   [default_tfv, speeds["sac_40m"], speeds["sac_9m"]], places=2)

    def test_the_correction_moved_the_returns_and_nothing_else(self):
        """Two arms of the same probe, one reward function apart: geometry identical, returns not.

        The environment-default arm is the retired protocol reproduced by the current code, so every
        figure the first version of this table published is still measurable - and the geometry
        comparison is what makes "the reward weights do not touch the trajectory" a checked claim
        rather than an assumption.
        """
        if not os.path.exists(self.DEFAULTS):
            self.skipTest("the environment-default arm of the probe has not been generated")
        with open(self.DEFAULTS, encoding="utf-8") as handle:
            defaults = json.load(handle)["models"]
        geometry = ("episodes_ever_in_band", "mean_max_z", "best_max_z_in_one_episode",
                    "lowest_peak_max_z", "mean_pct_steps_in_band",
                    "mean_pct_steps_upright_in_band", "mean_final_z", "mean_falls_per_episode")
        for key, art in self.posture.items():
            for field in geometry:
                self.assertEqual(art[field], defaults[key][field],
                                 f"{key}.{field} differs between the two reward arms, so the "
                                 f"trajectory moved and not only its score")
            self.assertEqual(art["reward_kwargs_applied"] == defaults[key]["reward_kwargs_applied"],
                             False, f"{key} was scored under the same reward in both arms")
        # The three retired figures the prose quotes, each against the arm that produces it: the two
        # rows the retired fallback gave the environment defaults, and the SAC row it already shaped.
        m = re.search(r"identical to the cent \((\d+(?:\.\d+)?)\)", self.flat)
        self.assertIsNotNone(m, "the unchanged-SAC-row sentence was reworded")
        self.check("SAC at 9M, shaped in both tables", m.group(1),
                   self.posture["sac_9m"]["mean_return"], places=2)
        m = re.search(r"the zero reference dropped (\d+(?:\.\d+)?) →\s*(\d+(?:\.\d+)?)", self.flat)
        self.assertIsNotNone(m, "the zero-reference sentence was reworded")
        self.check("zero reference, retired and current", [m.group(1), m.group(2)],
                   [defaults["commanding_zero"]["mean_return"],
                    self.posture["commanding_zero"]["mean_return"]], places=2)
        fix = self.readme_block_for_reward_fix()
        m = re.search(r"the mean moved (\d+(?:\.\d+)?) → \*\*(\d+(?:\.\d+)?)\*\*", fix)
        self.assertIsNotNone(m, "the 1M re-scoring sentence was reworded")
        self.check("1M dreamer, retired arm", m.group(1),
                   defaults["dreamer_v3_1m"]["mean_return"], places=2)
        self.check("1M dreamer, current arm", m.group(2),
                   self.posture["dreamer_v3_1m"]["mean_return"], places=2)
        self.assertEqual(defaults["commanding_zero"]["reward_kwargs_applied"], {},
                         "the retired arm's references must be the environment defaults")
        m = re.search(r"reproduces all five\s*retired returns exactly \(([^)]*)\)", self.flat)
        self.assertIsNotNone(m, "the five retired returns sentence was reworded")
        quoted = nums(m.group(1))
        retired = [defaults["dreamer_v3_1m"]["mean_return"], defaults["dreamer_v9_269k"]["mean_return"],
                   defaults["ars_v9_1m"]["mean_return"], defaults["commanding_zero"]["mean_return"],
                   defaults["uniform_random"]["mean_return"]]
        self.check("the five retired probe returns", quoted, retired, places=2)
        for key in ("dreamer_v3_1m", "dreamer_v9_269k", "ars_v9_1m"):
            self.assertNotEqual(defaults[key]["mean_return"], self.posture[key]["mean_return"],
                                f"{key}'s return is said to have moved and did not")

    def readme_block_for_reward_fix(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index("**That fallback was wrong for three of the four algorithms")
        return readme[start:readme.index("**A second defect fell out of checking the first", start)]

    def test_the_three_readings_are_what_the_artifact_says(self):
        d, z = self.posture["dreamer_v3_1m"], self.posture["commanding_zero"]
        m = re.search(r"as few episodes as commanding zero - (\d+) of 50 each -\s*and spends less "
                      r"time up than a policy that emits nothing at all\*\*\s*\(([\d.]+)% of steps "
                      r"against ([\d.]+)%\)", self.flat)
        self.assertIsNotNone(m, "reading 1 was reworded; re-point this test at it")
        self.assertEqual([int(m.group(1)), float(m.group(2)), float(m.group(3))],
                         [d["episodes_ever_in_band"], d["mean_pct_steps_in_band"],
                          z["mean_pct_steps_in_band"]], "reading 1 no longer matches the artifact")
        self.assertEqual(d["episodes_ever_in_band"], z["episodes_ever_in_band"],
                         "the paragraph claims the trained policy and doing nothing reach the band "
                         "in the same number of episodes; that is no longer true")
        self.assertEqual(d["mean_falls_per_episode"], z["mean_falls_per_episode"],
                         "the identical fall counts that the latch explanation rests on are no longer "
                         "identical")
        m = re.search(r"the trained policy and doing nothing report the \*\*same ([\d.]+) falls per "
                      r"episode\*\*", self.flat)
        self.assertIsNotNone(m, "the fall-count sentence was reworded")
        self.assertAlmostEqual(float(m.group(1)), d["mean_falls_per_episode"], places=2)

        # Reading 3 is now the exception, not the rule: the return ranks the two SAC rows and the
        # random reference correctly, and puts the 1M Dreamer above a policy that does nothing while
        # it stands for a third as long.
        m = re.search(r"outranks the zero-action reference on return while standing for a third as "
                      r"long\*\* \((\d+(?:\.\d+)?) against (\d+(?:\.\d+)?), ([\d.]+)% of steps "
                      r"against ([\d.]+)%\)", self.flat)
        self.assertIsNotNone(m, "reading 3 was reworded; re-point this test at it")
        self.check("reading 3 returns", [m.group(1), m.group(2)],
                   [d["mean_return"], z["mean_return"]], places=2)
        self.check("reading 3 band shares", [m.group(3), m.group(4)],
                   [d["mean_pct_steps_in_band"], z["mean_pct_steps_in_band"]], places=2)
        self.assertGreater(d["mean_return"], z["mean_return"],
                           "reading 3 says the trained policy outranks doing nothing on return")
        ratio = d["mean_pct_steps_in_band"] / z["mean_pct_steps_in_band"]
        self.assertTrue(0.2 < ratio < 0.45,
                        f"reading 3 says 'a third as long'; the artifact gives {ratio:.2f}")
        standers = sorted(((v["mean_pct_steps_in_band"], k) for k, v in self.posture.items()),
                          reverse=True)
        self.assertEqual([k for _, k in standers[:2]], ["sac_40m", "sac_9m"],
                         "the two SAC rows are the only meaningful standers in the table")
        self.assertLess(standers[2][0], 1.0,
                        "below the two SAC rows the table is meant to be a wall of ~1% or less")

    def test_the_artifact_records_the_two_things_that_make_it_repeatable(self):
        with open(self.POSTURE, encoding="utf-8") as handle:
            meta = json.load(handle)
        self.assertEqual(meta.get("torch_threads"), 1,
                         "the probe's thread pin is what makes it repeatable run to run")
        self.assertEqual({v["device"] for v in self.posture.values()}, {"cuda"},
                         "the published harness resolves to cuda on this box; mixing devices in one "
                         "artifact is how a cpu row next to a cuda row becomes invisible")

    def test_the_device_sensitivity_pair_is_the_run_that_ran(self):
        if not os.path.exists(self.SENS):
            self.skipTest("the device-sensitivity artifact has not been generated yet")
        with open(self.SENS, encoding="utf-8") as handle:
            pair = json.load(handle)["devices"]

        def spread(name):
            c, p = pair["cuda"][name]["mean_return"], pair["cpu"][name]["mean_return"]
            return 100.0 * abs(c - p) / ((c + p) / 2.0)

        m = re.search(r"scores \*\*(\d+(?:\.\d+)?) on cuda and\s*(\d+(?:\.\d+)?) on cpu\*\*, "
                      r"(\d+(?:\.\d+)?)% apart on the same ten episodes, and the 269k one "
                      r"(\d+(?:\.\d+)?) against (\d+(?:\.\d+)?)", self.flat)
        self.assertIsNotNone(m, "the device-sensitivity sentence was reworded")
        c1, p1 = pair["cuda"]["dreamer_v3_1m"], pair["cpu"]["dreamer_v3_1m"]
        self.check("1M dreamer cuda/cpu returns", [float(m.group(1)), float(m.group(2))],
                   [c1["mean_return"], p1["mean_return"]], places=2)
        self.check("1M dreamer spread", float(m.group(3)), round(spread("dreamer_v3_1m"), 1), places=1)
        c2, p2 = pair["cuda"]["dreamer_v9_269k"], pair["cpu"]["dreamer_v9_269k"]
        self.check("269k dreamer cuda/cpu returns", [float(m.group(4)), float(m.group(5))],
                   [c2["mean_return"], p2["mean_return"]], places=1)
        # The two entries that do not move are the ones that identify the mechanism: a float64 numpy
        # linear map and a policy that does no arithmetic. If either starts moving, the explanation
        # in the paragraph is wrong.
        for name in ("ars_v9_1m", "commanding_zero"):
            self.assertEqual(pair["cuda"][name]["mean_return"], pair["cpu"][name]["mean_return"],
                             f"{name} is device-independent by construction and no longer is")
        m = re.search(r"ARS at (\d+(?:\.\d+)?) on both devices and the zero-action reference at "
                      r"(\d+(?:\.\d+)?) on both", self.flat)
        self.assertIsNotNone(m, "the control sentence was reworded")
        self.check("control returns", [float(m.group(1)), float(m.group(2))],
                   [pair["cuda"]["ars_v9_1m"]["mean_return"],
                    pair["cuda"]["commanding_zero"]["mean_return"]], places=2)

        # The verdict survives the device; the posture figures do not always. Both halves are quoted.
        self.assertEqual(c1["mean_pct_steps_in_band"], p1["mean_pct_steps_in_band"],
                         "the 1M run's band share is claimed identical on both devices")
        m = re.search(r"\(([\d.]+)% of steps in the band on either device for the 1M\s*run, (\d+) of "
                      r"(\d+) episodes reaching it on both\)", self.flat)
        self.assertIsNotNone(m, "the band-share sentence was reworded")
        self.check("band share on either device", float(m.group(1)), c1["mean_pct_steps_in_band"],
                   places=2)
        self.assertEqual([int(m.group(2)), int(m.group(3))],
                         [c1["episodes_ever_in_band"], c1["episodes"]])
        self.assertEqual(p1["episodes_ever_in_band"], c1["episodes_ever_in_band"],
                         "the prose says 1 of 10 on both devices")
        m = re.search(r"the 269k checkpoint reads ([\d.]+)% of steps in the band on cuda against "
                      r"([\d.]+)% on cpu, (\d+) episodes against\s*(\d+)", self.flat)
        self.assertIsNotNone(m, "the 269k posture-divergence sentence was reworded")
        self.check("269k band share per device", [m.group(1), m.group(2)],
                   [c2["mean_pct_steps_in_band"], p2["mean_pct_steps_in_band"]], places=2)
        self.check("269k band episodes per device", [int(m.group(3)), int(m.group(4))],
                   [c2["episodes_ever_in_band"], p2["episodes_ever_in_band"]], places=0)

    def test_the_sac_device_pair_that_refutes_the_sampled_latent_explanation(self):
        """A deterministic actor, same checkpoint, same reward, two devices: 30% apart.

        The paragraph used to explain device sensitivity by the RSSM's reparametrised posterior. SAC
        has no posterior and `deterministic=True`, so if this pair ever agrees, the float32-chaos
        explanation is the one that needs revisiting, not the paragraph.
        """
        cuda = os.path.join(ROOT, "benchmarks", "phase1_evidence_eval.json")
        cpu = os.path.join(ROOT, "benchmarks", "phase1_evidence_eval_cpu_auto.json")
        if not (os.path.exists(cuda) and os.path.exists(cpu)):
            self.skipTest("one of the two evidence arms is missing")
        with open(cuda, encoding="utf-8") as handle:
            on_cuda = json.load(handle)
        with open(cpu, encoding="utf-8") as handle:
            on_cpu = json.load(handle)
        self.assertEqual(on_cuda["device"], "cuda")
        self.assertEqual(on_cpu["device"], "cpu")
        a, b = on_cuda["models"]["sac_target_40M"], on_cpu["models"]["sac_target_40M"]
        self.assertEqual(a["reward_kwargs"], b["reward_kwargs"],
                         "the pair only isolates the device if the reward is the same on both arms")
        m = re.search(r"scores\s*\*\*(\d+(?:\.\d+)?) on cpu against (\d+(?:\.\d+)?) on cuda\*\* - "
                      r"(\d+)% apart", self.flat)
        self.assertIsNotNone(m, "the SAC device sentence was reworded")
        self.check("SAC 40M cpu and cuda", [m.group(1), m.group(2)], [b["mean"], a["mean"]], places=2)
        self.assertEqual(int(m.group(3)),
                         round(100.0 * abs(b["mean"] - a["mean"]) / ((b["mean"] + a["mean"]) / 2.0)),
                         "the percentage is the midpoint spread, the same convention as the Dreamer "
                         "pair in the paragraph above it, so recompute it that way")
        ta = on_cuda["per_episode"]["sac_target_40M__telemetry"]
        tb = on_cpu["per_episode"]["sac_target_40M__telemetry"]
        differing = [i for i, (x, y) in enumerate(zip(ta, tb)) if x != y]
        m = re.search(r"with (\d+) of the (\d+) episodes on\s*completely different trajectories "
                      r"\(episode (\d+)'s closest approach ([\d.]+) m on cpu, ([\d.]+) m on cuda\)",
                      self.flat)
        self.assertIsNotNone(m, "the trajectory-divergence sentence was reworded")
        self.assertEqual(int(m.group(1)), len(differing),
                         "the prose counts the episodes whose telemetry differs between devices")
        self.assertEqual(int(m.group(2)), len(ta), "the prose names a different episode count")
        ep = int(m.group(3))
        self.check("the cited episode's closest approach", [m.group(4), m.group(5)],
                   [tb[ep]["min_target_distance"], ta[ep]["min_target_distance"]], places=3)
        self.assertIn(ep, differing, "the episode cited as divergent is one of the identical ones")


class TestReadmeAmortizeCells(ReadmeGate, unittest.TestCase):
    """The `--num-envs` ladder: wall clock and gradient steps have to move together in the prose.

    Raising `num_envs` is the only remaining way to cut Dreamer's wall clock without touching the
    update, and it is not free - it lowers the update-to-data ratio. The gate exists so the speedup
    can never be quoted without the gradient count that bought it.
    """

    AMORT = os.path.join(ROOT, "benchmarks", "dreamer_amortize_n.json")
    START = "**The last lever in the loop, and what it costs.**"
    END = "The batch axis is the second version"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.flat = re.sub(r"\s+", " ", self.block)
        self.art, = self.read_artifacts(self.AMORT)
        self.rate, self.split = self.read_artifacts(
            os.path.join(ROOT, "benchmarks", "dreamer_real_rate.json"),
            os.path.join(ROOT, "benchmarks", "dreamer_loop_split.json"))
        self.what = "num_envs amortisation"
        self.bad = []

    ROWS = {"4 (shipped)": 4, "8": 8, "16": 16}

    def test_each_row_of_the_ladder_is_the_measured_pair(self):
        for label, n in self.ROWS.items():
            row = self.art["per_num_envs"][str(n)]
            self.check(f"n={n} ms per iteration", self.cell(label, 1), row["ms_per_iteration"],
                       places=2)
            self.check(f"n={n} env-steps/s", self.cell(label, 2), row["env_steps_per_s"], places=1)
            self.check(f"n={n} hours per 1M", self.cell(label, 3), row["hours_per_1m_env_steps"],
                       places=2)
            self.check(f"n={n} gradient steps per 1k", self.cell(label, 4),
                       row["gradient_steps_per_1k_env_steps"], places=1)
            self.check(f"n={n} speedup", self.cell(label, 5).rstrip("x"),
                       row["speedup_vs_shipped_n4"], places=2)

    def test_the_gradient_column_is_just_the_env_step_budget_divided(self):
        """One update per collection step, so the column has to be 1000/n - not a second measurement."""
        for label, n in self.ROWS.items():
            self.check(f"n={n} gradient steps", self.cell(label, 4), 1000.0 / n, places=1)

    def test_the_control_agrees_with_the_other_two_instruments(self):
        """The n=4 row is the cross-check: if it drifts, the ladder's window moved, not the code."""
        n4 = self.art["per_num_envs"]["4"]["ms_per_iteration"]
        for other, name in ((self.rate["captured"]["ms_per_iteration"], "two-budget run"),
                            (self.split["whole_loop_ms"], "in-process split")):
            spread = abs(n4 - other) / other
            self.assertLess(spread, 0.10, f"n=4 ({n4} ms) and the {name} ({other} ms) are more than "
                                          "10% apart, so the prose's '9% apart' is no longer true")
        m = re.search(r"three instruments in three windows, (\d+)% apart", self.flat)
        self.assertIsNotNone(m, "the agreement sentence was reworded")
        worst = max(abs(n4 - x) / x for x in (self.rate["captured"]["ms_per_iteration"],
                                              self.split["whole_loop_ms"]))
        # The prose quotes one figure for a three-way comparison, so it has to be the worst pair
        # rounded the way a reader would round it - down, since claiming tighter agreement than
        # measured is the direction that misleads.
        self.assertEqual(int(m.group(1)), int(100 * worst), "the quoted agreement is not the worst pair")

    def test_the_hours_the_prose_promises_are_the_artifacts(self):
        m = re.search(r"costs \*\*(\d+(?:\.\d+)?) h\*\* where the shipped configuration costs "
                      r"(\d+(?:\.\d+)?) h",
                      self.flat)
        self.assertIsNotNone(m, "the headline hours sentence was reworded")
        self.check("the hours pair", [float(m.group(1)), float(m.group(2))],
                   [self.art["per_num_envs"]["16"]["hours_per_1m_env_steps"],
                    self.art["per_num_envs"]["4"]["hours_per_1m_env_steps"]], places=2)
        m = re.search(r"so (\d+) gradient steps per 1,000 environment steps\s*become (\d+(?:\.\d+)?)",
                      self.flat)
        self.assertIsNotNone(m, "the update-to-data sentence was reworded")
        self.check("the gradient-step pair", [float(m.group(1)), float(m.group(2))],
                   [self.art["per_num_envs"]["4"]["gradient_steps_per_1k_env_steps"],
                    self.art["per_num_envs"]["16"]["gradient_steps_per_1k_env_steps"]], places=1)


class TestReadmeNumEnvsLearningCells(ReadmeGate, unittest.TestCase):
    """The learning half of the `--num-envs` ladder: three schedules x two seeds at one budget.

    The timing half is `TestReadmeAmortizeCells`. This class exists because that half ends with a
    forward reference it has to honour, and because the first answer it gave - written from one seed
    per schedule, "8 envs is free, 16 is a cliff" - did not survive the second seed. So the gates
    here check the replicated design: every quoted figure is tied to an artifact, the two tables are
    tied to each other arm by arm, and the prose's central claim (that returns do not separate the
    schedules while forward speed does) is recomputed rather than remembered.
    """

    RATE = os.path.join(ROOT, "benchmarks", "training_rate_history.json")
    AMORTIZE = os.path.join(ROOT, "benchmarks", "dreamer_amortize_n.json")
    START = "**Does the cheaper update schedule still learn?**"
    END = "The batch axis is the second version of a sentence that was wrong before"

    # (schedule, seed) -> eval artifact, model key, run-id in the wall-clock table, table column
    ARMS = {
        ("4", "7"): ("phase1_dreamer_v3_250k.json", "dreamer_v3_250k", "dreamer_v3_1m", 1),
        ("4", "8"): ("phase1_dreamer_n4_s8_250k.json", "dreamer_n4_s8_250k", "dreamer_n4_s8_250k", 2),
        ("8", "7"): ("phase1_dreamer_n8_250k.json", "dreamer_n8_250k", "dreamer_n8_250k", 3),
        ("8", "8"): ("phase1_dreamer_n8_s8_250k.json", "dreamer_n8_s8_250k", "dreamer_n8_s8_250k", 4),
        ("16", "7"): ("phase1_dreamer_n16_250k.json", "dreamer_n16_250k", "dreamer_n16_250k", 5),
        ("16", "8"): ("phase1_dreamer_n16_s8_250k.json", "dreamer_n16_s8_250k",
                      "dreamer_n16_s8_250k", 6),
    }
    TIMING_ROWS = {"4 (shipped)": "4", "8": "8", "16": "16"}

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.flat = re.sub(r"\s+", " ", self.block)
        load = lambda name: json.load(
            open(os.path.join(ROOT, "benchmarks", name), encoding="utf-8"))
        self.art = {}
        for key, (name, model_key, run_id, col) in self.ARMS.items():
            top = load(name)
            self.art[key] = {"top": top, "model": top["models"][model_key],
                             "run_id": run_id, "col": col}
        self.rate_full = load("training_rate_history.json")
        self.rate = self.rate_full["checkpoint_wall_clock"]
        self.by_run = {r["run_id"]: r for r in self.rate["runs"]}
        self.amortize = load("dreamer_amortize_n.json")
        self.what = "num-envs learning"
        self.bad = []

    def test_every_arm_is_the_same_experiment_except_the_schedule(self):
        for (n, seed), arm in self.art.items():
            model, top = arm["model"], arm["top"]
            self.assertEqual(model["global_step"], 250000, f"n={n} seed={seed} budget")
            self.assertEqual(model["episodes"], 50, f"n={n} seed={seed} episodes")
            self.assertEqual(model["task_phase"], "target", f"n={n} seed={seed} phase")
            self.assertEqual(top["device"], "cuda", f"n={n} seed={seed} device")
        self.assertEqual({a["top"]["device"] for a in self.art.values()}, {"cuda"})
        rewards = {json.dumps(a["model"]["reward_kwargs"], sort_keys=True)
                   for a in self.art.values()}
        self.assertEqual(len(rewards), 1, "the arms were scored under more than one reward")
        for (n, seed), arm in self.art.items():
            run = self.by_run[arm["run_id"]]
            self.assertEqual(run["num_envs"], int(n), f"{arm['run_id']} labelled {run['num_envs']}")
            self.assertEqual(run["num_envs_from_normalizer"], int(n),
                             f"n={n} seed={seed}'s label disagrees with its checkpoint's normalizer "
                             f"counter ({run['normalizer_residue']}): mislabelled or resumed")
            mlflow_w = run["mlflow_window"]
            if mlflow_w:
                self.assertEqual(mlflow_w["seed"], seed,
                                 f"{arm['run_id']}: the MLflow row is seed {mlflow_w['seed']}")

    def test_the_timing_table(self):
        for label, n in self.TIMING_ROWS.items():
            self.check(f"n={n} gradient steps", nums(self.cell(label, 1))[0], 250000 // int(n),
                       places=0)
            for col, seed, timing_col in ((1, "7", 2), (2, "8", 3)):
                run = self.by_run[self.art[(n, seed)]["run_id"]]["window"]
                text = self.cell(label, timing_col)
                self.check(f"n={n} seed={seed} seconds", nums(text)[0], run["seconds"], places=1)
                self.check(f"n={n} seed={seed} rate", nums(text)[1], run["rate_steps_per_s"],
                           places=1)
                base = self.by_run[self.art[(n, seed)]["run_id"]]
                zero = self.by_run[self.art[("4", seed)]["run_id"]]
                want = round(base["window"]["rate_steps_per_s"]
                             / zero["window"]["rate_steps_per_s"], 2)
                self.check(f"n={n} seed={seed} speedup",
                           nums(self.cell(label, 4 + (1 if seed == "8" else 0)))[0], want,
                           places=2)
        self.assertEqual(self.rate["common_window"]["from_step"], 50000)
        self.assertEqual(self.rate["common_window"]["to_step"], 250000)

    def test_the_learning_table(self):
        self.assertEqual(nums(self.cell("seed", 1)), [7.0], "the seed row is out of step")
        for (n, seed), arm in self.art.items():
            model, col = arm["model"], arm["col"]
            self.assertEqual(int(nums(self.cell("seed", col))[0]), int(seed))
            self.check(f"n={n} seed={seed} mean", self.cell("mean return", col), model["mean"],
                       places=2)
            self.check(f"n={n} seed={seed} median", self.cell("median return", col),
                       model["median"], places=2)
            self.check(f"n={n} seed={seed} std", self.cell("std", col), model["std"], places=2)
            hit, of = re.match(r"(\d+) of (\d+)",
                               self.cell("episodes inside the 0.45 m radius", col)).groups()
            self.check(f"n={n} seed={seed} radius", [int(hit), int(of)],
                       [round(model["reached_target_pct"] / 100.0 * model["episodes"]),
                        model["episodes"]], places=0)
            self.check(f"n={n} seed={seed} closest", nums(self.cell("mean closest approach", col))[0],
                       model["mean_min_target_distance"], places=3)
            self.check(f"n={n} seed={seed} x-velocity", snums(self.cell("mean x-velocity", col)),
                       model["mean_x_velocity"], places=4)

    def test_the_replication_is_what_the_section_concludes(self):
        a = {k: v["model"] for k, v in self.art.items()}
        gap7 = a[("8", "7")]["mean"] - a[("4", "7")]["mean"]
        gap8 = a[("8", "8")]["mean"] - a[("4", "8")]["mean"]
        m = self.sentence(r"the 8-env arm led by (\d+) and the 16-env arm's median had collapsed to\s*"
                          r"([\d.]+)\. Re-run at seed 8, the 8-env arm \*trailed\* by (\d+)",
                          "the replication sentence")
        self.check("the two-seed gaps", [m.group(1), m.group(3)], [round(gap7), round(-gap8)],
                   places=0)
        self.check("the seed-7 16-env median", m.group(2), a[("16", "7")]["median"], places=2)
        self.assertGreater(gap7, 0, "the prose says 8 envs led at seed 7")
        self.assertLess(gap8, 0, "the prose says 8 envs trailed at seed 8 - the sign flip is the "
                                 "whole finding, so it has to hold")
        self.assertGreater(a[("16", "8")]["median"], 5.0 * a[("16", "7")]["median"],
                           "the prose says the 627.00 median was one draw, so seed 8 must not be low")
        spread4 = abs(a[("4", "7")]["mean"] - a[("4", "8")]["mean"])
        spread8 = abs(a[("8", "7")]["mean"] - a[("8", "8")]["mean"])
        m = self.sentence(r"(\d+\.\d+) across the 4-env mean and (\d+\.\d+) across the 8-env one -\s*"
                          r"is larger than any gap between schedules", "the seed-spread sentence")
        self.check("the within-schedule spreads", [m.group(1), m.group(2)],
                   [round(spread4, 2), round(spread8, 2)], places=2)
        means = {n: round(sum(a[(n, s)]["mean"] for s in ("7", "8")) / 2.0, 2) for n in ("4", "8", "16")}
        m = self.sentence(r"averaged over both seeds the means are ([\d.]+),\s*([\d.]+)\s*and ([\d.]+)",
                          "the per-schedule means")
        self.check("the per-schedule means", [m.group(1), m.group(2), m.group(3)],
                   [means["4"], means["8"], means["16"]], places=2)
        self.assertLess(max(gap7, -gap8), max(spread4, spread8),
                        "the section's claim is that no between-schedule gap exceeds the seed spread")

    def test_forward_speed_is_the_column_that_replicates(self):
        xv = {(n, s): self.art[(n, s)]["model"]["mean_x_velocity"]
              for (n, s) in self.art}
        m = self.sentence(r"the 16-env arm averaged\s*\*\*(-?[\d.]+) and (-?[\d.]+) m/s\*\*, two runs "
                          r"agreeing to ([\d.]+)%", "the stable x-velocity claim")
        self.check("the two 16-env velocities", [m.group(1), m.group(2)],
                   [xv[("16", "7")], xv[("16", "8")]], places=4)
        agreement = 100.0 * abs(xv[("16", "7")] - xv[("16", "8")]) / abs(xv[("16", "7")])
        self.check("their agreement", m.group(3), round(agreement, 1), places=1)
        self.assertLess(agreement, 1.0, "'agreeing to 0.3%' no longer describes the pair")
        self.assertLess(xv[("16", "7")], 0.0)
        self.assertLess(xv[("16", "8")], 0.0)
        hits = {n: sum(round(self.art[(n, s)]["model"]["reached_target_pct"] / 100.0 * 50)
                      for s in ("7", "8")) for n in ("4", "8", "16")}
        m = self.sentence(r"reached the radius in \*\*(\d+) of (\d+) episodes\*\*\.\s*The 4-env arm "
                          r"was positive in both seeds\s*\(\+?([\d.]+), \+?([\d.]+)\) and reached it "
                          r"in (\d+) of (\d+)", "the radius comparison")
        self.check("the 16-env radius record", [int(m.group(1)), int(m.group(2))],
                   [hits["16"], 100], places=0)
        self.check("the 4-env velocities and radius",
                   [float(m.group(3)), float(m.group(4)), int(m.group(5)), int(m.group(6))],
                   [xv[("4", "7")], xv[("4", "8")], hits["4"], 100], places=4)
        self.assertGreater(xv[("4", "7")], 0.0)
        self.assertGreater(xv[("4", "8")], 0.0, "the prose says the 4-env arm was positive twice")

    def test_the_confound_and_the_bench_are_disclosed(self):
        """At 16 envs the collector also switches backend, so the 16-env row is two changes at once."""
        import argparse
        import train_walker
        parser = argparse.ArgumentParser()
        train_walker.add_vec_env_args(parser)
        m = self.sentence(r"crosses\s*`--vec-parallel-threshold` \(default (\d+)\)", "the threshold")
        self.assertEqual(int(m.group(1)), parser.get_default("vec_parallel_threshold"),
                         "the disclosed threshold is not the trainer's default")
        self.assertLess(8, int(m.group(1)), "8 envs must stay under the threshold or the clean "
                                             "4-versus-8 comparison is not one")
        self.assertLessEqual(int(m.group(1)), 16, "16 envs must cross it, or the confound named "
                                                "in the prose does not exist")
        self.assertEqual(self.amortize["config"]["vec_backend"], "sync",
                         "the isolated bench is cited as holding one backend and does not")
        m = self.sentence(r"the isolated\s*bench's ([\d.]+)x sits inside it", "the bench ratio")
        self.check("the bench's 16-env ratio", m.group(1),
                   self.amortize["per_num_envs"]["16"]["speedup_vs_shipped_n4"], places=2)

    def test_each_run_started_into_a_recorded_window(self):
        m = self.sentence(r"\((\d+) MiB held by (\d+) other contexts for the 4-env seed-7 run, "
                          r"(\d+) by (\d+) for 8-env\s*seed 7, (\d+) by (\d+) for 16-env seed 7, and "
                          r"(\d+) by (\d+) for the 16-env seed-8 run", "the four windows")
        want = []
        for key, i in ((("4", "7"), 1), (("8", "7"), 3), (("16", "7"), 5), (("16", "8"), 7)):
            run = self.by_run[self.art[key]["run_id"]]
            w = run["mlflow_window"] or run["evidence_log"]
            if run["mlflow_window"]:
                want += [w["memory_used_mib"], w["other_cuda_contexts"]]
            else:
                held, _, others = re.search(r"(\d+)/(\d+) MiB in use by (\d+) other",
                                            w["gpu_window"]).groups()
                want += [int(held), int(others)]
        self.check("the four recorded windows", [int(m.group(i)) for i in range(1, 9)], want,
                   places=0)

    def test_four_of_the_six_checkpoints_record_their_own_reward(self):
        # The two that do not were trained before train_dreamer.py started writing reward_kwargs:
        # the shipped arm's 250k file is part of dreamer_v3_1m, and so is the 16-env seed-7 run.
        before = {("4", "7"), ("16", "7")}
        for key, arm in self.art.items():
            if key in before:
                self.assertIn("predates recording", arm["model"]["reward_source"],
                              f"n={key[0]} seed={key[1]} now records its reward; the README counts "
                              f"four and this test names the two exceptions")
            else:
                self.assertEqual(arm["model"]["reward_source"], "checkpoint",
                                 f"n={key[0]} seed={key[1]} fell back to an inferred reward")
        self.assertEqual(sum(1 for a in self.art.values()
                             if a["model"]["reward_source"] == "checkpoint"), 4,
                         "the prose counts four recording checkpoints")
        from envs.reward_shaping import TRAINING_REWARD_KWARGS
        self.assertEqual(self.art[("8", "7")]["model"]["reward_kwargs"],
                         dict(TRAINING_REWARD_KWARGS))
        self.sentence(r"The other four report\s*`reward_source = checkpoint`", "the recording sentence")


class TestReadmeTrainingRateCells(ReadmeGate, unittest.TestCase):
    """Was training faster before this work? The archive's own answer, and why the intuition differs.

    The claim is a 16.5x speedup measured from the trainers' logged metric timestamps, and the
    explanation for the opposite intuition is a quoted line of code. Both are checkable: the artifact
    carries every run it was built from, and the `sps` formula is a string in the trainer.
    """

    RATE = os.path.join(ROOT, "benchmarks", "training_rate_history.json")
    STACKED = os.path.join(ROOT, "benchmarks", "stacked_sac_n32_reps2.json")
    DB = os.path.join(ROOT, "mlruns.db")
    START = '**"It felt faster before all this work"'
    END = "**A third way to lose a run"

    def setUp(self):
        if not os.path.exists(self.RATE):
            self.skipTest("benchmarks/training_rate_history.json has not been generated")
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.flat = re.sub(r"\s+", " ", self.block)
        with open(self.RATE, encoding="utf-8") as handle:
            self.art = json.load(handle)
        self.before = self.art["cited"]["before_graph_capture"]
        self.after = self.art["cited"]["after_graph_capture"]
        self.what = "training-rate history"
        self.bad = []

    def test_the_two_cited_runs_are_matched_on_everything_but_the_code(self):
        for field in ("algo", "num_envs", "task_phase", "seed", "reset_mode", "total_timesteps"):
            self.assertEqual(self.before[field], self.after[field],
                             f"the two cited runs differ in {field}, so the ratio is not about code")
        self.assertNotEqual(self.before["update_graph"], self.after["update_graph"],
                            "the two runs do not differ in the thing the paragraph blames")

    def test_the_table_rows_are_the_cited_runs(self):
        for run, generation in ((self.before, "before capture"), (self.after, "after capture")):
            label = f"`{run['name']}`"
            self.assertEqual(self.cell(label, 1), generation,
                             f"{label} is in the wrong row of the pair")
            self.assertEqual(run["started"].replace("T", " ")[:16], self.cell(label, 2),
                             f"{label}'s start time is not the artifact's")
            self.check(f"{label} steps", nums(self.cell(label, 3))[0],
                       run["steps_between_samples"], places=0)
            self.check(f"{label} span", nums(self.cell(label, 4))[0], run["span_s"], places=1)
            self.check(f"{label} rate", nums(self.cell(label, 5))[0], run["rate_steps_per_s"],
                       places=1)
            self.check(f"{label} hours per 1M", nums(self.cell(label, 6))[0],
                       run["hours_per_1m_steps"], places=2)
            self.check(f"{label} first logged sps", nums(self.cell(label, 7))[0],
                       run["first_logged_sps"], places=0)
            self.check(f"{label} last logged sps", nums(self.cell(label, 8))[0],
                       run["last_logged_sps"], places=0)

    def test_the_ratio_and_the_population_it_comes_from(self):
        m = re.search(r"\*\*([\d.]+)x\*\*, and it is not two lucky draws: all (\d+) Dreamer runs in "
                      r"the archive from before the capture\s*landed sit between \*\*([\d.]+) and "
                      r"([\d.]+) env-steps/s\*\*", self.flat)
        self.assertIsNotNone(m, "the ratio sentence was reworded")
        self.check("the cited ratio", m.group(1), self.art["cited"]["ratio_after_over_before"],
                   places=2)
        pre = [r["rate_steps_per_s"] for r in self.art["runs"]
               if r["algo"] == "dreamer" and r["started"] < "2026-10-03"]
        self.check("the pre-capture population", [int(m.group(2)), m.group(3), m.group(4)],
                   [len(pre), min(pre), max(pre)], places=1)
        # The prose quotes no run count on purpose - the archive grows with every run - so the check
        # is that the artifact really is the list it is described as, and that the cited pair is in it.
        self.assertTrue(len(self.art["runs"]) >= 2, "the artifact holds no population to draw from")
        for run in (self.before, self.after):
            self.assertIn(run["run_uuid"], [r["run_uuid"] for r in self.art["runs"]],
                          "a cited run is not in the list the artifact publishes")
        m = re.search(r"recomputes a rate for every run\s*in `mlruns\.db` that logged `sps` twice or "
                      r"more - the artifact lists them all", self.flat)
        self.assertIsNotNone(m, "the population sentence was reworded")

    def test_the_span_cross_checks_the_published_hours(self):
        """The 1.27 h in the Phase-1 section comes from the trainer's clock; this is MLflow's."""
        m = re.search(r"The ([\d,]+\.?\d*) s span is also an independent\s*confirmation of the "
                      r"([\d.]+) h this README quotes", self.flat)
        self.assertIsNotNone(m, "the cross-check sentence was reworded")
        self.check("the span", m.group(1).replace(",", ""), self.after["span_s"], places=1)
        self.check("the hours", m.group(2), self.after["hours_per_1m_steps"], places=2)
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        other = re.search(r"the run above took \*\*([\d.]+) h\*\* of machine time", readme)
        self.assertIsNotNone(other, "the Phase-1 wall-clock sentence moved")
        self.assertEqual(float(other.group(1)), self.after["hours_per_1m_steps"],
                         "the two independent instruments no longer agree on the run's hours")

    def test_the_mechanism_is_the_line_of_code_it_quotes(self):
        m = re.search(r"`sps` is\s*`([^`]+)`", self.flat)
        self.assertIsNotNone(m, "the sps-definition sentence was reworded")
        with open(os.path.join(ROOT, "train_dreamer.py"), encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn(m.group(1), source,
                      "the README quotes a line of train_dreamer.py that is no longer in it")
        self.assertIn(m.group(1), self.art["sps_definition"],
                      "the artifact's recorded definition and the README's quote have diverged")
        # The shape of the claim: a cumulative average whose first sample precedes any update must
        # sit well above the settled rate. It is 18x above it in the old run and 3.2x in the new one;
        # anything under 2x would make "the chart decays" the wrong explanation.
        for run in (self.before, self.after):
            self.assertGreater(run["first_logged_sps"], 2.0 * run["last_logged_sps"],
                               f"{run['name']} does not show the decay the paragraph explains")
        m = re.search(r"the old trainer's chart read \*\*(\d+)\*\* and the new one reads \*\*(\d+)\*\*",
                      self.flat)
        self.assertIsNotNone(m, "the chart sentence was reworded")
        self.check("the two first samples", [m.group(1), m.group(2)],
                   [self.before["first_logged_sps"], self.after["first_logged_sps"]], places=0)

    def test_the_first_sample_is_the_one_logged_at_learning_starts(self):
        """The mechanism the paragraph names, checked rather than described.

        If the first `sps` sample is logged before any update runs, then every run's first sample is
        at the same step - `learning_starts` - and that step is the trainer's default. Both halves
        are in the archive and the source.
        """
        dreamer = [r for r in self.art["runs"] if r["algo"] == "dreamer"]
        with open(os.path.join(ROOT, "train_dreamer.py"), encoding="utf-8") as handle:
            source = handle.read()
        default = re.search(r'"--learning-starts", type=int, default=(\d+)', source)
        self.assertIsNotNone(default, "the --learning-starts default moved or was reworded")
        starts = int(default.group(1))
        # A run logs its first sample at the first global_step it reaches at or after
        # learning_starts, and global_step advances by num_envs, so the step is a multiple of
        # num_envs. 18 of 18 Dreamer runs in the archive land on exactly that grid.
        for r in dreamer:
            n = int(r["num_envs"])
            self.assertEqual(r["first_step"], n * -(-starts // n),
                             f"{r['name']}: first sample at {r['first_step']}, expected the first "
                             f"multiple of {n} at or above {starts}")
        m = re.search(r"its first sample is logged at the first step the run actually reaches at or "
                      r"after `learning_starts`\s*\((\d+) for a 4- or 8-env run, (\d+) for a "
                      r"16-env one", self.flat)
        self.assertIsNotNone(m, "the first-sample sentence was reworded")
        self.assertEqual([int(m.group(1)), int(m.group(2))], [starts, 16 * -(-starts // 16)],
                         "the two steps the prose names are not the grid the runs land on")

    def test_the_two_limits_on_the_claim(self):
        m = re.search(r"the (\d+) earlier runs span (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})",
                      self.flat)
        self.assertIsNotNone(m, "the archive-limit sentence was reworded")
        if not os.path.exists(self.DB):
            self.skipTest("mlruns.db is not in this checkout")
        cutoff_ms = int(datetime.datetime.fromisoformat(
            min(r["started"] for r in self.art["runs"])).timestamp() * 1000)
        con = sqlite3.connect("file:" + self.DB.replace(os.sep, "/") + "?mode=ro", uri=True,
                              timeout=10)
        try:
            count, first, last = con.execute(
                "SELECT COUNT(*), MIN(date(start_time/1000,'unixepoch','localtime')), "
                "MAX(date(start_time/1000,'unixepoch','localtime')) FROM runs "
                "WHERE start_time < ?", (cutoff_ms,)).fetchone()
        finally:
            con.close()
        self.check("the earlier-run count", int(m.group(1)), count, places=0)
        self.assertEqual([m.group(2), m.group(3)], [first, last],
                         "the date range of the pre-metric runs is not the database's")
        with open(self.STACKED, encoding="utf-8") as handle:
            stacked = json.load(handle)["n32"]
        # Same basis as TestReadmeStackedReplicationCells: the rate a reader re-divides from the
        # median printed beside it, not from the artifact's unrounded median.
        rates = [round(stacked["config"]["steps"] / round(statistics.median(v), 1))
                 for v in stacked["arms"].values()]
        m = re.search(r"measures \*\*([\d,]+)-([\d,]+) env-steps/s\*\*", self.flat)
        self.assertIsNotNone(m, "the cross-algorithm sentence was reworded")
        self.check("the stacked SAC range", [int(m.group(1).replace(",", "")),
                                             int(m.group(2).replace(",", ""))],
                   [min(rates), max(rates)], places=0)


class TestReadmeUnifiedPhaseCells(unittest.TestCase):
    """The "would one run do everything?" answer, against the environment and trainer source.

    The block's claim is that the target phase already *is* the unified task, and that what blocks a
    phase changing mid-run is the observation width. Both are structural facts about the code, so
    both are checked by constructing the thing rather than by reading a comment - including the
    curriculum hooks the block calls dead, which are the part most likely to be finished by someone
    who then finds this paragraph stale.
    """

    START = "**Would one run that does everything beat the phase split?"
    END = "The scale of that table matters"
    POSTURE = os.path.join(ROOT, "benchmarks", "phase1_posture_probe.json")

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.flat = re.sub(r"\s+", " ", readme[start:readme.index(self.END, start)])

    def test_the_phase_decides_the_width_and_the_width_is_three(self):
        import gymnasium as gym
        import envs.walker_ragdoll_env  # noqa: F401  registers WalkerRagdoll-v0
        shapes, envs_made = {}, []
        for phase in ("recovery", "balance", "walk", "target"):
            env = gym.make("WalkerRagdoll-v0", task_phase=phase)
            envs_made.append(env)
            shapes[phase] = env.observation_space.shape[0]
            env.close()
        self.assertEqual(shapes, {"recovery": 46, "balance": 46, "walk": 46, "target": 49},
                         "the observation width per phase is not what the block states")
        m = re.search(r"`observation_size = 49 if task_phase == \"target\" else\s*46`", self.flat)
        self.assertIsNotNone(m, "the width expression was reworded or the code moved")
        with open(os.path.join(ROOT, "envs", "walker_ragdoll_env.py"), encoding="utf-8") as handle:
            self.assertIn("observation_size = 49 if task_phase == \"target\" else 46", handle.read())

    def test_the_three_extra_components_are_the_target(self):
        import numpy as np
        import gymnasium as gym
        import envs.walker_ragdoll_env  # noqa: F401
        env = gym.make("WalkerRagdoll-v0", task_phase="target", reset_mode="mixed")
        try:
            obs, _ = env.reset(seed=11)
            rel_x, rel_y, distance = env.unwrapped._get_target_obs()
            np.testing.assert_allclose(np.asarray(obs)[-3:], [rel_x, rel_y, distance], rtol=0,
                                       atol=1e-12)
            self.assertLessEqual(distance, 5.0, "the distance component is documented as clipped")
        finally:
            env.close()

    def test_only_the_target_phase_terminates_on_a_fall(self):
        import train_walker
        args = lambda phase: type("A", (), {"reset_mode": "mixed", "fixed_reset_probability": 0.25,
                                            "upright_reset_probability": 0.15,
                                            "fallen_velocity_scale": 0.35, "task_phase": phase,
                                            "target_forward_velocity": 0.8})()
        got = {p: train_walker.env_common_kwargs(args(p))["terminate_when_unhealthy"]
               for p in ("recovery", "balance", "walk", "target")}
        self.assertEqual(got, {"recovery": False, "balance": False, "walk": False, "target": True},
                         "the block says target is the only phase the trainers terminate on a fall")

    def test_the_mixed_reset_proportions_are_the_constructor_s_defaults(self):
        import inspect
        from envs.walker_ragdoll_env import WalkerRagdollEnv
        params = inspect.signature(WalkerRagdollEnv.__init__).parameters
        fixed = params["fixed_reset_probability"].default
        upright = params["upright_reset_probability"].default
        m = re.search(r"which is (\d+)% fixed-fallen, (\d+)% upright and (\d+)% randomized-fallen",
                      self.flat)
        self.assertIsNotNone(m, "the reset-mixture sentence was reworded")
        self.assertEqual([int(m.group(i)) for i in (1, 2, 3)],
                         [round(100 * fixed), round(100 * upright),
                          round(100 * (1.0 - fixed - upright))])

    def test_the_automatic_curriculum_hooks_are_still_unused(self):
        """The automatic in-env promotion is unused; the *curriculum* is not - it is run
        manually across restarts, and `TestReadmeCurriculumProvenanceCells` gates that. If
        someone finishes these hooks, this test fails and both paragraphs get rewritten."""
        import gymnasium as gym
        import envs.walker_ragdoll_env  # noqa: F401
        env = gym.make("WalkerRagdoll-v0", task_phase="target", reset_mode="mixed",
                       target_curriculum_streak=3)
        try:
            self.assertFalse(hasattr(env.unwrapped, "_target_curriculum_streak"),
                             "target_curriculum_streak is now stored, so it is no longer dead")
            self.assertTrue(hasattr(env.unwrapped, "_target_fixed_until_curriculum"),
                            "the unused flag the block names has been removed - update the sentence")
            obs, info = env.reset(seed=11)
            for _ in range(3):
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample() * 0.0)
            self.assertEqual(info.get("target_success_streak"), 0,
                             "target_success_streak is no longer hard-coded to 0")
        finally:
            env.close()

    def test_the_target_resampling_range_is_what_the_block_says(self):
        import inspect
        from envs.walker_ragdoll_env import WalkerRagdollEnv
        lo, hi = inspect.signature(WalkerRagdollEnv.__init__).parameters[
            "target_distance_range"].default
        m = re.search(r"target resampling - (\d+)-(\d+) m away, within ±([\d.]+) rad", self.flat)
        self.assertIsNotNone(m, "the resampling sentence was reworded")
        self.assertEqual([float(m.group(1)), float(m.group(2))], [lo, hi])
        with open(os.path.join(ROOT, "envs", "walker_ragdoll_env.py"), encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn(f"self.np_random.uniform(-{m.group(3)}, {m.group(3)})", source,
                      "the angle bound the block quotes is not the one the env samples from")

    def test_the_two_rows_the_argument_rests_on(self):
        with open(self.POSTURE, encoding="utf-8") as handle:
            posture = json.load(handle)["models"]
        m = re.search(r"the difference between those two rows is ([\d.]+)% against ([\d.]+)%",
                      self.flat)
        self.assertIsNotNone(m, "the closing comparison was reworded")
        self.assertAlmostEqual(float(m.group(1)), posture["sac_40m"]["mean_pct_steps_in_band"],
                               places=2)
        self.assertAlmostEqual(float(m.group(2)), posture["dreamer_v3_1m"]["mean_pct_steps_in_band"],
                               places=2)
        m = re.search(r"cleared the standing rung on it \(([\d.]+)% of its steps\s*in the band\)",
                      self.flat)
        self.assertIsNotNone(m, "the standing-rung sentence was reworded")
        self.assertAlmostEqual(float(m.group(1)), posture["sac_40m"]["mean_pct_steps_in_band"],
                               places=2)



class TestReadmeTargetFromScratchCells(ReadmeGate, unittest.TestCase):
    """The from-zero 40M target run against the curriculum run, cell by cell.

    This is the arm the previous section did not have, and the block's whole point is a comparison -
    so every figure in both tables, in the reach-rate series, in the wall clock and in the claim that
    the control reproduced its own published row, is recomputed from the artifacts here. The two
    artifacts are also the ones where a silent protocol change would be invisible: if the pairing
    stopped being per-episode, or the posture control stopped reproducing, the prose would still read
    the same while meaning something else.
    """

    PAIRED = os.path.join(ROOT, "benchmarks",
                          "target_learning_curve_from_scratch_paired.json")
    POSTURE = os.path.join(ROOT, "benchmarks", "posture_target_from_scratch_v9.json")
    PUBLISHED_POSTURE = os.path.join(ROOT, "benchmarks", "phase1_posture_probe.json")
    CURVE = os.path.join(ROOT, "benchmarks", "target_learning_curve_from_scratch_v9.json")
    START = "**One phrase in that paragraph was doing more work than it could carry.**"
    END = "What the phase split does cost is that a phase cannot change mid-run."

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.flat = re.sub(r"\s+", " ", self.block)
        self.paired, self.posture, self.published, self.curve = self.read_artifacts(
            self.PAIRED, self.POSTURE, self.PUBLISHED_POSTURE, self.CURVE)
        self.bad = []
        self.what = "target-from-scratch comparison"

    def test_the_paired_table_is_the_paired_artifact(self):
        rows = {
            "episodes that got closer to the target":
                [self.paired["scratch_closer"], self.paired["curriculum_closer"]],
            "median closest approach":
                [self.paired["closest_approach"]["scratch"]["median"],
                 self.paired["closest_approach"]["curriculum"]["median"]],
            "best single episode":
                [self.paired["closest_approach"]["scratch"]["min"],
                 self.paired["closest_approach"]["curriculum"]["min"]],
            "episodes ending inside the radius":
                [self.paired["reached_target"]["scratch_only"],
                 self.paired["reached_target"]["curriculum_only"]],
        }
        for label, want in rows.items():
            m = re.search(rf"\| {label} \| ([\d.]+)(?: m)? \| ([\d.]+)(?: m)?", self.block)
            self.assertIsNotNone(m, f"the paired row {label!r} is gone")
            self.check(label, [m.group(1), m.group(2)], want, places=3)
        m = re.search(r"(\d+) pairs within 5 mm", self.flat)
        self.assertIsNotNone(m, "the within-5mm count is gone from the paired table")
        self.check("within 5 mm", m.group(1), self.paired["within_5mm"], places=0)

    def test_the_delta_and_episode_counts_are_recomputed(self):
        m = self.sentence(r"Median advantage ([\d.]+) m, mean ([\d.]+) m", "the paired delta")
        self.check("paired delta", [m.group(1), m.group(2)],
                   [abs(self.paired["median_delta_m"]), abs(self.paired["mean_delta_m"])], places=3)
        m = self.sentence(r"Over the (\d+) matched episodes", "the pair count")
        self.check("pair count", m.group(1), self.paired["pairs"], places=0)
        self.assertEqual(self.paired["reached_target"]["both"], 0,
                         "the block says the two arms never reach in the same episode")
        m = self.sentence(r"all (\d+) telemetry rows came back identical.*?"
                          r"\(`curriculum_rescore_matches_published`\)", "the reproduction claim")
        self.check("telemetry rows compared", m.group(1),
                   self.paired["curriculum_rescore_matches_published"]["episodes_compared"],
                   places=0)
        self.assertTrue(self.paired["curriculum_rescore_matches_published"]["identical"],
                        "the re-scored curriculum arm no longer reproduces its committed curve, so "
                        "the sentence that claims it does has to be rewritten")

    def test_the_run_provenance_and_rate_are_the_recorded_ones(self):
        run = self.paired["from_scratch_run"]
        m = self.sentence(r"(\d+)M steps in (\d+) h (\d+) min (\d+) s at ([\d,]+) env-steps/s",
                          "the run's wall clock")
        self.check("steps", m.group(1), run["config"]["total_timesteps"] // 1_000_000, places=0)
        typed = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
        self.assertEqual(typed, run["wall_clock_seconds"],
                         f"README says {m.group(2)} h {m.group(3)} min {m.group(4)} s, the artifact "
                         f"records {run['wall_clock_seconds']} s")
        self.check("sustained rate", m.group(5), run["sustained_env_steps_per_s"], places=0)
        for key in ("seed", "num_envs", "target_forward_velocity"):
            want = {"seed": 7, "num_envs": 32, "target_forward_velocity": 1.2}[key]
            self.assertEqual(run["config"][key], want,
                             f"the README names {key}={want}; the recorded run says "
                             f"{run['config'][key]}")
        self.assertIsNone(run["config"]["init_from"],
                          "the arm is only 'from zero' while no checkpoint seeds it")

    def test_the_reach_rate_series_is_the_curve_artifact(self):
        m = self.sentence(r"Reach rate by checkpoint \(([^)]*)\) is ([\d., ]+) percent",
                          "the reach-rate series")
        budgets = [int(x.strip().rstrip("M")) for x in m.group(1).split(",")]
        typed = [float(x.strip()) for x in m.group(2).split(",")]
        arms = [f"s{b}000000" for b in budgets]
        want = [self.curve["models"][a]["reached_target_pct"] for a in arms]
        self.assertEqual(len(typed), len(want), "the reach-rate series lost or gained an entry")
        self.check("reach rate", typed, want, places=1)

    def test_the_posture_table_is_the_posture_artifact(self):
        want = {
            "from scratch, 8M": "scratch_8m", "from scratch, 20M": "scratch_20m",
            "from scratch, 40M": "scratch_40m",
            "`walker_target_v1`, 40M (control)": "curriculum_40m",
        }
        for label, key in want.items():
            row = self.posture["models"][key]
            cells = re.search(rf"\| {re.escape(label)} \| ([\d.]+) \| (\d+)/50 \| ([\d.]+) m \| "
                              r"([\d.]+) m \|", self.block)
            self.assertIsNotNone(cells, f"the posture row {label!r} is gone or reformatted")
            self.check(label, [cells.group(1), cells.group(2), cells.group(3), cells.group(4)],
                       [row["mean_pct_steps_in_band"], row["episodes_ever_in_band"],
                        row["mean_max_z"], row["best_max_z_in_one_episode"]], places=3)

    def test_the_posture_control_row_is_the_published_row(self):
        """The three new rows are only comparable to the table because the control still reproduces."""
        published = self.published["models"]["sac_40m"]
        control = self.posture["models"]["curriculum_40m"]
        for field in ("mean_pct_steps_in_band", "mean_max_z", "best_max_z_in_one_episode",
                      "mean_falls_per_episode", "episodes_ever_in_band"):
            self.assertEqual(control[field], published[field],
                             f"the posture control no longer reproduces the published row on "
                             f"{field}: {published[field]} against {control[field]}, so the four "
                             "rows in this table are not in the same instrument")
        self.assertEqual(self.posture["torch_threads"], self.published["torch_threads"],
                         "the control changed its thread pin, which the published posture rows "
                         "were measured with")

    def test_the_standing_band_floor_is_the_environments_own_constant(self):
        m = self.sentence(r"does not reach the band.s floor at ([\d.]+) m", "the band floor")
        import inspect
        from envs.walker_ragdoll_env import WalkerRagdollEnv
        default = re.search(r"healthy_z_range: tuple = \(([\d.]+), ([\d.]+)\)",
                            inspect.getsource(WalkerRagdollEnv.__init__))
        self.assertIsNotNone(default, "the env no longer declares its healthy_z_range default")
        self.check("band floor", m.group(1), float(default.group(1)), places=2)
        stated = float(self.posture["protocol"].split("torso z in (")[1].split(",")[0].strip())
        self.assertAlmostEqual(stated, float(default.group(1)), places=2,
                               msg="the posture artifact measured a different band than the env "
                                   "documents")

    def test_the_return_caveat_names_both_task_speeds(self):
        m = self.sentence(r"records `target_forward_velocity` ([\d.]+) and the from-scratch "
                          r"checkpoints record ([\d.]+)", "the reward-function caveat")
        self.check("task speeds", [float(m.group(1)), float(m.group(2))],
                   [self.published["models"]["sac_40m"]["reward_kwargs_applied"]
                    ["target_forward_velocity"],
                    self.posture["models"]["scratch_40m"]["reward_kwargs_applied"]
                    ["target_forward_velocity"]], places=1)


class TestReadmePhysicsPresetCells(ReadmeGate, unittest.TestCase):
    """The physics-preset block: what each world costs, how far it drifts, what it buys end to end.

    The block makes two claims that could rot in opposite ways. The throughput ratios are a window,
    so they must keep matching the artifact that was measured back-to-back; and the end-to-end
    1.15x is a different measurement (two training runs) that must not be allowed to drift into
    sounding like the 2.71x env-step figure. Each cell is recomputed from its own artifact.
    """

    PRESETS = os.path.join(ROOT, "benchmarks", "physics_presets.json")
    SCREENS = os.path.join(ROOT, "benchmarks", "physics_presets_screen_paired.json")
    SCREENS5M = os.path.join(ROOT, "benchmarks", "physics_presets_screen5m_paired.json")
    START = "**That sentence now has a number on it, because a cheaper world was built and measured.**"
    END = "### \U0001f9ee The learner side"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.flat = re.sub(r"\s+", " ", self.block)
        self.presets, self.screens, self.screens5m = self.read_artifacts(
            self.PRESETS, self.SCREENS, self.SCREENS5M)
        self.bad = []
        self.what = "physics presets"

    def test_the_preset_table_is_the_ratio_block(self):
        for label, key in (("`euler`", "euler"), ("`fast`", "fast")):
            m = re.search(rf"\| {label} \|[^|]*\| ([\d.]+)x \| ([\d.]+)x \| ([\d.]+)x less \|",
                          self.block)
            self.assertIsNotNone(m, f"the preset table row {label} is gone")
            want = self.presets["throughput_ratio"][key]
            self.check(label, [m.group(1), m.group(2), m.group(3)],
                       [want["env_step_n1_vs_v9"], want["env_step_n8_vs_v9"],
                        want["mj_step_time_vs_v9"]], places=2)

    def test_the_absolute_rates_and_spread_are_the_measured_ones(self):
        m = self.sentence(r"were ([\d,]+) env-step/s for `v9` at n=1 against ([\d,]+) for `fast`, "
                          r"with ([\d.]+)-([\d.]+)% rep spread", "the absolute rate sentence")
        v9 = self.presets["throughput"]["v9"]["env_step_n1"]
        fast = self.presets["throughput"]["fast"]["env_step_n1"]
        self.check("n=1 rates", [m.group(1), m.group(2)],
                   [v9["median_steps_per_s"], fast["median_steps_per_s"]], places=0)
        spreads = sorted([v9["spread_pct"], fast["spread_pct"]])
        self.check("rep spread", [m.group(3), m.group(4)],
                   [min(spreads), max(spreads)], places=1)

    def test_the_divergence_cells_are_the_artifacts(self):
        div = self.presets["divergence"]
        m = self.sentence(r"`euler` separates by up to ([\d.]*\d) m of torso height and `fast` by "
                          r"([\d.]*\d) m, with mean observation L2 distance ([\d.]*\d) and "
                          r"([\d.]*\d)",
                          "the drift sentence")
        self.check("drift", m.groups(),
                   [div["euler"]["max_abs_torso_z_delta"], div["fast"]["max_abs_torso_z_delta"],
                    div["euler"]["mean_abs_obs_l2"], div["fast"]["mean_abs_obs_l2"]], places=2)
        m = self.sentence(r"over those ([\d,]+) steps the reference holds ([\d,]+) steps with a "
                          r"foot on the floor and ([\d,]+) with some other body on it, `euler` gets "
                          r"([\d,]+) and ([\d,]+), `fast` ([\d,]+) and ([\d,]+)",
                          "the standing-gate inputs")
        episodes = div["euler"]["episodes"]
        self.check("compared steps", m.group(1), episodes * 1000, places=0)
        self.check("contact totals", m.groups()[1:],
                   [div["euler"]["reference_foot_contact_steps_total"],
                    div["euler"]["reference_bad_support_steps_total"],
                    div["euler"]["foot_contact_steps_total"],
                    div["euler"]["bad_support_steps_total"],
                    div["fast"]["foot_contact_steps_total"],
                    div["fast"]["bad_support_steps_total"]], places=0)
        m = self.sentence(r"counter says ([\d,]+) and ([\d,]+)", "the per-step contact counter")
        self.check("contact counter", m.groups(),
                   [div["euler"]["floor_contact_disagreements"],
                    div["fast"]["floor_contact_disagreements"]], places=0)

    def test_the_end_to_end_screen_is_a_separate_measurement(self):
        arms = self.screens["arms"]
        m = self.sentence(r"\*\*(\d+) min (\d+) s in `v9`, (\d+) min (\d+) s in `fast`, which is "
                          r"([\d.]+)x\*\* where the environment step was ([\d.]+)x cheaper",
                          "the end-to-end sentence")
        typed = [int(m.group(1)) * 60 + int(m.group(2)), int(m.group(3)) * 60 + int(m.group(4))]
        self.check("screen wall clocks", typed,
                   [arms["v9"]["wall_clock_seconds"], arms["fast"]["wall_clock_seconds"]], places=0)
        self.check("end-to-end speedup", m.group(5),
                   self.screens["end_to_end_speedup_fast_vs_v9"], places=2)
        self.check("env-step ratio quoted beside it", m.group(6),
                   self.presets["throughput_ratio"]["fast"]["env_step_n8_vs_v9"], places=2)

    def test_the_screen_scores_and_the_caveat_are_the_paired_artifact(self):
        arms = self.screens["arms"]
        m = self.sentence(r"mean ([\d,.]+) against ([\d,.]+), std ([\d,.]+) against ([\d,.]+), "
                          r"mean closest approach ([\d.]+) m against ([\d.]+) m",
                          "the screen scores")
        self.check("screen scores", m.groups()[:4],
                   [arms["v9"]["mean"], arms["fast"]["mean"],
                    arms["v9"]["std"], arms["fast"]["std"]], places=2)
        self.check("closest approach", m.groups()[4:],
                   [arms["v9"]["mean_min_target_distance"], arms["fast"]["mean_min_target_distance"]],
                   places=3)
        m = self.sentence(r"reach the target in (\d+) of (\d+) episodes each", "the 0-of-20 caveat")
        self.assertEqual([int(m.group(1)), int(m.group(2))],
                         [0, 20], "the screens now differ from the 'both measure zero' claim")
        for preset in ("v9", "fast"):
            self.assertEqual(arms[preset]["reached_target_pct"], 0.0)
        self.assertEqual(arms["fast"]["scored_in_version"],
                         self.presets["protocol"].split("=")[1].split(")")[0].strip() + "_fast",
                         "the fast arm was not scored in the world it trained in")
        self.assertEqual(arms["fast"]["device"], arms["v9"]["device"],
                         "the two screens were scored on different devices")

    def test_the_dreamer_preset_gain_is_the_arithmetic_of_the_loop_split(self):
        pair, split = self.read_artifacts(
            os.path.join(ROOT, "benchmarks", "physics_presets_dreamer_pair.json"),
            os.path.join(ROOT, "benchmarks", "dreamer_loop_split.json"))
        m = self.sentence(r"\*\*(\d+) s in `v9` against (\d+) s in `fast`, ([\d.]+)x\*\*",
                          "the Dreamer preset pair")
        self.check("Dreamer wall clocks", [m.group(1), m.group(2)],
                   [pair["arms"]["v9"]["seconds"], pair["arms"]["fast"]["seconds"]], places=0)
        self.check("Dreamer speedup", m.group(3), pair["measured_speedup_fast_vs_v9"], places=3)
        # The prediction is recomputed here rather than trusted from the artifact: the point of the
        # paragraph is that the share of the loop, not the ratio of the step, sets the gain.
        predicted = 1.0 / (split["update_share_pct"] / 100.0
                           + (1 - split["update_share_pct"] / 100.0)
                           / pair["predicted_from_loop_split"]["env_step_ratio_used"])
        m = self.sentence(r"the update and the preset makes `env\.step` ([\d.]+)x cheaper at n=8, "
                          r"so leaving the update alone and\s*dividing only the collection part "
                          r"predicts \*\*([\d.]+)x\*\* - the run lands \*\*([\d.]+)%\*\*",
                          "the loop-split prediction")
        self.check("env ratio", m.group(1),
                   pair["predicted_from_loop_split"]["env_step_ratio_used"], places=3)
        self.check("update share", split["update_share_pct"],
                   pair["predicted_from_loop_split"]["update_share_pct"], places=1)
        self.check("predicted", m.group(2), round(predicted, 3), places=3)
        self.check("prediction error", m.group(3),
                   abs(pair["prediction_error_pct"]), places=1)
        # The same claim made for the two collection-bound trainers, from their own artifacts.
        trainer = json.load(open(os.path.join(ROOT, "benchmarks", "trainer_pair_ppo_sac.json"),
                                 encoding="utf-8"))
        ppo_gain = trainer["arms"]["ppo_v9"]["seconds"] / trainer["arms"]["ppo_fast"]["seconds"]
        self.assertAlmostEqual(ppo_gain, 1.156, places=3,
                               msg="the paragraph lists PPO's preset gain; it is not 1.156x")

    def test_the_1m_triple_is_the_three_worlds_measured_in_one_window(self):
        triple = json.load(open(os.path.join(ROOT, "benchmarks",
                                             "physics_presets_sac1m_triple.json"), encoding="utf-8"))
        arms = triple["arms"]
        m = self.sentence(r"\*\*([\d,]+) s in `v9`, ([\d,]+) s in `euler`, ([\d,]+) s in\s*`fast`\*\* "
                          r"- ([\d.]+)x and ([\d.]+)x", "the triple wall clocks")
        self.check("triple seconds", m.groups()[:3],
                   [arms["v9"]["train_seconds"], arms["euler"]["train_seconds"],
                    arms["fast"]["train_seconds"]], places=0)
        self.check("triple ratios", [m.group(4), m.group(5)],
                   [triple["train_seconds_over_v9"]["euler"], triple["train_seconds_over_v9"]["fast"]],
                   places=3)
        m = self.sentence(r"mean return \*\*([\d,.]+)\*\* in `v9`, \*\*([\d,.]+)\*\* in `euler`, "
                          r"\*\*([\d,.]+)\*\* in `fast`, with `fast` the only arm that ended any "
                          r"episode standing \(([\d.]+)% against ([\d.]+)%\) and the only one that "
                          r"got meaningfully closer to the target \(([\d.]+) m against ([\d.]+) m\), "
                          r"while also falling most \(([\d.]+) per episode against ([\d.]+)",
                          "the behavioural column")
        self.check("means", m.groups()[:3],
                   [arms["v9"]["mean"], arms["euler"]["mean"], arms["fast"]["mean"]], places=2)
        self.check("standing", m.groups()[3:5],
                   [arms["fast"]["standing_at_end_pct"], arms["v9"]["standing_at_end_pct"]], places=1)
        self.check("approach", m.groups()[5:7],
                   [arms["fast"]["mean_min_target_distance"], arms["v9"]["mean_min_target_distance"]],
                   places=3)
        self.check("falls", m.groups()[7:9],
                   [arms["fast"]["falls_per_episode"], arms["v9"]["falls_per_episode"]], places=2)
        m = self.sentence(r"evaluation of the three arms took (\d+) s, (\d+) s and\s*(\d+) s",
                          "the scorer's own clock")
        self.check("eval seconds", m.groups(),
                   [arms["v9"]["eval_seconds"], arms["euler"]["eval_seconds"],
                    arms["fast"]["eval_seconds"]], places=0)
        self.assertEqual(triple["train_seconds_over_v9"]["fast"],
                         round(arms["v9"]["train_seconds"] / arms["fast"]["train_seconds"], 3))

    def test_the_two_1m_windows_of_one_recipe_disagree_about_which_world_leads(self):
        """The level column at 1M is a second-hand measurement, and the second hand says "no"."""
        triple = json.load(open(os.path.join(ROOT, "benchmarks",
                                             "physics_presets_sac1m_triple.json"), encoding="utf-8"))
        first = json.load(open(os.path.join(ROOT, "benchmarks",
                                            "physics_presets_screen_paired.json"), encoding="utf-8"))
        t, f = triple["arms"], first["arms"]
        # The paragraph calls the two windows one recipe, so every instrument setting it names has to
        # actually match: a CUDA arm compared against a CPU one would be a different claim.
        for preset in ("v9", "fast"):
            self.assertEqual(t[preset]["device"], f[preset]["device"],
                             f"{preset}: the two windows are not the same device")
            self.assertEqual(t[preset]["torch_threads"], f[preset]["torch_threads"])
            self.assertEqual(t[preset]["scored_in_version"], f[preset]["scored_in_version"])
        for art in (triple, first):
            self.assertIn("seed 7", art["protocol"])
            self.assertIn("num_envs=8", art["protocol"])
            self.assertIn("target_forward_velocity=1.2", art["protocol"])
        m = self.sentence(r"it found \*\*([\d,.]+)\*\* in `v9` against \*\*([\d,.]+)\*\* in `fast`, "
                          r"where this window found\s*\*\*([\d,.]+)\*\* against \*\*([\d,.]+)\*\*",
                          "the two windows' 1M means")
        self.check("first window means", m.groups()[:2], [f["v9"]["mean"], f["fast"]["mean"]],
                   places=2)
        self.check("this window means", m.groups()[2:], [t["v9"]["mean"], t["fast"]["mean"]],
                   places=2)
        m = self.sentence(r"came down by a factor of ([\d.]+), the cheap one came up by ([\d.]+)",
                          "the reversal factors")
        self.check("reversal factors", [m.group(1), m.group(2)],
                   [round(f["v9"]["mean"] / t["v9"]["mean"], 1),
                    round(t["fast"]["mean"] / f["fast"]["mean"], 1)], places=1)
        self.assertLess(t["v9"]["mean"], f["v9"]["mean"] / 3.0,
                        "v9 no longer comes down by several-fold between the windows")
        self.assertGreater(f["v9"]["mean"], f["fast"]["mean"],
                           "the first window no longer has v9 ahead")
        self.assertGreater(t["fast"]["mean"], t["v9"]["mean"],
                           "this window no longer has fast ahead, so the ordering has not reversed")
        m = self.sentence(r"\*\*(\d+) of (\d+)\*\* episodes reach the target in every arm of both "
                          r"windows", "the replicated question")
        self.assertEqual(m.group(1), "0", "the two windows no longer both fail to reach")
        self.assertEqual(m.group(2), "20", "the eval protocol is no longer 20 episodes per arm")
        for art in (triple, first):
            self.assertIn("--num-episodes 20", art["protocol"])
        reaches = [arm["reached_target_pct"] for arm in list(t.values()) + list(f.values())
                   if "reached_target_pct" in arm]
        self.assertEqual(len(reaches), 5)
        self.assertEqual(set(reaches), {0.0},
                         f"some arm of the two 1M windows reaches the target ({reaches}), so the "
                         "question a 1M screen asks is no longer the only thing that replicated")

    def test_the_screens_environment_count_is_also_their_optimisation_dose(self):
        """1,798.4 against 467.4 s per 1M is the 4x update ratio, so the paragraph may call it that.

        Both rates come from artifacts that record their own `num_envs`, and the citation points at
        the loop lines, so this test also fails if the cited range stops being the UTD loop.
        """
        scratch = json.load(open(os.path.join(ROOT, "benchmarks",
                                             "target_learning_curve_from_scratch_paired.json"),
                                 encoding="utf-8"))
        run = scratch["from_scratch_run"]
        per_m_8 = self.screens5m["arms"]["v9"]["seconds_per_1m"]
        per_m_32 = round(run["wall_clock_seconds"] / (run["config"]["total_timesteps"] / 1e6), 1)
        self.check("s per 1M at 8 envs", self.sentence(
            r"the `v9` arm of the 5M pair ran at \*\*([\d,.]+) s per 1M steps\*\* at (\d+) envs",
            "the 8-env rate").group(1), per_m_8, places=1)
        m = self.sentence(r"the `v9` arm of the 5M pair ran at \*\*([\d,.]+) s per 1M steps\*\* at "
                          r"(\d+) envs while the 40M run at (\d+) envs sustained \*\*([\d,.]+) s per "
                          r"1M\*\*, and [\d,.]+/[\d,.]+ is \*\*([\d.]+)x\*\*", "the two dose rates")
        self.check("envs in the screens", m.group(2),
                   int(self.screens5m["protocol"].split("num_envs=")[1].split(",")[0]), places=0)
        self.check("envs in the 40M run", m.group(3), run["config"]["num_envs"], places=0)
        self.check("s per 1M at 32 envs", m.group(4), per_m_32, places=1)
        self.check("the rate ratio", m.group(5), round(per_m_8 / per_m_32, 2), places=2)
        self.assertEqual(int(m.group(3)) // int(m.group(2)), 4,
                         "the paragraph says the dose ratio is four; the recorded configs disagree")
        # The default that licenses "no committed run moves". Read from the source rather than by
        # importing the trainer: this whole file is a text-and-JSON gate and stays runnable while a
        # training job owns the machine.
        source = open(os.path.join(ROOT, "train_walker.py"), encoding="utf-8").read()
        self.assertIn('"UTD_RATIO": "1",', source,
                      "the shipped default is no longer one update per collection iteration")
        lines = source.splitlines()
        utd_lines = {i + 1 for i, text in enumerate(lines)
                     if "for _utd in range(max(1, args.utd_ratio))" in text}
        self.assertEqual(len(utd_lines), 2, "SAC and TD3 should each own one UTD loop")
        cited = re.findall(r"\(`?train_walker\.py:(\d+)-(\d+)`?\)", self.block)
        self.assertTrue(cited, "the paragraph no longer cites the loop it is describing")
        for lo, hi in ((int(lo), int(hi)) for lo, hi in cited):
            if any(lo <= n <= hi for n in utd_lines):
                break
        else:
            self.fail(f"none of the cited ranges {cited} contains a UTD loop line "
                      f"{sorted(utd_lines)}; the citation rotted with the edit that moved the loop")

    def test_the_second_draw_is_two_worlds_at_two_seeds_and_the_ordering_holds(self):
        """9 of 10 cells is a recomputed count, not a remembered one, and so is every figure."""
        draws = json.load(open(os.path.join(ROOT, "benchmarks",
                                            "physics_presets_screen5m_draws.json"), encoding="utf-8"))
        d7, d8 = draws["draws"]["seed7"], draws["draws"]["seed8"]
        m = self.sentence(r"ahead at \*\*(\d+) of the (\d+)\*\* world-by-budget cells, at 5M with "
                          r"\*\*(-?[\d,.]+)\*\* against the published world's \*\*(-?[\d,.]+)\*\*, "
                          r"reaching the target in (\d+) of (\d+) episodes against (\d+) of (\d+)",
                          "the replicated ordering")
        self.check("cells fast leads", m.group(1), draws["fast_ahead_cells_of_10"], places=0)
        self.check("cells total", m.group(2), 2 * 5, places=0)
        self.check("seed 8 means at 5M", [m.group(3), m.group(4)],
                   [d8["arms"]["fast"]["by_budget"]["s5000000"]["mean"],
                    d8["arms"]["v9"]["by_budget"]["s5000000"]["mean"]], places=2)
        reach = {p: [dr["arms"][p]["by_budget"]["s5000000"]["reached_target_pct"]
                     for dr in (d7, d8)] for p in ("v9", "fast")}
        self.check("seed 8 reaches per arm", m.groups()[4:8],
                   [reach["fast"][1] / 5.0, 20, reach["v9"][1] / 5.0, 20], places=0)
        m = self.sentence(r"The one cell where `v9` leads \(seed (\d+) at (\d+)M, ([\d,.]+) against "
                          r"([\d,.]+)\)", "the single crossing")
        self.check("crossing seed", m.group(1), 7, places=0)
        budget = f"s{int(m.group(2))}000000"
        self.check("crossing means", [m.group(3), m.group(4)],
                   [d7["arms"]["v9"]["by_budget"][budget]["mean"],
                    d7["arms"]["fast"]["by_budget"][budget]["mean"]], places=2)
        self.assertLess(d7["arms"]["v9"]["by_budget"][budget]["mean"],
                        d7["arms"]["v9"]["by_budget"]["s5000000"]["mean"],
                        "the crossing cell is no longer on the way up")

    def test_the_level_swing_inside_one_world_rivals_the_gap_between_worlds(self):
        draws = json.load(open(os.path.join(ROOT, "benchmarks",
                                            "physics_presets_screen5m_draws.json"), encoding="utf-8"))
        d7, d8 = draws["draws"]["seed7"], draws["draws"]["seed8"]
        at5 = lambda dr, p: dr["arms"][p]["by_budget"]["s5000000"]
        m = self.sentence(r"produced \*\*([\d,.]+)\*\* and ([\d.]+)% reach at one seed and "
                          r"\*\*(-?[\d,.]+)\*\* and ([\d.]+)% at the other - a swing of "
                          r"\*\*([\d,.]+)\*\* inside one world, while the cheap one swung "
                          r"\*\*([\d,.]+)\*\*\. For scale, the distance between the two worlds in the "
                          r"first draw was ([\d,.]+)", "the within-world swing")
        self.check("v9 both draws", [m.group(1), m.group(3)],
                   [at5(d7, "v9")["mean"], at5(d8, "v9")["mean"]], places=2)
        self.check("v9 reach both draws", [m.group(2), m.group(4)],
                   [at5(d7, "v9")["reached_target_pct"], at5(d8, "v9")["reached_target_pct"]],
                   places=1)
        swing = draws["mean_swing_between_draws_at_5m"]
        self.check("swings", [m.group(5), m.group(6)], [swing["v9"], swing["fast"]], places=2)
        gap = abs(at5(d7, "fast")["mean"] - at5(d7, "v9")["mean"])
        self.check("first-draw gap", m.group(7), gap, places=2)
        # The sentence's comparison, restated as an arithmetic check rather than prose.
        self.assertLess(abs(float(m.group(5).replace(",", "")) - gap), 0.01 * gap + 1,
                        "the published world's swing is no longer on the order of the gap")
        self.assertGreater(float(m.group(6).replace(",", "")), gap,
                           "the cheap world's swing is no longer larger than the gap")

    def test_the_two_draws_agree_on_the_speedup_and_disagree_on_everything_else(self):
        draws = json.load(open(os.path.join(ROOT, "benchmarks",
                                            "physics_presets_screen5m_draws.json"), encoding="utf-8"))
        d7, d8 = draws["draws"]["seed7"], draws["draws"]["seed8"]
        m = self.sentence(r"\*\*([\d.]+)x\*\* in the first window and \*\*([\d.]+)x\*\* in this one\s*"
                          r"\(([\d,]+) s against ([\d,]+) s - ([\d,.]+) s per 1M against ([\d,.]+) s"
                          r"\), and the scorer again felt the physics:\s*\*\*(\d+) s against (\d+) s\*\*",
                          "the two windows' speedups")
        self.check("speedups", [m.group(1), m.group(2)],
                   [d7["end_to_end_speedup_fast_vs_v9"], d8["end_to_end_speedup_fast_vs_v9"]],
                   places=3)
        self.check("seed 8 wall clocks", [m.group(3), m.group(4)],
                   [d8["arms"]["v9"]["wall_clock_seconds"], d8["arms"]["fast"]["wall_clock_seconds"]],
                   places=0)
        self.check("seed 8 per 1M", [m.group(5), m.group(6)],
                   [d8["arms"]["v9"]["seconds_per_1m"], d8["arms"]["fast"]["seconds_per_1m"]],
                   places=1)
        self.check("seed 8 scorer clocks", [m.group(7), m.group(8)],
                   [d8["arms"]["v9"]["scoring_seconds"], d8["arms"]["fast"]["scoring_seconds"]],
                   places=0)
        for dr in (d7, d8):
            for preset, arm in dr["arms"].items():
                self.assertEqual(arm["device"], "cuda")
                self.assertEqual(arm["torch_threads"], 24)
            self.assertIn("num_envs=8", draws["protocol"])
        # Both draws scored each arm in the world it trained in, or the ordering would be a
        # bookkeeping artefact rather than a behavioural one.
        for dr in (d7, d8):
            self.assertEqual(dr["arms"]["v9"]["scored_in_version"],
                             "standup_balance_walk_curriculum_v9")
            self.assertEqual(dr["arms"]["fast"]["scored_in_version"],
                             "standup_balance_walk_curriculum_v9_fast")

    def test_the_three_arms_share_one_reward_function_so_the_ordering_means_something(self):
        triple = json.load(open(os.path.join(ROOT, "benchmarks",
                                             "physics_presets_sac1m_triple.json"), encoding="utf-8"))
        for preset in ("v9", "euler", "fast"):
            scored = triple["arms"][preset]["scored_in_version"]
            self.assertEqual(scored, ("standup_balance_walk_curriculum_v9"
                                      + ("" if preset == "v9" else f"_{preset}")),
                             f"{preset} was not scored in the world it trained in")
        # The paragraph's claim is that these three are comparable in return because they share the
        # reward function - which is only true while all three record the same task speed.
        speeds = set()
        for preset in ("v9", "euler", "fast"):
            art = json.load(open(os.path.join(ROOT, "benchmarks",
                                              f"physics_presets_sac1m_{preset}.json"),
                                 encoding="utf-8"))
            model = art["models"][f"sac1m_{preset}"]
            speeds.add(model["reward_kwargs"]["target_forward_velocity"])
            self.assertEqual(model["reward_kwargs"],
                             art["models"][f"sac1m_{preset}"]["reward_kwargs"])
        self.assertEqual(speeds, {1.2},
                         f"the arms no longer share a task speed ({speeds}), so the returns are not "
                         "comparable across them and the paragraph has to be rewritten")

    def test_the_preset_refuses_to_be_scored_in_another_world(self):
        with open(os.path.join(ROOT, "eval_phase1.py"), encoding="utf-8") as handle:
            src = handle.read()
        self.assertIn("aliases an older env revision that has no --physics-preset", src,
                      "the refusal that the block describes is gone")
        self.assertIn('"physics_preset": args.physics_preset', src,
                      "the artifact stopped naming the preset it scored in")


    def test_the_5m_pair_agrees_on_the_question_and_not_on_the_level(self):
        arms = self.screens5m["arms"]
        v9, fast = arms["v9"]["by_budget"], arms["fast"]["by_budget"]
        m = self.sentence(r"each reaches the target in (\d+) of (\d+) episodes at (\d+)M",
                          "the 5M agreement")
        reached, total, budget = int(m.group(1)), int(m.group(2)), int(m.group(3))
        self.assertEqual(budget, 5)
        for name, arm in (("v9", v9), ("fast", fast)):
            pct = arm[f"s{budget}000000"]["reached_target_pct"]
            self.assertEqual(reached, round(pct * total / 100.0),
                             f"{name} at 5M: artifact gives {pct}% of {total} episodes, the README "
                             f"says {reached}")
        self.assertEqual([v9["s5000000"]["reached_target_pct"],
                          fast["s5000000"]["reached_target_pct"]], [10.0, 10.0],
                         "the two worlds no longer reach at the same rate at 5M")
        m = self.sentence(r"the `fast` arm one checkpoint earlier \(([\d.]+)% at (\d+)M\)",
                          "the earlier reach in fast")
        self.check("fast 4M reach", m.group(1), fast["s4000000"]["reached_target_pct"],
                   places=1)
        self.assertEqual(int(m.group(2)), 4)
        self.assertEqual(v9["s4000000"]["reached_target_pct"], 0.0,
                         "the published arm now also reaches at 4M, so 'first reach there and the "
                         "fast arm one earlier' is wrong")
        m = self.sentence(r"\*\*(\d+) h (\d+) min (\d+) s in `v9` against (\d+) h (\d+) min "
                          r"(\d+) s in `fast` - ([\d.]+)x\*\*", "the 5M wall clock")
        typed_v9 = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        typed_fast = int(m.group(4)) * 3600 + int(m.group(5)) * 60 + int(m.group(6))
        self.check("5M wall clocks", [typed_v9, typed_fast],
                   [arms["v9"]["wall_clock_seconds"], arms["fast"]["wall_clock_seconds"]], places=0)
        self.check("5M speedup", m.group(7),
                   self.screens5m["end_to_end_speedup_fast_vs_v9"], places=3)

    def test_the_window_caveat_names_the_two_v9_rates(self):
        m = self.sentence(r"ran at ([\d,\.]+) s\s*per 1M where its sibling at 1M ran at ([\d,]+) s "
                          r"per 1M", "the per-window caveat")
        self.check("5M v9 per 1M", m.group(1).replace(",", ""),
                   self.screens5m["arms"]["v9"]["seconds_per_1m"], places=1)
        self.check("1M v9 per 1M", m.group(2),
                   self.screens["arms"]["v9"]["wall_clock_seconds"], places=0)

    def test_the_level_difference_is_named_as_one_draw(self):
        arms = self.screens5m["arms"]
        m = self.sentence(r"closest\s*approach ([\d.]+) m against ([\d.]+) m at 5M, falls ([\d.]+) "
                          r"against ([\d.]+) per episode", "the level difference")
        self.check("level cells", [m.group(1), m.group(2), m.group(3), m.group(4)],
                   [arms["fast"]["by_budget"]["s5000000"]["mean_min_target_distance"],
                    arms["v9"]["by_budget"]["s5000000"]["mean_min_target_distance"],
                    arms["fast"]["by_budget"]["s5000000"]["falls_per_episode"],
                    arms["v9"]["by_budget"]["s5000000"]["falls_per_episode"]], places=2)


class TestReadmeUtdDoseCells(ReadmeGate, unittest.TestCase):
    """The `--utd-ratio` pair: what a gradient update costs, and what the score column refuses to say.

    Every cell is recomputed from `benchmarks/utd_dose_pair.json`, including the 56.9% share, which is
    *solved* from the two wall clocks rather than stored - the artifact keeps the equations and this
    class keeps the arithmetic honest, so a reworded paragraph cannot survive a changed measurement.
    """

    PAIR = os.path.join(ROOT, "benchmarks", "utd_dose_pair.json")
    DRAWS = os.path.join(ROOT, "benchmarks", "physics_presets_screen5m_draws.json")
    SCREENS5M = os.path.join(ROOT, "benchmarks", "physics_presets_screen5m_paired.json")
    START = "**The `num_envs=8` in those screens is not only a plumbing choice"
    END = "And the caveat that limits what a screen can ask."

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.pair, self.draws, self.screens5m = self.read_artifacts(
            self.PAIR, self.DRAWS, self.SCREENS5M)
        self.bad = []
        self.what = "UTD dose"

    def test_the_two_clocks_in_one_window_set_the_price_of_the_update(self):
        arms = self.pair["arms"]
        m = self.sentence(r"ratio 1 then ratio 4 back to back in one window - \*\*([\d,]+) s against "
                          r"([\d,]+) s, which is ([\d.]+)x and\s*not 4x\*\*", "the two wall clocks")
        self.check("ratio 1 clock", m.group(1), arms["n32_r1"]["wall_clock_seconds"], places=0)
        self.check("ratio 4 clock", m.group(2), arms["n32_r4"]["wall_clock_seconds"], places=0)
        self.check("cost ratio", m.group(3), self.pair["ratio_4_over_ratio_1_seconds"], places=3)
        self.assertEqual(self.pair["ratio_4_over_ratio_1_seconds"],
                         round(arms["n32_r4"]["wall_clock_seconds"]
                               / arms["n32_r1"]["wall_clock_seconds"], 3))

    def test_the_update_share_is_derived_from_the_pair_and_not_remembered(self):
        arms, derived = self.pair["arms"], self.pair["derived_from_the_two_measurements"]
        m = self.sentence(r"puts \*\*([\d,]+) s in the gradient step and ([\d,]+) s in\s*"
                          r"collection: ([\d.]+)%\*\*", "the decomposition")
        self.check("update and collection seconds", [m.group(1), m.group(2)],
                   [derived["update_seconds_at_ratio_1"], derived["collection_seconds"]], places=1)
        self.check("update share", m.group(3), derived["update_share_of_ratio_1_clock_pct"], places=1)
        # The two equations the artifact names have to close against both measured clocks.
        u, c = (derived["update_seconds_at_ratio_1"], derived["collection_seconds"])
        self.assertAlmostEqual(c + u, arms["n32_r1"]["wall_clock_seconds"], delta=1)
        self.assertAlmostEqual(c + 4 * u, arms["n32_r4"]["wall_clock_seconds"], delta=1)
        self.assertLess(4 * u / (c + 4 * u), 1, "the ratio-4 run is not update-dominated in the way "
                                               "the sentence says")

    def test_the_dose_matched_cross_window_figure_carries_its_caveat(self):
        cross = self.pair["dose_matched_against_the_committed_8_env_arm"]
        m = self.sentence(r"cost \*\*([\d,]+) s where that arm took ([\d,]+) s "
                          r"\(([\d.]+)x\)\*\*", "the dose-matched comparison")
        self.check("n32 ratio 4", m.group(1), self.pair["arms"]["n32_r4"]["wall_clock_seconds"],
                   places=0)
        self.check("n8 ratio 1", m.group(2), cross["n8_ratio1_seconds"], places=0)
        self.check("speedup", m.group(3), cross["speedup_of_n32_ratio4"], places=3)
        # The committed 8,992 s the paragraph reuses has to still be the number the 5M pair stores.
        self.assertEqual(cross["n8_ratio1_seconds"],
                         self.screens5m["arms"]["v9"]["wall_clock_seconds"],
                         "the 8-env arm quoted here is no longer the one the 5M pair recorded")
        m = self.sentence(r"ran at ([\d,.]+) s per 1M in one of them and ([\d,.]+) s per 1M in the "
                          r"other", "the per-window spread of the same world")
        self.check("v9 per 1M, first window", m.group(1),
                   self.draws["draws"]["seed7"]["arms"]["v9"]["seconds_per_1m"], places=1)
        self.check("v9 per 1M, second window", m.group(2),
                   self.draws["draws"]["seed8"]["arms"]["v9"]["seconds_per_1m"], places=1)

    def test_draw_one_ranked_the_two_columns_in_opposite_directions(self):
        """Not a prose gate: the paragraph this checked was replaced by the two-draw version.

        The claim stays here as a data check because the new paragraph is built on it - draw 1 is the
        reason two draws were bought, and the opposite ordering of return and reach in that draw is
        what the second draw then had to reproduce.
        """
        arms = self.pair["arms"]
        at5 = {name: arm["by_budget"]["s5000000"] for name, arm in arms.items()}
        self.assertGreater(at5["n32_r1"]["mean"], at5["n32_r4"]["mean"],
                           "draw 1 no longer has the under-dosed arm ahead on return")
        self.assertLess(at5["n32_r1"]["reached_target_pct"], at5["n32_r4"]["reached_target_pct"],
                        "draw 1 no longer has the dose-matched arm ahead on reaching")
        upper = {name: [arm["by_budget"][f"s{m}000000"]["reached_target_pct"] for m in (3, 4, 5)]
                 for name, arm in arms.items()}
        self.assertEqual(upper["n32_r1"], [5.0, 15.0, 5.0])
        self.assertEqual(upper["n32_r4"], [25.0, 10.0, 20.0])

    def test_the_dose_of_each_arm_is_what_the_protocol_claims(self):
        arms = self.pair["arms"]
        for name, ratio in (("n32_r1", 1), ("n32_r4", 4)):
            arm = arms[name]
            self.assertEqual(arm["num_envs"], 32)
            self.assertEqual(arm["utd_ratio"], ratio)
            self.assertEqual(arm["updates_per_environment_step"], round(ratio / 32.0, 4),
                             f"{name}: the dose the artifact records is not ratio/num_envs")
            self.assertEqual(arm["seed"], 7)
            self.assertEqual(arm["device"], "cuda")
            self.assertEqual(arm["torch_threads"], 24)
            self.assertEqual(arm["scored_in_version"], "standup_balance_walk_curriculum_v9")
        # Ratio 4 at 32 envs is the same dose the committed 8-env screens ran at - that is the whole
        # reason the pair is comparable to them, and it is arithmetic on two recorded configs.
        self.assertEqual(arms["n32_r4"]["updates_per_environment_step"], round(1 / 8.0, 4))
        self.assertIn("num_envs=32", self.pair["protocol"])
        self.assertIn("--utd-ratio 1 and one at", self.pair["protocol"])


class TestReadmeUtdDrawReplicationCells(ReadmeGate, unittest.TestCase):
    """The dose pair drawn at two seeds: every cell against benchmarks/utd_dose_draws.json.

    This is the paragraph that replaced a "one draw each, so directional" hedge, so its gate is
    deliberately about the two draws disagreeing or agreeing: the replication figure, the cell count,
    and the arithmetic that turns two per-seed lists into the averages the prose quotes.
    """

    DRAWS = os.path.join(ROOT, "benchmarks", "utd_dose_draws.json")
    PAIR = os.path.join(ROOT, "benchmarks", "utd_dose_pair.json")
    START = "**The pair drawn a second time at seed 8 reproduces"
    END = "And the caveat that limits what a screen can ask."

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.draws, self.pair = self.read_artifacts(self.DRAWS, self.PAIR)
        self.bad = []
        self.what = "UTD dose draws"

    def test_the_cost_arithmetic_replicates_across_two_independent_windows(self):
        d8, d7 = self.draws["draws"]["seed8"], self.draws["draws"]["seed7"]
        m = self.sentence(r": ([\d,]+) s against ([\d,]+) s, which is \*\*([\d.]+)x\*\*, "
                          r"0\.3% from the first draw's (?:\*\*)?([\d.]+)x(?:\*\*)?, and the same two "
                          r"equations put \*\*([\d.]+)%\*\*", "the second draw's cost")
        self.check("second draw clocks", [m.group(1), m.group(2)],
                   [d8["arms"]["1"]["wall_clock_seconds"], d8["arms"]["4"]["wall_clock_seconds"]],
                   places=0)
        self.check("cost ratios", [m.group(3), m.group(4)],
                   [self.draws["cost_ratios_per_draw"]["seed8"],
                    self.draws["cost_ratios_per_draw"]["seed7"]], places=3)
        self.check("update share, draw 2", m.group(5),
                   self.draws["update_share_of_ratio_1_clock_pct"]["seed8"], places=1)
        self.assertEqual(self.draws["cost_ratios_per_draw"]["seed8"],
                         round(d8["arms"]["4"]["wall_clock_seconds"]
                               / d8["arms"]["1"]["wall_clock_seconds"], 3))
        self.assertEqual(self.draws["reproduced_to_pct"], 0.3,
                         "the paragraph quotes 0.3%; the two draws moved apart by more than that")
        # Draw 1's figures, re-checked from the other artifact so the sentence cannot half-rot.
        self.assertEqual(self.draws["cost_ratios_per_draw"]["seed7"],
                         self.pair["ratio_4_over_ratio_1_seconds"])
        self.assertEqual(self.draws["update_share_of_ratio_1_clock_pct"]["seed7"],
                         self.pair["derived_from_the_two_measurements"]
                         ["update_share_of_ratio_1_clock_pct"])

    def test_the_reach_advantage_and_the_cell_count_come_from_the_same_grid(self):
        cells = self.draws["dose4_cells"]
        m = self.sentence(r"reaches the target more often in \*\*(\d+) of the (\d+) cells that are "
                          r"not ties\*\*", "the cell count")
        self.check("cells ahead", m.group(1), cells["ahead"], places=0)
        self.check("non-tie cells", m.group(2), cells["ahead"] + cells["behind"], places=0)
        self.assertEqual(cells["ahead"] + cells["behind"] + cells["tied"], 10,
                         "two draws over five budgets is ten cells")
        at5 = self.draws["at_5m"]
        m = self.sentence(r"\*\*(\d+) of (\d+) episodes against (\d+) of (\d+)\*\* "
                          r"\(([\d.]+)% and ([\d.]+)% against ([\d.]+)% and ([\d.]+)%\)",
                          "the 5M reach columns")
        self.check("episodes at 5M", [m.group(1), m.group(3)],
                   [at5["4"]["reached_episodes_of_40"], at5["1"]["reached_episodes_of_40"]], places=0)
        self.check("episodes denominators", [m.group(2), m.group(4)], [40, 40], places=0)
        self.check("percentages", [m.group(5), m.group(6), m.group(7), m.group(8)],
                   at5["4"]["reaches_pct"] + at5["1"]["reaches_pct"], places=1)
        self.assertEqual(int(m.group(1)) + int(m.group(3)),
                         sum(p * 40 // 100 for p in at5["4"]["reaches_pct"]
                             + at5["1"]["reaches_pct"]))
        upper = self.draws["mean_over_upper_budgets_pct"]
        m = self.sentence(r"reaches in \*\*([\d.]+)%\*\* of episodes against \*\*([\d.]+)%\*\*",
                          "the upper-budget average")
        self.check("upper budget means", [m.group(1), m.group(2)],
                   [upper["4"], upper["1"]], places=2)

    def test_the_return_column_still_points_the_other_way_after_two_draws(self):
        at5 = self.draws["at_5m"]
        avg = {r: round(sum(at5[r]["means"]) / 2.0, 1) for r in ("1", "4")}
        m = self.sentence(r"on average\s*\*\*([\d,.]+)\*\* against \*\*([\d,.]+)\*\* across the two "
                          r"5M draws", "the averaged mean returns")
        self.check("averaged means", [m.group(1), m.group(2)], [avg["1"], avg["4"]], places=1)
        self.assertGreater(avg["1"], avg["4"],
                           "the under-dosed arm no longer averages the higher return, so the "
                           "opposite-ranking claim has to be rewritten rather than re-pointed")
        self.assertLess(at5["4"]["reached_episodes_of_40"] - at5["1"]["reached_episodes_of_40"], 12,
                        "the reach gap is quoted as ten episodes out of eighty")

    def test_both_draws_are_the_same_recipe_and_one_world(self):
        for tag, draw in self.draws["draws"].items():
            for ratio, arm in draw["arms"].items():
                self.assertEqual(arm["num_envs"], 32, f"{tag}/{ratio}")
                self.assertEqual(arm["device"], "cuda")
                self.assertEqual(arm["torch_threads"], 24)
                self.assertEqual(arm["scored_in_version"], "standup_balance_walk_curriculum_v9")
                self.assertEqual(arm["physics_preset"], "v9")
                self.assertEqual(arm["updates_per_environment_step"],
                             round(int(ratio) / 32.0, 4))
            self.assertEqual(draw["arms"]["1"]["seed"], draw["arms"]["4"]["seed"],
                             f"{tag}: the two arms of a draw are not the same seed")
        self.assertNotEqual(self.draws["draws"]["seed7"]["arms"]["1"]["seed"],
                            self.draws["draws"]["seed8"]["arms"]["1"]["seed"])
        self.assertIn("num_envs=32", self.draws["protocol"])
        self.assertIn("--utd-ratio 1 and one at 4", self.draws["protocol"])


class TestReadmeEulerDrawsCells(ReadmeGate, unittest.TestCase):
    """`euler` at 5M, two draws, against benchmarks/physics_presets_screen5m_euler_draws.json.

    The paragraph's whole point is that draw 1 was misread, so its gate has to fail if someone
    re-quotes draw 1 as a property of the world: the two draws are checked against each other, not
    just against the numbers in the prose.
    """

    EULER = os.path.join(ROOT, "benchmarks", "physics_presets_screen5m_euler_draws.json")
    TRIPLE = os.path.join(ROOT, "benchmarks", "physics_presets_sac1m_triple.json")
    START = "**`euler` has now been trained at the budget"
    END = "### \U0001f9ee The learner side"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.art, self.triple = self.read_artifacts(self.EULER, self.TRIPLE)
        self.bad = []
        self.what = "euler 5M draws"

    def test_draw_one_was_a_flat_line_and_draw_two_was_not(self):
        d7, d8 = self.art["draws"]["seed7"], self.art["draws"]["seed8"]
        m = self.sentence(r"mean ([\d,]+\.[\d]+) falling to \*\*(-?[\d,.]+)\*\* across its five "
                          r"checkpoints,\s*forward speed ~0, ([\d.]+) falls per episode, the target "
                          r"reached in \*\*(\d+) of (\d+)\*\* scored episodes", "draw 1's flat line")
        self.check("draw 1 first and last mean", [m.group(1), m.group(2)],
                   [d7["by_budget"]["s1000000"]["mean"], d7["by_budget"]["s5000000"]["mean"]],
                   places=2)
        self.check("draw 1 falls", m.group(3), d7["by_budget"]["s5000000"]["falls_per_episode"],
                   places=2)
        reached7 = sum(v["reached_target_pct"] for v in d7["by_budget"].values())
        self.check("draw 1 reaches", [m.group(4), m.group(5)],
                   [int(reached7 * 20 / 100), 5 * 20], places=0)
        m = self.sentence(r"it climbs to \*\*([\d,]+\.[\d]+)\*\* at (\d)M and reaches the target in "
                          r"\*\*([\d.]+)%\*\*", "draw 2's climb")
        self.check("draw 2 4M mean", m.group(1), d8["by_budget"]["s4000000"]["mean"], places=2)
        self.check("draw 2 budget", m.group(2), 4, places=0)
        self.check("draw 2 5M reach", m.group(3),
                   d8["by_budget"]["s5000000"]["reached_target_pct"], places=1)
        # The claim "the flat line was the run" has to stay true of the data, not just of the prose.
        self.assertGreater(d8["by_budget"]["s5000000"]["mean"], 5000,
                           "draw 2 no longer rescues draw 1, so the paragraph must be rewritten")

    def test_the_cost_replicates_inside_each_window(self):
        d7, d8 = self.art["draws"]["seed7"], self.art["draws"]["seed8"]
        m = self.sentence(r"\*\*([\d,]+) s against the published world's ([\d,]+) s at seed (\d+)\s*"
                          r"\(([\d.]+)x\)\*\* and \*\*([\d,]+) s against ([\d,]+) s at seed (\d+) "
                          r"\(([\d.]+)x\)\*\*", "the two cost ratios")
        self.check("seed 7 clocks and ratio", [m.group(1), m.group(2), m.group(4)],
                   [d7["wall_clock_seconds"], d7["v9_same_draw_seconds"],
                    d7["speedup_euler_over_v9_in_this_window"]], places=1)
        self.check("seed 8 clocks and ratio", [m.group(5), m.group(6), m.group(8)],
                   [d8["wall_clock_seconds"], d8["v9_same_draw_seconds"],
                    d8["speedup_euler_over_v9_in_this_window"]], places=1)
        self.assertEqual(m.group(3), "7")
        self.assertEqual(m.group(7), "8")
        m = self.sentence(r"bracketing the\s*\*\*([\d.]+)x\*\* the 1M triple measured", "the 1M ratio")
        self.check("1M triple euler ratio", m.group(1),
                   self.triple["train_seconds_over_v9"]["euler"], places=3)
        lo = min(d7["speedup_euler_over_v9_in_this_window"],
                 d8["speedup_euler_over_v9_in_this_window"])
        hi = max(d7["speedup_euler_over_v9_in_this_window"],
                 d8["speedup_euler_over_v9_in_this_window"])
        self.assertLessEqual(lo, self.triple["train_seconds_over_v9"]["euler"],
                             "the 1M ratio no longer sits inside the two 5M windows")
        self.assertGreaterEqual(hi, self.triple["train_seconds_over_v9"]["euler"])

    def test_euler_tracks_v9_on_the_criterion_and_fast_does_not(self):
        reach = self.art["reached_target_pct_at_5m"]
        m = self.sentence(r"`euler`'s two-draw reach record at 5M is \*\*(\d+)% then (\d+)%\*\* where "
                          r"`v9`'s is \*\*(\d+)% then\s*(\d+)%\*\*", "the two reach records")
        self.check("euler record", [m.group(1), m.group(2)], reach["euler"], places=0)
        self.check("v9 record", [m.group(3), m.group(4)], reach["v9"], places=0)
        self.assertEqual(sorted(reach["euler"]), sorted(reach["v9"]),
                         "the two worlds' reach records are no longer the same multiset, so the "
                         "'indistinguishable' sentence has to be replaced")
        fast = json.load(open(os.path.join(ROOT, "benchmarks",
                                           "physics_presets_screen5m_draws.json"),
                              encoding="utf-8"))
        fast5 = [fast["draws"][t]["arms"]["fast"]["by_budget"]["s5000000"]["reached_target_pct"]
                 for t in fast["draws"]]
        v9_5 = [fast["draws"][t]["arms"]["v9"]["by_budget"]["s5000000"]["mean"] for t in fast["draws"]]
        fast_5 = [fast["draws"][t]["arms"]["fast"]["by_budget"]["s5000000"]["mean"]
                  for t in fast["draws"]]
        self.assertTrue(all(f > v for f, v in zip(fast_5, v9_5)),
                        "fast no longer leads both draws on mean return at 5M")
        self.assertGreater(min(fast_5), max(self.art["mean_at_5m"]["euler"]),
                           "euler's best draw now outruns fast, so 'the divergence lives in fast' "
                           "needs re-reading")

    def test_the_instruments_of_both_draws_match_the_published_protocol(self):
        for tag, draw in self.art["draws"].items():
            self.assertEqual(draw["device"], "cuda", tag)
            self.assertEqual(draw["torch_threads"], 24, tag)
            self.assertEqual(draw["physics_preset"], "euler", tag)
            self.assertEqual(draw["scored_in_version"],
                             "standup_balance_walk_curriculum_v9_euler", tag)
            self.assertEqual(draw["seconds_per_1m"], round(draw["wall_clock_seconds"] / 5.0, 1), tag)
            self.assertGreater(draw["scoring_seconds"], 0, tag)
        self.assertIn("num_envs=8", self.art["protocol"])
        self.assertIn("target_forward_velocity=1.2", self.art["protocol"])
        gap = max(self.art["mean_at_5m"]["euler"]) - min(self.art["mean_at_5m"]["euler"])
        m = self.sentence(r"the two `euler` draws differ by more than ([\d,]+) of mean return",
                          "the within-world gap")
        self.assertGreater(gap, float(m.group(1).replace(",", "")),
                           f"the paragraph says more than {m.group(1)}; the draws differ by {gap:.2f}")


class TestReadmeArrivalDefinitionCells(ReadmeGate, unittest.TestCase):
    """The two arrival rules, against benchmarks/reach_upright_arrival.json.

    This is the paragraph that says what the other reach numbers in the README are *not* measuring, so
    it is gated on the strict column and on the fidelity claim: every row's distance-only figure has to
    still equal the committed artifact it was re-scored from, or the two columns are not two readings
    of one measurement.
    """

    ARRIVAL = os.path.join(ROOT, "benchmarks", "reach_upright_arrival.json")
    DOSE = os.path.join(ROOT, "benchmarks", "utd_sac_n32_seed8_r4_5m.json")
    START = "**Every reach number above and below is distance-only"
    END = "**An upright arrival says the body was standing"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.art = self.read_artifacts(self.ARRIVAL)[0]
        self.dose = self.read_artifacts(self.DOSE)[0]
        self.rows = {(r["artifact"].split("/")[-1], r["model"]): r for r in self.art["rows"]}
        self.bad = []
        self.what = "arrival definitions"

    def row(self, artifact, model):
        key = (artifact.split("/")[-1], model)
        self.assertIn(key, self.rows, f"{artifact}:{model} is no longer measured")
        return self.rows[key]

    def test_the_distance_only_column_still_reproduces_what_was_published(self):
        for r in self.art["rows"]:
            art = json.load(open(os.path.join(ROOT, r["artifact"]), encoding="utf-8"))
            self.assertEqual(r["reached_distance_only_pct"],
                             art["models"][r["model"]]["reached_target_pct"],
                             f"{r['label']}: the re-score disagrees with the committed row, so the "
                             "two columns are not two definitions of one measurement")

    def test_the_best_published_arm_arrives_standing_three_times(self):
        m = self.sentence(r"the best arm in the published world for everything published before the "
                          r"reward-override experiment arrives standing in \*\*(\d+) of (\d+) episodes "
                          r"\((\d+)%\)\*\* -\s*SAC at `--utd-ratio 4` seed 7 / (\d)M and at ratio 1 seed "
                          r"8 / (\d)M, whose\s*distance-only rows read ([\d.]+)% and ([\d.]+)%",
                          "the strict best among the pre-override arms")
        # The ceiling the override arm broke, so it is read off the pre-override rows only.
        ceiling = [r for r in self.art["rows"] if r["physics_preset"] == "v9"
                   and not r["label"].startswith("stability 60")]
        best_before = max(ceiling, key=lambda r: r["reached_upright_count"])
        self.check("upright episodes", [m.group(1), m.group(3)],
                   [best_before["reached_upright_count"], best_before["reached_upright_pct"]],
                   places=0)
        self.check("episodes per arm", m.group(2), 20, places=0)
        d4 = self.row("benchmarks/utd_sac_n32_r4_5m.json", "s3000000")
        d1 = self.row("benchmarks/utd_sac_n32_seed8_r1_5m.json", "s4000000")
        self.check("the two arms' loose rows", [m.group(6), m.group(7)],
                   [d4["reached_distance_only_pct"], d1["reached_distance_only_pct"]], places=1)
        self.assertEqual((m.group(4), m.group(5)), ("3", "4"))
        self.assertEqual(d4["reached_upright_count"], d1["reached_upright_count"],
                         "the paragraph names two arms tied on the strict column")

    def test_the_other_quoted_arms_are_the_measured_ones(self):
        m = self.sentence(r"the 40M from-scratch curve's ([\d.]+)% at 5M is \*\*([\d.]+)%\*\* upright",
                          "the from-scratch pair")
        scratch = self.row("benchmarks/target_learning_curve_from_scratch_v9.json", "s5000000")
        self.check("from scratch", [m.group(1), m.group(2)],
                   [scratch["reached_distance_only_pct"], scratch["reached_upright_pct"]], places=1)
        words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        m = self.sentence(r"PPO at 10M is the extreme: ([\d.]+)% and\s*([\d.]+)% by distance, "
                          r"\*\*([\d.]+)% and ([\d.]+)%\*\* standing, because (one|two|three|four|"
                          r"five) of its (one|two|three|four|five) arrivals are collapses",
                          "the PPO pair")
        p8 = self.row("benchmarks/ppo_learning_curve_10m_v9_seed8.json", "s7012352")
        p7 = self.row("benchmarks/ppo_learning_curve_10m_v9.json", "s5046272")
        self.check("ppo loose", [m.group(1), m.group(2)],
                   [p8["reached_distance_only_pct"], p7["reached_distance_only_pct"]], places=1)
        self.check("ppo upright", [m.group(3), m.group(4)],
                   [p8["reached_upright_pct"], p7["reached_upright_pct"]], places=1)
        self.check("ppo collapses", [words[m.group(5)], words[m.group(6)]],
                   [p8["collapse_on_target_only"],
                    int(p8["reached_distance_only_pct"] * 20 / 100)], places=0)
        m = self.sentence(r"the `fast` world at 5M seed 8 \(([\d.]+)% by both rules, (\d+) collapses\)",
                          "the clean fast arm")
        fast = self.row("benchmarks/physics_presets_screen5m_seed8_fast.json", "s5000000")
        self.check("fast arm", [m.group(1), m.group(2)],
                   [fast["reached_upright_pct"], fast["collapse_on_target_only"]], places=0)
        self.assertEqual(fast["reached_upright_pct"], fast["reached_distance_only_pct"],
                         "the fast arm is no longer the one where every arrival was clean")

    def test_the_near_miss_that_motivated_the_video_is_the_measured_one(self):
        tele = self.dose["per_episode"]["s4000000__telemetry"]
        misses = [t["min_target_distance"] for t in tele
                  if not t["reached_target"] and t["min_target_distance"] <= 0.5]
        self.assertEqual(misses, [0.468], "the 1.8 cm near miss the paragraph cites is gone")
        reached = [t["min_target_distance"] for t in tele if t["reached_target"]]
        self.assertIn(0.299, reached, "the clip the paragraph quotes as a collapse arrival is gone")
        # The sentences that quote the clips (the 1.8 cm miss, the 0.299 m collapse) are gated in
        # TestReadmeApproachMechanismCells, which has the per-step traces they are read from.


class TestReadmeApproachMechanismCells(ReadmeGate, unittest.TestCase):
    """The step-level "walking or toppling?" block, against benchmarks/approach_mechanism.json.

    This paragraph exists to take a claim back, so its figures carry more weight than a new
    measurement would: they are the reason "it walks" was replaced. Every one of them is read out of the
    pooled per-step traces the script writes, the clip sentences against the records of the four
    episodes that were rendered into video, and the reach counts against the source artifacts the rows
    were re-scored from - so a row that no longer reproduces its published telemetry fails here too, not
    only at write time.
    """

    MECH = os.path.join(ROOT, "benchmarks", "approach_mechanism.json")
    DOSE = os.path.join(ROOT, "benchmarks", "utd_sac_n32_seed8_r4_5m.json")
    START = "**Every reach number above and below is distance-only"
    END = "The standing rung splits the other way"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.art, self.dose = self.read_artifacts(self.MECH, self.DOSE)
        self.pooled = self.art["pooled_v9_reach_episodes"]
        self.clips = {c["seed"]: c for c in self.art["rendered_clips"]}
        self.v9 = [r for r in self.art["rows"] if r["physics_preset"] == "v9"]
        self.bad = []
        self.what = "approach mechanism"

    def test_the_trace_covers_every_quoted_checkpoint(self):
        m = self.sentence(r"which replays all (\d+) scored episodes of the (\d+) quoted checkpoints",
                          "the coverage claim")
        episodes = sum(r["episodes"] for r in self.art["rows"])
        self.check("episodes traced", [m.group(1), m.group(2)], [episodes, len(self.art["rows"])],
                   places=0)
        self.assertEqual(episodes, 20 * len(self.art["rows"]),
                         "the rows no longer carry the same number of episodes")

    def test_each_row_still_reproduces_the_reach_count_it_was_scored_from(self):
        for r in self.art["rows"]:
            art = json.load(open(os.path.join(ROOT, r["artifact"]), encoding="utf-8"))
            published = art["models"][r["model"]]["reached_target_pct"] * r["episodes"] / 100.0
            self.assertEqual(r["reach_episodes"]["n"], round(published),
                             f"{r['label']}: the traced reach count moved away from the published row")

    def test_the_reach_episodes_split_of_ground_between_standing_and_down(self):
        m = self.sentence(r"the (\d+) episodes the distance rule counts as reaches in the published "
                          r"world close \*\*([\d.]+) m at in-band steps against ([\d.]+) m with the "
                          r"torso below the band\*\* \(([\d.]+)% of the ground\)", "the closing split")
        standing, down = self.pooled["closing_while_standing_m"], self.pooled["closing_while_down_m"]
        self.check("reach episodes", m.group(1), self.pooled["episodes"], places=0)
        self.check("metres standing / metres down", [m.group(2), m.group(3)], [standing, down],
                   places=2)
        self.check("share standing", m.group(4), 100.0 * standing / (standing + down), places=1)
        self.check("pooled share field", self.pooled["share_while_standing"],
                   standing / (standing + down), places=3)

    def test_most_closest_approaches_come_after_the_first_fall(self):
        m = self.sentence(r"and \*\*(\d+) of those (\d+) closest approaches come after the episode's "
                          r"first fall\*\*", "the after-fall count")
        self.check("after a fall", [m.group(1), m.group(2)],
                   [self.pooled["closest_approach_after_first_fall"], self.pooled["episodes"]],
                   places=0)

    def test_the_standing_time_is_short_and_broken_into_runs(self):
        m = self.sentence(r"Those episodes spend ([\d.]+)% of their steps in the band but never more "
                          r"than \*\*([\d.]+) s\*\* at a stretch", "the continuity claim")
        self.check("in-band steps", m.group(1), self.pooled["mean_pct_steps_in_band"], places=1)
        self.check("longest run", m.group(2), self.pooled["max_longest_band_run_s"], places=2)
        self.assertLess(self.pooled["max_longest_band_run_s"], 2.0,
                        "a reaching episode now stands for two continuous seconds, so the README's "
                        "'never stands for two continuous seconds' has to be rewritten")
        self.assertEqual(self.pooled["episodes_with_band_run_ge_2s"], 0)

    def test_direction_survives_where_the_gait_does(self):
        m = self.sentence(r"the body moves toward the target at ([\d.]+) to ([\d.]+) m/s depending on "
                          r"the arm, and the best near miss in the repository comes to ([\d.]+) cm "
                          r"inside the radius", "the heading-velocity range")
        # The sentence describes the arms published before the reward override, so the range is read
        # off those rows only - the gait arms are quoted in their own paragraph below.
        rates = [r["reach_episodes"]["mean_heading_velocity_in_band_mps"] for r in self.v9
                 if r["reach_episodes"]["n"] and not r["arm_class"]]
        self.check("min/max in-band heading velocity", [m.group(1), m.group(2)],
                   [min(rates), max(rates)], places=2)
        tele = self.dose["per_episode"]["s4000000__telemetry"]
        miss = [t["min_target_distance"] for t in tele
                if not t["reached_target"] and t["min_target_distance"] <= 0.5]
        self.assertEqual(len(miss), 1, "the near miss the sentence cites is no longer unique")
        self.check("centimetres", m.group(3), round((miss[0] - 0.45) * 100, 1), places=1)

    def test_the_clip_quoted_as_a_collapse_arrival_is_the_measured_episode(self):
        m = self.sentence(r"the clip that .reached. at ([\d.]+) m got there at a step with the torso "
                          r"at ([\d.]+) m, and it ends ([\d.]+) m from the marker, motionless, "
                          r"([\d.]+) s later", "the collapse-arrival clip")
        clip = self.clips.get(25)
        self.assertIsNotNone(clip, "the rendered clip at seed 25 is no longer in the artifact")
        self.assertEqual(clip["rendered_as"], "reached")
        self.check("closest approach", m.group(1), clip["min_target_distance"], places=3)
        self.check("torso height at arrival", m.group(2), clip["min_dist_z"], places=3)
        self.check("distance at the end", m.group(3), clip["final_distance"], places=2)
        self.check("seconds after arrival", m.group(4),
                   clip["seconds_between_arrival_and_end"], places=1)
        self.assertLess(clip["final_root_speed_mps"], 0.01,
                        "the episode is no longer motionless at the end, so 'motionless' is wrong")
        self.assertFalse(clip["reached_target_upright"],
                         "this arrival is now upright, so it is no longer the collapse the paragraph "
                         "quotes")


class TestReadmeGaitArmCells(ReadmeGate, unittest.TestCase):
    """The two gait arms, against the arrival table and the step-level traces they were read from.

    This is the first paragraph in the repository that claims an intervention *worked*, so every figure
    in it is tied to one of the two artifacts, including the negative one. It also gates the claim that
    killed the other candidate arm before it was run: that `stillness_penalty` is dead code in the target
    phase, which is a statement about the environment's source and is checked against the source.
    """

    ARRIVAL = os.path.join(ROOT, "benchmarks", "reach_upright_arrival.json")
    MECH = os.path.join(ROOT, "benchmarks", "approach_mechanism.json")
    START = "**So the experiment was to change the gait"
    END = "The standing rung splits the other way"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.arr, self.mech = self.read_artifacts(self.ARRIVAL, self.MECH)
        self.rows = {(r["artifact"].split("/")[-1], r["model"]): r for r in self.arr["rows"]}
        self.mrows = {(r["artifact"].split("/")[-1], r["model"]): r for r in self.mech["rows"]}
        self.bad = []
        self.what = "gait arms"

    def arrival(self, artifact, model):
        key = (artifact.split("/")[-1], model)
        self.assertIn(key, self.rows, f"{artifact}:{model} is no longer re-scored")
        return self.rows[key]

    def mech_row(self, artifact, model):
        key = (artifact.split("/")[-1], model)
        self.assertIn(key, self.mrows, f"{artifact}:{model} is no longer traced")
        return self.mrows[key]

    def test_the_dead_candidate_arm_is_dead_in_the_source_not_just_in_the_prose(self):
        with open(os.path.join(ROOT, "envs", "walker_ragdoll_env.py"), encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        m = self.sentence(r"`stillness_penalty` is only\s*computed when `task_phase == \"balance\"` "
                          r"\(`envs/walker_ragdoll_env\.py:(\d+)-(\d+)`\)", "the dead-arm citation")
        lo, hi = int(m.group(1)), int(m.group(2))
        window = "\n".join(lines[lo - 1:hi])
        self.assertIn("stillness_penalty = (", window,
                      "the cited range no longer holds the stillness penalty's only assignment")
        self.assertIn('if task_phase == "balance":', window,
                      "the cited range no longer shows the balance guard that makes it dead in target")
        outside = "\n".join(lines[:lo - 1] + lines[hi:])
        self.assertNotIn("stillness_penalty = (", outside,
                         "a second non-zero assignment appeared, so the term is live in another phase "
                         "and the arm is not dead after all")

    def test_arm_a_moves_the_strict_column_for_the_first_time(self):
        m = self.sentence(r"\*\*(\d+) of (\d+) episodes \((\d+)%\) arrive\s*standing\*\* at its 4M "
                          r"checkpoint on seed 7, where (\d+) of 20 had been the ceiling, and its seed 8 "
                          r"best cell lands back on (\d+) of 20", "arm A's arrival counts")
        a7 = self.arrival("benchmarks/stab60_sac_n32_r4_5m_curve.json", "s4000000")
        a8 = self.arrival("benchmarks/stab60_sac_n32_r4_5m_seed8_curve.json", "s4000000")
        self.check("seed 7 upright", [m.group(1), m.group(3)],
                   [a7["reached_upright_count"], a7["reached_upright_pct"]], places=0)
        self.check("episodes", m.group(2), a7["episodes"], places=0)
        self.check("seed 8 upright", m.group(5), a8["reached_upright_count"], places=0)
        pre = [r for r in self.arr["rows"] if r["physics_preset"] == "v9"
               and not r["label"].startswith(("stability 60", "curriculum"))]
        self.check("the ceiling it broke", m.group(4),
                   max(r["reached_upright_count"] for r in pre), places=0)
        self.assertGreater(a7["reached_upright_count"], int(m.group(4)),
                           "arm A no longer beats the pre-override ceiling")
        self.assertEqual((a7["reached_upright_count"], a8["reached_upright_count"]), (4, 3),
                         "the paragraph's replicate story (peak, then back to the ceiling) changed")

    def test_arm_b_is_reported_as_the_negative_it_is(self):
        m = self.sentence(r"(\d+) of 20 upright at its best in-task cell and (\d+) of 20 when the same "
                          r"checkpoints are scored on the published 2-5 m, having spent ([\d.]+)-"
                          r"([\d.]+)% of its steps in the band against the published arms' ([\d.]+)% and"
                          r"\s*falling ([\d.]+) to ([\d.]+) times an episode against arm A's ([\d.]+) to "
                          r"([\d.]+) per episode", "arm B's negative result")
        in_task = [self.arrival(f, k) for f, k in
                   (("benchmarks/tcur_sac_n32_r4_5m_curve.json", "s4000000"),
                    ("benchmarks/tcur_sac_n32_r4_5m_seed8_curve.json", "s2000000"))]
        transfer = [self.arrival(f, k) for f, k in
                    (("benchmarks/tcur_sac_n32_r4_5m_transfer.json", "x4000000"),
                     ("benchmarks/tcur_sac_n32_r4_5m_seed8_transfer.json", "x5000000"))]
        self.check("best in-task / best transfer", [m.group(1), m.group(2)],
                   [max(r["reached_upright_count"] for r in in_task),
                    max(r["reached_upright_count"] for r in transfer)], places=0)
        banded = [self.mech_row(f, k)["mean_pct_steps_in_band"] for f, k in
                  (("benchmarks/tcur_sac_n32_r4_5m_curve.json", "s4000000"),
                   ("benchmarks/tcur_sac_n32_r4_5m_seed8_curve.json", "s2000000"),
                   ("benchmarks/tcur_sac_n32_r4_5m_transfer.json", "x4000000"),
                   ("benchmarks/tcur_sac_n32_r4_5m_seed8_transfer.json", "x5000000"))]
        self.check("curriculum band share", [m.group(3), m.group(4)],
                   [min(banded), max(banded)], places=1)
        self.check("published pool band share", m.group(5),
                   self.mech["pooled_v9_reach_episodes"]["mean_pct_steps_in_band"], places=1)
        falls_b = [self.arrival(f, k) for f, k in
                   (("benchmarks/tcur_sac_n32_r4_5m_curve.json", "s4000000"),
                    ("benchmarks/tcur_sac_n32_r4_5m_seed8_curve.json", "s2000000"),
                    ("benchmarks/tcur_sac_n32_r4_5m_transfer.json", "x4000000"),
                    ("benchmarks/tcur_sac_n32_r4_5m_seed8_transfer.json", "x5000000"))]
        falls_a = [self.arrival(f, k) for f, k in
                   (("benchmarks/stab60_sac_n32_r4_5m_curve.json", "s4000000"),
                    ("benchmarks/stab60_sac_n32_r4_5m_curve.json", "s5000000"),
                    ("benchmarks/stab60_sac_n32_r4_5m_seed8_curve.json", "s4000000"))]
        self.check("falls per episode, arm B", [m.group(6), m.group(7)],
                   [min(r["falls_per_episode"] for r in falls_b),
                    max(r["falls_per_episode"] for r in falls_b)], places=1)
        self.check("falls per episode, arm A", [m.group(8), m.group(9)],
                   [min(r["falls_per_episode"] for r in falls_a),
                    max(r["falls_per_episode"] for r in falls_a)], places=1)

    def test_the_trace_says_arm_a_bought_more_than_a_lucky_arrival_frame(self):
        m = self.sentence(r"the reach\s*episodes close \*\*([\d.]+)% to ([\d.]+)% of their ground inside "
                          r"the band\*\* against the published pool's ([\d.]+)%, they\s*are in the band "
                          r"for ([\d.]+)-([\d.]+)% of their steps against ([\d.]+)%, and the best cell's "
                          r"upright arrivals hold the\s*band for ([\d.]+) s at a stretch",
                          "arm A's step-level split")
        a_rows = [self.mech_row(f, k) for f, k in
                  (("benchmarks/stab60_sac_n32_r4_5m_curve.json", "s4000000"),
                   ("benchmarks/stab60_sac_n32_r4_5m_curve.json", "s5000000"),
                   ("benchmarks/stab60_sac_n32_r4_5m_seed8_curve.json", "s4000000"))]
        shares = [100.0 * r["reach_episodes"]["mean_share_while_standing"] for r in a_rows]
        self.check("share closed in band", [m.group(1), m.group(2)],
                   [min(shares), max(shares)], places=1)
        self.check("published pool share", m.group(3),
                   100.0 * self.mech["pooled_v9_reach_episodes"]["share_while_standing"], places=1)
        banded = [r["mean_pct_steps_in_band"] for r in a_rows]
        self.check("arm A band time", [m.group(4), m.group(5)], [min(banded), max(banded)], places=1)
        self.check("published pool band time", m.group(6),
                   self.mech["pooled_v9_reach_episodes"]["mean_pct_steps_in_band"], places=1)
        self.check("best cell upright hold", m.group(7),
                   a_rows[0]["upright_arrival_episodes"]["mean_longest_band_run_s"], places=2)

    def test_the_gait_still_does_not_last_two_seconds(self):
        m = self.sentence(r"over all\s*\*\*(\d+)\*\* reaching episodes now traced in the published world, "
                          r"the longest continuous stand is still \*\*([\d.]+) s\*\* and none of them "
                          r"reaches two seconds", "the open problem")
        pub, gait = (self.mech["pooled_v9_reach_episodes"],
                     self.mech["pooled_v9_gait_reach_episodes"])
        self.check("reaching episodes", m.group(1),
                   pub["episodes"] + gait["episodes"], places=0)
        longest = max(pub["max_longest_band_run_s"], gait["max_longest_band_run_s"])
        self.check("longest continuous stand", m.group(2), longest, places=2)
        self.assertEqual(pub["episodes_with_band_run_ge_2s"] + gait["episodes_with_band_run_ge_2s"], 0,
                         "an arm now holds the band for two seconds, so the paragraph's open problem "
                         "is solved and has to be rewritten")


class TestReadmeMpcBaselineCells(ReadmeGate, unittest.TestCase):
    """The sampling-MPC baseline against the learned arms, from benchmarks/mpc_target_rows.json.

    Two things are gated here that are easy to get wrong in prose. The first is the arrival/gait split:
    the planner wins the arrival column and loses the gait column, so quoting either one alone turns a
    measurement into a verdict. The second is the reproducibility claim, which is not inherited from the
    learned arms - the constraint solver is not bit-deterministic, so the sentence "drawn a second time
    it is identical" is only true if the two committed cells actually agree, column by column.
    """

    ROWS = os.path.join(ROOT, "benchmarks", "mpc_target_rows.json")
    START = "**A planner with simulator access reaches the marker"
    END = "So the residual-learning question now has a measured target"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.art, = self.read_artifacts(self.ROWS)
        self.small, self.draw2, self.big = self.art["mpc_rows"]
        self.learned = self.art["learned"]
        self.bad = []
        self.what = "MPC baseline"

    def test_the_two_arrival_counts_of_the_small_cell_are_the_measured_ones(self):
        m = self.sentence(r"reaches in \*\*(\d+) of (\d+)\*\* and arrives standing in \*\*(\d+) of "
                          r"(\d+)\*\* - the published ceiling", "the small-search cell")
        self.check("reach", [m.group(1), m.group(2)],
                   [self.small["reach"], self.small["episodes"]], places=0)
        self.check("upright", [m.group(3), m.group(4)],
                   [self.small["upright_arrivals"], self.small["episodes"]], places=0)
        best_before = max(r["reached_upright_count"] for r in
                          json.load(open(os.path.join(ROOT, "benchmarks",
                                                       "reach_upright_arrival.json"),
                                          encoding="utf-8"))["rows"]
                          if r["physics_preset"] == "v9"
                          and not r["label"].startswith(("stability 60", "curriculum")))
        self.assertEqual(self.small["upright_arrivals"], best_before,
                         "the small cell is no longer at the pre-override ceiling")

    def test_the_replication_claim_is_the_two_committed_cells(self):
        m = self.sentence(r"Drawn a second time it\s*is identical on every column",
                          "the replication claim")
        agree = self.art["baseline_two_draws_agree"]
        for column, equal in agree.items():
            self.assertTrue(equal, f"the two baseline draws disagree on {column}, so 'identical on "
                                   "every column' is false")
        for key in ("reach", "upright_arrivals", "pct_steps_in_band",
                    "share_closed_standing_all", "max_longest_band_run_s"):
            self.assertEqual(self.small[key], self.draw2[key], f"the {key} column moved between draws")

    def test_more_search_buys_arrival_and_not_gait(self):
        m = self.sentence(r"puts it at \*\*(\d+) of (\d+) by\s*distance and (\d+) of (\d+) standing "
                          r"\(([\d.]+)%\)\*\*, above the stability arm's (\d+) of (\d+)",
                          "the bigger-search cell")
        self.check("reach", [m.group(1), m.group(2)], [self.big["reach"], self.big["episodes"]],
                   places=0)
        self.check("upright", [m.group(3), m.group(4), m.group(5)],
                   [self.big["upright_arrivals"], self.big["episodes"], self.big["upright_pct"]],
                   places=1)
        self.check("stability arm", [m.group(6), m.group(7)],
                   [self.learned["best_learned_arm"]["upright_arrivals"], self.big["episodes"]],
                   places=0)
        self.assertGreater(self.big["upright_arrivals"],
                           self.learned["best_learned_arm"]["upright_arrivals"],
                           "the planner no longer beats the best learned arm on arrival, so the "
                           "paragraph's asymmetry has to be rewritten")
        m = self.sentence(r"it closes \*\*([\d.]+)%\*\* of its ground inside the band where the "
                          r"published pool closes ([\d.]+)% and the stability\s*arm ([\d.]+)%",
                          "the gait split")
        self.check("planner share standing", m.group(1),
                   100.0 * self.big["share_closed_standing_all"], places=1)
        self.check("published pool share", m.group(2),
                   100.0 * self.learned["published_pool_reach_episodes"]["share_closed_standing"],
                   places=1)
        self.check("stability arm share", m.group(3),
                   100.0 * self.learned["stability60_seed7_4m_share_closed_standing"], places=1)
        self.assertLess(self.big["share_closed_standing_all"],
                        self.learned["published_pool_reach_episodes"]["share_closed_standing"],
                        "the planner now closes more ground standing than the learned pool, which "
                        "reverses the sentence")
        m = self.sentence(r"its longest continuous stand is ([\d.]+) s against the learned arms' "
                          r"([\d.]+) s", "the continuity comparison")
        self.check("planner longest stand", m.group(1), self.big["max_longest_band_run_s"], places=2)
        self.check("learned longest stand", m.group(2),
                   self.learned["published_pool_reach_episodes"]["max_longest_band_run_s"], places=2)

    def test_the_cost_is_stated_in_the_unit_that_reproduces(self):
        m = self.sentence(r"It also costs\s*([\d.]+) s and ([\d.]+) s of one CPU core per 10 s episode "
                          r"\((\d+)k and (\d+)k simulated steps\)", "the planning cost")
        # Simulated steps per episode are fixed by the config; wall clock is not, so the seconds are
        # allowed to move with the window and the k-steps are not.
        self.check("steps per episode", [m.group(3), m.group(4)],
                   [self.small["sim_steps_per_episode"] / 1e3, self.big["sim_steps_per_episode"] / 1e3],
                   places=0)
        self.check("planning seconds, small cell", m.group(1), self.small["planning_s_per_episode"],
                   slack=0.25 * self.small["planning_s_per_episode"])
        self.check("planning seconds, big cell", m.group(2), self.big["planning_s_per_episode"],
                   slack=0.25 * self.big["planning_s_per_episode"])


class TestReadmeBcTeacherCells(ReadmeGate, unittest.TestCase):
    """The planner-as-teacher benchmark, against benchmarks/bc_teacher_benchmark.json.

    This paragraph reports a failure, and failures are where a prose claim most easily outruns its
    evidence: "the teacher is bad" and "the teacher cannot be cloned" are different sentences with the
    same-looking evidence, so the gate pins both the teacher's own arrival counts and the fact that no
    predictor of its action beats a constant. The last test is the one that keeps the second claim honest -
    it fails if any regressor ever separates itself from the trivial baseline, because then the paragraph
    would be describing a student that could have learned and did not.
    """

    BENCH = os.path.join(ROOT, "benchmarks", "bc_teacher_benchmark.json")
    STUDENTS = os.path.join(ROOT, "benchmarks", "bc_mpc_students.json")
    START = "**Cloning the planner fails"
    END = 'So the honest answer to "does an agent walk'

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.art, self.stu = self.read_artifacts(self.BENCH, self.STUDENTS)
        self.mse = self.art["predictability_val_mse"]
        self.bad = []
        self.what = "planner-as-teacher benchmark"

    def test_the_teacher_is_scored_on_its_own_episodes(self):
        m = self.sentence(r"the teacher reaches in \*\*(\d+) of (\d+)\*\* and arrives standing in\s*"
                          r"\*\*(\d+) of \d+\*\*, with ([\d.]+)% of its steps inside the band",
                          "the teacher's own numbers")
        t = self.art["teacher"]
        self.check("teacher reached", [m.group(1), m.group(2)],
                   [t["reached"], t["episodes"]], places=0)
        self.check("teacher upright arrivals", m.group(3), t["upright_arrivals"], places=0)
        self.check("teacher band time", m.group(4), t["pct_steps_in_band"], places=1)
        self.assertGreater(t["upright_arrivals"], 0,
                           "the teacher no longer arrives standing at all, so 'it is the cloning that "
                           "fails' is no longer the right diagnosis")

    def test_the_two_students_and_the_data_they_saw(self):
        m = self.sentence(r"one imitating only\s*the in-band steps \(([\d,]+) of ([\d,]+) transitions\)"
                          r"\. Both score \*\*(\d+) of (\d+)\*\* on the published protocol, at mean\s*"
                          r"return ([\d,.]*\d) and ([\d,.]*\d)", "the student cells")
        t = self.art["teacher"]
        self.check("in-band over all transitions", [m.group(1).replace(",", ""),
                                                    m.group(2).replace(",", "")],
                   [t["transitions_in_band"], t["transitions"]], places=0)
        self.check("student reach", [m.group(3), m.group(4)], [0, 20], places=0)
        for key, value in (("bc_raw", m.group(5)), ("bc_band", m.group(6))):
            self.check(f"{key} mean return", value.replace(",", ""),
                       self.stu["models"][key]["mean"], places=2)
            self.assertEqual(self.stu["models"][key]["reached_target_pct"], 0.0,
                             f"{key} now reaches, so the paragraph's negative result is stale")
        self.assertTrue(self.art["split"]["val_rows"] > 0)

    def test_no_predictor_of_the_action_beats_a_constant_by_more_than_noise(self):
        m = self.sentence(r"predicting a constant zero action gives validation MSE\s*\*\*([\d.]+)\*\*, "
                          r"the per-dimension mean ([\d.]+), ridge ([\d.]+), nearest-neighbour "
                          r"([\d.]+) at k=1 falling to\s*([\d.]+)\s*at k=32, and the trained students "
                          r"([\d.]+) and ([\d.]+)", "the predictability table")
        want = {"zero": m.group(1), "per_dim_mean_of_train": m.group(2),
                "ridge_alpha100": m.group(3), "knn_k1": m.group(4), "knn_k32": m.group(5),
                "student_bc_raw": m.group(6), "student_bc_band": m.group(7)}
        for name, typed in want.items():
            self.check(name, typed, self.mse[name], places=3)
        baseline = min(self.mse["zero"], self.mse["per_dim_mean_of_train"])
        for name, value in self.mse.items():
            if name in ("zero", "per_dim_mean_of_train"):
                continue
            self.assertGreater(value, baseline * 0.98,
                               f"{name} separates itself from the constant baseline, so 'nothing "
                               "recovers any state-dependence' is no longer true")

    def test_the_jitter_claim_is_the_recorded_signal_not_an_illustration(self):
        m = self.sentence(r"consecutive recorded actions differ by MSE\s*\*\*([\d.]+)\*\* against the "
                          r"action's own energy of ([\d.]+)", "the jitter")
        j = self.art["teacher_jitter"]
        self.check("consecutive difference", m.group(1), j["consecutive_action_mse"], places=3)
        self.check("action energy", m.group(2), j["action_energy"], places=3)
        self.assertGreater(j["consecutive_action_mse"], j["action_energy"],
                           "the recorded action is no longer more variable step to step than its own "
                           "magnitude, so the mechanism the paragraph names is gone")

    def test_the_elite_mean_comparison_comes_from_a_committed_trajectory(self):
        m = self.sentence(r"on a 4-episode probe the jitter moves from ([\d.]+) to\s*"
                          r"([\d.]+), ridge \(([\d.]+)\) and nearest-neighbour \(([\d.]+)\) still lose to "
                          r"the constant predictor \(([\d.]+)\), and\s*the teacher drops to (\d+) of "
                          r"(\d+) reaching with no upright arrival", "the elite-mean comparison")
        cmp_block = self.art.get("action_extraction_compare")
        self.assertIsNotNone(cmp_block, "the elite-mean trajectories were never committed, so the "
                                        "numbers in that clause have no artifact behind them")
        self.check("jitter argmin -> elite-mean", [m.group(1), m.group(2)],
                   [self.art["teacher_jitter"]["consecutive_action_mse"],
                    cmp_block["jitter"]["consecutive_action_mse"]], places=3)
        p2 = cmp_block["predictability_val_mse"]
        self.check("ridge / knn / zero on the elite-mean split",
                   [m.group(3), m.group(4), m.group(5)],
                   [p2["ridge_alpha100"], p2["knn_k32"], p2["zero"]], places=3)
        self.assertGreater(p2["ridge_alpha100"], p2["zero"],
                           "ridge now beats the constant predictor on the elite-mean actions, so the "
                           "extraction may be learnable after all and the clause has to be rewritten")
        self.check("elite-mean reached", [m.group(6), m.group(7)],
                   [cmp_block["reached"], cmp_block["episodes"]], places=0)
        self.assertEqual(cmp_block["upright_arrivals"], 0)


class TestReadmeTrainerPairCells(ReadmeGate, unittest.TestCase):
    """PPO against SAC: the rate claim and the behaviour claim are gated together or not at all.

    The danger in this pair is that one half gets quoted alone. "PPO is 2.071x faster" is true and
    useless without "-985.97 against 22,020.06", and the reverse is true too, so the gate insists
    both numbers are present, in the same section, from the same artifact.
    """

    PAIR = os.path.join(ROOT, "benchmarks", "trainer_pair_ppo_sac.json")
    START = "| `train_walker.py --algo ppo` |"
    END = "The Dreamer row was the whole story of the eager build"

    def setUp(self):
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = readme[start:readme.index(self.END, start)]
        self.flat = re.sub(r"\s+", " ", self.block)
        self.art, = self.read_artifacts(self.PAIR)
        self.bad = []
        self.what = "PPO against SAC"

    def test_the_two_table_rows_are_the_measured_rates(self):
        arms = self.art["arms"]
        m = re.search(r"\| `train_walker\.py --algo ppo` \|[^|]*\| ([\d,.]+) steps in (\d+) s = "
                      r"([\d,.]+)/s \| ([\d.]+) min \|", self.block)
        self.assertIsNotNone(m, "the PPO row of the trainer table is gone or reformatted")
        self.check("PPO row", [m.group(1), m.group(2), m.group(3), m.group(4)],
                   [arms["ppo_v9"]["collected_steps"], arms["ppo_v9"]["seconds"],
                    arms["ppo_v9"]["env_steps_per_second"],
                    arms["ppo_v9"]["seconds_per_1m_env_steps"] / 60], places=1)
        m = re.search(r"\| `train_walker\.py --algo sac` \|[^|]*\| ([\d,.]+) steps in (\d+) s = "
                      r"([\d,.]+)/s \| ([\d.]+) min \|", self.block)
        self.assertIsNotNone(m, "the SAC row of the trainer table is gone or reformatted")
        self.check("SAC row", [m.group(1), m.group(2), m.group(3), m.group(4)],
                   [arms["sac_v9"]["collected_steps"], arms["sac_v9"]["seconds"],
                    arms["sac_v9"]["env_steps_per_second"],
                    arms["sac_v9"]["seconds_per_1m_env_steps"] / 60], places=1)

    def test_the_speed_and_the_score_are_stated_together(self):
        m = self.sentence(r"PPO collects ([\d.]+)[x×] the environment steps per second that SAC "
                          r"does - ([\d,.]+)/s against ([\d,.]+)/s", "the rate claim")
        self.check("rate ratio", m.group(1), self.art["ppo_steps_per_second_over_sac"], places=3)
        self.check("rates", [m.group(2), m.group(3)],
                   [self.art["arms"]["ppo_v9"]["env_steps_per_second"],
                    self.art["arms"]["sac_v9"]["env_steps_per_second"]], places=0)
        m = self.sentence(r"PPO averages \*\*(-?[\d.]+)\*\* where SAC averages \*\*([\d,.]+)\*\*, "
                          r"its closest approach to the target is \*\*([\d.]+) m\*\* against SAC's "
                          r"\*\*([\d.]+) m\*\*", "the behaviour claim")
        ppo, sac = self.art["arms"]["ppo_v9"], self.art["arms"]["sac_v9"]
        self.check("means and distances", [m.group(1), m.group(2), m.group(3), m.group(4)],
                   [ppo["mean"], sac["mean"], ppo["mean_min_target_distance"],
                    sac["mean_min_target_distance"]], places=2)
        m = self.sentence(r"PPO's \*\*falls per episode are ([\d.]+) against SAC's ([\d.]+)\*\*",
                          "the falls sentence")
        self.check("falls", [m.group(1), m.group(2)],
                   [ppo["falls_per_episode"], sac["falls_per_episode"]], places=2)
        self.assertEqual(ppo["standing_at_end_pct"], 0.0,
                         "the paragraph explains PPO's low fall count by 0% standing; that is no "
                         "longer what the artifact says")
        self.assertEqual(sac["standing_at_end_pct"], 0.0)

    def test_the_rollout_boundary_is_stated_as_the_reason_for_983040(self):
        m = self.sentence(r"its\s*\n?\"1M\" run ends at \*\*([\d,]+) steps\*\*, not ([\d,]+)",
                          "the checkpoint-boundary caveat")
        self.check("collected steps", m.group(1), self.art["arms"]["ppo_v9"]["collected_steps"],
                   places=0)
        self.assertEqual(int(m.group(2).replace(",", "")),
                         self.art["arms"]["sac_v9"]["collected_steps"],
                         "the nominal budget no longer matches the SAC arm's")
        self.assertIn("rollout boundaries", self.flat,
                      "the reason 983,040 is not a bug has to travel with the number")

    def test_the_earlier_budget_pair_carries_the_same_sign(self):
        earlier = self.art["earlier_budget"]
        m = self.sentence(r"`ppo_524288` against `sac_500000`, puts PPO at \*\*([\d,.]+)\*\* mean "
                          r"and \*\*([\d.]+) m\*\*\s*closest against SAC's \*\*([\d,.]+)\*\* and "
                          r"\*\*([\d.]+) m\*\*", "the earlier-budget pair")
        self.check("earlier pair", [m.group(1), m.group(2), m.group(3), m.group(4)],
                   [earlier["ppo_524288"]["mean"], earlier["ppo_524288"]["mean_min_target_distance"],
                    earlier["sac_500000"]["mean"],
                    earlier["sac_500000"]["mean_min_target_distance"]], places=2)
        self.assertLess(earlier["ppo_524288"]["mean"], earlier["sac_500000"]["mean"],
                        "the README claims the ordering holds at 5% of budget too; it no longer "
                        "does at this budget")
        self.assertLess(self.art["arms"]["ppo_v9"]["mean"], self.art["arms"]["sac_v9"]["mean"])

    def test_both_arms_were_scored_in_the_world_they_trained_in(self):
        arms = self.art["arms"]
        self.assertEqual(arms["ppo_v9"]["scored_in_version"],
                         "standup_balance_walk_curriculum_v9")
        self.assertEqual(arms["ppo_fast"]["scored_in_version"],
                         "standup_balance_walk_curriculum_v9_fast")
        self.assertEqual(arms["ppo_fast"]["env_steps_per_second"] >
                         arms["ppo_v9"]["env_steps_per_second"], True,
                         "the fast world is quoted as cheaper for PPO too; that has to stay true "
                         "or the preset section's claim needs re-checking")


class TestReadmeCurriculumProvenanceCells(ReadmeGate, unittest.TestCase):
    """The manual phase curriculum, gated against `benchmarks/curriculum_provenance.json`.

    The claim being protected is a factual one the README had backwards: that recovery was fine-tuned
    from the walking policy. The checkpoint metadata says the opposite, and the weights themselves are
    gitignored, so the *metadata* had to be committed for the correction to be checkable rather than
    remembered. Every date, width and step count quoted in the paragraph is read from the artifact here.
    """

    PROV = os.path.join(ROOT, "benchmarks", "curriculum_provenance.json")
    START = "**That is not the same as the curriculum being absent"
    END = "The scale of that table matters"

    def setUp(self):
        if not os.path.exists(self.PROV):
            self.skipTest("benchmarks/curriculum_provenance.json has not been generated")
        with open(self.PROV, encoding="utf-8") as handle:
            self.art = json.load(handle)
        self.runs = {r["run_id"]: r for r in self.art["runs"]}
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index(self.START)
        self.block = re.sub(r"\s+", " ", readme[start:readme.index(self.END, start)])
        self.flat = self.block
        self.what = "curriculum provenance"
        self.bad = []

    def test_the_order_and_the_gap_are_the_artifact_s(self):
        order = self.art["order"]
        self.assertEqual([order["first"], order["second"]],
                         ["walker_recovery_v1", "walker_target_v1"])
        m = self.sentence(r"So recovery precedes target by ([\d.]+) days", "the ordering sentence")
        self.check("the gap in days", m.group(1), order["gap_days"], places=1)
        m = self.sentence(r"the recovery\s*files are ([\d.]+) days older and (\d+)-wide against (\d+)",
                          "the restatement of the order")
        self.check("the same gap, restated", [m.group(1), m.group(2), m.group(3)],
                   [order["gap_days"], order["widths"]["walker_recovery_v1"],
                    order["widths"]["walker_target_v1"]], places=1)

    def test_the_two_runs_are_described_as_their_checkpoints_record_them(self):
        rec, tgt = self.runs["walker_recovery_v1"], self.runs["walker_target_v1"]
        m = self.sentence(r"`walker_recovery_v1` holds (\d+) checkpoints of a\s*\*\*(\d+)-wide\*\* "
                          r"policy stamped `env_version = ([A-Za-z0-9_]+)`, written\s*"
                          r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}) \((\d+)M\s*steps\) and\s*"
                          r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}) \((\d+)M\)", "the recovery sentence")
        self.check("recovery: count, width, steps",
                   [int(m.group(1)), int(m.group(2)), m.group(6), m.group(9)],
                   [rec["distinct_steps"], rec["actor_obs_width"],
                    rec["first_step"] // 1_000_000, rec["last_step"] // 1_000_000], places=0)
        self.assertEqual(m.group(3), rec["env_version"],
                         "the env revision the prose names is not the one the checkpoint records")
        self.assertEqual(m.group(4) + " " + m.group(5), rec["first_written"].replace("T", " ")[:16],
                         "recovery's first checkpoint date/time is not the file's mtime")
        self.assertEqual(m.group(7) + " " + m.group(8), rec["last_written"].replace("T", " ")[:16],
                         "recovery's last checkpoint date/time is not the file's mtime")
        m = self.sentence(r"`walker_target_v1` holds (\d+) step positions of a \*\*(\d+)-wide\*\*\s*"
                          r"policy stamped `([A-Za-z0-9_]+)` with `task_phase=([a-z]+)`,\s*"
                          r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}) \((\d+)M\)\s*through\s*"
                          r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}) \((\d+)M\)", "the target sentence")
        self.check("target: count, width, steps",
                   [int(m.group(1)), int(m.group(2)), m.group(7), m.group(10)],
                   [tgt["distinct_steps"], tgt["actor_obs_width"],
                    tgt["first_step"] // 1_000_000, tgt["last_step"] // 1_000_000], places=0)
        self.assertEqual([m.group(3), m.group(4)], [tgt["env_version"], tgt["task_phase"]],
                         "the env revision or phase the prose names is not the checkpoint's")
        self.assertEqual(m.group(5) + " " + m.group(6), tgt["first_written"].replace("T", " ")[:16])
        self.assertEqual(m.group(8) + " " + m.group(9), tgt["last_written"].replace("T", " ")[:16])

    def test_the_mid_run_stage_change_is_pinned_to_two_files(self):
        switches = self.runs["walker_target_v1"]["target_forward_velocity_switches"]
        self.assertEqual(len(switches), 1, "the prose describes one restart with a new speed")
        s = switches[0]
        m = self.sentence(r"nominal walking speed recorded in the checkpoints is \*\*([\d.]+) m/s at "
                          r"the\s*([\d,]+)-step file \((\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}):\d{2}\) and "
                          r"([\d.]+) m/s at\s*([\d,]+) \((\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}):\d{2}\)\*\*",
                          "the stage-change sentence")
        self.check("the switch: speeds and steps",
                   [m.group(1), int(m.group(2).replace(",", "")), m.group(5),
                    int(m.group(6).replace(",", ""))],
                   [s["from_velocity"], s["from_step"], s["to_velocity"], s["to_step"]], places=1)
        self.assertEqual([m.group(3), m.group(4), m.group(7), m.group(8)],
                         [s["from_written"][:10], s["from_written"][11:16],
                          s["to_written"][:10], s["to_written"][11:16]],
                         "the restart's clock times are not the files' mtimes")
        self.assertLess(s["from_velocity"], s["to_velocity"],
                        "the prose says the speed was raised; the metadata says otherwise")
        self.assertGreater(datetime.datetime.fromisoformat(s["to_written"])
                           - datetime.datetime.fromisoformat(s["from_written"]),
                           datetime.timedelta(hours=6),
                           "the prose calls it a night's gap; the mtimes do not")

    def test_the_widening_is_three_columns_and_the_zero_fill_is_in_the_code(self):
        widths = self.art["order"]["widths"]
        self.assertEqual(widths["walker_target_v1"] - widths["walker_recovery_v1"], 3,
                         "the prose says three new target columns; the widths say otherwise")
        with open(os.path.join(ROOT, "train_walker.py"), encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("target_value.data.zero_()", source,
                      "load_state_dict_with_expanded_input no longer zero-fills the new columns")
        self.sentence(r"copies the overlapping columns and zero-fills the three new\s*ones, and "
                      r"`adapt_obs_rms` pads the\s*normalizer statistics", "the widening sentence")

    def test_the_prose_does_not_claim_more_than_the_metadata_shows(self):
        """The one sentence that must stay a hedge: order is proven, initialization is not."""
        self.sentence(r"They do not establish that target was initialized \*from\* recovery",
                      "the limit-of-the-evidence sentence")
        self.assertIn("Phase 3 now points here", self.flat,
                      "the Phase-3 correction was removed but its old ordering claim may be back")
        with open(README, encoding="utf-8") as handle:
            readme = handle.read()
        start = readme.index("**The Transfer Learning Curriculum")
        phase3 = re.sub(r"\s+", " ", readme[start:start + 2600])
        self.assertNotIn("recovery agent was fine-tuned from the walking agent", phase3,
                         "Phase 3 still states the reversed lineage that the checkpoint dates refute")

if __name__ == "__main__":
    unittest.main()
