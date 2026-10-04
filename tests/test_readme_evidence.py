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
        self.block = re.sub(r"\s+", " ", readme[start:start + 900])

    def tearDown(self):
        self.con.close()

    def find(self, pattern, what):
        m = re.search(pattern, self.block)
        self.assertIsNotNone(m, f"the {what} sentence was reworded; re-point this test at it")
        return m

    def test_the_run_and_experiment_counts(self):
        active = self.con.execute("SELECT COUNT(*) FROM runs WHERE lifecycle_stage='active'").fetchone()[0]
        exps = self.con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        m = self.find(r"\*\*(\d+) active runs\*\* in (\d+) experiments", "the run and experiment counts")
        self.assertEqual(int(m.group(1)), active, "the README's active-run count is not the database's")
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
            self.assertEqual(value, want[name], f"{name}: README says {value}, database has {want[name]}")

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
        self.run = load(self.P1)["models"]["dreamer_v3_1m"]
        self.partial = load(self.P9)["models"]["dreamer_v9_269k"]
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
        m = self.find(r"the full budget moved\s*the mean from ([\d.]+) down to ([\d.]+) and the std "
                      r"from ([\d.]+) up to ([\d.]+), on (\d+) episodes rather\s+than the (\d+) the "
                      r"partial one was scored on", "the partial-versus-full comparison")
        self.check("partial versus full mean and std",
                   [float(m.group(i)) for i in (1, 2, 3, 4)],
                   [self.partial["mean"], self.run["mean"], self.partial["std"], self.run["std"]],
                   places=2)
        self.check("episode counts", [int(m.group(5)), int(m.group(6))],
                   [self.run["episodes"], self.partial["episodes"]], places=0)
        self.assertEqual(self.partial["global_step"], 269404, "the partial checkpoint's step count")
        m = self.find(r"\(2\.893 m against ([\d.]+) m\) with a lower return", "the ARS comparison")
        self.assertAlmostEqual(float(m.group(1)), self.ars["mean_min_target_distance"], places=3)
        self.assertLess(self.run["mean"], self.ars["mean"],
                        "the paragraph says Dreamer's return is lower than ARS at the same budget")

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


if __name__ == "__main__":
    unittest.main()
