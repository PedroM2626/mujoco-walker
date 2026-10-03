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
            cells = self.row(label, 3)
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
        m = self.sentence(r"one imagination step costs \*\*([\d.]+) ms\*\* eager against "
                          r"\*\*([\d.]+) ms\*\* captured", "imagination marginal")
        marg = self.art["marginal_ms_per_unit"]
        self.check("per imagination step", [float(m.group(1)), float(m.group(2))],
                   [marg["imag_horizon_eager"], marg["imag_horizon_captured"]], places=3)
        m = self.sentence(r"one world-model timestep \*\*([\d.]+) ms\*\* against "
                          r"\*\*([\d.]+) ms\*\*", "world-model marginal")
        self.check("per world-model timestep", [float(m.group(1)), float(m.group(2))],
                   [marg["seq_len_eager"], marg["seq_len_captured"]], places=3)

    def test_the_baseline_and_batch_rows_are_the_measured_medians(self):
        base = self.art["configs"][self.BASE]
        m = self.sentence(r"the whole update ([\d.]+) ms against \*\*([\d.]+) ms\*\*", "baseline")
        self.check("update at the shipped sizes", [float(m.group(1)), float(m.group(2))],
                   [base["eager"]["median_ms"], base["captured"]["median_ms"]], places=2)
        for arm, pattern in (("eager", r"16 -> ([\d.]+) ms,\s*64 -> ([\d.]+) ms, "
                                      r"256 -> ([\d.]+) ms"),
                            ("captured", r"it grows \(([\d.]+) -> ([\d.]+) -> ([\d.]+) ms\)")):
            m = self.sentence(pattern, f"the {arm} batch row")
            want = self.art[f"batch_scaling_{arm}"]
            self.check(f"{arm} batch scaling", [float(m.group(i)) for i in (1, 2, 3)],
                       [want["16"], want["64"], want["256"]], places=2)

    def test_the_two_loop_shares_are_multiplied_from_the_sweep_not_remembered(self):
        """`about 3.9 ms` and `about 7.1 ms` are the marginals times the shipped loop lengths."""
        marg = self.art["marginal_ms_per_unit"]
        seq = self.art["configs"][self.BASE]["seq_len"]
        imag = self.art["configs"][self.BASE]["imag_horizon"]
        m = self.sentence(r"the imagination loop is\s*about ([\d.]+) ms and the world model "
                          r"about ([\d.]+) ms", "loop shares")
        self.check("loop shares", [float(m.group(1)), float(m.group(2))],
                   [round(marg["imag_horizon_captured"] * imag, 1),
                    round(marg["seq_len_captured"] * seq, 1)], places=1)


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


if __name__ == "__main__":
    unittest.main()
