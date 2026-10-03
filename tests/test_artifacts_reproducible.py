"""Every committed benchmark artifact must be exactly what its logs say - regenerated, not trusted.

`summarize_benchmarks.py` parses the evaluator logs in this repository into `benchmarks/*.json`, and
the README quotes those JSON files (gated cell by cell in `test_readme_evidence.py`). This test closes
the other half of that chain: it runs every builder and compares its data against the committed file,
so a hand-edited artifact, an edited log, or a builder whose derivation changed all fail here rather
than in a citation months later.

It also asserts the evidence logs are actually in the repository: `.gitignore` excludes `*.log` with
named exceptions, and a clone that lacks them silently loses the ability to reproduce any number the
README quotes.
"""

import json
import os
import unittest

import summarize_benchmarks as sb

ROOT = sb.ROOT
EVIDENCE_LOGS = [
    "eval_phase3_20ep.log", "eval_phase3_100ep.log", "eval_phase3_100ep_v9.log",
    "eval_phase3_100ep_rawobs.log", os.path.join("openai_walker", "eval_phase4_20ep.log"),
    os.path.join("openai_walker", "eval_phase4_50ep.log"),
    os.path.join("openai_walker", "eval_extratrees_50ep.log"),
    "ars_evidence.log", "redq_evidence.log", "dreamer_smoke.log",
]


class TestArtifactsRegenerate(unittest.TestCase):
    def test_the_evidence_logs_are_present_and_not_ignored(self):
        """`.gitignore` excludes `*.log` with named exceptions; these ten must stay excepted.

        Checked without shelling out to git (this repository is not owned by the user that runs the
        tests, so git refuses to read it). Whether the logs are actually committed is covered
        indirectly and more strongly: on a CI checkout the builders below would raise
        FileNotFoundError if a clone lacked them, and that fails the next test.
        """
        with open(os.path.join(ROOT, ".gitignore"), encoding="utf-8") as handle:
            negated = {line[1:].strip() for line in handle if line.startswith("!")}
        for path in EVIDENCE_LOGS:
            self.assertTrue(os.path.exists(os.path.join(ROOT, path)), f"{path} is missing")
            self.assertIn(path.replace(os.sep, "/"), negated,
                          f"{path} is not excepted from the *.log rule, so a clone cannot "
                          "rebuild the artifacts this test checks")

    def test_every_builder_reproduces_its_committed_artifact(self):
        drift, unreadable = [], []
        for builder in sb.BENCHMARKS:
            try:
                data, target = builder()
            except FileNotFoundError as exc:
                unreadable.append(f"{getattr(builder, '__name__', builder)}: {exc}")
                continue
            path = os.path.join(ROOT, target)
            self.assertTrue(os.path.exists(path), f"{target} is built but not committed")
            with open(path, encoding="utf-8") as handle:
                committed = json.load(handle)
            # Compare through JSON so equal-but-differently-typed data is still comparable.
            if json.loads(json.dumps(data, sort_keys=True)) != json.loads(
                    json.dumps(committed, sort_keys=True)):
                drift.append(target)
        self.assertEqual([], unreadable, "a builder could not read its log")
        self.assertEqual([], drift, "these artifacts no longer match what their logs produce; "
                                    "run python summarize_benchmarks.py and review the diff")

    def test_the_published_protocol_line_matches_the_command_that_ran(self):
        """The protocol string is derived from the log; check it still says what the log says."""
        data, _target = sb.phase3("eval_phase3_100ep.log", seed=11,
                                  out="benchmarks/phase3_merging_100ep.json")
        with open(os.path.join(ROOT, "benchmarks", "phase3_merging_100ep.json"),
                  encoding="utf-8") as handle:
            committed = json.load(handle)
        self.assertEqual(committed["protocol"], data["protocol"])
        self.assertIn("--num-episodes 100", committed["protocol"])
        self.assertIn("--seed 11", committed["protocol"])


if __name__ == "__main__":
    unittest.main()
