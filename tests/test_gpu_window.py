"""The GPU-window guard has to be right about three things: what it reads, when it refuses, and
what it says it did.

None of it needs a GPU to test. It matters because the guard is what stops a 1M-step run from
starting into the contention that killed the last one at 84k steps, and because a refusal that
fires on a 5k-step test - or an approval that claims the wrong clause let it through - is worse
than no check at all.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import gpu_window as gw  # noqa: E402

CLEAN = {"gpu": {"name": "RTX 4070 Laptop GPU", "memory_used_mib": 371,
                 "memory_total_mib": 8188, "driver_version": "556.29"},
         "other_cuda_contexts": 5, "other_context_memory_mib_reported": False}
CONTENDED = {"gpu": {"name": "RTX 4070 Laptop GPU", "memory_used_mib": 2100,
                     "memory_total_mib": 8188, "driver_version": "556.29"},
             "other_cuda_contexts": 7, "other_context_memory_mib_reported": True}


class TestParsers(unittest.TestCase):
    def test_a_real_query_line_parses(self):
        line = "NVIDIA GeForce RTX 4070 Laptop GPU, 371, 8188, 556.29"
        got = gw.parse_gpu_line(line)
        self.assertEqual({"name": "NVIDIA GeForce RTX 4070 Laptop GPU", "memory_used_mib": 371,
                          "memory_total_mib": 8188, "driver_version": "556.29"}, got)

    def test_unparseable_gpu_lines_are_none_not_an_exception(self):
        # A driver that prints "No devices were found" must not take a trainer down with it.
        for line in ("", "No devices were found", "gpu, used, total", "x, n/a, 8188, 1.0"):
            self.assertIsNone(gw.parse_gpu_line(line), line)

    def test_context_memory_reports_as_na_for_desktop_processes(self):
        # This is the measured shape on Windows: five contexts, no memory figures.
        self.assertEqual((20820, None), gw.parse_context_line("20820, [N/A]"))
        self.assertEqual((2548, 478), gw.parse_context_line("2548, 478 MiB"))
        for line in ("pid, used_gpu_memory [MiB]", "", "abc, 12 MiB"):
            self.assertIsNone(gw.parse_context_line(line), line)


class TestDecision(unittest.TestCase):
    def test_a_clean_window_is_allowed_and_a_contended_one_refused(self):
        self.assertTrue(gw.decide_gpu_exclusivity(CLEAN, allow_shared=False)[0])
        ok, reason = gw.decide_gpu_exclusivity(CONTENDED, allow_shared=False)
        self.assertFalse(ok)
        self.assertIn("2100 MiB", reason)
        self.assertIn("--allow-shared-gpu", reason)

    def test_the_flag_and_a_missing_tool_both_let_the_run_through(self):
        self.assertTrue(gw.decide_gpu_exclusivity(CONTENDED, allow_shared=True)[0])
        ok, reason = gw.decide_gpu_exclusivity(None, allow_shared=False)
        self.assertTrue(ok)
        self.assertIn("could not be checked", reason)

    def test_the_ceiling_is_the_number_the_module_publishes(self):
        at_ceiling = dict(CONTENDED, gpu=dict(CONTENDED["gpu"], memory_used_mib=gw.DEFAULT_CEILING_MIB))
        self.assertTrue(gw.decide_gpu_exclusivity(at_ceiling, allow_shared=False)[0],
                        "the ceiling is exclusive; a window exactly at it is still allowed")


class TestCheckRefusesOnlyLongRuns(unittest.TestCase):
    def call(self, window, total_steps, allow_shared=False):
        with mock.patch.object(gw, "read_gpu_window", return_value=window):
            lines = []
            try:
                got = gw.check_gpu_window("cuda", allow_shared, total_steps, log=lines.append)
            except SystemExit as exc:
                return "refused", str(exc), lines
            return "ok", got, lines

    def test_cpu_runs_are_not_checked_at_all(self):
        with mock.patch.object(gw, "read_gpu_window") as reader:
            self.assertIsNone(gw.check_gpu_window("cpu", False, 10_000_000))
            reader.assert_not_called()

    def test_a_contended_window_refuses_a_long_run(self):
        verdict, message, _ = self.call(CONTENDED, 1_000_000)
        self.assertEqual("refused", verdict)
        self.assertIn("refusing to start", message)

    def test_a_contended_window_only_warns_for_a_short_one(self):
        verdict, window, lines = self.call(CONTENDED, 5_000)
        self.assertEqual("ok", verdict)
        self.assertIn("long-run threshold", " ".join(lines))
        self.assertNotIn("--allow-shared-gpu was passed", " ".join(lines))

    def test_the_flag_is_named_when_it_is_what_allowed_the_run(self):
        _, _, lines = self.call(CONTENDED, 1_000_000, allow_shared=True)
        self.assertIn("--allow-shared-gpu was passed", " ".join(lines))

    def test_a_clean_long_run_says_the_window_is_clear(self):
        verdict, window, lines = self.call(CLEAN, 1_000_000)
        self.assertEqual("ok", verdict)
        self.assertIn("window is clear", " ".join(lines))
        self.assertEqual(CLEAN, window)

    def test_the_window_is_published_as_run_params(self):
        params = gw.window_params(CLEAN)
        self.assertEqual(371, params["gpu_memory_used_mib_before_run"])
        self.assertEqual(5, params["gpu_other_cuda_contexts"])
        self.assertEqual({}, gw.window_params(None))


if __name__ == "__main__":
    unittest.main()
