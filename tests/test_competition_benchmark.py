"""Offline regression checks for the competition-facing comparison report."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_competition import benchmark, format_report  # noqa: E402
from text_baseline import TEXT_ROUTE, measure  # noqa: E402


class CompetitionBenchmarkTests(unittest.TestCase):
    def test_pure_text_baseline_has_the_full_eight_step_workflow(self):
        report = measure()
        self.assertEqual(report["case_count"], 10)
        self.assertEqual(report["message_count"], 10 * len(TEXT_ROUTE))
        self.assertEqual(report["non_text_transfer_count"], 0)
        self.assertGreater(report["total_message_bytes"], 0)
        self.assertGreater(report["total_message_tokens_estimate"], 0)

    def test_comparison_checks_equal_quality_and_emits_markdown(self):
        report = benchmark(include_process=False)
        self.assertTrue(report["comparison"]["same_cases"])
        self.assertTrue(report["comparison"]["same_exact_matches"])
        self.assertIn("纯文本协作与结构化协议对比", format_report(report))
        self.assertEqual(report["non_text_state_transfer"]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
