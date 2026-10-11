import unittest
from dataclasses import replace

from perf_scope.config import Case, Metric, Suite, Variant
from perf_scope.regression import compare_case

CASE = Case(
    "gemm",
    "gemm",
    (128, 128, 128),
    "fp32",
    Metric.LATENCY,
    Variant("base", ("base",)),
    Variant("new", ("new",)),
)
SUITE = Suite(42, 12, 5, 20, 120, "fixture", 0.05, (CASE,))


def rows_for(case, baseline, candidate):
    return [
        {
            "source": "fixture",
            "case": case.id,
            "pair": pair,
            "side": side,
            "variant": variant,
            "device": "fixture-device",
            "metric": case.metric,
            "status": "ok",
            "value": value,
        }
        for pair, (first, second) in enumerate(zip(baseline, candidate, strict=True))
        for side, variant, value in (
            ("baseline", case.baseline.id, first),
            ("candidate", case.candidate.id, second),
        )
    ]


class RegressionTest(unittest.TestCase):
    def test_classifies_paired_latency_and_exact_margin(self):
        for current, status in (
            (125, "regression"),
            (105, "within_threshold"),
            (96, "within_threshold"),
        ):
            with self.subTest(current=current):
                rows = rows_for(CASE, [100] * SUITE.pairs, [current] * SUITE.pairs)
                result = compare_case(SUITE, CASE, rows)
                self.assertEqual(result.status, status)
                self.assertAlmostEqual(result.slowdown, current / 100 - 1)
                self.assertEqual(result, compare_case(SUITE, CASE, rows))

    def test_throughput_has_inverse_direction(self):
        case = replace(CASE, metric=Metric.BANDWIDTH)
        rows = rows_for(case, [100] * SUITE.pairs, [80] * SUITE.pairs)
        result = compare_case(SUITE, case, rows)
        self.assertEqual(result.status, "regression")
        self.assertAlmostEqual(result.slowdown, 0.25)

    def test_inconclusive_noisy_and_short_runs(self):
        rows = rows_for(CASE, [100] * 12, [95, 120] * 6)
        result = compare_case(SUITE, CASE, rows)
        self.assertEqual(result.status, "inconclusive")
        self.assertLess(result.lower, SUITE.slowdown_margin)
        self.assertGreater(result.upper, SUITE.slowdown_margin)
        short = compare_case(replace(SUITE, pairs=2), CASE, rows[:4])
        self.assertEqual(short.status, "inconclusive")
        self.assertIsNone(short.lower)

    def test_rejects_mixed_devices_duplicate_pairs_and_incomplete_rows(self):
        rows = rows_for(CASE, [100] * SUITE.pairs, [125] * SUITE.pairs)
        for invalid in (
            [rows[0] | {"device": "other"}, *rows[1:]],
            [rows[1], *rows[1:]],
            rows[:-1],
        ):
            with self.subTest(rows=len(invalid)):
                result = compare_case(SUITE, CASE, invalid)
                self.assertEqual(result.status, "invalid")
                self.assertIsNone(result.slowdown)
