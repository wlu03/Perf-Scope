import unittest

from perf_scope.config import Case, Metric, Variant
from perf_scope.profiling import diagnose, parse_launches

CASE = Case(
    "softmax",
    "softmax",
    (128, 259),
    "fp32",
    Metric.LATENCY,
    Variant("fused", ("bench", "softmax", "fused")),
    Variant("multi-pass", ("bench", "softmax", "multi-pass")),
)


class TraceTest(unittest.TestCase):
    def test_counts_softmax_kernels_and_diagnoses_extra_launches(self):
        fused = parse_launches(
            'Processing report\n"Name","Instances"\n"perf_scope::softmax_fused(float *)",1\n'
        )
        multi = parse_launches(
            "Instances,Name\n1,softmax_reduce_max\n1,softmax_exp_sum\n1,softmax_normalize\n3,other\n"
        )
        self.assertEqual(fused, {"softmax_fused": 1})
        decision = diagnose(
            CASE,
            [
                {"status": "ok", "launches": fused},
                {"status": "ok", "launches": multi},
            ],
        )
        self.assertEqual(decision["cause"], "increased CUDA kernel launches")
        self.assertEqual(
            decision["evidence"], {"baseline_launches": 1, "candidate_launches": 3}
        )

    def test_rejects_empty_or_invalid_kernel_counts(self):
        for report in (
            "Instances,Name\n3,other\n",
            "Instances,Name\n0,softmax_fused\n",
        ):
            with self.subTest(report=report), self.assertRaises(ValueError):
                parse_launches(report)
