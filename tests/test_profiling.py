import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from perf_scope.config import Case, Metric, Suite, Variant
from perf_scope.profiling import (
    READS,
    STORES,
    TIME,
    WRITES,
    diagnose,
    parse_counters,
    profile_run,
)
from perf_scope.regression import Comparison
from perf_scope.roofline import roofline_points

CASE = Case(
    "transpose",
    "transpose",
    (32, 32),
    "fp32",
    Metric.LATENCY,
    Variant("tiled", ("bench", "transpose", "tiled")),
    Variant("naive", ("bench", "transpose", "naive")),
)


class ProfilingTest(unittest.TestCase):
    def test_reads_raw_and_long_csv(self):
        wide = (
            f'==PROF== Connected\n"ID","{TIME}","{READS}","{WRITES}","{STORES}"\n'
            '"","nsecond","byte","byte","sector"\n"0","2,500","100","200","64"\n'
        )
        long = (
            f"Metric Name,Metric Unit,Metric Value\n{TIME},usecond,2.5\n"
            f"{READS},byte,100\n{WRITES},byte,200\n{STORES},sector,64\n"
        )
        wanted = {TIME: 2500, READS: 100, WRITES: 200, STORES: 64}
        self.assertEqual(parse_counters(wide), wanted)
        self.assertEqual(parse_counters(long), wanted)
        with self.assertRaisesRegex(ValueError, "lacks duration"):
            parse_counters("==ERROR== ERR_NVGPUCTRPERM")

    def test_diagnoses_comparable_store_traffic(self):
        baseline = {"status": "ok", "counters": {STORES: 64}}
        current = {"status": "ok", "counters": {STORES: 100}}
        self.assertEqual(
            diagnose(CASE, [baseline, current])["cause"],
            "increased global store sector traffic",
        )
        self.assertEqual(
            diagnose(CASE, [baseline, {"status": "unavailable"}])["cause"],
            "undetermined",
        )

    def test_missing_profiler_preserves_comparison(self):
        suite = Suite(42, 8, 5, 20, 120, "measured", 0.05, (CASE,))
        decision = Comparison(
            CASE.id, Metric.LATENCY, "tiled", "naive", 0.05, "regression", device="x"
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("perf_scope.profiling.shutil.which", return_value=None),
        ):
            report = profile_run(suite, Path(tmp), [decision], "regressions")
            self.assertEqual(report["diagnoses"][0]["cause"], "undetermined")
            self.assertEqual(report["profiles"][0]["status"], "unavailable")
            self.assertTrue((Path(tmp) / "profiles.json").is_file())
        self.assertEqual(decision.status, "regression")

    def test_roofline_uses_profile_traffic_and_duration(self):
        case = Case(
            "gemm",
            "gemm",
            (4, 8, 16),
            "fp32",
            Metric.LATENCY,
            CASE.baseline,
            CASE.candidate,
        )
        suite = Suite(42, 8, 5, 20, 120, "fixture", 0.05, (case,))
        profile = {
            "case": "gemm",
            "variant": "naive",
            "status": "ok",
            "device": "x",
            "counters": {TIME: 100, READS: 128, WRITES: 128},
        }
        (point,) = roofline_points(
            suite, [profile, profile | {"device": "y"}], "x", 50, 3
        )
        self.assertEqual(point["intensity_flops_per_byte"], 4)
        self.assertAlmostEqual(point["achieved_gflops"], 10.24)
        self.assertEqual(point["ceiling_gflops"], 12)
