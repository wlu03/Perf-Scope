import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from perf_scope.cli import main
from perf_scope.config import ConfigError, Metric, load_suite

SUITE = """\
version = 1
seed = 16
source = "fixture"

[[cases]]
id = "partial-tile"
workload = "transpose"
shape = [67, 99]

[cases.baseline]
id = "simple"
command = ["perf-scope-cuda", "transpose", "naive"]

[cases.candidate]
id = "shared"
command = ["perf-scope-cuda", "transpose", "tiled"]
"""


class SuiteTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "suite.toml"

    def test_valid_suite_with_defaults(self):
        self.path.write_text(SUITE)
        suite = load_suite(self.path)
        self.assertEqual(suite.seed, 16)
        self.assertEqual(suite.pairs, 20)
        self.assertEqual(suite.source, "fixture")
        (case,) = suite.cases
        self.assertEqual(case.shape, (67, 99))
        self.assertEqual(case.dtype, "fp32")
        self.assertEqual(case.metric, Metric.LATENCY)
        self.assertEqual(case.candidate.command[-1], "tiled")

    def test_rejects_invalid_suite(self):
        for invalid, message in (
            (SUITE.replace("version = 1", "version = 2"), "version 1"),
            (SUITE.replace("seed = 16", "seed = true"), "seed must be"),
            (SUITE.replace("[67, 99]", "[0, 99]"), "shape must be"),
            (SUITE.replace('id = "partial-tile"', 'id = "../outside"'), "must start"),
            (SUITE.replace('id = "shared"', 'id = "simple"'), "distinct"),
            (SUITE.replace('source = "fixture"', 'source = "unknown"'), "source must"),
            (SUITE.replace("shape = [67, 99]", 'shape = [67, 99]\nmetric = "seconds"'), "metric"),
            (SUITE.replace("shape = [67, 99]", "shape = [67, 99]\niteratons = 10"), "unknown keys"),
            (SUITE.replace('["perf-scope-cuda", "transpose", "tiled"]', "[]"), "argument array"),
            (SUITE + SUITE[SUITE.index("[[cases]]"):], "unique"),
        ):
            with self.subTest(message=message):
                self.path.write_text(invalid)
                with self.assertRaisesRegex(ConfigError, message):
                    load_suite(self.path)

    def test_validation_exit_status(self):
        self.path.write_text(SUITE)
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(["validate", str(self.path)]), 0)
        self.assertIn("Validated suite", stream.getvalue())
        self.path.write_text("unparseable")
        with contextlib.redirect_stderr(io.StringIO()) as stream:
            self.assertEqual(main(["validate", str(self.path)]), 2)
        self.assertIn(str(self.path), stream.getvalue())
