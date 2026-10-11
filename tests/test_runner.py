import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from perf_scope.cli import main

ADAPTER = Path(__file__).with_name("fixtures") / "adapter.py"


class RunnerTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workspace = Path(directory.name)
        self.config = self.workspace / "case.toml"

    def execute(self, candidate="ok", pairs=1, fail_on_regression=False):
        baseline_argv = json.dumps([sys.executable, str(ADAPTER.resolve()), "ok"])
        candidate_argv = json.dumps([sys.executable, str(ADAPTER.resolve()), candidate])
        self.config.write_text(f"""\
version = 1
source = "fixture"
seed = 13
pairs = {pairs}
timeout_s = 1

[[cases]]
id = "edge-shape"
workload = "transpose"
shape = [65, 31]

[cases.baseline]
id = "base"
command = {baseline_argv}

[cases.candidate]
id = "candidate"
command = {candidate_argv}
""")
        command = ["run", str(self.config), "--output", str(self.workspace / "runs")]
        if fail_on_regression:
            command.append("--fail-on-regression")
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            status = main(command)
        output = Path(stream.getvalue().strip())
        manifest = json.loads((output / "manifest.json").read_text())
        rows = [
            json.loads(line)
            for line in (output / "measurements.jsonl").read_text().splitlines()
        ]
        return status, output, manifest, rows

    def test_regression_exit_and_report_replay(self):
        status, output, _, _ = self.execute(
            candidate="slow", pairs=8, fail_on_regression=True
        )
        self.assertEqual(status, 3)
        decisions = json.loads((output / "comparisons.json").read_text())
        self.assertEqual(decisions["source"], "fixture")
        self.assertEqual(decisions["comparisons"][0]["status"], "regression")
        self.assertIn("<strong>fixture</strong>", (output / "report.html").read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["compare", str(output), "--fail-on-regression"]), 3)
        self.assertEqual(
            json.loads((output / "comparisons.json").read_text()), decisions
        )

    def test_balanced_reproducible_pairs(self):
        status, output, manifest, rows = self.execute(pairs=4)
        self.assertEqual(status, 0)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["source"], "fixture")
        first_sides = [row["side"] for row in rows if row["order"] == 0]
        self.assertEqual(first_sides.count("baseline"), 2)
        self.assertEqual(len(rows), 8)
        for i, row in enumerate(rows):
            self.assertEqual(row["source"], "fixture")
            self.assertEqual(row["pair"], i // 2)
            self.assertEqual(row["value"], 1.5)
            self.assertEqual(row["measurement"]["args"]["shape"], "65,31")
            self.assertEqual(row["measurement"]["args"]["seed"], 13)
        _, second_output, _, second_rows = self.execute(pairs=4)
        self.assertNotEqual(output, second_output)
        self.assertEqual(
            first_sides, [row["side"] for row in second_rows if row["order"] == 0]
        )

    def test_persists_failed_samples_and_logs(self):
        for mode, expected in (
            ("incorrect", "incorrect"),
            ("garbled", "invalid_output"),
            ("timeout", "timeout"),
        ):
            with self.subTest(mode=mode):
                status, output, manifest, rows = self.execute(candidate=mode)
                self.assertEqual(status, 1)
                self.assertEqual(manifest["status"], "invalid")
                self.assertEqual(manifest["failed_measurements"], 1)
                candidate = next(row for row in rows if row["side"] == "candidate")
                self.assertEqual(candidate["status"], expected)
                if mode == "timeout":
                    self.assertIn(
                        "timeout fixture started",
                        (output / candidate["stderr"]).read_text(),
                    )
