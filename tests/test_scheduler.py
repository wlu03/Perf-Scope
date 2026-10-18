import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from test_regression import CASE, SUITE, rows_for

from perf_scope.cli import main
from perf_scope.profiling import STORES
from perf_scope.scheduler import evaluate

FAULT = replace(CASE, id="fault", workload="transpose", shape=(32, 32))
CONTROL = replace(FAULT, id="control")
CAUSE = "increased global store sector traffic"


def bundle(path, pairs, first, second):
    path.mkdir()
    suite = replace(SUITE, pairs=pairs, cases=(FAULT, CONTROL))
    manifest = {
        "schema_version": 1,
        "source": "fixture",
        "status": "complete",
        "started_at": path.name,
        "suite": asdict(suite),
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    rows = rows_for(FAULT, [100] * pairs, first) + rows_for(
        CONTROL, [100] * pairs, second
    )
    (path / "measurements.jsonl").write_text(
        "".join(json.dumps(row | {"wall_s": 0.05}) + "\n" for row in rows)
    )
    profiles = [
        {
            "case": "fault",
            "side": side,
            "device": "fixture-device",
            "status": "ok",
            "counters": {STORES: sectors},
            "wall_s": 0.1,
        }
        for side, sectors in (("baseline", 100), ("candidate", 200))
    ]
    (path / "profiles.json").write_text(
        json.dumps({"source": "fixture", "profiles": profiles})
    )


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.reference = self.root / "reference"
        self.evaluation = self.root / "evaluation"
        self.report = self.root / "report.json"
        bundle(self.reference, 16, [125] * 16, [100] * 16)

    def test_cli_scores_budget_and_diagnosis(self):
        bundle(self.evaluation, 10, [125] * 10, [100] * 10)
        expected = self.root / "expected.json"
        expected.write_text(json.dumps({"fault": CAUSE}))
        with contextlib.redirect_stdout(io.StringIO()):
            status = main(
                [
                    "evaluate",
                    str(self.reference),
                    str(self.evaluation),
                    "--budget-s",
                    "1.8",
                    "--screen-pairs",
                    "2",
                    "--confirm-pairs",
                    "8",
                    "--expected-causes",
                    str(expected),
                    "--output",
                    str(self.report),
                ]
            )
        self.assertEqual(status, 0)
        report = json.loads(self.report.read_text())
        self.assertEqual(report["source"], "fixture")
        exhaustive, fixed, adaptive = report["policies"]
        self.assertEqual(exhaustive["recall"], 1)
        self.assertEqual(adaptive["recall"], 1)
        self.assertEqual(adaptive["diagnosis_accuracy"], 1)
        self.assertLessEqual(fixed["replayed_wall_s"], 1.8)
        self.assertLessEqual(adaptive["replayed_wall_s"], 1.8)

    def test_ranking_uses_screening_only(self):
        bundle(self.evaluation, 10, [90, 90] + [125] * 8, [150, 150] + [100] * 8)
        evaluate(self.reference, self.evaluation, self.report, 1.8, 2, 8, set(), {})
        exhaustive, _, adaptive = json.loads(self.report.read_text())["policies"]
        self.assertEqual(exhaustive["recall"], 1)
        self.assertEqual(adaptive["recall"], 0)
        self.assertEqual(adaptive["cases"][0]["case"], "control")
        self.assertEqual(adaptive["cases"][0]["decision"], "within_threshold")

    def test_rejects_reusing_reference_run(self):
        with self.assertRaisesRegex(ValueError, "independent runs"):
            evaluate(self.reference, self.reference, self.report, 1.8, 2, 8, set(), {})
