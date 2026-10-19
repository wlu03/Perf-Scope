import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from test_runner import ADAPTER

from perf_scope.cli import main


class CampaignTest(unittest.TestCase):
    def test_records_independent_runs_and_replays_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "suite.toml"
            program = [sys.executable, str(ADAPTER.resolve())]
            config.write_text(f"""\
version = 1
source = "fixture"
[[cases]]
id = "fault"
workload = "transpose"
shape = [32, 32]
[cases.baseline]
id = "base"
command = {json.dumps([*program, "ok"])}
[cases.candidate]
id = "slow"
command = {json.dumps([*program, "slow"])}
""")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(
                    [
                        "campaign",
                        str(config),
                        "--budget-s",
                        "30",
                        "--observation-pairs",
                        "10",
                        "--reference-pairs",
                        "12",
                        "--screen-pairs",
                        "2",
                        "--confirm-pairs",
                        "8",
                        "--output",
                        str(root / "runs"),
                    ]
                )
            self.assertEqual(status, 0)
            campaign = Path(output.getvalue().strip())
            state = json.loads((campaign / "campaign.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertNotEqual(state["observations"], state["reference"])
            for stage in ("observations", "reference"):
                self.assertTrue((campaign / state[stage] / "report.html").is_file())
            scores = json.loads((campaign / state["evaluation"]).read_text())
            self.assertEqual(scores["source"], "fixture")
            self.assertEqual(scores["policies"][0]["recall"], 1)
