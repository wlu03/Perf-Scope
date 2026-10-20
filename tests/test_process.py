import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from perf_scope.runner import run_command


class ProcessTest(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    def test_timeout_terminates_subprocess_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "child-survived"
            grandchild = "import pathlib,sys,time; time.sleep(2); pathlib.Path(sys.argv[1]).touch()"
            parent = (
                "import subprocess,sys; "
                "child = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
                "print('started', flush=True); child.wait()"
            )
            with self.assertRaises(subprocess.TimeoutExpired) as error:
                run_command(
                    [sys.executable, "-c", parent, grandchild, str(marker)], timeout=1
                )
            self.assertIn(b"started", error.exception.stdout)
            time.sleep(1.25)
            self.assertFalse(marker.exists())
