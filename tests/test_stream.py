import contextlib
import io
import unittest
from unittest.mock import patch

from perf_scope.stream import main, parse_triad

LOG = """\
NVIDIA-STREAM 26.2.1
Device Selected 0: "Fixture GPU"
Array size (float)=2048*4*8
Function   Rate (MB/s)   Avg time    Min time   Max time
Triad:     50,125.5      0.01        0.01       0.01
"""


class StreamTest(unittest.TestCase):
    def test_parses_triad_and_decimal_units(self):
        bandwidth, release = parse_triad(LOG, 2048, "26.02.01")
        self.assertAlmostEqual(bandwidth, 50.1255)
        self.assertEqual(release, "26.2.1")

    def test_rejects_invalid_upstream_results(self):
        for output in (
            LOG.replace("Device Selected 0", "Device Selected 1"),
            LOG.replace("2048*4", "4096*4"),
            LOG.replace("26.2.1", "25.9.0"),
            LOG.replace("MB/s", "GB/s"),
            LOG.replace("50,125.5", "nan"),
            LOG + "Triad: 1000 0.01 0.01 0.01\n",
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_triad(output, 2048, "26.02.01")

    def test_missing_launcher_is_a_readable_error(self):
        argv = [
            "--release",
            "26.02.01",
            "--shape",
            "2048",
            "--dtype",
            "fp32",
            "--seed",
            "42",
            "--warmups",
            "0",
            "--iterations",
            "1",
        ]
        with (
            patch("perf_scope.stream.shutil.which", return_value=None),
            contextlib.redirect_stderr(io.StringIO()) as error,
        ):
            self.assertEqual(main(argv), 1)
        self.assertIn("STREAM launcher is unavailable", error.getvalue())
