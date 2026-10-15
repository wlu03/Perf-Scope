import contextlib
import io
import unittest

from perf_scope.resnet import _arguments

ARGUMENTS = [
    "--shape",
    "8",
    "--dtype",
    "fp32",
    "--seed",
    "12",
    "--warmups",
    "10",
    "--iterations",
    "25",
]


class ResNetArgumentsTest(unittest.TestCase):
    def test_accepts_valid_batch_and_cache_policy(self):
        args = _arguments(["inductor", "--cold-cache", *ARGUMENTS])
        self.assertEqual(args.shape, 8)
        self.assertTrue(args.cold_cache)
        self.assertEqual(args.iterations, 25)

    def test_rejects_invalid_batch_precision_and_cache_policy(self):
        for arguments in (
            ["eager", "--cold-cache", *ARGUMENTS],
            ["inductor", *ARGUMENTS, "--shape", "0"],
            ["inductor", *ARGUMENTS, "--dtype", "fp16"],
            ["inductor", *ARGUMENTS, "--seed", str(2**64)],
        ):
            with (
                self.subTest(arguments=arguments),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    _arguments(arguments)
