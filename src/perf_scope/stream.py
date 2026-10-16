import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from uuid import UUID


def _gpu_identity() -> str:
    driver = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    uuid = (ctypes.c_ubyte * 16)()
    for name, arguments in (
        ("cuInit", (0,)),
        ("cuDeviceGet", (ctypes.byref(device), 0)),
        ("cuDeviceGetUuid_v2", (ctypes.byref(uuid), device)),
    ):
        status = getattr(driver, name)(*arguments)
        if status != 0:
            raise RuntimeError(f"{name} failed with CUDA status {status}")
    return f"GPU-{UUID(bytes=bytes(uuid))}"


def parse_triad(output: str, elements: int, release: str) -> tuple[float, str]:
    version = re.search(
        r"^\s*NVIDIA-STREAM\s+(\d+\.\d+(?:\.\d+)?)\b", output, re.MULTILINE
    )
    if version is None:
        raise ValueError("STREAM output lacks an NVIDIA-STREAM version")
    actual = version.group(1)
    if tuple(map(int, actual.split(".")[:2])) != tuple(
        map(int, release.split(".")[:2])
    ):
        raise ValueError(
            f"STREAM version {actual} differs from package release {release}"
        )
    devices = re.findall(r"^\s*Device Selected\s+(\d+)\s*:", output, re.MULTILINE)
    if devices != ["0"]:
        raise ValueError("STREAM output must identify CUDA device 0")
    shapes = re.findall(r"Array size\s*\(float\)\s*=\s*([\d,]+)\s*\*\s*4\b", output)
    if len(shapes) != 1 or int(shapes[0].replace(",", "")) != elements:
        raise ValueError("STREAM output does not match the requested FP32 array size")
    table = re.search(r"^\s*Function\s+[^\n]*MB/s", output, re.MULTILINE)
    if table is None:
        raise ValueError("STREAM output lacks an MB/s rate table")
    rows = re.findall(r"^\s*Triad:\s+(\S+)", output[table.end() :], re.MULTILINE)
    if len(rows) != 1:
        raise ValueError("STREAM output must contain one TRIAD measurement")
    bandwidth = float(rows[0].replace(",", "")) / 1000
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("STREAM TRIAD bandwidth must be finite and positive")
    return bandwidth, actual


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m perf_scope.stream")
    parser.add_argument("--launcher", default="stream-gpu-test.sh")
    parser.add_argument("--release", required=True)
    parser.add_argument("--shape", type=int, metavar="ELEMENTS", required=True)
    parser.add_argument("--dtype", choices=("fp32",), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", args.release):
        parser.error("--release must identify a numeric NVIDIA HPC Benchmarks release")
    if args.shape < 1 or args.iterations < 1 or args.warmups < 0 or args.seed < 0:
        parser.error(
            "shape and iterations must be positive and seed and warmups must be nonnegative"
        )
    return args


def _benchmark(args: argparse.Namespace) -> dict:
    found = shutil.which(args.launcher)
    if found is None:
        raise ValueError(f"STREAM launcher is unavailable: {args.launcher}")
    launcher = Path(found).resolve()
    device = _gpu_identity()
    bandwidths = []
    version = None
    for attempt in range(args.warmups + args.iterations):
        result = subprocess.run(
            [
                str(launcher),
                "--d",
                "0",
                "--n",
                str(args.shape),
                "--dt",
                "fp32",
                "--t",
                "T",
            ],
            cwd=launcher.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env={**os.environ, "LC_ALL": "C"},
        )
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode:
            raise RuntimeError(f"NVIDIA STREAM exited with status {result.returncode}")
        value, actual = parse_triad(result.stdout, args.shape, args.release)
        if version is not None and actual != version:
            raise ValueError("NVIDIA STREAM version changed across invocations")
        version = actual
        if attempt >= args.warmups:
            bandwidths.append(value)
    return {
        "device": device,
        "correct": True,
        "metrics": {"bandwidth_gb_s": statistics.median(bandwidths)},
        "bandwidth_samples_gb_s": bandwidths,
        "timing_scope": "stream_triad_reported_rate",
        "validation": "launcher_exit_and_output",
        "package_release": args.release,
        "stream_version": version,
        "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        print(json.dumps(_benchmark(args), allow_nan=False))
    except (
        AttributeError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.TimeoutExpired,
    ) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
