import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from perf_scope.config import Case, Suite, Variant
from perf_scope.regression import Comparison
from perf_scope.runner import _text, command_for, run_command, write_json

TIME = "gpu__time_duration.sum"
READS = "dram__bytes_read.sum"
WRITES = "dram__bytes_write.sum"
STORES = "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
UNITS = {
    TIME: {"nsecond": 1, "ns": 1, "usecond": 1e3, "us": 1e3, "msecond": 1e6, "ms": 1e6},
    READS: {"byte": 1, "bytes": 1, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9},
    WRITES: {"byte": 1, "bytes": 1, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9},
    STORES: {"": 1, "sector": 1, "sectors": 1},
}


def _scaled(metric: str, value: str, unit: str) -> float:
    try:
        result = float(value.replace(",", "")) * UNITS[metric][unit.strip().lower()]
    except (KeyError, ValueError) as error:
        raise ValueError(f"unusable Nsight value for {metric}") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"unusable Nsight value for {metric}")
    return result


def parse_counters(export: str) -> dict[str, float]:
    rows = list(csv.reader(io.StringIO(export)))
    values = {}
    for index, header in enumerate(rows):
        if TIME in header and index + 2 < len(rows):
            units, samples = rows[index + 1 : index + 3]
            if len(header) != len(units) or len(header) != len(samples):
                raise ValueError("incomplete Nsight raw CSV row")
            fields = zip(header, units, samples, strict=True)
        elif {"Metric Name", "Metric Unit", "Metric Value"}.issubset(header):
            fields = (
                tuple(
                    row[header.index(column)]
                    for column in ("Metric Name", "Metric Unit", "Metric Value")
                )
                for row in rows[index + 1 :]
                if len(row) == len(header)
            )
        else:
            continue
        for metric, unit, value in fields:
            if metric in UNITS and value not in ("", "N/A"):
                if metric in values:
                    raise ValueError("Nsight CSV contains multiple profiled launches")
                values[metric] = _scaled(metric, value, unit)
        break
    if values.get(TIME, 0) <= 0 or READS not in values or WRITES not in values:
        raise ValueError("Nsight export lacks duration or DRAM counters")
    return values


def _kernel(case: Case, variant: Variant) -> str | None:
    if len(variant.command) < 3 or variant.command[-2] != case.workload:
        return None
    implementation = variant.command[-1]
    if case.workload in ("transpose", "gemm") and implementation in ("naive", "tiled"):
        return f"{case.workload}_{implementation}"
    if case.workload == "triad" and implementation == "streaming":
        return "stream_triad"
    return None


def _device_from_output(output: str) -> str | None:
    for line in output.splitlines():
        if line.lstrip().startswith("{"):
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if (
                isinstance(payload, dict)
                and "device" in payload
                and payload.get("correct") is True
            ):
                return payload["device"]
    return None


def _collect(
    executable: str,
    suite: Suite,
    case: Case,
    side: str,
    variant: Variant,
    expected_device: str,
    root: Path,
) -> dict:
    kernel = _kernel(case, variant)
    profile = {
        "case": case.id,
        "side": side,
        "variant": variant.id,
        "device": expected_device,
        "status": "unavailable",
        "wall_s": 0.0,
    }
    if kernel is None:
        return profile | {"reason": "variant has no single-kernel profile target"}
    folder = root / "profiles" / case.id
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / f"{side}.csv"
    metrics = [TIME, READS, WRITES]
    if case.workload == "transpose":
        metrics.append(STORES)
    command = [
        executable,
        "--csv",
        "--page",
        "raw",
        "--print-units",
        "base",
        "--clock-control",
        "none",
        "--cache-control",
        "none",
        "--kernel-name-base",
        "function",
        "--kernel-name",
        f"regex:{kernel}",
        "--launch-skip",
        str(suite.warmups),
        "--launch-count",
        "1",
        "--metrics",
        ",".join(metrics),
        "--log-file",
        str(raw.resolve()),
        *command_for(case, variant, replace(suite, iterations=1)),
    ]
    profile["argv"] = command
    started = time.monotonic()
    stdout = stderr = ""
    try:
        completed = run_command(
            command,
            timeout=max(suite.timeout_s, 120),
            env={**os.environ, "LC_ALL": "C"},
        )
        stdout, stderr = completed.stdout, completed.stderr
        export = (
            raw.read_text(encoding="utf-8", errors="replace") if raw.is_file() else ""
        )
        if "ERR_NVGPUCTRPERM" in export + stdout + stderr:
            raise ValueError(
                "NVIDIA performance counters are unavailable to this process"
            )
        if completed.returncode:
            raise ValueError(f"ncu exited with status {completed.returncode}")
        if _device_from_output(stdout + "\n" + export) != expected_device:
            raise ValueError("profile target did not validate on the measured device")
        profile.update(status="ok", counters=parse_counters(export))
    except subprocess.TimeoutExpired as error:
        stdout, stderr = _text(error.stdout), _text(error.stderr)
        profile["reason"] = "ncu timed out"
    except (ValueError, OSError, csv.Error) as error:
        profile["reason"] = str(error)
    profile["wall_s"] = time.monotonic() - started
    (folder / f"{side}.stdout").write_text(stdout, encoding="utf-8")
    (folder / f"{side}.stderr").write_text(stderr, encoding="utf-8")
    if raw.is_file():
        profile["raw"] = raw.relative_to(root).as_posix()
    return profile


def parse_launches(export: str) -> dict[str, int]:
    rows = list(csv.reader(io.StringIO(export)))
    launches: dict[str, int] = {}
    for index, header in enumerate(rows):
        if "Instances" not in header or "Name" not in header:
            continue
        for row in rows[index + 1 :]:
            if len(row) != len(header):
                continue
            match = re.search(
                r"\bsoftmax_(?:fused|reduce_max|exp_sum|normalize)\b",
                row[header.index("Name")],
            )
            if match:
                count = int(row[header.index("Instances")].replace(",", ""))
                if count < 1:
                    raise ValueError("invalid CUDA kernel instance count")
                kernel = match.group()
                launches[kernel] = launches.get(kernel, 0) + count
        break
    if not launches:
        raise ValueError("Nsight Systems export lacks softmax kernel launches")
    return launches


def _trace_softmax(
    executable: str,
    suite: Suite,
    case: Case,
    side: str,
    variant: Variant,
    expected_device: str,
    root: Path,
) -> dict:
    profile = {
        "case": case.id,
        "side": side,
        "variant": variant.id,
        "device": expected_device,
        "status": "unavailable",
    }
    folder = root / "profiles" / case.id
    folder.mkdir(parents=True, exist_ok=True)
    prefix = folder / side
    trace = prefix.with_suffix(".nsys-rep")
    raw = prefix.with_suffix(".csv")
    command = [
        executable,
        "profile",
        "--trace=cuda",
        "--sample=none",
        "--cpuctxsw=none",
        "--stats=false",
        "--force-overwrite=true",
        "-o",
        str(prefix.resolve()),
        *command_for(case, variant, replace(suite, warmups=0, iterations=1)),
    ]
    stats_command = [
        executable,
        "stats",
        "--report",
        "cuda_gpu_kern_sum",
        "--format",
        "csv",
        "--output",
        "-",
        str(trace.resolve()),
    ]
    profile.update(argv=command, stats_argv=stats_command)
    started = time.monotonic()
    stdout = stderr = ""
    try:
        capture = run_command(
            command,
            timeout=max(suite.timeout_s, 120),
            env={**os.environ, "LC_ALL": "C"},
        )
        stdout, stderr = capture.stdout, capture.stderr
        if capture.returncode or not trace.is_file():
            raise ValueError(f"nsys capture failed with status {capture.returncode}")
        if _device_from_output(stdout + "\n" + stderr) != expected_device:
            raise ValueError("trace target did not validate on the measured device")
        stats = run_command(
            stats_command,
            timeout=120,
            env={**os.environ, "LC_ALL": "C"},
        )
        stderr += stats.stderr
        raw.write_text(stats.stdout, encoding="utf-8")
        if stats.returncode:
            raise ValueError(f"nsys report failed with status {stats.returncode}")
        profile.update(status="ok", launches=parse_launches(stats.stdout))
    except subprocess.TimeoutExpired as error:
        stdout += _text(error.stdout)
        stderr += _text(error.stderr)
        profile["reason"] = "nsys timed out"
    except (ValueError, OSError, csv.Error) as error:
        profile["reason"] = str(error)
    profile["wall_s"] = time.monotonic() - started
    prefix.with_suffix(".stdout").write_text(stdout, encoding="utf-8")
    prefix.with_suffix(".stderr").write_text(stderr, encoding="utf-8")
    if raw.is_file():
        profile["raw"] = raw.relative_to(root).as_posix()
    if trace.is_file():
        profile["trace"] = trace.relative_to(root).as_posix()
    return profile


def diagnose(case: Case, profiles: list[dict]) -> dict:
    result = {"case": case.id, "cause": "undetermined"}
    if len(profiles) != 2 or any(row["status"] != "ok" for row in profiles):
        return result
    if case.workload == "transpose":
        base, current = [row["counters"].get(STORES) for row in profiles]
        if (
            base is not None
            and current is not None
            and base > 0
            and current > base * 1.10
        ):
            return result | {
                "cause": "increased global store sector traffic",
                "evidence": {
                    "baseline_store_sectors": base,
                    "candidate_store_sectors": current,
                },
            }
    if case.workload == "softmax":
        baseline, candidate = [sum(row["launches"].values()) for row in profiles]
        if candidate > baseline:
            return result | {
                "cause": "increased CUDA kernel launches",
                "evidence": {
                    "baseline_launches": baseline,
                    "candidate_launches": candidate,
                },
            }
    return result


def profile_run(
    suite: Suite,
    root: Path,
    comparisons: list[Comparison],
    mode: str,
) -> dict:
    results = {
        "schema_version": 1,
        "source": suite.source,
        "profiles": [],
        "diagnoses": [],
    }
    if mode == "off" or suite.source != "measured":
        return results
    executables = {}
    for case, comparison in zip(suite.cases, comparisons, strict=True):
        if comparison.device is None or (
            mode == "regressions" and comparison.status != "regression"
        ):
            continue
        collector = _trace_softmax if case.workload == "softmax" else _collect
        tool = "nsys" if case.workload == "softmax" else "ncu"
        if tool not in executables:
            executables[tool] = shutil.which(tool)
        executable = executables[tool]
        if executable is None:
            results["profiles"].append(
                {
                    "case": case.id,
                    "status": "unavailable",
                    "reason": f"{tool} executable was not found",
                }
            )
            if comparison.status == "regression":
                results["diagnoses"].append({"case": case.id, "cause": "undetermined"})
            continue
        profiles = [
            collector(executable, suite, case, side, variant, comparison.device, root)
            for side, variant in (
                ("baseline", case.baseline),
                ("candidate", case.candidate),
            )
        ]
        results["profiles"].extend(profiles)
        if comparison.status == "regression":
            results["diagnoses"].append(diagnose(case, profiles))
    if results["profiles"]:
        for tool, executable in executables.items():
            if executable is None:
                continue
            try:
                version = subprocess.run(
                    [executable, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                results[f"{tool}_version"] = version.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                results[f"{tool}_version"] = None
        write_json(root / "profiles.json", results)
    return results
