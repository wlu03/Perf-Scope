import json
import math
import os
import platform
import random
import subprocess
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from perf_scope.config import Case, Suite, Variant


def command_for(case: Case, variant: Variant, suite: Suite) -> list[str]:
    return [
        *variant.command,
        "--shape",
        ",".join(str(size) for size in case.shape),
        "--dtype",
        case.dtype,
        "--seed",
        str(suite.seed),
        "--warmups",
        str(suite.warmups),
        "--iterations",
        str(suite.iterations),
    ]


def _revision() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "--no-optional-locks", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(changes)}


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _measurement(output: str, case: Case) -> tuple[dict, float]:
    payload = json.loads(output)
    if not isinstance(payload, dict) or type(payload.get("correct")) is not bool:
        raise ValueError("expected an object with a Boolean correctness result")
    device = payload.get("device")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("expected a nonempty device identifier")
    metrics = payload.get("metrics")
    value = metrics.get(case.metric) if isinstance(metrics, dict) else None
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"expected a finite positive {case.metric} measurement")
    json.dumps(payload, allow_nan=False)
    return payload, float(value)


def _invoke(
    suite: Suite,
    case: Case,
    side: str,
    variant: Variant,
    pair: int,
    order: int,
    directory: Path,
) -> dict:
    argv = command_for(case, variant, suite)
    started = time.monotonic()
    stdout, stderr = "", ""
    row = {
        "source": suite.source,
        "case": case.id,
        "pair": pair,
        "order": order,
        "side": side,
        "variant": variant.id,
        "metric": case.metric,
        "argv": argv,
        "iterations": suite.iterations,
    }
    try:
        process = subprocess.run(
            argv,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=suite.timeout_s,
        )
        stdout, stderr = process.stdout, process.stderr
    except subprocess.TimeoutExpired as error:
        stdout, stderr = _text(error.stdout), _text(error.stderr)
        row.update(
            status="timeout", error=f"process exceeded {suite.timeout_s} seconds"
        )
    except OSError as error:
        row.update(status="unavailable", error=str(error))
    else:
        row["returncode"] = process.returncode
        if process.returncode:
            row.update(
                status="process_error",
                error=f"process exited with status {process.returncode}",
            )
        else:
            try:
                payload, value = _measurement(stdout, case)
            except (ValueError, OverflowError) as error:
                row.update(status="invalid_output", error=str(error))
            else:
                row.update(measurement=payload, value=value, device=payload["device"])
                if payload["correct"]:
                    row["status"] = "ok"
                else:
                    row.update(
                        status="incorrect", error="adapter reported incorrect output"
                    )
    row["wall_s"] = time.monotonic() - started
    stem = directory / "logs" / case.id / f"{pair:04d}-{side}"
    for suffix, content in (("stdout", stdout), ("stderr", stderr)):
        path = stem.with_suffix(f".{suffix}")
        path.write_text(content, encoding="utf-8")
        row[suffix] = path.relative_to(directory).as_posix()
    return row


def _order(suite: Suite, case: Case) -> list[bool]:
    rng = random.Random(f"{suite.seed}:{case.id}")
    reversed_pairs = [False, True] * (suite.pairs // 2)
    if suite.pairs % 2:
        reversed_pairs.append(rng.choice([False, True]))
    rng.shuffle(reversed_pairs)
    return reversed_pairs


def run_suite(suite: Suite, source: Path, output: Path) -> tuple[Path, int]:
    started = datetime.now(UTC)
    manifest = {
        "schema_version": 1,
        "source": suite.source,
        "suite_path": str(source.resolve()),
        "cwd": str(Path.cwd()),
        "suite": asdict(suite),
        "started_at": started.isoformat(),
        "revision": _revision(),
        "host": {
            "os": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "status": "running",
    }
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(
            prefix=started.strftime("%Y%m%dT%H%M%SZ-"), dir=output.resolve()
        )
    )
    write_json(directory / "manifest.json", manifest)
    failures = 0
    with (directory / "measurements.jsonl").open("w", encoding="utf-8") as stream:
        for case in suite.cases:
            (directory / "logs" / case.id).mkdir(parents=True)
            variants = [("baseline", case.baseline), ("candidate", case.candidate)]
            for pair, reverse in enumerate(_order(suite, case)):
                order = variants[::-1] if reverse else variants
                for index, (side, variant) in enumerate(order):
                    row = _invoke(suite, case, side, variant, pair, index, directory)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    failures += row["status"] != "ok"
    manifest.update(
        status="invalid" if failures else "complete",
        failed_measurements=failures,
        finished_at=datetime.now(UTC).isoformat(),
    )
    write_json(directory / "manifest.json", manifest)
    return directory, failures
