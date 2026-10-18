import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from perf_scope.config import Case, Metric, Suite, parse_suite
from perf_scope.runner import write_json

MIN_PAIRS = 8
BOOTSTRAP_DRAWS = 4000


@dataclass(frozen=True)
class Comparison:
    case: str
    metric: Metric
    baseline: str
    candidate: str
    margin: float
    status: str
    pairs: int = 0
    device: str | None = None
    baseline_median: float | None = None
    candidate_median: float | None = None
    slowdown: float | None = None
    lower: float | None = None
    upper: float | None = None
    reason: str | None = None


def _paired_values(
    suite: Suite, case: Case, rows: list[dict]
) -> tuple[list[float], list[float], str]:
    if len(rows) != 2 * suite.pairs:
        raise ValueError(
            f"expected {2 * suite.pairs} observations, received {len(rows)}"
        )
    pairs: dict[tuple[int, str], float] = {}
    devices = set()
    variants = {"baseline": case.baseline.id, "candidate": case.candidate.id}
    for row in rows:
        if row.get("status") != "ok":
            raise ValueError(
                f"{row.get('variant')}: {row.get('error', 'measurement failed')}"
            )
        side = row.get("side")
        index = row.get("pair")
        if (
            type(index) is not int
            or not 0 <= index < suite.pairs
            or not isinstance(side, str)
            or side not in variants
            or row.get("variant") != variants[side]
            or row.get("metric") != case.metric
            or row.get("source") != suite.source
        ):
            raise ValueError("measurement identity differs from the resolved suite")
        if (index, side) in pairs:
            raise ValueError("duplicate observation for a pair and side")
        value = row.get("value")
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("measurement must be finite and positive")
        device = row.get("device")
        if not isinstance(device, str) or not device:
            raise ValueError("measurement lacks a device identifier")
        devices.add(device)
        pairs[index, side] = float(value)
    if len(devices) != 1:
        raise ValueError("paired observations must use the same device")
    return (
        [pairs[index, "baseline"] for index in range(suite.pairs)],
        [pairs[index, "candidate"] for index in range(suite.pairs)],
        devices.pop(),
    )


def _quantile(sorted_values: list[float], probability: float) -> float:
    index = (len(sorted_values) - 1) * probability
    start = math.floor(index)
    end = math.ceil(index)
    return sorted_values[start] + (index - start) * (
        sorted_values[end] - sorted_values[start]
    )


def _interval(values: list[float], seed: str) -> tuple[float, float]:
    rng = random.Random(seed)
    bootstrap = sorted(
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_DRAWS)
    )
    return _quantile(bootstrap, 0.025), _quantile(bootstrap, 0.975)


def paired_ratios(
    case: Case, baseline: list[float], candidate: list[float]
) -> list[float]:
    numerators, denominators = (
        (candidate, baseline)
        if case.metric == Metric.LATENCY
        else (baseline, candidate)
    )
    ratios = [a / b for a, b in zip(numerators, denominators, strict=True)]
    if not all(math.isfinite(value) for value in ratios):
        raise ValueError("slowdown is outside numeric range")
    return ratios


def compare_case(suite: Suite, case: Case, rows: list[dict]) -> Comparison:
    identity = {
        "case": case.id,
        "metric": case.metric,
        "baseline": case.baseline.id,
        "candidate": case.candidate.id,
        "margin": suite.slowdown_margin,
    }
    try:
        baseline, candidate, device = _paired_values(suite, case, rows)
        values = paired_ratios(case, baseline, candidate)
    except (OverflowError, ValueError) as error:
        return Comparison(**identity, status="invalid", reason=str(error))

    effect = {
        "pairs": suite.pairs,
        "device": device,
        "baseline_median": statistics.median(baseline),
        "candidate_median": statistics.median(candidate),
        "slowdown": statistics.median(values) - 1,
    }
    if suite.pairs < MIN_PAIRS:
        return Comparison(
            **identity,
            **effect,
            status="inconclusive",
            reason=f"requires at least {MIN_PAIRS} pairs",
        )
    lower, upper = _interval(values, f"{suite.seed}:{case.id}")
    limit = 1 + suite.slowdown_margin
    if lower > limit:
        status = "regression"
    elif upper <= limit:
        status = "within_threshold"
    else:
        status = "inconclusive"
    return Comparison(
        **identity, **effect, status=status, lower=lower - 1, upper=upper - 1
    )


def read_run(directory: Path) -> tuple[dict, Suite, dict[str, list[dict]]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or not isinstance(manifest.get("suite"), dict)
        or manifest.get("status") not in ("complete", "invalid")
    ):
        raise ValueError("comparison requires a finished version 1 run")
    suite = parse_suite({"version": 1, **manifest["suite"]})
    if manifest.get("source") != suite.source:
        raise ValueError("run provenance differs from the resolved suite")
    by_case: dict[str, list[dict]] = {case.id: [] for case in suite.cases}
    with (directory / "measurements.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            case = row.get("case") if isinstance(row, dict) else None
            if not isinstance(case, str) or case not in by_case:
                raise ValueError("measurement has an unknown case")
            by_case[case].append(row)
    return manifest, suite, by_case


def analyze_run(directory: Path) -> tuple[dict, list[Comparison]]:
    manifest, suite, by_case = read_run(directory)
    results = [compare_case(suite, case, by_case[case.id]) for case in suite.cases]
    write_json(
        directory / "comparisons.json",
        {
            "schema_version": 1,
            "source": suite.source,
            "comparisons": [asdict(result) for result in results],
        },
    )
    return manifest, results
