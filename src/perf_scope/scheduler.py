import json
import math
import random
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

from perf_scope.config import Case, Suite
from perf_scope.profiling import diagnose
from perf_scope.regression import (
    MIN_PAIRS,
    Comparison,
    _paired_values,
    compare_case,
    paired_ratios,
    read_run,
)
from perf_scope.runner import write_json

SCREENING_SHARE = 0.25


@dataclass(frozen=True)
class RecordedCase:
    id: str
    label: str
    screening_cost: float
    potential: float
    confirmation_cost: float
    confirmation: Comparison
    profile_cost: float | None
    cause: str


def _seconds(rows: list[dict]) -> float:
    costs = [row.get("wall_s") for row in rows]
    if any(
        type(cost) not in (float, int) or not math.isfinite(cost) or cost < 0
        for cost in costs
    ):
        raise ValueError("replay requires finite nonnegative wall_s measurements")
    return math.fsum(costs)


def _window(rows: list[dict], start: int, count: int) -> list[dict]:
    return [
        row | {"pair": row["pair"] - start}
        for row in rows
        if start <= row["pair"] < start + count
    ]


def _recorded_profiles(directory: Path, source: str) -> dict[str, list[dict]]:
    path = directory / "profiles.json"
    if not path.is_file():
        return {}
    profiles = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profiles, dict) or profiles.get("source") != source:
        raise ValueError("profile provenance differs from the evaluation run")
    collected: dict[str, list[dict]] = {}
    for row in profiles["profiles"]:
        collected.setdefault(row["case"], []).append(row)
    return collected


def _recorded_case(
    case: Case,
    reference: Comparison,
    suite: Suite,
    rows: list[dict],
    profiles: list[dict],
    screen_pairs: int,
    confirm_pairs: int,
) -> RecordedCase:
    _, _, device = _paired_values(suite, case, rows)
    if reference.device != device:
        raise ValueError(
            f"{case.id} reference and evaluation require valid pairs on the same device"
        )
    screening = _window(rows, 0, screen_pairs)
    baseline, candidate, _ = _paired_values(
        replace(suite, pairs=screen_pairs), case, screening
    )
    ratios = paired_ratios(case, baseline, candidate)
    center = statistics.median(ratios)
    spread = statistics.median(abs(ratio - center) for ratio in ratios)
    confirmation = _window(rows, screen_pairs, confirm_pairs)
    confirmed = compare_case(replace(suite, pairs=confirm_pairs), case, confirmation)
    by_side = {
        row["side"]: row
        for row in profiles
        if row.get("side") in ("baseline", "candidate") and row.get("device") == device
    }
    profile_cost = None
    cause = "undetermined"
    if len(by_side) == 2:
        selected = [by_side["baseline"], by_side["candidate"]]
        profile_cost = _seconds(selected)
        cause = diagnose(case, selected)["cause"]
    return RecordedCase(
        case.id,
        reference.status,
        _seconds(screening),
        center - 1 + spread / math.sqrt(screen_pairs),
        _seconds(confirmation),
        confirmed,
        profile_cost,
        cause,
    )


def _policy(
    name: str,
    records: list[RecordedCase],
    budget: float,
    priority: set[str],
    expected_causes: dict[str, str],
) -> dict:
    screened = set()
    spent = 0.0
    if name == "adaptive":
        for record in sorted(records, key=lambda item: item.id not in priority):
            if spent + record.screening_cost > budget * SCREENING_SHARE:
                break
            spent += record.screening_cost
            screened.add(record.id)
        records = sorted(
            records,
            key=lambda item: (
                item.id not in priority,
                item.id not in screened,
                -item.potential if item.id in screened else 0,
            ),
        )
    entries = []
    for record in records:
        result = {
            "case": record.id,
            "reference": record.label,
            "screened": record.id in screened,
            "decision": "unmeasured",
            "detected_at_s": None,
            "profile": None,
            "diagnosis": None,
        }
        if spent + record.confirmation_cost <= budget:
            spent += record.confirmation_cost
            result["decision"] = record.confirmation.status
            if record.confirmation.status == "regression":
                result["detected_at_s"] = spent
                if record.profile_cost is None:
                    result["profile"] = "unavailable"
                elif spent + record.profile_cost <= budget:
                    spent += record.profile_cost
                    result.update(profile="attempted", diagnosis=record.cause)
                else:
                    result["profile"] = "deferred"
        elif spent < budget:
            spent = budget
            result["decision"] = "interrupted"
        entries.append(result)

    positives = [row for row in entries if row["reference"] == "regression"]
    negatives = [row for row in entries if row["reference"] == "within_threshold"]
    detected = [row for row in positives if row["decision"] == "regression"]
    false_positives = sum(row["decision"] == "regression" for row in negatives)
    eligible_diagnoses = [row for row in detected if row["case"] in expected_causes]
    correct_causes = sum(
        row["diagnosis"] == expected_causes[row["case"]] for row in eligible_diagnoses
    )
    return {
        "policy": name,
        "budget_s": budget if math.isfinite(budget) else None,
        "replayed_wall_s": spent,
        "replayed_gpu_hours": spent / 3600,
        "true_positives": len(detected),
        "reference_positives": len(positives),
        "false_positives": false_positives,
        "reference_negatives": len(negatives),
        "unlabeled": len(entries) - len(positives) - len(negatives),
        "recall": len(detected) / len(positives) if positives else None,
        "false_positive_rate": false_positives / len(negatives) if negatives else None,
        "median_detection_s": statistics.median(
            row["detected_at_s"] for row in detected
        )
        if detected
        else None,
        "missed_regressions": len(positives) - len(detected),
        "correct_diagnoses": correct_causes,
        "expected_diagnoses": len(eligible_diagnoses),
        "diagnosis_accuracy": correct_causes / len(eligible_diagnoses)
        if eligible_diagnoses
        else None,
        "undetermined_diagnoses": sum(
            row["diagnosis"] in (None, "undetermined") for row in eligible_diagnoses
        ),
        "cases": entries,
    }


def evaluate(
    reference_path: Path,
    evaluation_path: Path,
    output: Path,
    budget: float,
    screen_pairs: int,
    confirm_pairs: int,
    priority: set[str],
    expected_causes: dict[str, str],
) -> Path:
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("budget must be finite and positive")
    if screen_pairs < 2 or confirm_pairs < MIN_PAIRS:
        raise ValueError(
            f"evaluation requires at least two screening and {MIN_PAIRS} confirmation pairs"
        )
    reference_meta, reference, truth_rows = read_run(reference_path)
    evaluation_meta, evaluation, observations = read_run(evaluation_path)
    if reference_meta.get("started_at") == evaluation_meta.get("started_at"):
        raise ValueError("reference and evaluation must come from independent runs")
    reference_programs = reference_meta.get("executables")
    evaluation_programs = evaluation_meta.get("executables")
    if (
        reference_programs is not None
        and evaluation_programs is not None
        and reference_programs != evaluation_programs
    ):
        raise ValueError(
            "reference and evaluation runs used different executable identities"
        )
    if replace(reference, pairs=evaluation.pairs) != evaluation:
        raise ValueError(
            "reference and evaluation suites may differ only in pair count"
        )
    if (
        reference.pairs <= confirm_pairs
        or evaluation.pairs < screen_pairs + confirm_pairs
    ):
        raise ValueError(
            "provide a longer reference run and disjoint evaluation windows"
        )
    known = {case.id for case in evaluation.cases}
    if (priority | set(expected_causes)) - known:
        raise ValueError("priority or diagnosis expectation refers to an unknown case")
    profiles = _recorded_profiles(evaluation_path, evaluation.source)
    records = [
        _recorded_case(
            case,
            compare_case(reference, case, truth_rows[case.id]),
            evaluation,
            observations[case.id],
            profiles.get(case.id, []),
            screen_pairs,
            confirm_pairs,
        )
        for case in evaluation.cases
    ]
    if len({record.confirmation.device for record in records}) != 1:
        raise ValueError("replay requires one device across all cases")
    random.Random(evaluation.seed).shuffle(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output,
        {
            "schema_version": 1,
            "source": evaluation.source,
            "reference": str(reference_path.resolve()),
            "evaluation": str(evaluation_path.resolve()),
            "screen_pairs": screen_pairs,
            "confirm_pairs": confirm_pairs,
            "screening_budget_fraction": SCREENING_SHARE,
            "priority": sorted(priority),
            "expected_causes": expected_causes,
            "policies": [
                _policy(name, records, cost, priority, expected_causes)
                for name, cost in (
                    ("exhaustive", math.inf),
                    ("fixed_budget", budget),
                    ("adaptive", budget),
                )
            ],
        },
    )
    return output
