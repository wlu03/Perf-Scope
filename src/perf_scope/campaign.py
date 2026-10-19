import math
import tempfile
from dataclasses import replace
from pathlib import Path

from perf_scope.config import load_suite
from perf_scope.profiling import profile_run
from perf_scope.regression import MIN_PAIRS, analyze_run
from perf_scope.report import render_report
from perf_scope.runner import run_suite, write_json
from perf_scope.scheduler import evaluate


def capture_campaign(
    config: Path,
    output: Path,
    budget: float,
    screen_pairs: int,
    confirm_pairs: int,
    priority: set[str],
    causes: dict[str, str],
    *,
    reference_pairs: int,
    observation_pairs: int,
) -> Path:
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("budget must be finite and positive")
    if screen_pairs < 2 or confirm_pairs < MIN_PAIRS:
        raise ValueError(
            f"campaign requires at least two screening and {MIN_PAIRS} confirmation pairs"
        )
    if (
        reference_pairs <= observation_pairs
        or observation_pairs < screen_pairs + confirm_pairs
    ):
        raise ValueError(
            "reference pairs must exceed observation pairs, which must cover both windows"
        )
    suite = load_suite(config)
    if (priority | set(causes)) - {case.id for case in suite.cases}:
        raise ValueError("priority or diagnosis expectation refers to an unknown case")

    output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="campaign-", dir=output.resolve()))
    state = {
        "schema_version": 1,
        "source": suite.source,
        "status": "running",
        "config": str(config.resolve()),
        "reference_pairs": reference_pairs,
        "observation_pairs": observation_pairs,
        "screen_pairs": screen_pairs,
        "confirm_pairs": confirm_pairs,
        "replay_budget_s": budget,
    }
    metadata = root / "campaign.json"
    write_json(metadata, state)
    paths = {}
    try:
        for name, count, mode in (
            ("observations", observation_pairs, "all"),
            ("reference", reference_pairs, "off"),
        ):
            capture_suite = replace(suite, pairs=count)
            path, failures = run_suite(capture_suite, config, root / name)
            paths[name] = path
            state[name] = path.relative_to(root).as_posix()
            write_json(metadata, state)
            manifest, decisions = analyze_run(path)
            try:
                profile_run(capture_suite, path, decisions, mode)
            finally:
                render_report(path, manifest, decisions)
            if failures or any(decision.status == "invalid" for decision in decisions):
                raise ValueError(f"{name} contain invalid measurements")
        results = evaluate(
            paths["reference"],
            paths["observations"],
            root / "evaluation.json",
            budget,
            screen_pairs,
            confirm_pairs,
            priority,
            causes,
        )
        state.update(status="complete", evaluation=results.name)
    except (OSError, ValueError) as error:
        state.update(status="failed", error=str(error))
        raise ValueError(f"{error}. Campaign artifacts are in {root}") from error
    finally:
        write_json(metadata, state)
    return root
