import argparse
import json
import math
import sys
from pathlib import Path

from perf_scope.config import load_suite
from perf_scope.profiling import profile_run
from perf_scope.regression import analyze_run
from perf_scope.report import render_report
from perf_scope.roofline import write_roofline
from perf_scope.runner import run_suite
from perf_scope.scheduler import evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="perf-scope")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a TOML benchmark suite")
    validate.add_argument("suite", type=Path)
    run = commands.add_parser("run", help="measure all paired cases in a suite")
    run.add_argument("suite", type=Path)
    run.add_argument("--output", type=Path, default=Path("runs"))
    run.add_argument("--fail-on-regression", action="store_true")
    run.add_argument(
        "--profile", choices=("regressions", "all", "off"), default="regressions"
    )
    run.add_argument("--bandwidth-run", type=Path)
    run.add_argument("--fp32-peak-gflops", type=float)
    run.add_argument("--peak-source")
    compare = commands.add_parser(
        "compare", help="recompute comparisons from a completed run"
    )
    compare.add_argument("directory", type=Path)
    compare.add_argument("--fail-on-regression", action="store_true")
    replay = commands.add_parser(
        "evaluate", help="replay GPU measurement policies from two runs"
    )
    replay.add_argument("reference", type=Path)
    replay.add_argument("observations", type=Path)
    replay.add_argument("--budget-s", type=float, required=True)
    replay.add_argument("--screen-pairs", type=int, default=4)
    replay.add_argument("--confirm-pairs", type=int, default=16)
    replay.add_argument("--priority", action="append", default=[])
    replay.add_argument("--expected-causes", type=Path)
    replay.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            load_suite(args.suite)
            print(f"Validated suite {args.suite}")
            return 0
        if args.command == "evaluate":
            causes = {}
            if args.expected_causes is not None:
                causes = json.loads(args.expected_causes.read_text(encoding="utf-8"))
                if not isinstance(causes, dict) or not all(
                    isinstance(cause, str) and cause for cause in causes.values()
                ):
                    raise ValueError(
                        "diagnosis expectations must map case IDs to cause strings"
                    )
            print(
                evaluate(
                    args.reference,
                    args.observations,
                    args.output,
                    args.budget_s,
                    args.screen_pairs,
                    args.confirm_pairs,
                    set(args.priority),
                    causes,
                )
            )
            return 0
        if args.command == "run":
            ceiling = (args.bandwidth_run, args.fp32_peak_gflops, args.peak_source)
            if any(value is not None for value in ceiling) and (
                not all(ceiling)
                or not math.isfinite(args.fp32_peak_gflops)
                or args.fp32_peak_gflops <= 0
                or args.profile == "off"
            ):
                raise ValueError(
                    "roofline requires profiling, bandwidth run, positive FP32 peak, and peak source"
                )
            suite = load_suite(args.suite)
            directory, _ = run_suite(suite, args.suite, args.output)
        else:
            directory = args.directory
        manifest, decisions = analyze_run(directory)
        try:
            if args.command == "run":
                profiles = profile_run(suite, directory, decisions, args.profile)
                if args.bandwidth_run is not None:
                    write_roofline(
                        directory,
                        suite,
                        profiles["profiles"],
                        args.bandwidth_run,
                        args.fp32_peak_gflops,
                        args.peak_source,
                    )
        finally:
            render_report(directory, manifest, decisions)
    except (ValueError, OSError) as error:
        print(error, file=sys.stderr)
        return 2
    print(directory)
    if any(decision.status == "invalid" for decision in decisions):
        return 1
    if args.fail_on_regression and any(
        decision.status == "regression" for decision in decisions
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
