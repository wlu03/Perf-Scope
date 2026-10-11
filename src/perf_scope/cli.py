import argparse
import sys
from pathlib import Path

from perf_scope.config import load_suite
from perf_scope.regression import analyze_run
from perf_scope.report import render_report
from perf_scope.runner import run_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="perf-scope")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a TOML benchmark suite")
    validate.add_argument("suite", type=Path)
    run = commands.add_parser("run", help="measure all paired cases in a suite")
    run.add_argument("suite", type=Path)
    run.add_argument("--output", type=Path, default=Path("runs"))
    run.add_argument("--fail-on-regression", action="store_true")
    compare = commands.add_parser(
        "compare", help="recompute comparisons from a completed run"
    )
    compare.add_argument("directory", type=Path)
    compare.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            load_suite(args.suite)
            print(f"Validated suite {args.suite}")
            return 0
        if args.command == "run":
            suite = load_suite(args.suite)
            directory, _ = run_suite(suite, args.suite, args.output)
        else:
            directory = args.directory
        manifest, decisions = analyze_run(directory)
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
