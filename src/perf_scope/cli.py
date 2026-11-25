import argparse
import sys
from pathlib import Path

from perf_scope.config import ConfigError, load_suite
from perf_scope.runner import run_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="perf-scope")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a TOML benchmark suite")
    validate.add_argument("suite", type=Path)
    run = commands.add_parser("run", help="measure all paired cases in a suite")
    run.add_argument("suite", type=Path)
    run.add_argument("--output", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)

    try:
        suite = load_suite(args.suite)
        if args.command == "run":
            directory, failures = run_suite(suite, args.suite, args.output)
            print(directory)
            return 1 if failures else 0
    except (ConfigError, OSError) as error:
        print(error, file=sys.stderr)
        return 2
    print(f"Validated suite {args.suite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
