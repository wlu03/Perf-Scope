import argparse
import sys
from pathlib import Path

from perf_scope.config import ConfigError, load_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="perf-scope")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a TOML benchmark suite")
    validate.add_argument("suite", type=Path)
    args = parser.parse_args(argv)

    try:
        load_suite(args.suite)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    print(f"Validated suite {args.suite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
