import re
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from sys import float_info


class ConfigError(ValueError):
    """The suite cannot be translated into benchmark cases."""


class Metric(StrEnum):
    LATENCY = "latency_ms"
    BANDWIDTH = "bandwidth_gb_s"
    THROUGHPUT = "items_s"


@dataclass(frozen=True)
class Variant:
    id: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class Case:
    id: str
    workload: str
    shape: tuple[int, ...]
    dtype: str
    metric: Metric
    baseline: Variant
    candidate: Variant


@dataclass(frozen=True)
class Suite:
    seed: int
    pairs: int
    warmups: int
    iterations: int
    timeout_s: int
    source: str
    slowdown_margin: float
    cases: tuple[Case, ...]


def _table(value: object, label: str, keys: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a table")
    unknown = value.keys() - set(keys.split())
    if unknown:
        raise ConfigError(f"{label} has unknown keys {', '.join(sorted(unknown))}")
    return value


def _integer(value: object, label: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{label} must be an integer >= {minimum}")
    return value


def _identifier(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value) is None
    ):
        raise ConfigError(
            f"{label} must start with a letter or digit and use letters, digits, _ or -"
        )
    return value


def _variant(value: object, label: str) -> Variant:
    table = _table(value, label, "id command")
    identifier = _identifier(table.get("id"), f"{label}.id")
    command = table.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(arg, str) and "\x00" not in arg for arg in command)
        or not command[0].strip()
    ):
        raise ConfigError(
            f"{label}.command must be an argument array with a program name"
        )
    return Variant(identifier, tuple(command))


def _case(value: object, index: int) -> Case:
    label = f"cases[{index}]"
    table = _table(value, label, "id workload shape dtype metric baseline candidate")
    identifier = _identifier(table.get("id"), f"{label}.id")
    workload = _identifier(table.get("workload"), f"{label}.workload")
    shape = table.get("shape")
    if not isinstance(shape, list) or not shape:
        raise ConfigError(f"{label}.shape must be a nonempty list of dimensions")
    dimensions = tuple(_integer(size, f"{label}.shape", 1) for size in shape)
    dtype = _identifier(table.get("dtype", "fp32"), f"{label}.dtype")
    try:
        metric = Metric(table.get("metric", Metric.LATENCY))
    except ValueError as error:
        raise ConfigError(
            f"{label}.metric must be one of {', '.join(Metric)}"
        ) from error
    baseline = _variant(table.get("baseline"), f"{label}.baseline")
    candidate = _variant(table.get("candidate"), f"{label}.candidate")
    if baseline.id == candidate.id:
        raise ConfigError(f"{label} must use distinct baseline and candidate IDs")
    return Case(identifier, workload, dimensions, dtype, metric, baseline, candidate)


def parse_suite(value: object) -> Suite:
    table = _table(
        value,
        "suite",
        "version seed pairs warmups iterations timeout_s source slowdown_margin cases",
    )
    if _integer(table.get("version"), "version", 1) != 1:
        raise ConfigError("only suite version 1 is supported")
    seed = _integer(table.get("seed", 0), "seed", 0)
    pairs = _integer(table.get("pairs", 20), "pairs", 1)
    warmups = _integer(table.get("warmups", 5), "warmups", 0)
    iterations = _integer(table.get("iterations", 25), "iterations", 1)
    timeout = _integer(table.get("timeout_s", 120), "timeout_s", 1)
    source = table.get("source", "measured")
    if source not in ("measured", "fixture"):
        raise ConfigError("source must be measured or fixture")
    margin = table.get("slowdown_margin", 0.05)
    if type(margin) not in (int, float) or not 0 <= margin <= float_info.max:
        raise ConfigError("slowdown_margin must be a finite nonnegative fraction")
    entries = table.get("cases")
    if not isinstance(entries, list) or not entries:
        raise ConfigError("cases must contain at least one case")
    cases = tuple(_case(entry, i) for i, entry in enumerate(entries))
    if len({case.id for case in cases}) != len(cases):
        raise ConfigError("case IDs must be unique")
    return Suite(
        seed, pairs, warmups, iterations, timeout, source, float(margin), cases
    )


def load_suite(path: Path) -> Suite:
    try:
        with path.open("rb") as stream:
            return parse_suite(tomllib.load(stream))
    except (ConfigError, OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"{path}: {error}") from error
