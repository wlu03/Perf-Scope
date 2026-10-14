import math
import statistics
from pathlib import Path

from perf_scope.config import Metric, Suite
from perf_scope.profiling import READS, TIME, WRITES
from perf_scope.regression import compare_case, read_run
from perf_scope.runner import write_json


def _bandwidth(directory: Path) -> tuple[float, str]:
    _, suite, by_case = read_run(directory)
    cases = [
        case
        for case in suite.cases
        if case.workload == "triad" and case.metric == Metric.BANDWIDTH
    ]
    if suite.source != "measured" or len(cases) != 1:
        raise ValueError("bandwidth calibration needs one measured triad case")
    case = cases[0]
    rows = by_case[case.id]
    comparison = compare_case(suite, case, rows)
    if comparison.device is None or case.baseline.command != case.candidate.command:
        raise ValueError(
            "bandwidth calibration requires valid paired runs of the same executable"
        )
    for row in rows:
        measurement = row["measurement"]
        l2 = measurement.get("cuda", {}).get("l2_bytes")
        working_set = measurement.get("logical_bytes")
        if (
            not isinstance(l2, (int, float))
            or not isinstance(working_set, (int, float))
            or not 0 < l2 < working_set
        ):
            raise ValueError("triad working set must exceed the reported L2 capacity")
    return statistics.median(row["value"] for row in rows), comparison.device


def roofline_points(
    suite: Suite, profiles: list[dict], device: str, peak: float, bandwidth: float
) -> list[dict]:
    cases = {
        case.id: case
        for case in suite.cases
        if case.workload == "gemm" and case.dtype == "fp32"
    }
    points = []
    for profile in profiles:
        case = cases.get(profile["case"])
        if case is None or profile["status"] != "ok" or profile["device"] != device:
            continue
        work = 2 * math.prod(case.shape)
        counters = profile["counters"]
        traffic = counters[READS] + counters[WRITES]
        if traffic <= 0:
            continue
        intensity = work / traffic
        points.append(
            {
                "case": case.id,
                "variant": profile["variant"],
                "intensity_flops_per_byte": intensity,
                "achieved_gflops": work / counters[TIME],
                "ceiling_gflops": min(peak, bandwidth * intensity),
            }
        )
    return points


def _plot(path: Path, points: list[dict], peak: float, bandwidth: float) -> None:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ValueError(
            "install perf-scope[plot] to generate a roofline figure"
        ) from error
    figure = Figure(figsize=(9, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    ridge = peak / bandwidth
    intensities = [point["intensity_flops_per_byte"] for point in points]
    x = [min(*intensities, ridge) / 2, ridge, max(*intensities, ridge) * 2]
    axes.loglog(
        x,
        [min(peak, bandwidth * value) for value in x],
        label="DRAM / scalar FP32 ceiling",
    )
    for point in points:
        axes.scatter(
            point["intensity_flops_per_byte"],
            point["achieved_gflops"],
            label=f"{point['case']} / {point['variant']}",
        )
    axes.set(xlabel="Arithmetic intensity (FLOPs / DRAM byte)", ylabel="GFLOP/s")
    axes.grid(visible=True, which="both", alpha=0.25)
    axes.legend()
    figure.savefig(path, dpi=150)


def write_roofline(
    directory: Path,
    suite: Suite,
    profiles: list[dict],
    calibration: Path,
    peak: float,
    peak_source: str,
) -> None:
    bandwidth, device = _bandwidth(calibration)
    points = roofline_points(suite, profiles, device, peak, bandwidth)
    write_json(
        directory / "roofline.json",
        {
            "source": suite.source,
            "device": device,
            "fp32_peak_gflops": peak,
            "peak_source": peak_source,
            "bandwidth_gb_s": bandwidth,
            "bandwidth_run": str(calibration.resolve()),
            "points": points,
        },
    )
    if points:
        _plot(directory / "roofline.png", points, peak, bandwidth)
