from html import escape
from pathlib import Path

from perf_scope.regression import Comparison


def _number(value: float | None, format_spec: str = ".6g") -> str:
    return "—" if value is None else format(value, format_spec)


def render_report(
    directory: Path, manifest: dict, comparisons: list[Comparison]
) -> Path:
    rows = []
    cases = {case["id"]: case for case in manifest["suite"]["cases"]}
    for comparison in comparisons:
        case = cases[comparison.case]
        interval = (
            f"{_number(comparison.lower, '.2%')} to {_number(comparison.upper, '.2%')}"
        )
        cells = (
            comparison.case,
            " × ".join(map(str, case["shape"])),
            case["dtype"],
            comparison.device or "—",
            comparison.metric,
            f"{comparison.baseline} ({_number(comparison.baseline_median)})",
            f"{comparison.candidate} ({_number(comparison.candidate_median)})",
            str(comparison.pairs),
            _number(comparison.slowdown, ".2%"),
            interval if comparison.lower is not None else "—",
            _number(comparison.margin, ".2%"),
            comparison.status.replace("_", " "),
            comparison.reason or "—",
        )
        rows.append(
            "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in cells) + "</tr>"
        )
    labels = (
        "Case",
        "Shape",
        "Precision",
        "Device",
        "Metric",
        "Baseline median",
        "Candidate median",
        "Pairs",
        "Slowdown",
        "95% interval",
        "Margin",
        "Status",
        "Detail",
    )
    headers = "".join(f'<th scope="col">{label}</th>' for label in labels)
    source = escape(manifest["source"])
    report = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Perf-Scope run report</title>
<style>
body {{ font: 1rem system-ui; margin: 2rem; }}
table {{ border-collapse: collapse; text-align: left; }}
th, td {{ padding: .5rem; border-bottom: 1px solid #aaa; }}
</style>
<h1>Perf-Scope run report</h1>
<p>Data source <strong>{source}</strong></p>
<table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>
<p><a href="manifest.json">Manifest</a> · <a href="measurements.jsonl">Measurements</a>
 · <a href="comparisons.json">Decisions</a></p>
</html>
"""
    output = directory / "report.html"
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(output)
    return output
