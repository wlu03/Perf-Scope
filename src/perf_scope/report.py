import json
from html import escape
from pathlib import Path

from perf_scope.regression import Comparison


def _number(value: float | None, format_spec: str = ".6g") -> str:
    return "—" if value is None else format(value, format_spec)


def _profile_section(directory: Path) -> str:
    sections = []
    source = directory / "profiles.json"
    if source.is_file():
        profiles = json.loads(source.read_text(encoding="utf-8"))
        entries = []
        for row in profiles["profiles"]:
            label = f"{row['case']} {row.get('variant', '')}".strip()
            detail = row.get("reason", row["status"])
            entries.append(f"<li>{escape(label)} · {escape(detail)}</li>")
        for row in profiles["diagnoses"]:
            evidence = row.get("evidence", {})
            detail = ", ".join(
                f"{key.replace('_', ' ')} {value:g}" for key, value in evidence.items()
            )
            text = f"{row['case']} · {row['cause']}" + (
                f" ({detail})" if detail else ""
            )
            entries.append(f"<li>{escape(text)}</li>")
        sections.append(
            "<h2>Profiler evidence</h2><ul>" + "".join(entries) + "</ul>"
            '<p><a href="profiles.json">Counter records and logs</a></p>'
        )
    if (directory / "roofline.png").is_file():
        sections.append(
            '<h2>GEMM roofline</h2><img src="roofline.png" '
            'alt="Scalar FP32 DRAM roofline with achieved GEMM throughput" '
            'style="max-width:100%"><p><a href="roofline.json">Roofline data</a></p>'
        )
    elif (directory / "roofline.json").is_file():
        sections.append('<p><a href="roofline.json">Roofline data availability</a></p>')
    return "".join(sections)


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
{_profile_section(directory)}
</html>
"""
    output = directory / "report.html"
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(output)
    return output
