"""
Generate comparison report from benchmark results.

Results are loaded recursively from benchmark/results/**/*.json.
Each file is expected to have the new 8-dimension schema:
  platform, model, integration_path, capability,
  discovery, setup_friction, integration_complexity, feature_coverage,
  error_recovery, output_quality, token_efficiency, mcp_compatibility,
  total_score, max_score, platform_category (optional)

Run after scoring: python generate_report.py
Output: report/results_matrix.md (also printed to stdout)
"""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results"
REPORT_DIR = Path(__file__).parent.parent.parent / "report"

DIMENSIONS = [
    "discovery",
    "setup_friction",
    "integration_complexity",
    "feature_coverage",
    "error_recovery",
    "output_quality",
    "token_efficiency",
    "mcp_compatibility",
]

# Platforms hosted via Ollama (for OSS model table separation)
OLLAMA_PLATFORMS = {"Llama 3.3", "Qwen 2.5", "DeepSeek Coder", "Phi-4", "Mistral"}


def load_results() -> list[dict]:
    results = []
    if not RESULTS_DIR.exists():
        return results
    for f in sorted(RESULTS_DIR.rglob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            # Tag OSS results: only direct children of oss_models/, not framework subdirs
            # e.g. results/oss_models/foo__tool_calling__extract.json → oss_model
            # but results/oss_models/langchain/langchain__framework__extract.json → not oss_model
            rel = f.relative_to(RESULTS_DIR)
            if rel.parts[0] == "oss_models" and len(rel.parts) == 2:
                data.setdefault("platform_category", "oss_model")
            results.append(data)
        except (json.JSONDecodeError, OSError):
            pass
    return results


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def fmt(v: float) -> str:
    return f"{v:.2f}" if v else "-"


def is_oss(r: dict) -> bool:
    return r.get("platform_category") == "oss_model" or r.get("platform") in OLLAMA_PLATFORMS


def _group_key(platform: str) -> str:
    """Group 'LangChain/ModelName' → 'LangChain' for the summary table."""
    return platform.split("/")[0]


def platform_summary_table(results: list[dict]) -> str:
    """One row per platform, avg score per dimension + total avg."""
    # Aggregate by platform (group framework/model combos under the framework name)
    by_platform: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_platform[_group_key(r["platform"])].append(r)

    # Sort by avg total_score descending
    platform_order = sorted(
        by_platform.keys(),
        key=lambda p: avg([r.get("total_score", 0) for r in by_platform[p]]),
        reverse=True,
    )

    dim_headers = " | ".join(d.replace("_", " ").title() for d in DIMENSIONS)
    header = f"| Platform | {dim_headers} | Avg Total |"
    sep_cols = " | ".join([":-----------:"] * len(DIMENSIONS))
    sep = f"|----------|{sep_cols}|:---------:|"

    rows = [header, sep]
    for platform in platform_order:
        entries = by_platform[platform]
        dim_avgs = [avg([r.get(d, 0) for r in entries]) for d in DIMENSIONS]
        total_avg = avg([r.get("total_score", 0) for r in entries])
        max_possible = avg([r.get("max_score", 40) for r in entries])
        dim_cells = " | ".join(fmt(v) for v in dim_avgs)
        rows.append(f"| {platform} | {dim_cells} | **{fmt(total_avg)}/{fmt(max_possible)}** |")

    return "\n".join(rows)


def integration_path_matrix(results: list[dict]) -> str:
    """Rows = integration paths, columns = platforms, cells = avg output_quality."""
    platforms = sorted({_group_key(r["platform"]) for r in results})
    paths = sorted({r.get("integration_path", "unknown") for r in results})

    # Build lookup: (path, platform) -> list of output_quality scores
    lookup: dict[tuple, list[float]] = defaultdict(list)
    for r in results:
        key = (r.get("integration_path", "unknown"), _group_key(r["platform"]))
        lookup[key].append(r.get("output_quality", 0))

    # Only include platforms that appear in the results
    active_platforms = [p for p in platforms if any((path, p) in lookup for path in paths)]

    header = "| Integration Path | " + " | ".join(active_platforms) + " |"
    sep = "|" + "---|" * (len(active_platforms) + 1)
    rows = [header, sep]

    for path in paths:
        cells = []
        for p in active_platforms:
            vals = lookup.get((path, p), [])
            cells.append(fmt(avg(vals)) if vals else "-")
        rows.append(f"| {path} | " + " | ".join(cells) + " |")

    return "\n".join(rows)


def capability_coverage_table(results: list[dict]) -> str:
    """Rows = capabilities, columns = integration paths, cells = avg output_quality."""
    paths = sorted({r.get("integration_path", "unknown") for r in results})
    capabilities = sorted({r.get("capability", "unknown") for r in results})

    lookup: dict[tuple, list[float]] = defaultdict(list)
    for r in results:
        key = (r.get("capability", "unknown"), r.get("integration_path", "unknown"))
        lookup[key].append(r.get("output_quality", 0))

    active_paths = [p for p in paths if any((cap, p) in lookup for cap in capabilities)]

    header = "| Capability | " + " | ".join(active_paths) + " |"
    sep = "|" + "---|" * (len(active_paths) + 1)
    rows = [header, sep]

    for cap in capabilities:
        cells = []
        for path in active_paths:
            vals = lookup.get((cap, path), [])
            cells.append(fmt(avg(vals)) if vals else "-")
        rows.append(f"| {cap} | " + " | ".join(cells) + " |")

    return "\n".join(rows)


def oss_models_table(results: list[dict]) -> str:
    """Same as platform summary but filtered to OSS/Ollama models only."""
    oss_results = [r for r in results if is_oss(r)]
    if not oss_results:
        return "_No OSS model results recorded yet._"
    return platform_summary_table(oss_results)


def main():
    results = load_results()

    if not results:
        print("No results found in benchmark/results/. Run benchmarks first.")
        return

    # Split OSS from the main table so it doesn't clutter the primary comparison
    main_results = [r for r in results if not is_oss(r)]
    oss_results = [r for r in results if is_oss(r)]

    sections = []

    sections.append("# Reducto Agent Benchmark — Results Matrix\n")

    sections.append("## Platform Summary\n")
    sections.append("Average score per dimension across all runs for each platform. "
                    "Sorted by total score descending.\n")
    sections.append(platform_summary_table(main_results or results))
    sections.append("")

    sections.append("## Integration Path vs Output Quality\n")
    sections.append("Average `output_quality` score broken down by integration path and platform.\n")
    sections.append(integration_path_matrix(results))
    sections.append("")

    sections.append("## Capability Coverage vs Integration Path\n")
    sections.append("Average `output_quality` for each capability type per integration path.\n")
    sections.append(capability_coverage_table(results))
    sections.append("")

    sections.append("## OSS / Community Models (OpenRouter)\n")
    sections.append(oss_models_table(oss_results))
    sections.append("")

    output = "\n".join(sections)
    print(output)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "results_matrix.md"
    out_path.write_text(output)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
