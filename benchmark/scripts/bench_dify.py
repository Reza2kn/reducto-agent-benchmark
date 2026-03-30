"""
Benchmark: Dify + Reducto

Dify is an open-source LLM application platform with a visual workflow
builder. Reducto is accessed via HTTP Request nodes — no SDK, no MCP.
This tests whether low-code/no-code users can assemble a Reducto document
pipeline in a drag-and-drop editor.

Setup:
  1. Self-host Dify:
     git clone https://github.com/langgenius/dify
     cd dify/docker
     cp .env.example .env
     docker compose up -d
     Open http://localhost/install

     OR use Dify Cloud: https://cloud.dify.ai (free tier available)

  2. Import the workflow DSL:
     Studio → + Create App → Import DSL file
     → benchmark/automation/dify-reducto-app.yml

  3. In the imported workflow, open each HTTP Request node and verify
     the Authorization header uses the {{#start.api_key#}} variable.
     No hardcoded keys needed — the API key is supplied at runtime.

  4. Click "Run" (top-right). In the dialog:
     - doc_url:  https://cdn.reducto.ai/samples/fidelity-example.pdf
     - api_key:  YOUR_REDUCTO_API_KEY

  5. Copy the workflow output (the formatted report from the End node)
     and save it to: benchmark/results/dify_output.json

     Minimal expected shape:
     {
       "report": "# Reducto Benchmark Report\\n...",
       "job_id": "...",
       "extract": { ... },
       "split_sections": { ... }
     }

  6. Run: python benchmark/scripts/bench_dify.py

Capabilities tested: parse, extract, split, jobid_chaining
Integration path: workflow (HTTP nodes + Code node)
"""

from bench_utils import (
    BenchmarkScore,
    evaluate_extract_output,
    evaluate_split_output,
    save_result,
)
import json
from pathlib import Path

PLATFORM = "Dify"
MODEL = "n/a"           # Dify workflow mode; no LLM in the data path for this test
INTEGRATION_PATH = "workflow"
SUBDIR = "automation"

# Fixed scores for dimensions that are properties of Dify's integration path,
# not of any particular execution result.
SETUP_SCORES = {
    # Docker + compose up + install wizard + import DSL + supply credential at runtime = 5 steps
    "setup_friction":         2,
    # Visual editor with HTTP nodes and a Python Code node; no raw API client code
    "integration_complexity": 4,
    # Covers parse/extract/split/upload; classify absent from DSL but trivially addable;
    # async polling requires a Loop node (doable but awkward); MCP not supported
    "feature_coverage":       3,
    # Dify has no MCP support in workflow mode as of 2026-03
    "mcp_compatibility":      0,
    # No LLM in the loop for these HTTP-node workflows
    "token_efficiency":       0,
    # Endpoint names must be known in advance; Dify offers no Reducto-specific discovery
    "discovery":              2,
}


def print_setup_instructions():
    print("""
╔══════════════════════════════════════════════════════╗
║  Dify + Reducto Benchmark Setup                      ║
╚══════════════════════════════════════════════════════╝

Option A — Self-hosted (Docker):
  git clone https://github.com/langgenius/dify
  cd dify/docker
  cp .env.example .env
  docker compose up -d
  → Open http://localhost/install

Option B — Dify Cloud:
  → https://cloud.dify.ai  (free tier, no Docker required)

Steps after Dify is running:
  1. Studio → + Create App → Import DSL file
     → benchmark/automation/dify-reducto-app.yml

  2. In the workflow canvas, inspect each HTTP Request node.
     The Authorization header should read:
       Bearer {{#start.api_key#}}
     This is passed in at runtime — no credential store needed.

  3. Click "Run" (top-right corner of the canvas).
     Fill in the dialog:
       doc_url : https://cdn.reducto.ai/samples/fidelity-example.pdf
       api_key : YOUR_REDUCTO_API_KEY

  4. The workflow runs: Upload → Parse → (Extract || Split) → Format Report

  5. Copy the final output from the End node panel.
     Save it to: benchmark/results/dify_output.json

     Expected shape:
     {
       "report": "# Reducto Benchmark Report\\n...",
       "job_id": "...",
       "extract": { ... },
       "split_sections": { ... }
     }

  6. Run: python benchmark/scripts/bench_dify.py
""")


def score_from_output(output_path: Path) -> list[BenchmarkScore]:
    """
    Parse the saved Dify workflow output and emit one BenchmarkScore
    per tested capability.
    """
    if not output_path.exists():
        print(f"  No output file at {output_path}")
        print("  Run the Dify workflow and save output first (see setup above)")
        return []

    with open(output_path) as f:
        output = json.load(f)

    scores = []

    # The formatted report lives at output["report"]; individual node outputs
    # may also be present if the user copied the full execution data.
    report_text = output.get("report", "")
    extract_node = output.get("extract", output)
    split_node = output.get("split_sections", output)

    for capability in ["extract", "split", "jobid_chaining"]:
        if capability == "extract":
            # Prefer the dedicated extract node output; fall back to the
            # formatted report which includes extracted field values.
            source = json.dumps(extract_node) if extract_node != output else report_text
            quality, _ = evaluate_extract_output(source)

        elif capability == "split":
            source = json.dumps(split_node) if split_node != output else report_text
            quality, _ = evaluate_split_output(source)

        else:
            # jobid_chaining: the Code node embeds the job_id in the report and
            # the extract/split nodes are configured with jobid:// inputs.
            # Check both the report text and the raw job_id field.
            combined = report_text + json.dumps(output)
            quality = 4 if ("jobid://" in combined or output.get("job_id")) else 2

        # error_recovery not explicitly tested in the Dify DSL (no error test nodes),
        # but Dify surfaces HTTP errors in the node output panel — credit partial.
        error_recovery = 3

        score = BenchmarkScore(
            platform=PLATFORM,
            model=MODEL,
            integration_path=INTEGRATION_PATH,
            capability=capability,
            error_recovery=error_recovery,
            output_quality=quality,
            notes="Scored from Dify workflow execution output",
            **SETUP_SCORES,
        )
        scores.append(score)

    return scores


def main():
    print(f"Reducto Benchmark — {PLATFORM}")
    print("=" * 55)

    output_path = Path(__file__).parent.parent / "results" / "dify_output.json"

    if not output_path.exists():
        print_setup_instructions()
        # Save placeholder scores so Dify appears in the summary report
        # before anyone has executed the workflow.
        for capability in ["extract", "split", "jobid_chaining"]:
            score = BenchmarkScore(
                platform=PLATFORM,
                model=MODEL,
                integration_path=INTEGRATION_PATH,
                capability=capability,
                notes="Not yet tested — run workflow and re-execute this script",
                **SETUP_SCORES,
            )
            save_result(score, subdir=SUBDIR)
    else:
        scores = score_from_output(output_path)
        for score in scores:
            save_result(score, subdir=SUBDIR)
            print(f"  {score.capability}: {score.total_score}/{score.max_score}")


if __name__ == "__main__":
    main()
