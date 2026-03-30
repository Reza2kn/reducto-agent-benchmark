"""
Benchmark: n8n + Reducto

n8n is open-source workflow automation (self-hosted or n8n.cloud).
Reducto is accessed via HTTP Request nodes — no SDK, no MCP.
This tests whether automation-first users can build Reducto pipelines
without writing any code.

Setup:
  1. Self-host n8n: docker run -it --rm -p 5678:5678 n8nio/n8n
     OR use n8n.cloud free tier
  2. Import: benchmark/automation/n8n-reducto-workflow.json
  3. Configure credential: Settings → Credentials → New → Header Auth
     Name: "Reducto API" | Header: Authorization | Value: Bearer <YOUR_KEY>
  4. Execute the workflow manually
  5. Paste the execution output into benchmark/results/n8n_output.json
  6. Run: python bench_n8n.py

Capabilities tested: parse, extract, split, classify, jobid_chaining, error_handling
Integration path: workflow (HTTP nodes)
"""

from bench_utils import BenchmarkScore, evaluate_extract_output, evaluate_split_output, save_result, PROMPTS
import json
from pathlib import Path

PLATFORM = "n8n"
MODEL = "n/a"
INTEGRATION_PATH = "workflow"
SUBDIR = "automation"

# n8n scores for dimensions that don't depend on output content.
# These are fixed properties of the n8n integration path — scored once,
# regardless of which capability is being evaluated.
SETUP_SCORES = {
    "setup_friction":         2,  # Docker + import workflow + configure credential + execute = 4 steps
    "integration_complexity": 4,  # No code; just configure HTTP nodes in UI
    "feature_coverage":       3,  # Reaches parse/extract/split/classify/upload; MCP not supported; async awkward
    "mcp_compatibility":      0,  # n8n has no MCP support as of 2026-03
    "token_efficiency":       0,  # Not applicable — no LLM in the loop
    "discovery":              2,  # Must know Reducto API endpoints; no automatic discovery
}


def print_setup_instructions():
    print("""
╔══════════════════════════════════════════════════════╗
║  n8n + Reducto Benchmark Setup                       ║
╚══════════════════════════════════════════════════════╝

1. Run n8n locally:
   docker run -it --rm --name n8n -p 5678:5678 \\
     -e N8N_BASIC_AUTH_ACTIVE=false \\
     n8nio/n8n

2. Open http://localhost:5678

3. Import workflow:
   Workflows → ⋮ → Import from File
   → benchmark/automation/n8n-reducto-workflow.json

4. Set credential:
   Settings → Credentials → + New → Header Auth
   Name: Reducto API
   Name (header): Authorization
   Value: Bearer YOUR_REDUCTO_API_KEY

5. Open the imported workflow, set the credential on each
   HTTP Request node, then click "Execute Workflow"

6. In the output panel, copy the final node's JSON output
   and save it to: benchmark/results/n8n_output.json

   The expected shape is:
   {
     "job_id": "...",
     "extract": { ... },
     "split_sections": { ... },
     "classification": { ... },
     "error_tests_passed": true
   }

7. Run: python benchmark/scripts/bench_n8n.py
""")


def score_from_output(output_path: Path) -> list[BenchmarkScore]:
    """
    Read the saved n8n execution output and produce BenchmarkScore objects.
    One score per tested capability.
    """
    if not output_path.exists():
        print(f"  No output file at {output_path}")
        print("  Run the n8n workflow and save output first (see setup above)")
        return []

    with open(output_path) as f:
        output = json.load(f)

    scores = []

    # error_recovery is common across capabilities — did both error nodes behave?
    error_recovery = 4 if output.get("error_tests_passed") else 2

    for capability in ["extract", "split", "jobid_chaining"]:
        node_output = output.get(capability, output)
        text = json.dumps(node_output)

        if capability == "extract":
            quality, _ = evaluate_extract_output(text)
        elif capability == "split":
            quality, _ = evaluate_split_output(text)
        else:
            # jobid_chaining: check that jobid:// appears in the recorded execution data,
            # confirming that extract/split used the cached parse result.
            quality = 4 if "jobid://" in text else 2

        score = BenchmarkScore(
            platform=PLATFORM,
            model=MODEL,
            integration_path=INTEGRATION_PATH,
            capability=capability,
            error_recovery=error_recovery,
            output_quality=quality,
            notes="Scored from n8n workflow execution output",
            **SETUP_SCORES,
        )
        scores.append(score)

    return scores


def main():
    print(f"Reducto Benchmark — {PLATFORM}")
    print("=" * 55)

    output_path = Path(__file__).parent.parent / "results" / "n8n_output.json"

    if not output_path.exists():
        print_setup_instructions()
        # Save placeholder scores so this platform appears in the summary report
        # even before anyone has run the workflow.
        for capability in ["extract", "split", "classify", "jobid_chaining"]:
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
