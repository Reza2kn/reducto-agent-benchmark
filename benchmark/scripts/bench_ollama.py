"""
Benchmark: Open-source models via Ollama + Reducto code generation

For each model in MODELS, asks the model to write a Python script that calls
Reducto's /parse endpoint. The generated code is extracted, executed in a
subprocess, and scored on:

  - integration_complexity  (lines of code)
  - error_recovery          (static: has try/except, status checks)
  - output_quality          (from actual execution result)

No LLM framework required — just the Ollama REST API at localhost:11434.

Run:
    ollama serve                   # ensure Ollama is running
    ollama pull llama3.3:70b       # pull whichever models you want to test
    REDUCTO_API_KEY=... python bench_ollama.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Optional

import requests

from bench_utils import (
    BenchmarkScore,
    TEST_DOC_URL,
    evaluate_parse_output,
    save_result,
)

# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------

MODELS = [
    ("llama3.3:70b",      "Llama 3.3 70B"),
    ("qwen2.5-coder:32b", "Qwen 2.5 Coder 32B"),
    ("deepseek-coder-v2", "DeepSeek Coder V2"),
    ("phi4:14b",          "Phi-4 14B"),
    ("mistral:7b",        "Mistral 7B"),
]

OLLAMA_BASE = "http://localhost:11434"

# The generation prompt is fixed so all models get an identical task.
CODE_GEN_PROMPT = f"""\
Write a complete, runnable Python script that:
1. Sends a POST request to https://platform.reducto.ai/parse
2. Uses Authorization: Bearer header from os.environ["REDUCTO_API_KEY"]
3. Body: {{"input": "{TEST_DOC_URL}"}}
4. Prints the first 500 chars of the response content
5. Prints the number of tables found (blocks with type "Table")
Use only the `requests` library. Handle HTTP errors. No placeholders.
Output ONLY the Python code block, nothing else.\
"""

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _check_ollama() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def _list_local_models() -> list[str]:
    """Return model names available locally in Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def _generate(model_id: str, prompt: str) -> Optional[str]:
    """
    Call /api/generate (non-streaming) and return the full response text.
    Returns None on error.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model_id, "prompt": prompt, "stream": False},
            timeout=300,  # large models can be slow
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.HTTPError as exc:
        print(f"  HTTP error from Ollama: {exc}")
        return None


# ---------------------------------------------------------------------------
# Code extraction & static analysis
# ---------------------------------------------------------------------------

def _extract_code(text: str) -> str:
    """Pull the first Python code block from a markdown-wrapped response."""
    # Try fenced code block first
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: the whole response (model may have followed instructions)
    return text.strip()


def _count_loc(code: str) -> int:
    """Count non-blank, non-comment lines of code."""
    return sum(
        1 for line in code.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def _static_error_score(code: str) -> tuple[int, dict]:
    """
    Simple static analysis for error-handling quality.
    Returns (score 1-5, details_dict).
    """
    has_try_except    = "try:" in code and "except" in code
    checks_status     = ".raise_for_status()" in code or "status_code" in code
    handles_non_200   = (
        "raise_for_status" in code
        or re.search(r"status_code\s*!=\s*200", code) is not None
        or re.search(r"if\s+resp\.ok", code) is not None
    )
    has_env_var_read  = 'os.environ' in code or 'os.getenv' in code

    details = {
        "has_try_except":   has_try_except,
        "checks_status":    checks_status,
        "handles_non_200":  handles_non_200,
        "reads_env_var":    has_env_var_read,
    }
    score = sum(1 for v in details.values() if v)
    return max(1, min(5, score + 1)), details  # base of 1 for attempting


def _integration_complexity_score(loc: int) -> int:
    """Map lines-of-code to the 1-5 complexity scale from bench_utils rubric."""
    if loc <= 5:
        return 5
    if loc <= 15:
        return 4
    if loc <= 30:
        return 3
    if loc <= 60:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

def _run_code(code: str, timeout: int = 60) -> tuple[str, list[str]]:
    """
    Write code to a temp file and run it in a subprocess.
    Returns (stdout_combined, errors_list).
    """
    errors: list[str] = []
    env = {**os.environ, "REDUCTO_API_KEY": os.environ.get("REDUCTO_API_KEY", "")}

    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        combined = result.stdout + result.stderr
        if result.returncode != 0:
            errors.append(f"exit code {result.returncode}: {result.stderr[:300]}")
        return combined, errors
    except subprocess.TimeoutExpired:
        errors.append(f"execution timed out after {timeout}s")
        return "", errors
    except Exception as exc:
        errors.append(str(exc))
        return "", errors
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Per-model benchmark
# ---------------------------------------------------------------------------

def bench_model(model_id: str, display_name: str) -> None:
    print(f"\n  Model: {display_name} ({model_id})")
    print("  " + "-" * 50)

    local_models = _list_local_models()
    # Ollama tag matching: "llama3.3:70b" or "llama3.3" both valid
    tag_base = model_id.split(":")[0]
    is_available = any(
        model_id == m or tag_base == m.split(":")[0]
        for m in local_models
    )
    if not is_available:
        print(f"  Model not pulled. Run:  ollama pull {model_id}")
        print("  Skipping.")
        return

    # --- Generate code ---
    print("  Generating code...", end=" ", flush=True)
    gen_start = time.time()
    raw_response = _generate(model_id, CODE_GEN_PROMPT)
    gen_time = time.time() - gen_start

    if raw_response is None:
        print("failed (Ollama connection error).")
        return

    print(f"done ({gen_time:.1f}s)")
    code = _extract_code(raw_response)
    loc = _count_loc(code)
    print(f"  Generated {loc} LOC.")

    # --- Static analysis ---
    error_score, error_details = _static_error_score(code)

    # --- Execute code ---
    print("  Executing generated code...", end=" ", flush=True)
    exec_start = time.time()
    exec_output, exec_errors = _run_code(code, timeout=90)
    exec_time = time.time() - exec_start
    print(f"done ({exec_time:.1f}s)")

    if exec_errors:
        print(f"  Execution errors: {exec_errors}")

    # --- Score output quality ---
    quality_score, quality_details = evaluate_parse_output(exec_output)

    complexity_score = _integration_complexity_score(loc)

    all_errors = exec_errors  # static analysis issues are in error_details

    score = BenchmarkScore(
        platform=f"Ollama/{display_name}",
        model=model_id,
        integration_path="rest_api",
        capability="parse",
        discovery=3,        # model knows REST but not necessarily Reducto
        setup_friction=3,   # needs Ollama running + model pulled
        integration_complexity=complexity_score,
        feature_coverage=2,  # REST only; no SDK, no MCP
        error_recovery=error_score,
        output_quality=quality_score,
        token_efficiency=0,  # not measurable for local inference
        mcp_compatibility=0,
        wall_time_seconds=round(gen_time + exec_time, 2),
        lines_of_code=loc,
        errors_encountered=all_errors,
        notes=(
            f"Code generation via Ollama /api/generate. "
            f"Static analysis: {error_details}. "
            f"Output quality: {quality_details}. "
            f"LOC: {loc}, gen_time: {gen_time:.1f}s, exec_time: {exec_time:.1f}s."
        ),
    )

    print(f"  Score: {score.total_score}/{score.max_score}")
    save_result(score, subdir="ollama", raw_output=exec_output)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Reducto Agent Benchmark — Ollama (open-source models)")
    print("=" * 60)

    if not os.environ.get("REDUCTO_API_KEY"):
        print("Error: REDUCTO_API_KEY environment variable is not set.")
        sys.exit(1)

    if not _check_ollama():
        print(
            "Error: Ollama is not running or not reachable at "
            f"{OLLAMA_BASE}.\n"
            "Start it with:  ollama serve"
        )
        sys.exit(1)

    print(f"Ollama is running at {OLLAMA_BASE}.")
    print(f"Testing {len(MODELS)} models.\n")

    for model_id, display_name in MODELS:
        bench_model(model_id, display_name)

    print("\nDone. Results saved to benchmark/results/ollama/")


if __name__ == "__main__":
    main()
