"""
Quick sanity test for tiny local models via llama-server (OpenAI-compatible API).
Tests tool-calling capability — specifically whether the model can invoke reducto_parse.

Usage:
    python bench_local_models.py --model-path /path/to/model.gguf --model-name "FunctionGemma-270M-Q6"
    python bench_local_models.py --hf-repo airev-ae/Qwen-0.8B-AgentJSON --hff model.safetensors --model-name "Qwen-0.8B-AgentJSON"
"""

import argparse
import json
import os
import subprocess
import sys
import time
import requests as req

REDUCTO_BASE = "https://platform.reducto.ai"
TEST_DOC = "https://cdn.reducto.ai/samples/fidelity-example.pdf"
SERVER_PORT = 18080
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "reducto_parse",
            "description": "Parse a document URL with Reducto. Returns extracted markdown text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_url": {"type": "string", "description": "URL of the document to parse"}
                },
                "required": ["input_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reducto_extract",
            "description": "Extract structured JSON from a document. Pass URL and a JSON Schema string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_url": {"type": "string"},
                    "schema_json": {"type": "string", "description": "JSON Schema string"},
                },
                "required": ["input_url", "schema_json"],
            },
        },
    },
]

PROMPTS = [
    ("tool_call",   f"Please parse this document and return its text: {TEST_DOC}"),
    ("json_output", f"Extract the account number from this PDF: {TEST_DOC}. Use reducto_extract with a schema that has an 'account_number' field."),
    ("reasoning",   f"What document type is this? Classify it: {TEST_DOC}"),
]


def wait_for_server(timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = req.get(f"{SERVER_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def call_server(prompt: str, tools: list | None = None, max_tokens: int = 4096) -> dict:
    payload: dict = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    r = req.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def run_test(model_name: str) -> dict:
    results = {}
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    for test_name, prompt in PROMPTS:
        print(f"\n[{test_name}] {prompt[:80]}...")
        try:
            resp = call_server(prompt, tools=TOOLS if test_name != "reasoning" else None)
            choice = resp["choices"][0]
            finish = choice.get("finish_reason", "?")
            msg = choice["message"]

            if finish == "tool_calls" and msg.get("tool_calls"):
                tc = msg["tool_calls"][0]["function"]
                print(f"  ✅ TOOL CALL: {tc['name']}({tc['arguments'][:80]}...)")
                results[test_name] = {"status": "tool_call", "tool": tc["name"], "args": tc["arguments"]}
            elif msg.get("content"):
                content = msg["content"][:200]
                print(f"  ⚠️  TEXT: {content}...")
                results[test_name] = {"status": "text", "content": content}
            else:
                print(f"  ❌ EMPTY: finish={finish}")
                results[test_name] = {"status": "empty", "finish": finish}
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results[test_name] = {"status": "error", "error": str(e)}

    # Summary
    tool_calls = sum(1 for r in results.values() if r.get("status") == "tool_call")
    print(f"\nSummary: {tool_calls}/{len(PROMPTS)} prompts triggered tool calls")
    print(f"Verdict: {'✅ Worth testing in main benchmark' if tool_calls >= 2 else '⚠️  Marginal' if tool_calls == 1 else '❌ Skip — cannot do tool calling'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Quick tool-calling test for local GGUF models")
    parser.add_argument("--model-path", help="Path to local GGUF file")
    parser.add_argument("--hf-repo", help="HuggingFace repo (e.g. unsloth/functiongemma-270m-it-GGUF)")
    parser.add_argument("--hff", help="HuggingFace file (e.g. functiongemma-270m-it-Q6_K.gguf)")
    parser.add_argument("--model-name", required=True, help="Display name for the model")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    args = parser.parse_args()

    if not args.model_path and not args.hf_repo:
        print("Error: --model-path or --hf-repo required")
        sys.exit(1)

    # Build llama-server command
    cmd = [
        "llama-server",
        "--port", str(SERVER_PORT),
        "--host", "127.0.0.1",
        "--ctx-size", str(args.ctx_size),
        "--n-gpu-layers", str(args.n_gpu_layers),
        "--jinja",          # use model's built-in jinja chat template (enables tool-call format)
        "--log-disable",
    ]
    if args.model_path:
        cmd += ["--model", args.model_path]
    elif args.hf_repo:
        cmd += ["--hf-repo", args.hf_repo]
        if args.hff:
            cmd += ["--hf-file", args.hff]

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_API_KEY")
    if hf_token:
        cmd += ["--hf-token", hf_token]

    print(f"Starting llama-server: {' '.join(cmd[:6])} ...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        print("Waiting for server to be ready...")
        if not wait_for_server(120):
            print("ERROR: Server failed to start within 120s")
            sys.exit(1)
        print("Server ready.")

        results = run_test(args.model_name)

        # Save results
        out_path = f"/tmp/local_model_test_{args.model_name.replace(' ', '_').replace('/', '_')}.json"
        with open(out_path, "w") as f:
            json.dump({"model": args.model_name, "results": results}, f, indent=2)
        print(f"\nResults saved to: {out_path}")

    finally:
        proc.terminate()
        proc.wait()
        print("Server stopped.")


if __name__ == "__main__":
    main()
