#!/usr/bin/env python3
"""Small OpenAI-compatible endpoint benchmark for llama.cpp servers."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
import uuid


def post_json(url: str, payload: dict, timeout: int = 240) -> tuple[float, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read())
    return (time.perf_counter() - start) * 1000, data


def chat_payload(model: str, prompt: str, max_tokens: int, thinking: bool = False) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": thinking,
            "preserve_thinking": thinking,
        },
    }


def forced_decode(base_url: str, model: str, max_tokens: int, reps: int) -> list[dict]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    rows = []
    for _ in range(reps):
        prompt = (
            f"Benchmark nonce {uuid.uuid4()}. Output the word token repeatedly, "
            "separated by single spaces. Do not count, explain, punctuate, or stop early."
        )
        wall_ms, data = post_json(url, chat_payload(model, prompt, max_tokens))
        usage = data.get("usage", {})
        timings = data.get("timings", {})
        rows.append(
            {
                "kind": "forced_decode",
                "model": model,
                "max_tokens": max_tokens,
                "finish": data["choices"][0]["finish_reason"],
                "wall_ms": wall_ms,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_tps": timings.get("prompt_per_second"),
                "decode_tps": timings.get("predicted_per_second"),
            }
        )
    return rows


def prefill(base_url: str, model: str, input_words: int, reps: int) -> list[dict]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    vocab = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega".split()
    filler = " ".join(vocab[i % len(vocab)] for i in range(input_words))
    rows = []
    for _ in range(reps):
        prompt = (
            f"Prefill benchmark nonce {uuid.uuid4()}. Read the following filler and "
            f"then reply OK. FILLER:\n{filler}\nAnswer OK only."
        )
        wall_ms, data = post_json(url, chat_payload(model, prompt, 1))
        usage = data.get("usage", {})
        timings = data.get("timings", {})
        rows.append(
            {
                "kind": "prefill",
                "model": model,
                "input_words": input_words,
                "wall_ms": wall_ms,
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens"),
                "prompt_ms": timings.get("prompt_ms"),
                "prompt_tps": timings.get("prompt_per_second"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for key in ("wall_ms", "prompt_tokens", "completion_tokens", "prompt_ms", "prompt_tps", "decode_tps"):
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[key] = {
                "avg": sum(values) / len(values),
                "p50": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    all_rows = []
    for max_tokens in (128, 512):
        rows = forced_decode(args.base_url, args.model, max_tokens, args.reps)
        all_rows.extend(rows)
        print(json.dumps({"summary": "forced_decode", "max_tokens": max_tokens, "stats": summarize(rows)}, sort_keys=True))

    for input_words in (1024, 4096, 16384, 32768):
        reps = args.reps if input_words < 32768 else 1
        rows = prefill(args.base_url, args.model, input_words, reps)
        all_rows.extend(rows)
        print(json.dumps({"summary": "prefill", "input_words": input_words, "stats": summarize(rows)}, sort_keys=True))

    for row in all_rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()

