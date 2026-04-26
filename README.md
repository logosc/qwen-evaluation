# Qwen3.6 llama.cpp Evaluation on RTX 5090

Benchmarks from a local RTX 5090 server running `llama.cpp` as an OpenAI-compatible endpoint.

## Environment

- Host: `192.168.3.21`
- GPU: NVIDIA GeForce RTX 5090, 32 GB VRAM
- Driver: 570.133.07
- CUDA toolkit: 12.8
- llama.cpp: `b8934` / `b760272f1`
- Build flags: CUDA enabled, flash attention enabled, SM120 target
- Server shape: `llama-server`, OpenAI-compatible `/v1`, `--parallel 1`

Common server settings:

```bash
--ctx-size 131072
--gpu-layers 999
--flash-attn on
--cache-type-k q8_0
--cache-type-v q8_0
--batch-size 2048
--ubatch-size 512
--threads 16
--no-mmap
--jinja
--reasoning-format deepseek
```

Thinking was disabled by default for endpoint usability:

```json
{"enable_thinking": false, "preserve_thinking": false}
```

## Models

| Model | Quant | File | Runtime VRAM |
| --- | --- | --- | --- |
| Qwen3.6-27B | Unsloth `UD-Q4_K_XL` | `Qwen3.6-27B-UD-Q4_K_XL.gguf` | ~21.8 GiB |
| Qwen3.6-35B-A3B | Unsloth `UD-Q4_K_XL` | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | ~23.4 GiB |

The 35B-A3B MoE model uses more model weight memory, but less KV memory at this configuration, so the runtime VRAM delta was only about 1.6 GiB.

## Decode Throughput

Forced decode prompts were used to reduce early-stop artifacts.

| Model | 128 token decode | 512 token decode |
| --- | ---: | ---: |
| Qwen3.6-27B | ~69 tok/s | ~68.5 tok/s |
| Qwen3.6-35B-A3B | ~213 tok/s | ~213 tok/s |

The 35B-A3B model decoded roughly 3.1x faster than the dense 27B model on this RTX 5090 setup.

## Prefill Throughput

These runs used `max_tokens=1` and randomized prompts to avoid prompt-cache hits.

| Model | Prompt tokens | Prompt time | Prefill throughput |
| --- | ---: | ---: | ---: |
| Qwen3.6-27B | 1,265 | 415 ms | 3,048 tok/s |
| Qwen3.6-27B | 4,848 | 1,381 ms | 3,511 tok/s |
| Qwen3.6-27B | 19,183 | 5,611 ms | 3,419 tok/s |
| Qwen3.6-27B | 38,295 | 12,267 ms | 3,122 tok/s |
| Qwen3.6-35B-A3B | 1,264 | 203 ms | 6,284 tok/s |
| Qwen3.6-35B-A3B | 4,849 | 566 ms | 8,563 tok/s |
| Qwen3.6-35B-A3B | 19,184 | 2,214 ms | 8,664 tok/s |
| Qwen3.6-35B-A3B | 38,299 | 4,672 ms | 8,198 tok/s |

35B-A3B was about 2.6x faster for long prompt prefill.

## Streaming Latency

Short streaming prompts, thinking disabled:

| Model | Avg TTFT | Median TTFT |
| --- | ---: | ---: |
| Qwen3.6-27B | 206 ms | 179 ms |
| Qwen3.6-35B-A3B | 109 ms | 107 ms |

## Concurrency

The server was launched with `--parallel 1`, so concurrent HTTP requests queue. Four concurrent 256-token requests showed serial tail behavior. This is expected and should not be interpreted as multi-user throughput.

For multi-client serving, retest with `--parallel 2` or higher and re-evaluate VRAM.

## Intelligence Smoke Tests

Eight deterministic tests were used:

- exact arithmetic
- pigeonhole logic
- ordered list transform
- instruction/data boundary
- strict JSON schema
- one-line code bug fix
- long-context needle retrieval
- tool-call JSON shape

| Model | Thinking mode | Score | Avg latency |
| --- | --- | ---: | ---: |
| Qwen3.6-27B | off | 5/8 | 698 ms |
| Qwen3.6-27B | on | 7/8 | 7,331 ms |
| Qwen3.6-35B-A3B | off | 5/8 | 310 ms |
| Qwen3.6-35B-A3B | on | 7/8 | 3,514 ms |

Both models missed the same small reasoning tasks with thinking disabled. With thinking enabled, both fixed those misses but one code-debug test failed because the model spent the full token budget in reasoning and never emitted final content.

Operational takeaway: keep thinking disabled for low-latency tool/API use, and enable it per request for reasoning-heavy tasks with a sufficiently high token budget.

## Expanded Coding and Reasoning Tests

I added a larger auto-scored suite in `scripts/coding_reasoning_eval.py`:

- 10 reasoning tasks covering arithmetic, graph distance, modular constraints, set logic, substring counting, instruction boundary handling, and strict JSON output.
- 8 coding tasks that execute generated Python against deterministic unit assertions.
- three prompt modes: direct final-answer prompts, visible scratchpad with `FINAL:`, and a capped hidden-thinking diagnostic.

| Mode | Model | Reasoning | Coding | Overall | Avg latency |
| --- | --- | ---: | ---: | ---: | ---: |
| Direct, thinking off | Qwen3.6-27B | 6/10 | 8/8 | 14/18 | 957 ms |
| Direct, thinking off | Qwen3.6-35B-A3B | 2/10 | 8/8 | 10/18 | 452 ms |
| Visible scratchpad, thinking off | Qwen3.6-27B | 10/10 | n/a | 10/10 | 4,563 ms |
| Visible scratchpad, thinking off | Qwen3.6-35B-A3B | 10/10 | n/a | 10/10 | 1,686 ms |
| Hidden thinking, 512-token cap | Qwen3.6-27B | 4/10 | n/a | 4/10 | 6,888 ms |
| Hidden thinking, 512-token cap | Qwen3.6-35B-A3B | 1/10 | n/a | 1/10 | 2,445 ms |

The direct results show an interesting split: 27B is better on final-only reasoning prompts, while both models pass the coding set and 35B-A3B is faster. When allowed visible scratch work, both models solved all 10 reasoning tasks, but 35B-A3B stayed much faster.

The hidden-thinking rows should be treated as an endpoint usability diagnostic, not a best-quality result. With a tight 512-token cap both models often spent the whole budget in `reasoning_content` and emitted no final answer.

Raw results are in `data/coding_reasoning_*.jsonl`; a concise report is in `data/coding_reasoning_summary.md`.

## TurboQuant Status

I would not use TurboQuant as the primary endpoint path for this setup yet.

- Mainline `llama.cpp` did not expose `tbq*` KV cache types in the compiled CUDA server.
- The upstream TurboQuant KV PR was still CPU-only in its initial scope, with CUDA listed as follow-up work.
- Qwen3.6 TurboQuant weight GGUFs exist, but they depend on custom forks rather than a mature mainline CUDA path.

The mature baseline today is latest mainline `llama.cpp` plus Unsloth GGUF quants and normal KV cache types such as `q8_0`.

## Recommendation

For this RTX 5090 host, Qwen3.6-35B-A3B `UD-Q4_K_XL` is the better performance candidate for throughput. It is faster on decode, prefill, TTFT, coding latency, and visible scratchpad reasoning, while using only modestly more VRAM than the 27B dense model under the same 128K/q8-KV setup.

Qwen3.6-27B did better on direct final-only reasoning in the expanded suite, so I would not call 35B-A3B strictly smarter across every interaction style. The practical recommendation is: use 35B-A3B for default serving, and use visible scratchpad prompting or higher thinking budgets for reasoning-heavy requests.
