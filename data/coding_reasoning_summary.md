# Expanded Coding and Reasoning Evaluation

Run date: 2026-04-26

Endpoint: `llama.cpp` OpenAI-compatible `/v1/chat/completions` on RTX 5090, `temperature=0`.

## Test Sets

Reasoning tasks:

- arithmetic expression with mixed operations
- pigeonhole sock draw guarantee
- ordered list transform
- binary plus hexadecimal conversion
- small Chinese remainder problem
- weighted shortest path
- overlapping substring count
- set symmetric-difference sum
- instruction/data boundary
- strict JSON reasoning output

Coding tasks:

- first duplicate index
- merge intervals
- top-k frequent words
- Unix path normalization
- one-line CSV parser
- string-based big integer addition
- longest valid parentheses
- minimum meeting rooms

The coding tests execute generated Python in a temporary directory with deterministic unit assertions. The reasoning tests use exact or JSON-field checkers.

## Results

### Direct Final-Answer Prompts

These prompts ask for only the final answer or only Python code. This matches low-latency API use with thinking disabled, but it punishes models that need visible scratch work for arithmetic and symbolic tasks.

| Model | Reasoning | Coding | Overall | Avg latency |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6-27B | 6/10 | 8/8 | 14/18 | 957 ms |
| Qwen3.6-35B-A3B | 2/10 | 8/8 | 10/18 | 452 ms |

Direct-mode misses:

| Model | Misses |
| --- | --- |
| Qwen3.6-27B | arithmetic, binary+hex, shortest path, symmetric difference |
| Qwen3.6-35B-A3B | arithmetic, pigeonhole, list transform, binary+hex, Chinese remainder, shortest path, substring count, symmetric difference |

### Visible Scratchpad Reasoning

These prompts allow visible work and require a final line of `FINAL: <answer>`. The checker scores only the value after `FINAL:`.

| Model | Reasoning | Avg latency |
| --- | ---: | ---: |
| Qwen3.6-27B | 10/10 | 4,563 ms |
| Qwen3.6-35B-A3B | 10/10 | 1,686 ms |

Both models solved the expanded reasoning set when allowed visible scratch work. Qwen3.6-35B-A3B was about 2.7x faster on these reasoning prompts.

### Hidden Thinking Diagnostic

These requests enabled Qwen thinking with a 512-token completion cap. This is not a maximum-quality setting; it is a latency and usability diagnostic for the endpoint behavior.

| Model | Reasoning | Avg latency | Main failure mode |
| --- | ---: | ---: | --- |
| Qwen3.6-27B | 4/10 | 6,888 ms | Spent token budget in `reasoning_content` and often emitted no final content |
| Qwen3.6-35B-A3B | 1/10 | 2,445 ms | Spent token budget in `reasoning_content` and often emitted no final content |

Hidden thinking needs a higher token budget and explicit per-request control. For this endpoint shape, visible scratchpad prompting was more predictable than enabling hidden thinking under a tight cap.

## Takeaways

- Qwen3.6-27B is less brittle than Qwen3.6-35B-A3B on direct, final-only reasoning prompts in this expanded suite.
- Both models passed all 8 coding tasks after the interval prompt was disambiguated; Qwen3.6-35B-A3B had about 2.2x lower average coding latency.
- With visible scratch work, both models reached 10/10 reasoning, but Qwen3.6-35B-A3B was much faster.
- The expanded tests do not show a broad intelligence win for 27B. They show a narrower direct-answer reasoning advantage for 27B, and a performance/coding advantage for 35B-A3B.
