# Intelligence Smoke Tests

The tests were intended to be small, deterministic, and easy to score without another judge model.

| Test | Expected |
| --- | --- |
| exact arithmetic | `390` |
| pigeonhole socks | `5` |
| ordered list transform | `12` |
| instruction/data boundary | `kiwi` |
| JSON schema | valid JSON with `winner=Ada`, `score=42` |
| one-line code bug fix | `seen.add(x)` |
| long-context needle | `ORCHID-7319` |
| tool JSON | valid JSON for `read_file` and `/tmp/report.txt` |

## Results

| Model | Thinking off | Thinking on |
| --- | ---: | ---: |
| Qwen3.6-27B `UD-Q4_K_XL` | 5/8 | 7/8 |
| Qwen3.6-35B-A3B `UD-Q4_K_XL` | 5/8 | 7/8 |

Both models missed the same three reasoning tasks with thinking disabled:

- exact arithmetic
- pigeonhole socks
- ordered list transform

With thinking enabled, both models fixed those three misses. Both failed the code-debug test in thinking mode because reasoning consumed the full token limit and no final answer was emitted. With thinking disabled, both solved the code-debug test quickly.

