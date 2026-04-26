# Agentic Action-Loop Evaluation

Run date: 2026-04-26

Endpoint: `llama.cpp` OpenAI-compatible `/v1/chat/completions` on RTX 5090, `temperature=0`, thinking disabled.

## Harness

The model is placed in an action/observation loop and must respond with exactly one JSON object per turn:

- tool call: `{"action":"tool","tool":"tool_name","args":{...}}`
- final answer: `{"action":"final","answer":"..."}`

The harness executes deterministic local tools and scores whether the model chose the right tools, used observations correctly, handled missing data, made requested side effects, and stopped with the correct final answer.

Tasks:

- lookup facts and calculate invoice total
- retrieve a policy from docs
- recover from a stale ticket id by searching
- intersect two calendars
- use one bulk inventory lookup
- close only the duplicate issue
- update only one note task
- abstain from tools when instructed

## Results

| Model | Score | Avg latency | Avg steps | Avg tool calls | Schema errors | Invalid JSON |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-27B | 8/8 | 1,478 ms | 2.62 | 1.62 | 0 | 0 |
| Qwen3.6-35B-A3B | 5/8 | 785 ms | 3.75 | 1.25 | 14 | 0 |

Qwen3.6-35B-A3B was faster and always produced syntactically valid JSON, but it often put the tool name in the `action` field instead of using the required `action=tool` schema. It also missed the calendar intersection by answering a slot when Ava was busy.

Qwen3.6-27B was slower, but followed the action schema exactly, recovered from tool misses, made the requested side effects, and completed all tasks.

## Takeaway

For a Hermes-style local agent executor, Qwen3.6-27B is the better current choice from this test. Qwen3.6-35B-A3B is attractive for throughput, but needs either a schema-repair wrapper or a stronger tool-calling prompt before it is safe as the primary executor.

