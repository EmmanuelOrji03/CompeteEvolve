# Running CompeteEvolve

## Requirements

- Rust 1.85 or newer. `rust-toolchain.toml` selects stable, so `rustup` installs it on first build.
- Python 3.10 or newer with `numpy` and `scikit-learn`. The harness uses scikit-learn only for datasets, cross-validation and the reference models. Candidates may not import it.
- A Gemini API key.

```bash
pip install numpy scikit-learn
cp .env.example .env      # then set GEMINI_API_KEY
```

If `python` is not the interpreter with scikit-learn, set `PYTHON=/path/to/python` in `.env`.

## Commands

```bash
# List tasks
cargo run --release -- --list-tasks

# Measure the seed and scikit-learn reference only. No API key needed.
cargo run --release -- --task logistic_regression --baseline-only

# A full competition
cargo run --release -- --task logistic_regression --agents 3 --rounds 3

# Optional: a goal statement and reference material for the agents
cargo run --release -- --task knn \
  --goal "vectorise predict without losing accuracy" \
  --context context/knn.md
```

All flags:

| Flag | Meaning |
|---|---|
| `--task NAME` | `kmeans`, `logistic_regression` or `knn`. Prompts if omitted. |
| `--agents N` | Number of competing agents. Default 3. |
| `--rounds N` | Number of rounds. Default 3. |
| `--goal TEXT` | Optimisation goal shown to the agents. |
| `--context FILE` | File excerpted into the first prompt (capped by `context_max_chars`). |
| `--config FILE` | JSON config. Default `competeevolve.json` if present. |
| `--run-id ID` | Name of the run directory. Default `<task>-<timestamp>`. |
| `--list-tasks` | Print tasks and exit. |
| `--baseline-only` | Measure the seed and scikit-learn reference, then exit. |

## What a run looks like

1. The task, its datasets, and the harness settings.
2. Baselines measured by the harness: the seed implementation and scikit-learn.
3. The run directory under `runs/`.
4. Per round: each agent's tool calls, one line per evaluated candidate, then each agent's notes.
5. A leaderboard after every round with position, score, performance, reward and best candidate ID.
6. A `RESULT` block with the best candidate, its speedup and accuracy change against the seed, and the path to `best.py`.

A candidate line:

```
[agent-1] agent-1-r1-c005: ok valid=true quality 0.9703 time 0.0083s performance 14.6119 novelty 0.176 score 0.5503
```

- `valid=true` means every fold ran and quality stayed within tolerance of the baseline.
- `performance` is relative to the seed. 1.0 equals the seed, higher is better.
- `novelty` is one minus the mean similarity to the seed, the agent's previous candidate and the rivals' best.
- `score` is what agents are ranked by. Invalid candidates score 0.

## Output files

Everything lives under `runs/<run-id>/`, which git ignores:

| File | Contents |
|---|---|
| `candidates.jsonl` | One line per evaluated candidate: code, metrics, parents, operator. |
| `candidates/*.py` | The full source file of every candidate, runnable on its own. |
| `events.jsonl` | Rejections, round summaries, LLM call counts. |
| `summary.json` | Standings, best candidate, top five, improvement over baseline. |
| `best.py` | The best valid candidate as a complete file. |
| `config.json` | The configuration used. |

Re-evaluate any candidate independently:

```bash
python harness/evaluate.py --task logistic_regression --candidate runs/<run-id>/best.py --folds 5 --repeats 3
```

Copy anything worth keeping into `results/`, which is tracked.

## Configuration

`competeevolve.json`, every field optional:

| Field | Default | Meaning |
|---|---|---|
| `agents`, `rounds` | 3, 3 | Competition size. |
| `samples_per_evolve` | 3 | Candidates one `evolve` tool call generates. |
| `max_tool_calls_per_round` | 5 | Per agent. Bounds LLM usage. |
| `mu_threshold` | 0.7 | Stop early when squash(best performance) reaches this. 0.7 is performance 2.33. |
| `seed_parent_probability` | 0.2 | Chance a mutation starts from the seed. |
| `novelty_reject_threshold` | 0.98 | Similarity above which a candidate is rejected unevaluated. |
| `fitness.quality_exponent` | 4 | Higher makes accuracy loss costlier than speed gain. |
| `fitness.speed_exponent` | 1 | Exponent on the speedup. |
| `fitness.novelty_weight` | 0.5 | Weight of novelty in the rank score. |
| `fitness.quality_tolerance` | 0.01 | Valid only if quality >= baseline minus this. |
| `harness.folds`, `harness.repeats` | 3, 2 | Cross-validation size. More is slower but less noisy. |
| `harness.timeout_seconds` | 180 | A candidate exceeding this is killed and marked invalid. |
| `gemini.agent_model` | gemini-3.7-flash | Runs the agent loop. |
| `gemini.generation_model` | gemini-3.7-flash | Writes candidates inside `evolve`. |
| `gemini.requests_per_minute` | 5 | Shared by every call in the run. |

## Rate limits and cost

The free Gemini tier allows about 10 requests and 250k tokens per minute. Agent prompts are large, so the token cap is usually hit first. A `429` is retried with the delay the API requests, up to `max_retries` times. A few per run are normal. A 2-agent, 1-round run used 14 requests. Budget roughly `agents x rounds x (max_tool_calls_per_round x samples_per_evolve + 2)` requests as an upper bound.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `GEMINI_API_KEY is not set` | Create `.env` from `.env.example`. |
| `spawning python for the harness` | Set `PYTHON` in `.env` to an interpreter with numpy and scikit-learn. |
| HTTP 400 on the first call | Model name not available to your key. Set `GEMINI_MODEL` in `.env`. |
| Every candidate invalid with `forbidden imports` | The model keeps wrapping scikit-learn. Add a stronger `--goal`. |
| Constant `429` retries | Lower `gemini.requests_per_minute` or `agents`. |
| Seed fit time varies a lot between runs | Machine load. Timings are per-dataset medians, but keep the machine idle for numbers you plan to report. |

## Tests

```bash
cargo test                                            # unit tests plus a harness integration test
python harness/evaluate.py --task knn --seed-file     # harness smoke test
```
