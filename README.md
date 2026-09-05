# CompeteEvolve

Competing LLM agents that improve machine-learning algorithm implementations
through evolutionary search, with every candidate measured for real.

Several Gemini agents share one archive. Each round, every agent generates
candidates (mutation of one parent, or crossover of two parents with one taken
from a rival), the Python harness runs each candidate on scikit-learn datasets
and reports quality and wall time, and the agents are ranked on their best
valid candidate. The top half earn a reward, everyone sees the leaderboard and
the best rival's code, and the next round begins.

Documentation:

- [docs/RUNNING.md](docs/RUNNING.md): setup, commands, output files, configuration, troubleshooting.
- [docs/RESULTS.md](docs/RESULTS.md): measured baselines and every live run, with the exact commands.
- [ROADMAP.md](ROADMAP.md): what the first draft did, what was wrong with it, and what is still to do.

## Layout

```
harness/
  evaluate.py          real evaluation: ast check, import allowlist, k-fold timing + quality
  tasks/kmeans.py      seed implementations (plain NumPy) with EVOLVE-BLOCK markers
  tasks/logistic_regression.py
  tasks/knn.py
src/
  main.rs              CLI
  config.rs            competeevolve.json + CLI flags
  gemini.rs            rate-limited Gemini client with retries
  harness.rs           subprocess bridge to evaluate.py, EVOLVE-BLOCK template
  fitness.rs           performance / novelty / rank / reward (unit-tested)
  archive.rs           persistent JSONL archive, parent sampling, leaderboard
  evolution.rs         mutate + crossover operators, novelty rejection, evaluation
  agent.rs             the tool-calling agent loop
  prompts.rs           all prompt text
  orchestrator.rs      rounds, barrier, feedback, summary
context/               scikit-learn source dumps, optional reference material (--context)
runs/<run-id>/         candidates.jsonl, candidates/*.py, events.jsonl, summary.json, best.py
```

## Requirements

- Rust 1.85 or newer (`rust-toolchain.toml` selects stable).
- Python 3.10+ with `numpy` and `scikit-learn` on the interpreter named by
  `PYTHON` (default `python`). scikit-learn is used only by the harness for
  datasets, cross-validation and the reference models; candidates may not
  import it.
- A Gemini API key in `.env` (copy `.env.example`).

## Run

```bash
cargo run --release -- --list-tasks
cargo run --release -- --task kmeans --baseline-only        # no API key needed
cargo run --release -- --task logistic_regression --agents 3 --rounds 3
cargo run --release -- --task knn --goal "vectorise predict without losing accuracy" --context context/knn.md
```

Everything about a run is written under `runs/<task>-<timestamp>/`. The best
valid candidate is saved as `best.py`, ready to evaluate again with

```bash
python harness/evaluate.py --task kmeans --candidate runs/<run-id>/best.py
```

## How fitness works

For a candidate with quality `Q`, wall time `T` (fit + predict per fold), and
baseline (seed) values `Q0`, `T0`:

```
performance = 2^-(syntax + runtime errors) * (folds ok / folds) * (Q/Q0)^4 * (T0/T)
valid       = every fold ran and Q >= Q0 - 0.01
novelty     = 1 - mean similarity to (seed, own previous candidate, rivals' best)
rank score  = performance/(1+performance) * (0.5 + 0.5 * novelty)      (valid only)
reward      = 1 for the top half of agents by best rank score, else 0
```

Exponents, weights and tolerance live in `competeevolve.json`.

## First result

One live run on `logistic_regression` (2 agents, 1 round, 14 Gemini calls)
produced a pure-NumPy L-BFGS softmax regression. Re-measured independently
with 5 folds x 3 repeats:

| Implementation | Accuracy | Fit time per fold |
|---|---|---|
| Seed (gradient descent) | 0.9525 | 137 ms |
| scikit-learn LogisticRegression | 0.9711 | 28 ms |
| Evolved best | 0.9730 | 12.6 ms |

The code, summary and every candidate of that run are kept under
`results/logistic_regression/`. Runs write to `runs/`, which is ignored by
git; copy anything worth keeping into `results/`.

## Rate limits

The free Gemini tier allows roughly 10 requests and 250k tokens per minute.
The default `gemini.requests_per_minute` is 5 because agent prompts are large
and the token limit is usually hit first. A 429 is retried with the delay the
API asks for; a few per run are normal.

## Tests

```bash
cargo test
python harness/evaluate.py --task knn --seed-file
```
