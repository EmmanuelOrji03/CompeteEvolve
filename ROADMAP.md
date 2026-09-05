# CompeteEvolve Roadmap

Status of the project as of 2026-09-05, and the plan to reach the goal stated in
the paper: a measurable speed or accuracy improvement of a real machine-learning
algorithm, produced by competing LLM agents running evolutionary search.

Legend: `[x]` done, `[~]` in progress, `[ ]` not started. Last updated 2026-09-05.

---

## 1. Where the project stood before this roadmap

What the code did:

- A Rust CLI spawned N Gemini tool-calling agents that shared four tools:
  `database`, `evolution`, `evaluator`, `reinforcement`.
- The user picked one scikit-learn source dump from `context/` and typed a prompt.
  Both went to Gemini once to produce a "refined prompt".
- The agents received only the refined prompt. They never saw the original code,
  and the shared database started empty.
- The "database" was an in-memory vector of structs plus a hashed bag-of-words
  embedding. Nothing was persisted.
- The "evolution" tool asked Gemini for a few independent rewrites. No crossover,
  no mutation of best traits, no islands, no archive.
- Fitness was simulated. "Time" was a busy loop proportional to word count.
  "Accuracy" was 0.5 plus bonuses for the substring `return` and for braces.
  No Python was ever executed. Syntax errors were bracket balance. Runtime
  errors were counts of Rust idioms (`.unwrap()`, `panic!`) in Python code.
- The "reinforcement" tool returned a sentence of feedback and nothing else
  changed. No loop ran until the threshold mu was reached.
- `benchmark.rs` was empty and not compiled. `gemini-rust` was a dependency
  but unused. The crate required Rust 1.85+ (edition 2024) while the machine
  had 1.79.

Inverted formulas (in code and in the paper draft):

| Quantity            | Was                                   | Effect                                |
|---------------------|---------------------------------------|---------------------------------------|
| Performance         | time / accuracy                       | Slower code scored higher             |
| Novelty             | cosine similarity to original/rivals  | Copying scored as most novel          |
| Reward              | rank position >= n/2 and not 1        | Worse half rewarded, winner got 0     |
| Accuracy (paper)    | divided by code length L              | Shorter beat correct                  |
| DB "top" ordering   | similarity to earlier entries         | Least original entry returned first   |

Other defects: candidate IDs collided across agents; paper novelty has three
terms, code averaged two; quota retry existed only in the agent loop, not in
sample generation; rank positions were computed over whichever agents had
finished, so results were nondeterministic; `Q` declared in (0,1) but only took
values 0 and 1; paper appendix described a Python/Node layout that does not
match this repo; Implementation, Results and Conclusion sections were empty.

---

## 2. Phase 0 - Make it build and run

- [x] Upgrade Rust toolchain to 1.85+ and pin it with `rust-toolchain.toml`.
- [x] Switch `reqwest` to `native-tls` so the build does not need cmake/nasm
      (`aws-lc-sys`).
- [x] Remove the unused `gemini-rust` dependency.
- [x] Fix `.gitignore` (inline comment made the `Cargo.lock` line a no-op; the
      lock file should be committed for a binary anyway).
- [x] Add `.env.example` and a README that explains how to run.
- [x] Make the model name configurable (`GEMINI_MODEL`, default `gemini-3.7-flash`).

## 3. Phase 1 - Real evaluation (the single biggest unblock)

- [x] Python harness `harness/evaluate.py`: imports a candidate file, checks it
      with `ast.parse`, fits it on a fixed set of scikit-learn datasets with
      repeated stratified k-fold, and reports fit time, predict time, accuracy,
      exceptions, and a JSON verdict. Hard timeout and deterministic seeds.
- [x] Seed tasks in `harness/tasks/`: self-contained NumPy implementations of
      `kmeans`, `logistic_regression`, `knn` with `# EVOLVE-BLOCK-START/END`
      markers, plus a scikit-learn reference for each so the harness can
      report the baseline alpha.
- [x] Rust `evaluator` calls the harness in a subprocess with a timeout instead
      of simulating. Syntax errors come from `ast.parse`, runtime errors from
      harness exceptions.
- [x] Baseline (alpha) computed once per run from the seed and cached.
- [ ] Optional Docker sandbox (`harness/Dockerfile`) for untrusted candidates.
- [ ] Extend the dataset set toward OMEGA's 20-dataset Infinity-Bench style
      benchmark with min-max normalized scores.

## 4. Phase 2 - Correct fitness, novelty and reward

- [x] Novelty = 1 - mean cosine similarity (to original, to own previous
      candidate, to rivals), matching the three terms in the paper.
- [x] Performance = (accuracy / baseline accuracy) x (baseline time / time),
      penalised by 2^-(syntax errors + runtime errors). Higher is better.
- [x] Rank score = weighted combination of performance and novelty, with
      weights in config.
- [x] Reward goes to the top half of the leaderboard (rank position <= n/2),
      plus a continuous "improvement over baseline" signal.
- [x] Database "top" ordering uses rank score, not similarity.
- [x] Unique candidate IDs (agent + round + generation + sample).
- [x] Unit tests asserting: faster and equally accurate beats original; a
      verbatim copy has novelty 0; the best agent gets reward 1.

## 5. Phase 3 - Real evolution

- [x] Persistent archive: JSONL on disk under `runs/<run-id>/` with every
      candidate, its metrics, parent IDs and the operator used. Survives
      restarts.
- [x] Parent sampling from the archive weighted by fitness and novelty
      (ShinkaEvolve-style) instead of always rewriting the original.
- [x] Two operators: `mutate` (edit inside the EVOLVE block of one parent) and
      `crossover` (two parents plus an "inspiration" prompt, CodeEvolve-style).
- [x] Novelty rejection: a candidate whose similarity to any archived entry
      exceeds a threshold is discarded before evaluation.
- [x] Only the EVOLVE block is regenerated; the rest of the file is kept, so
      the harness interface cannot drift.
- [ ] Islands with periodic migration.
- [ ] Real embedding model (Gemini embedding endpoint) instead of hashed
      bag-of-words, behind a feature flag.

## 6. Phase 4 - Feedback that changes behaviour, and true competition

- [x] Agents receive the actual seed code, the task description and the
      baseline metrics in their first prompt. Nothing has to be invented.
- [x] The run is organised in rounds. After each round every agent gets the
      leaderboard, its reward, and the best rival's EVOLVE block, injected
      into its next prompt. This is the in-context "reinforcement" step.
- [x] Shared rate limiter across all Gemini calls (free tier is ~10 RPM), with
      retry on `RESOURCE_EXHAUSTED` everywhere, not only in the agent loop.
- [x] Leaderboard is computed only at round barriers, so ranks are
      deterministic for a given set of candidates.
- [x] Cheap model for generation, stronger model for planning/repair, as the
      paper describes (`gemini.generation_model` vs `gemini.agent_model`; both
      default to Flash, set `agent_model` to a Pro model to use the split).
- [x] Live end-to-end run with a real `GEMINI_API_KEY`. First run
      (2026-09-05, `logistic_regression`, 2 agents, 1 round, 14 LLM calls,
      7 candidates, all valid) produced an L-BFGS softmax implementation
      that beats both the seed and scikit-learn. Re-measured in a fresh
      process with 5 folds x 3 repeats:

      | Implementation | Accuracy | Fit time per fold |
      |---|---|---|
      | Seed (gradient descent) | 0.9525 | 137 ms |
      | scikit-learn LogisticRegression | 0.9711 | 28 ms |
      | Evolved best | 0.9730 | 12.6 ms |

      Artifacts: `results/logistic_regression/`. Still to do: `kmeans` and
      `knn`, multi-round runs, and repeated runs with different seeds so the
      paper can report variance.
- [ ] Ablation runner: n competing agents vs 1 agent with n x budget vs n
      collaborating agents (shared notes). CORAL's evidence points toward
      collaboration; the paper's rivalry claim needs this test.

## 7. Phase 5 - Engineering hygiene

- [x] Config file (`competeevolve.json`) for agents, rounds, populations,
      thresholds, weights, model names, timeouts.
- [x] Structured run log (`runs/<id>/events.jsonl`) and a final `summary.json`.
- [x] `cargo test` covers the database, fitness math and reward.
- [x] CI workflow (fmt, clippy, test, harness smoke test).
- [ ] `benchmark.rs` either becomes the ablation runner or is deleted.

## 8. Phase 6 - Paper

- [ ] Fill Implementation (Section 4) from this repo, not the old Python layout.
- [ ] Results (Section 5): report baseline vs best evolved candidate per task
      (fit time, accuracy, CV std), number of LLM samples used, and the
      ablation above. Compare against OpenEvolve/ShinkaEvolve on the same seed.
- [ ] Fix notation: novelty as 1 - sim, reward Q in {0,1}, remove /L from
      functional accuracy or justify it, R is real-valued not natural.
- [ ] Align Appendix A.2 with this repository.
- [ ] Only keep "significant improvement" in the abstract if Section 5 shows it.

---

## 9. References that shaped this plan

- AlphaEvolve (Novikov et al., 2025) - EVOLVE-BLOCK markers, evaluator cascade.
- OpenEvolve - open reimplementation of the above.
- ShinkaEvolve (Lange et al., 2025, ICLR 2026) - parent sampling, novelty
  rejection sampling, bandit LLM ensemble. https://arxiv.org/abs/2509.19349
- CodeEvolve (Assumpcao et al., 2025) - islands, inspiration crossover,
  meta-prompting. https://arxiv.org/abs/2510.14150
- CORAL (Qu et al., 2026) - multi-agent evolution through shared memory;
  collaboration beat independent agents. https://arxiv.org/abs/2604.01658
- OMEGA (ICLR 2026) - generating scikit-learn compatible classifiers and
  benchmarking them on 20 datasets. https://arxiv.org/abs/2604.26211
