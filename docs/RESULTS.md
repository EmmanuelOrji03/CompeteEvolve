# Results

Every number here was produced by `harness/evaluate.py` on real code. Nothing is estimated.

## Setup

| Item | Value |
|---|---|
| Date | 2026-09-05 |
| Machine | Windows 11, Python 3.12.4, numpy 2.1.3, scikit-learn 1.6.1 |
| Model | gemini-3.7-flash for both agent loop and generation |
| Datasets | iris, wine, breast_cancer, digits (classification); iris, wine, digits (clustering) |
| Preprocessing | StandardScaler fit on the training fold |
| Quality metric | Held-out accuracy (classification), adjusted Rand index (clustering) |
| Timing | Wall time per fold, median within a dataset, mean across datasets |

## Baselines

Measured with 3 folds x 2 repeats. These are what the agents must beat.

| Task | Seed quality | Seed time | scikit-learn quality | scikit-learn time |
|---|---|---|---|---|
| logistic_regression | 0.9526 | 108 ms | 0.9689 | 35 ms |
| knn | 0.9636 | 40 ms | 0.9636 | 28 ms |
| kmeans (ARI) | 0.673 | 71 ms | 0.662 | 433 ms |

Seed times vary by up to 3x with machine load. Compare candidates only against a baseline measured in the same run, which is what the system does.

## Run 1: logistic_regression, 2 agents, 1 round

Command:

```bash
cargo run --release -- --task logistic_regression --agents 2 --rounds 1 --run-id live-smoke-1
```

| Item | Value |
|---|---|
| LLM calls | 14 |
| Candidates evaluated | 7, all valid |
| Near-duplicates rejected before evaluation | 1 |
| Rate-limit retries | 2, both recovered |
| Threshold mu reached | Yes, 0.936 against 0.7 |

Both agents began by reading the leaderboard, then wrote candidates themselves with `submit_code` instead of delegating to the generator. Both independently chose a multinomial softmax formulation solved by L-BFGS. Agent 1 iterated on memory layout and allocation. Agent 2 tried cutting iterations, had one attempt rejected as a near-duplicate of its own code, and switched to per-class binary fits with an adaptive step.

Leaderboard after the round:

| Position | Agent | Best score | Best performance | Reward | Valid |
|---|---|---|---|---|---|
| 1 | agent-1 | 0.5503 | 14.61 | 1 | 4/4 |
| 2 | agent-2 | 0.5164 | 8.50 | 0 | 3/3 |

All candidates in that run, from `candidates.jsonl`:

| Candidate | Agent | Quality | Time | Performance | Novelty | Score |
|---|---|---|---|---|---|---|
| agent-1-r1-c001 | 1 | 0.9708 | 16.4 ms | 7.42 | 0.225 | 0.540 |
| agent-2-r1-c002 | 2 | 0.9708 | 23.2 ms | 5.23 | 0.197 | 0.502 |
| agent-1-r1-c003 | 1 | 0.9704 | 15.4 ms | 7.89 | 0.178 | 0.523 |
| agent-2-r1-c004 | 2 | 0.9706 | 14.3 ms | 8.50 | 0.154 | 0.516 |
| agent-1-r1-c005 | 1 | 0.9703 | 8.3 ms | 14.61 | 0.176 | 0.550 |
| agent-2-r1-c006 | 2 | 0.9723 | 38.7 ms | 3.16 | 0.235 | 0.469 |
| agent-1-r1-c007 | 1 | 0.9703 | 7.1 ms | 17.17 | 0.155 | 0.546 |

Candidate c007 is faster than the winner c005 but scores lower because it is a close relative of c005, so novelty discounts it. That is the intended effect of the rank score.

### Independent re-measurement

The winning file was re-evaluated in a fresh process with more folds:

```bash
python harness/evaluate.py --task logistic_regression \
  --candidate results/logistic_regression/live-smoke-1_best.py --folds 5 --repeats 3
```

| Implementation | Accuracy | Fit time per fold | Predict time per fold |
|---|---|---|---|
| Seed (gradient descent, one-vs-rest) | 0.9525 | 137 ms | 0.06 ms |
| scikit-learn LogisticRegression (lbfgs, max_iter 1000) | 0.9711 | 28 ms | 0.41 ms |
| Evolved best (agent-1-r1-c005) | 0.9730 | 12.6 ms | 0.05 ms |

Per dataset, evolved best:

| Dataset | Accuracy | Fit time |
|---|---|---|
| iris | 0.9667 | 6.6 ms |
| wine | 0.9813 | 3.5 ms |
| breast_cancer | 0.9736 | 9.5 ms |
| digits | 0.9703 | 30.8 ms |

The evolved implementation is 156 lines, imports only NumPy, and beats scikit-learn's own logistic regression on this benchmark in both accuracy and fit time.

Artifacts: `results/logistic_regression/live-smoke-1_best.py`, `live-smoke-1_summary.json`, `live-smoke-1_candidates.jsonl`.

## Caveats

- One run is one data point. Report variance over several seeds before drawing conclusions.
- Logistic regression was the easiest target: the seed used fixed-step gradient descent, so a proper solver was an obvious win. The kNN seed already matches scikit-learn's accuracy, so gains there can only be speed.
- The comparison is against scikit-learn on four small datasets with standardised features. It says nothing about large, sparse or ill-conditioned problems.
- Both agents chose the same algorithm family. The novelty term separated them, but a run with more agents or rounds is needed to see whether competition produces diverse approaches.

## Still to run

- `kmeans` and `knn` with 3 agents and 3 rounds.
- Five seeds per task for mean and standard deviation.
- The ablation the paper needs: n competing agents against one agent with n times the budget, and against n collaborating agents.
