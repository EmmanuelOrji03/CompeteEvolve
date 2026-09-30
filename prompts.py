"""Prompts — from the paper's Appendix A.2.

Three prompts, kept here rather than inlined at their call sites so the
wording stays reviewable against the paper:

  A.2.1  SYSTEM_PROMPT    - the agent's role and tool ordering
  A.2.2  EVOLUTION_PROMPT - annotate traits, then mutate/cross over
  A.2.3  FEEDBACK_PROMPT  - reward/punishment, with the top-3 agents'
                            full trait breakdowns

There is no metaprompt step: the task prompt is pre-written and filled
in with the algorithm and dataset names.
"""

from __future__ import annotations

from typing import Optional

# The 13 primary/secondary traits the evolution prompt asks the agent to
# annotate, in the paper's own order (A.2.2).
TRAIT_CHECKLIST = [
    "Inference Time",
    "Syntax errors",
    "Runtime errors",
    "Logical errors",
    "Loss",
    "Model Accuracy",
    "Memory usage",
    "Computational cost",
    "Novelty",
    "Resources",
    "Model accuracy",
    "Time",
]


SYSTEM_PROMPT = """You are {name}, one of several competing agents in an evolutionary \
code-improvement system focused on optimizing machine learning algorithms from \
scikit-learn, evaluated on the '{dataset}' dataset. You must actually IMPROVE code by \
calling your tools — do not just describe what you would do; call the functions.

Your tools and the order they're normally used in:
1. memory — read existing code or the dataset (action: 'read'), or write new code \
(action: 'write'). Use this first to check what already exists for '{algorithm}'.
2. evolution — generates and scores candidate implementations of '{algorithm}' seeded \
from memory, and writes the best one back. Call this to produce an improved candidate.
3. evaluator — actually runs your evolved candidate against the real dataset and scores \
it against the baseline and other agents' candidates.
4. reinforcement — fetches your reward and rank position after evaluation, as feedback \
on how you're doing relative to the other agents.

Call these tools whenever the task calls for producing, testing, or comparing code — do \
not simply answer in prose when a tool call would make real progress on the task.

## STEPS
1. Start by recalling the current baseline code (memory tool, action="read", \
category="code", algorithm="{algorithm}", no "agent" filter).
2. It defines `def run(data_path: str)`, which loads the dataset CSV, splits it into \
train/test itself, trains, and reports test-set accuracy along with its loss and its \
training and inference times. Read the label column the way the baseline already does — \
do not assume a column name.
3. Lines marked "# TRAIT:" mark the parts you are expected to change — hyperparameters, \
model choice, preprocessing/feature handling, or the train/test split itself.
4. Then use your tools, in this order, to make and prove real progress.
5. Iterate again if you still have turns left and your rank isn't first.

### WARNING
1. Do not remove the `run(data_path)` entry point or change its return contract; the \
sandbox that scores your candidates depends on it."""


TASK_PROMPT = """Optimize the '{algorithm}' ({display_name}) scikit-learn algorithm, \
evaluated on the '{dataset}' dataset, which has already been remembered for you.

Work through your tools: recall the baseline from memory, run `evolution` to produce an \
improved candidate, run `evaluator` to score it for real, then `reinforcement` to see \
your rank against the other agents. Keep iterating while you have turns left and your \
rank isn't first."""


# A.2.2 — the annotate-then-mutate prompt. Three modes, matching the
# paper's clauses 5-7: annotate only, first-generation mutation, and
# mutation guided by the best traits so far.
EVOLUTION_PROMPT = """You are an evolutionary agent in a competition with other agents, \
focused on generating and mutating machine learning algorithms from scikit-learn.

1. You are given code. Annotate, using comments, the sections of the code most likely \
responsible for each of the following:
{trait_list}
2. These annotated sections are known as traits.
3. The code is run in a sandbox and numerical values are attached to each trait.
4. Your job is to improve the numerical value of the traits. Devise strategies to achieve \
a HIGH value for positive traits (e.g. model accuracy) and a LOW value for negative traits \
(e.g. loss, time, memory usage). Implement those strategies by generating code with \
improved traits.
5. Aim for at least a 25 percent improvement in performance over the originally annotated \
code.

## DATASET
{dataset_note}

## OUTPUT CONTRACT (do not break this)
The code MUST define exactly this entry point:

    def run(data_path: str) -> dict:
        # loads the CSV at data_path, splits it into train/test ITSELF,
        # trains, evaluates, and returns:
        #   {{"accuracy": <test accuracy 0-1>,
        #     "loss": <log loss, or 1-accuracy if unavailable>,
        #     "training_time": <seconds spent in fit()>,
        #     "inference_time": <seconds spent in predict()>}}

Measure training_time around fit() only and inference_time around predict() only — they \
are scored as separate traits. Never evaluate on the training split, and never fit any \
preprocessing on data before splitting: both are logical errors and are penalised heavily.

Respond with ONLY the Python code — keep your `# TRAIT:` annotation comments in the code, \
but add no explanation outside it and no markdown fences.

{mode_instruction}

{reference_section}"""

_MODE_ANNOTATE_ONLY = """### THIS GENERATION
You have been given the original file. Annotate it thoroughly and carefully using the \
traits listed above, then produce an improved version of it."""

_MODE_FIRST_GENERATION = """### THIS GENERATION
You have the annotated file but no best-trait file yet — this is the first generation. \
Generate modifications of the annotated file that would increase its trait values."""

_MODE_CROSSOVER = """### THIS GENERATION
You have the annotated original code AND a best-traits file from the top samples so far. \
Mutate the annotated original by CROSSING OVER the best-performing sections from those \
top samples, so the result improves on the previous generation. Take the specific \
sections credited with each top sample's best traits and combine them."""


def build_system_prompt(name: str, algorithm: str, dataset: str) -> str:
    return SYSTEM_PROMPT.format(name=name, algorithm=algorithm, dataset=dataset)


def build_task_prompt(algorithm: str, display_name: str, dataset: str) -> str:
    return TASK_PROMPT.format(algorithm=algorithm, display_name=display_name, dataset=dataset)


DEFAULT_DATASET_NOTE = (
    "The dataset CSV has a label column named 'target'; every other column is a feature."
)


def build_evolution_prompt(
    original_code: str,
    mode: str,
    best_traits_markdown: Optional[str] = None,
    dataset_note: Optional[str] = None,
) -> str:
    """`mode` is one of "annotate", "first", "crossover" — the paper's
    clauses 5, 6 and 7 respectively. `best_traits_markdown` is the
    top-3 traits file, included only in crossover mode.

    `dataset_note` describes the actual dataset's shape and which column
    holds the label. It matters for user-supplied CSVs, which may not
    have a column literally called 'target' — without it, generated
    candidates assume one exists and fail with a KeyError."""
    mode_instruction = {
        "annotate": _MODE_ANNOTATE_ONLY,
        "first": _MODE_FIRST_GENERATION,
        "crossover": _MODE_CROSSOVER,
    }.get(mode, _MODE_FIRST_GENERATION)

    reference_section = f"### ORIGINAL ANNOTATED CODE\n{original_code}\n"
    if best_traits_markdown:
        reference_section += (
            f"\n### BEST TRAITS SO FAR (top samples, with their measured trait values)\n"
            f"{best_traits_markdown}\n"
        )

    trait_list = "\n".join(f"    {i + 1}. {t}" for i, t in enumerate(TRAIT_CHECKLIST))
    return EVOLUTION_PROMPT.format(
        trait_list=trait_list,
        dataset_note=dataset_note or DEFAULT_DATASET_NOTE,
        mode_instruction=mode_instruction,
        reference_section=reference_section,
    )


def build_feedback_prompt(rank_position: int, agent_count: int, reward: int, leaderboard_markdown: str) -> str:
    """A.2.3 — the reward/punishment feedback, including the top 3
    agents' full trait breakdowns so the agent can see exactly which
    traits it is losing on."""
    if reward == 1:
        closing = (
            "Keep exploring in this direction and aim at improving your code traits to beat "
            "the traits of other agents."
        )
    else:
        closing = (
            "You are lagging behind. Consider trying a different generation and mutation "
            "strategy to improve your rank. Improve the performance of your code one trait at "
            "a time, starting from the traits with the worst values. Aim to beat the other "
            "agents across all traits."
        )

    return (
        f"Reinforcement feedback: your last sample ranked position {rank_position} of "
        f"{agent_count} agents and earned a reward of {reward}.\n\n"
        f"The top agents scored:\n{leaderboard_markdown}\n{closing}"
    )
