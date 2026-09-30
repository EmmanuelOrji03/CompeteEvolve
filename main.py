from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

import benchmark
import sklearn_algorithms
import user_inputs
from agent import Agent, SharedState, memory
from agent.llm_client import ModelRoles, create_client
from prompts import DEFAULT_DATASET_NOTE, build_task_prompt

# Paper §2 defaults.
DEFAULT_AGENTS = 5
DEFAULT_ISLANDS = 4
DEFAULT_SAMPLES_PER_ISLAND = 5
DEFAULT_GENERATIONS = 3

VALID_PROVIDERS = ("gemini", "openrouter")


@dataclass
class RunInputs:
    algorithm: str
    display_name: str
    dataset_name: str
    baseline_code: str
    dataset_csv: str
    dataset_note: str


async def main() -> None:
    load_dotenv()
    options = _parse_args()

    roles = _choose_roles(options)
    missing = roles.missing_keys()
    if missing:
        print(f"Error: {', '.join(missing)} not set (add it to your .env file)", file=sys.stderr)
        sys.exit(1)
    print(f"LLM roles — {roles.describe()}")

    try:
        inputs = await _resolve_inputs(options)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Optimizing: {inputs.display_name} on {inputs.dataset_name}")

    shared = SharedState(dataset_note=inputs.dataset_note)
    await _seed_memory(shared, inputs)

    num_agents = options.agents or _read_agent_count()
    task_prompt = build_task_prompt(inputs.algorithm, inputs.display_name, inputs.dataset_name)
    task_prompt += (
        f"\n\nWhen you call `evolution`, use generations={DEFAULT_GENERATIONS}, "
        f"population_size={DEFAULT_ISLANDS}, samples_per_population={DEFAULT_SAMPLES_PER_ISLAND}."
    )

    agents = [
        Agent(f"agent-{i + 1}", inputs.algorithm, inputs.dataset_name, roles, shared)
        for i in range(num_agents)
    ]

    print(f"\nRunning {num_agents} agent(s) concurrently...\n")
    results = await asyncio.gather(
        *(agent.send(task_prompt) for agent in agents), return_exceptions=True
    )

    for agent, result in zip(agents, results):
        if isinstance(result, Exception):
            print(f"{agent.name}: error - {result}")
        else:
            print(f"{agent.name}: {result}")

    for agent in agents:
        await agent.aclose()

    # A dedicated client for the report's annotation and logical-error
    # review, since every agent's client is closed by now.
    annotation_client = create_client(
        roles.evolution_provider, roles.evolution_api_key, roles.evolution_model, shared.rate_limiter
    )
    try:
        await benchmark.run_and_print(
            shared, inputs.algorithm, agent_names=[a.name for a in agents], llm_client=annotation_client
        )
    except RuntimeError as e:
        print(f"\nSkipping benchmark report: {e}")
    finally:
        await annotation_client.aclose()


async def _resolve_inputs(options) -> RunInputs:
    code_path, data_path = options.code, options.data

    # Only ask which source to use when the flags didn't already settle it.
    if not code_path and not data_path:
        if not _prompt_wants_custom():
            return _builtin_inputs()
        code_path = input("Path to your baseline algorithm (.py): ").strip()
        data_path = input("Path to your dataset (.csv): ").strip()

    if not code_path or not data_path:
        raise RuntimeError(
            "custom input needs BOTH a baseline .py file and a dataset .csv file "
            "(pass --code and --data, or leave both out to use a built-in algorithm)"
        )

    print("Loading your algorithm and dataset...")
    custom = await user_inputs.prepare_custom_inputs(code_path, data_path)
    return RunInputs(
        algorithm=custom.algorithm,
        display_name=custom.display_name,
        dataset_name=custom.dataset_name,
        baseline_code=custom.baseline_code,
        dataset_csv=custom.dataset_csv,
        dataset_note=custom.description.as_prompt_note(),
    )


def _builtin_inputs() -> RunInputs:
    algorithm = sklearn_algorithms.choose_algorithm()
    spec = sklearn_algorithms.get_spec(algorithm)
    return RunInputs(
        algorithm=algorithm,
        display_name=spec.display_name,
        dataset_name=spec.dataset_name,
        baseline_code=spec.baseline_code,
        dataset_csv=sklearn_algorithms.dataset_csv(algorithm),
        dataset_note=DEFAULT_DATASET_NOTE,
    )


def _prompt_wants_custom() -> bool:
    print("Where should the algorithm and data come from?")
    print("  1) Built-in scikit-learn algorithm on the Iris dataset")
    print("  2) My own algorithm (.py) and my own dataset (.csv)")
    return input("> ").strip() == "2"


async def _seed_memory(shared: SharedState, inputs: RunInputs) -> None:
    async with shared.memory_lock:
        code_result = memory.remember_text(
            shared.memory, inputs.baseline_code, category="code", algorithm=inputs.algorithm
        )
        print(f"  seeded baseline code: {code_result}")
        dataset_result = memory.remember_text(
            shared.memory, inputs.dataset_csv, category="dataset", algorithm=inputs.algorithm
        )
        print(f"  seeded dataset: {dataset_result}")


def _parse_args():
    parser = argparse.ArgumentParser(add_help=True, description="CompeteEvolve")
    parser.add_argument("agents", nargs="?", type=int, default=None, help="number of agents")
    parser.add_argument("chat_provider", nargs="?", default=None, choices=[*VALID_PROVIDERS, None])
    parser.add_argument("evolution_provider", nargs="?", default=None, choices=[*VALID_PROVIDERS, None])
    parser.add_argument("--code", default=None, help="path to your own baseline algorithm (.py)")
    parser.add_argument("--data", default=None, help="path to your own dataset (.csv)")
    return parser.parse_args()


def _choose_roles(options) -> ModelRoles:
    chat_provider = options.chat_provider
    evolution_provider = options.evolution_provider

    if chat_provider and not evolution_provider:
        evolution_provider = chat_provider  # one provider given: use it for both

    if not chat_provider:
        print("Select the LLM provider for the agent conversation loop (high volume):")
        print("  1) Gemini")
        print("  2) OpenRouter  (friendlier free tier)")
        chat_provider = "openrouter" if input("> ").strip() == "2" else "gemini"

    if not evolution_provider:
        print("Select the provider for evolution/mutation (code quality matters most here):")
        print("  1) Gemini")
        print("  2) OpenRouter")
        print(f"  3) Same as the conversation loop ({chat_provider})")
        choice = input("> ").strip()
        evolution_provider = {"1": "gemini", "2": "openrouter"}.get(choice, chat_provider)

    return ModelRoles.from_env_and_choice(chat_provider, evolution_provider)


def _read_agent_count() -> int:
    raw = input(f"How many agents? (default {DEFAULT_AGENTS}) ").strip()
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_AGENTS


if __name__ == "__main__":
    asyncio.run(main())
