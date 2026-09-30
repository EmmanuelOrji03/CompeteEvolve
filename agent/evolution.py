from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import memory, sandbox, traits as traits_module
from .llm_client import LLMClient
from .logic_check import LogicChecker
from .traits import TraitSet

import sys

# `prompts` lives at the project root, not inside the `agent` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts import build_evolution_prompt  # noqa: E402

TRAITS_DIR = Path("traits")
TOP_N_SAMPLES = 3


@dataclass
class Sample:


    id: str
    island: int
    generation: int
    trait_set: TraitSet

    @property
    def code(self) -> str:
        return self.trait_set.code

    @property
    def trait(self) -> float:
        return self.trait_set.trait


async def run(
    args: dict,
    agent_name: str,
    algorithm: str,
    llm_client: LLMClient,
    shared,  # agent.SharedState — untyped to avoid a circular import
) -> str:
    """Entry point called from `Agent._call_tool`.

    Expected `args` (all optional except `generations`, which the tool
    schema marks required; the paper's §2 defaults are used otherwise):
        {
          "generations": 3,            # Gn
          "population_size": 4,        # NI, number of islands
          "samples_per_population": 5, # NS, samples per island
          "mu_threshold": 0.5556,      # μ target (25% improvement)
        }
    """
    generations = int(args.get("generations") or 3)
    island_count = int(args.get("population_size") or 4)          # NI
    samples_per_island = int(args.get("samples_per_population") or 5)  # NS
    mu_threshold = float(args.get("mu_threshold") or traits_module.MU_THRESHOLD)

    if generations < 1 or island_count < 1 or samples_per_island < 1:
        raise ValueError("'generations', 'population_size' and 'samples_per_population' must be >= 1")

    # ---- CO and the dataset ----
    original_code = args.get("original_code") or await _fetch_baseline_code(shared, algorithm)
    dataset_csv = args.get("dataset") or await _fetch_dataset(shared, algorithm)
    time_limit = sandbox.default_time_limit()
    logic_checker = LogicChecker(llm_client)

    # ---- P1: the original code's own performance, for μ = P2/(P1+P2) ----
    original_traits = await _measure(
        original_code, original_code, None, dataset_csv, time_limit, logic_checker, is_reference=True
    )
    p1 = original_traits.secondary.task_performance

    print(
        f"[evolution:{agent_name}] '{algorithm}': {generations} generation(s) x {island_count} island(s) "
        f"x {samples_per_island} sample(s); P1(original)={p1:.6g}, target mu>{mu_threshold:.4f}"
    )

    best_overall: Optional[Sample] = None
    top_samples: list[Sample] = []          # TB — the best 3 so far
    previous_generation_code: Optional[str] = None  # Ci^(t-1) for Nθ
    mu = 0.0
    traits_file: Optional[Path] = None
    generations_run = 0

    # ---- 18-31: islands x generations x samples ----
    for generation in range(1, generations + 1):
        if mu > mu_threshold:
            print(f"[evolution:{agent_name}] mu={mu:.4f} exceeds threshold; stopping early")
            break
        generations_run = generation
        generation_samples: list[Sample] = []

        for island in range(1, island_count + 1):
            # Mode per the Evolution Prompt's clauses 5-7.
            if generation == 1 and island == 1 and not top_samples:
                mode = "annotate"
            elif top_samples:
                mode = "crossover"
            else:
                mode = "first"

            best_traits_md = _render_best_traits(top_samples) if top_samples else None
            print(
                f"[evolution:{agent_name}]   gen {generation}/{generations} island {island}/{island_count} "
                f"({mode}): requesting {samples_per_island} sample(s)..."
            )

            generated = await _generate_samples(
                llm_client,
                original_code,
                samples_per_island,
                mode,
                best_traits_md,
                getattr(shared, "dataset_note", "") or None,
            )

            for idx, code in enumerate(generated):
                trait_set = await _measure(
                    code, original_code, previous_generation_code, dataset_csv, time_limit, logic_checker
                )
                sample = Sample(
                    id=f"gen{generation}_island{island}_sample{idx}",
                    island=island,
                    generation=generation,
                    trait_set=trait_set,
                )
                generation_samples.append(sample)

                p = trait_set.primary
                print(
                    f"[evolution:{agent_name}]     {sample.id}: Ma={p.model_accuracy:.4f} "
                    f"L={p.loss:.4f} T={trait_set.secondary.time:.4f}s Nθ={trait_set.novelty:.4f} "
                    f"SE={p.syntax_errors} RE={p.runtime_errors} EL={p.logical_errors} "
                    f"Tθ={trait_set.trait:.6g}"
                )

                if best_overall is None or sample.trait > best_overall.trait:
                    best_overall = sample

        if not generation_samples:
            continue

        # ---- 25-27: rank by Tθ, keep the top 3, write them out as Mθ ----
        pool = sorted(generation_samples + top_samples, key=lambda s: s.trait, reverse=True)
        top_samples = pool[:TOP_N_SAMPLES]
        traits_file = _write_traits_file(algorithm, agent_name, generation, top_samples, original_traits)

        # Ci^(t-1) for the next generation's novelty term.
        previous_generation_code = top_samples[0].code

        # ---- 23: μ ← P2 / (P1 + P2) ----
        if best_overall is not None:
            mu = traits_module.compute_mu(p1, best_overall.trait_set.secondary.task_performance)
        print(
            f"[evolution:{agent_name}]   gen {generation} complete: best Tθ={top_samples[0].trait:.6g} "
            f"mu={mu:.4f}"
        )

    if best_overall is None:
        raise RuntimeError("evolution produced no viable samples")

    stb = best_overall
    print(
        f"[evolution:{agent_name}] finished after {generations_run} generation(s): "
        f"Ma={stb.trait_set.accuracy:.4f} Tθ={stb.trait:.6g} mu={mu:.4f}"
    )

    # ---- remember the winning sample ----
    async with shared.memory_lock:
        try:
            memory.run(
                shared.memory,
                {
                    "action": "write",
                    "code": stb.code,
                    "functional_accuracy": stb.trait_set.accuracy,
                    "rank": stb.trait,
                    "category": "code",
                    "agent": agent_name,
                    "algorithm": algorithm,
                    "id": f"{agent_name}_{stb.id}",
                },
            )
        except memory.MemoryError:
            pass  # best-effort write-back

    p, s = stb.trait_set.primary, stb.trait_set.secondary
    return json.dumps(
        {
            "status": "ok",
            "agent": agent_name,
            "algorithm": algorithm,
            "id": stb.id,
            "mu": mu,
            "mu_threshold": mu_threshold,
            "reached_threshold": mu > mu_threshold,
            "performance_original": p1,
            "generations_run": generations_run,
            "primary_traits": {
                "training_time": p.training_time,
                "inference_time": p.inference_time,
                "syntax_errors": p.syntax_errors,
                "runtime_errors": p.runtime_errors,
                "logical_errors": p.logical_errors,
                "loss": p.loss,
                "model_accuracy": p.model_accuracy,
                "memory_usage": p.memory_usage,
                "compute_cost": p.compute_cost,
            },
            "secondary_traits": {
                "time": s.time,
                "model_performance": s.model_performance,
                "novelty": s.novelty,
                "accuracy_term": s.accuracy_term,
                "resource": s.resource,
                "performance": s.performance,
                "trait": s.trait,
            },
            "traits_file": str(traits_file) if traits_file else None,
            "code": stb.code,
        }
    )


async def _measure(
    code: str,
    original_code: str,
    previous_code: Optional[str],
    dataset_csv: str,
    time_limit: float,
    logic_checker: LogicChecker,
    is_reference: bool = False,
) -> TraitSet:
    """Algorithm 2's TraitEvaluation: run the candidate, ask the LLM
    reviewer for EL, then derive every secondary trait. `is_reference`
    scores the original code CO as the P1 baseline - see traits.derive."""
    result = await sandbox.run(code, dataset_csv, time_limit)
    primary = result.primary

    # Only review code that actually ran — a candidate that failed to
    # compile or crashed is already penalised through SE/RE, and asking
    # for a logical-error count on top would double-count the same fault.
    if result.ok:
        logical_errors, _reasons = await logic_checker.count(code)
        primary.logical_errors = logical_errors

    return traits_module.build(
        primary=primary,
        sample_code=code,
        original_code=original_code,
        previous_code=previous_code,
        stderr=result.stderr,
        is_reference=is_reference,
    )


def _render_best_traits(top_samples: list[Sample]) -> str:
    """The `Mθ` context: the top samples' trait tables and their code."""
    blocks = []
    for position, sample in enumerate(top_samples, start=1):
        blocks.append(
            traits_module.format_traits_markdown(
                f"Top sample {position} (id: {sample.id}, generation {sample.generation})",
                sample.trait_set,
            )
        )
        blocks.append(f"\n```python\n{sample.code}\n```\n")
    return "\n".join(blocks)


def _write_traits_file(
    algorithm: str,
    agent_name: str,
    generation: int,
    top_samples: list[Sample],
    original_traits: TraitSet,
) -> Path:
    """Writes the top-3 traits markdown file used as crossover context
    (and kept on disk as a record of the search)."""
    directory = TRAITS_DIR / algorithm
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{agent_name}_gen{generation:02d}_best_traits.md"

    body = [
        f"# Best traits — {algorithm}, {agent_name}, generation {generation}\n",
        traits_module.format_traits_markdown("Original code (CO)", original_traits),
        "\n",
        _render_best_traits(top_samples),
    ]
    path.write_text("\n".join(body), encoding="utf-8")
    return path


async def _generate_samples(
    llm_client: LLMClient,
    original_code: str,
    n: int,
    mode: str,
    best_traits_markdown: Optional[str],
    dataset_note: Optional[str] = None,
) -> list[str]:
    """Algorithm 2 line 21: `Ci ← G(P, CO, Mθ)`."""
    prompt = build_evolution_prompt(original_code, mode, best_traits_markdown, dataset_note)

    samples: list[str] = []
    for _ in range(n):
        try:
            response = await llm_client.generate(
                system_instruction="", history=[{"role": "user", "text": prompt}]
            )
        except Exception as e:  # noqa: BLE001 - one failed generation shouldn't end the search
            print(f"[evolution] sample generation failed ({e}); skipping this sample")
            continue
        cleaned = _strip_code_fences(response.text)
        if cleaned.strip():
            samples.append(cleaned)

    if not samples:
        raise RuntimeError("the LLM returned no usable samples")
    return samples


def _strip_code_fences(text: str) -> str:
    """Strips a leading/trailing ``` fence (with an optional language
    tag) if the model wrapped its answer in one despite being asked not to."""
    trimmed = text.strip()
    if trimmed.startswith("```"):
        without_lang = trimmed[3:]
        newline = without_lang.find("\n")
        first_line = without_lang[:newline] if newline != -1 else ""
        if first_line.isalnum() or first_line == "":
            without_lang = without_lang[newline + 1 :] if newline != -1 else without_lang
        end = without_lang.rfind("```")
        if end != -1:
            return without_lang[:end].strip()
        return without_lang.strip()
    return trimmed


async def _fetch_baseline_code(shared, algorithm: str) -> str:
    """CO — the fixed, agent-less 'code' entry. Every generation mutates
    from this, never from whatever currently ranks highest (which could
    be another agent's submission)."""
    async with shared.memory_lock:
        raw = memory.run(
            shared.memory, {"action": "read", "top_n": 100, "category": "code", "algorithm": algorithm}
        )
    results = json.loads(raw).get("results", [])
    hit = next((r for r in results if r.get("agent") is None), None)
    if hit is None:
        raise RuntimeError(
            f"no baseline (agent-less) 'code' entry found for algorithm '{algorithm}' in memory"
        )
    return hit["code"]


async def _fetch_dataset(shared, algorithm: str) -> str:
    async with shared.memory_lock:
        raw = memory.run(
            shared.memory, {"action": "read", "top_n": 1, "category": "dataset", "algorithm": algorithm}
        )
    results = json.loads(raw).get("results", [])
    if not results:
        raise RuntimeError(f"no 'dataset' entry found for algorithm '{algorithm}' in memory")
    return results[0]["code"]
