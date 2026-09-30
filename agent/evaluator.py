
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from . import memory, sandbox, traits as traits_module
from .logic_check import LogicChecker
from .traits import TraitSet


@dataclass
class AgentEvaluation:


    code: str
    trait_set: TraitSet

    @property
    def trait(self) -> float:
        return self.trait_set.trait


async def run(args: dict, shared, agent_name: str, algorithm: str, llm_client=None) -> str:
    """Entry point called from `Agent._call_tool`."""
    sample_code = args.get("candidate") or await _fetch_agent_sample(shared, agent_name, algorithm)
    original_code = args.get("original_code") or await _fetch_original_code(shared, algorithm)
    dataset_csv = args.get("dataset") or await _fetch_dataset(shared, algorithm)

    time_limit = sandbox.default_time_limit()
    logic_checker = LogicChecker(llm_client)

    print(f"[evaluator:{agent_name}] measuring baseline ('{algorithm}' original code)...")
    original_traits = await _measure(
        original_code, original_code, None, dataset_csv, time_limit, logic_checker, is_reference=True
    )
    baseline_ok = not (
        original_traits.primary.syntax_errors or original_traits.primary.runtime_errors
    )
    if not baseline_ok:
        print(
            f"[evaluator:{agent_name}] WARNING: original '{algorithm}' code failed in the sandbox — "
            "continuing without halting."
        )

    print(f"[evaluator:{agent_name}] measuring this agent's evolved sample...")
    # The agent's own previous submission is Ci^(t-1) for the novelty term.
    previous_code = await _fetch_agent_previous_sample(shared, agent_name, algorithm, sample_code)
    sample_traits = await _measure(
        sample_code, original_code, previous_code, dataset_csv, time_limit, logic_checker
    )

    p, s = sample_traits.primary, sample_traits.secondary
    print(
        f"[evaluator:{agent_name}] Ma={p.model_accuracy:.4f} L={p.loss:.4f} T={s.time:.4f}s "
        f"Nθ={s.novelty:.4f} Aθ={s.accuracy_term:.4f} R={s.resource:.4f} Tθ={s.trait:.6g}"
    )

    async with shared.eval_lock:
        shared.agent_evaluations[agent_name] = AgentEvaluation(
            code=sample_code, trait_set=sample_traits
        )

    positions = await compute_rank_positions(shared)
    agent_count = len(positions)
    rank_position = next((rp for name, rp in positions if name == agent_name), None)
    if rank_position is None:
        raise RuntimeError(f"internal error: agent '{agent_name}' missing from its own rank computation")

    mu = traits_module.compute_mu(original_traits.secondary.task_performance, sample_traits.secondary.task_performance)

    notes: list[str] = []
    if p.syntax_errors or p.runtime_errors or p.logical_errors:
        notes.append(
            f"Your code scored SE={p.syntax_errors}, RE={p.runtime_errors}, EL={p.logical_errors}. "
            f"These reduce Aθ (and therefore your performance) sharply — logical errors most of all. "
            "Fix them and evolve a new sample."
        )
    if mu <= traits_module.MU_THRESHOLD:
        notes.append(
            f"Your mu is {mu:.4f}; the target is above {traits_module.MU_THRESHOLD:.4f} "
            "(a 25% performance improvement over the original). Keep evolving."
        )
    if not baseline_ok:
        notes.append(
            f"The original '{algorithm}' baseline failed in the sandbox, so the comparison against "
            "it is unreliable this round."
        )

    return json.dumps(
        {
            "status": "ok",
            "agent": agent_name,
            "algorithm": algorithm,
            "rank_position": rank_position,
            "agent_count": agent_count,
            "mu": mu,
            "baseline_ok": baseline_ok,
            "performance_original": original_traits.performance,
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
            "notes": notes,
            "sandbox_stderr": sample_traits.stderr or None,
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
    result = await sandbox.run(code, dataset_csv, time_limit)
    primary = result.primary
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


async def compute_rank_positions(shared) -> list[tuple[str, int]]:
    
    async with shared.eval_lock:
        snapshot = dict(shared.agent_evaluations)

    if not snapshot:
        raise RuntimeError("no agent has been evaluated yet; run the 'evaluator' tool for at least one agent first")

    return traits_module.rank_positions([(name, ev.trait) for name, ev in snapshot.items()])


async def _fetch_agent_sample(shared, agent_name: str, algorithm: str) -> str:
    async with shared.memory_lock:
        raw = memory.run(
            shared.memory,
            {"action": "read", "top_n": 1, "category": "code", "agent": agent_name, "algorithm": algorithm},
        )
    results = json.loads(raw).get("results", [])
    if not results:
        raise RuntimeError(
            f"no '{algorithm}' sample found in memory for agent '{agent_name}'; run 'evolution' first"
        )
    return results[0]["code"]


async def _fetch_agent_previous_sample(
    shared, agent_name: str, algorithm: str, exclude_code: str
) -> Optional[str]:

    async with shared.memory_lock:
        raw = memory.run(
            shared.memory,
            {"action": "read", "top_n": 5, "category": "code", "agent": agent_name, "algorithm": algorithm},
        )
    results = json.loads(raw).get("results", [])
    for r in results:
        code = r.get("code")
        if code and code != exclude_code:
            return code
    return None


async def _fetch_original_code(shared, algorithm: str) -> str:
    async with shared.memory_lock:
        raw = memory.run(
            shared.memory, {"action": "read", "top_n": 100, "category": "code", "algorithm": algorithm}
        )
    results = json.loads(raw).get("results", [])
    hit = next((r for r in results if r.get("agent") is None), None)
    if hit is None:
        raise RuntimeError(f"no baseline (agent-less) 'code' entry found for algorithm '{algorithm}'")
    return hit["code"]


async def _fetch_dataset(shared, algorithm: str) -> str:
    async with shared.memory_lock:
        raw = memory.run(
            shared.memory, {"action": "read", "top_n": 1, "category": "dataset", "algorithm": algorithm}
        )
    results = json.loads(raw).get("results", [])
    if not results:
        raise RuntimeError(f"no 'dataset' entry found for algorithm '{algorithm}'")
    return results[0]["code"]
