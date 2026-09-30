from __future__ import annotations

import json
import sys
from pathlib import Path

from . import evaluator, traits as traits_module

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts import build_feedback_prompt  # noqa: E402

TOP_N_IN_FEEDBACK = 3


async def run(args: dict, shared, agent_name: str) -> str:

    episodes = int(args.get("episodes", 1) or 1)
    if episodes < 1:
        raise ValueError("'episodes' must be at least 1")

    positions = await evaluator.compute_rank_positions(shared)
    n = len(positions)

    async with shared.eval_lock:
        snapshot = dict(shared.agent_evaluations)

    leaderboard = [
        {
            "agent": name,
            "rank_position": rp,
            "reward": traits_module.reward_for_rank(rp),
            "trait": snapshot[name].trait if name in snapshot else 0.0,
        }
        for name, rp in positions
    ]
    leaderboard.sort(key=lambda e: e["rank_position"])

    self_entry = next((e for e in leaderboard if e["agent"] == agent_name), None)
    if self_entry is None:
        raise RuntimeError(f"agent '{agent_name}' has not been evaluated yet")

    rank_position = self_entry["rank_position"]
    reward = self_entry["reward"]

    # The top-3 agents' full trait tables, as the A.2.3 prompt requires.
    leaderboard_markdown = _render_leaderboard(leaderboard[:TOP_N_IN_FEEDBACK], snapshot)
    feedback_prompt = build_feedback_prompt(rank_position, n, reward, leaderboard_markdown)

    return json.dumps(
        {
            "status": "ok",
            "agent": agent_name,
            "rank_position": rank_position,
            "reward": reward,
            "agent_count": n,
            "feedback_prompt": feedback_prompt,
            "leaderboard": [
                {k: v for k, v in entry.items() if k != "trait"} | {"trait": round(entry["trait"], 9)}
                for entry in leaderboard
            ],
        }
    )


def _render_leaderboard(top_entries: list[dict], snapshot: dict) -> str:
    blocks = []
    for entry in top_entries:
        name = entry["agent"]
        evaluation = snapshot.get(name)
        if evaluation is None:
            continue
        blocks.append(
            traits_module.format_traits_markdown(
                f"{name} — rank {entry['rank_position']}, reward {entry['reward']}",
                evaluation.trait_set,
            )
        )
    return "\n".join(blocks) if blocks else "(no evaluated agents yet)"
