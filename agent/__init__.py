
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import evaluator, evolution, memory, rate_limiter as rate_limiter_module, reinforcement
from .llm_client import LLMClient, ModelRoles, create_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts import build_system_prompt  # noqa: E402


@dataclass
class SharedState:
 
    dataset_note: str = ""

    memory: memory.AgentMemory = field(default_factory=memory.AgentMemory)
    memory_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    agent_evaluations: dict = field(default_factory=dict)
    eval_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    rate_limiter: "rate_limiter_module.RateLimiter" = field(
        default_factory=lambda: rate_limiter_module.RateLimiter.from_env(10)
    )


_TOOLS = [
    {
        "name": "evolution",
        "description": (
            "Generates, scores and mutates candidate implementations across islands and "
            "generations, and remembers the best one"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "generations": {"type": "integer"},
                "population_size": {"type": "integer"},
                "samples_per_population": {"type": "integer"},
            },
            "required": ["generations"],
        },
    },
    {
        "name": "evaluator",
        "description": "Runs your evolved candidate for real and scores its traits and rank",
        "parameters": {
            "type": "object",
            "properties": {"candidate": {"type": "string"}},
        },
    },
    {
        "name": "reinforcement",
        "description": "Fetches your reward, rank position, and the top agents' trait breakdowns",
        "parameters": {
            "type": "object",
            "properties": {"episodes": {"type": "integer"}},
            "required": ["episodes"],
        },
    },
    {
        "name": "memory",
        "description": "Reads from or writes to the shared memory store",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


class Agent:


    def __init__(self, name: str, algorithm: str, dataset_name: str, roles: ModelRoles, shared: SharedState) -> None:
        self.name = name
        self.algorithm = algorithm
        self.dataset_name = dataset_name
        self.shared = shared
        self.roles = roles

        self.chat_client: LLMClient = create_client(
            roles.chat_provider, roles.chat_api_key, roles.chat_model, shared.rate_limiter
        )
        # Reuse one client when both roles resolve to the same endpoint.
        if roles.same_endpoint():
            self.evolution_client: LLMClient = self.chat_client
            self._owns_evolution_client = False
        else:
            self.evolution_client = create_client(
                roles.evolution_provider, roles.evolution_api_key, roles.evolution_model, shared.rate_limiter
            )
            self._owns_evolution_client = True

        self.history: list[dict] = []
        self.tools = _TOOLS
        self.system_instruction = build_system_prompt(name, algorithm, dataset_name)

    async def aclose(self) -> None:
        await self.chat_client.aclose()
        if self._owns_evolution_client:
            await self.evolution_client.aclose()

    async def send(self, message: str) -> str:

        print(f"[{self.name}] sending message, awaiting the LLM...")
        self.history.append({"role": "user", "text": message})

        while True:
            response = await self.chat_client.generate(self.system_instruction, self.history, self.tools)

            if not response.function_calls:
                self.history.append({"role": "assistant", "text": response.text, "raw": response.raw})
                print(f"[{self.name}] done, no further tool calls")
                return response.text

            self.history.append(
                {
                    "role": "assistant",
                    "text": response.text,
                    "function_calls": response.function_calls,
                    "raw": response.raw,
                }
            )

            for fc in response.function_calls:
                fc_name = fc.get("name", "")
                fc_args = fc.get("args", {})
                print(f"[{self.name}] calling tool '{fc_name}' with args: {fc_args}")
                result = await self._call_tool(fc_name, fc_args)
                print(f"[{self.name}] tool '{fc_name}' finished")
                self.history.append(
                    {"role": "tool", "id": fc.get("id", ""), "name": fc_name, "result": result}
                )

            print(f"[{self.name}] awaiting the LLM again after tool result(s)...")

    async def _call_tool(self, name: str, args: dict) -> str:

        try:
            if name == "evolution":
                return await evolution.run(args, self.name, self.algorithm, self.evolution_client, self.shared)
            if name == "evaluator":
                return await evaluator.run(
                    args, self.shared, self.name, self.algorithm, self.evolution_client
                )
            if name == "reinforcement":
                return await reinforcement.run(args, self.shared, self.name)
            if name == "memory":
                async with self.shared.memory_lock:
                    return memory.run(self.shared.memory, args)
            return f"no such tool: {name}"
        except (memory.MemoryError, RuntimeError, ValueError) as e:
            # A tool-level failure becomes a normal result the agent can
            # see and react to next turn, rather than a hard exception
            # that ends its turn.
            return json.dumps({"status": "error", "message": str(e)})
