from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from typing import Optional


EMBEDDING_DIM = 256


def embed(code: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    for token in code.split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, "little") % EMBEDDING_DIM
        vector[idx] += 1.0

    norm = sum(v * v for v in vector) ** 0.5
    if norm > 0.0:
        vector = [v / norm for v in vector]
    return vector


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

@dataclass
class _VectorEntry:
    id: str
    embedding: list[float]


class _VectorIndex:
    def __init__(self) -> None:
        self.entries: list[_VectorEntry] = []

    def remember_and_score(self, entry_id: str, code: str) -> float:
        embedding = embed(code)
        similarity = max(
            (cosine_similarity(e.embedding, embedding) for e in self.entries),
            default=0.0,
        )
        self.entries.append(_VectorEntry(id=entry_id, embedding=embedding))
        return similarity

@dataclass
class _MemoryEntry:
    id: str
    code: str
    cosine_similarity: float
    functional_accuracy: float
    rank: Optional[float]
    category: str
    agent: Optional[str]
    algorithm: Optional[str]

    def rank_score(self) -> float:
        return (self.cosine_similarity + self.functional_accuracy) / 2.0


class _RankedStore:
    def __init__(self) -> None:
        self.entries: list[_MemoryEntry] = []

    def write(
        self,
        entry_id: str,
        code: str,
        cosine_similarity: float,
        functional_accuracy: float,
        rank: Optional[float],
        category: str,
        agent: Optional[str],
        algorithm: Optional[str],
    ) -> None:
        self.entries.append(
            _MemoryEntry(
                id=entry_id,
                code=code,
                cosine_similarity=cosine_similarity,
                functional_accuracy=functional_accuracy,
                rank=rank,
                category=category,
                agent=agent,
                algorithm=algorithm,
            )
        )

    def recall_top(
        self,
        n: int,
        category: str,
        agent: Optional[str],
        algorithm: Optional[str],
    ) -> list[dict]:
        candidates = [
            e
            for e in self.entries
            if e.category == category
            and (agent is None or e.agent == agent)
            and (algorithm is None or e.algorithm == algorithm)
        ]
        candidates.sort(key=lambda e: e.rank_score(), reverse=True)

        return [
            {
                "id": e.id,
                "code": e.code,
                "cosine_similarity": e.cosine_similarity,
                "functional_accuracy": e.functional_accuracy,
                "rank": e.rank,
                "category": e.category,
                "agent": e.agent,
                "algorithm": e.algorithm,
                "rank_score": e.rank_score(),
            }
            for e in candidates[: max(n, 1)]
        ]

@dataclass
class AgentMemory:

    vector_index: _VectorIndex = field(default_factory=_VectorIndex)
    ranked_store: _RankedStore = field(default_factory=_RankedStore)
    _next_id: int = 0

    def fresh_id(self) -> str:
        self._next_id += 1
        return f"mem_{self._next_id}"


class MemoryError(Exception):
    """Raised for any memory-level failure (bad CSV, missing entry, etc.)."""


def _validate_csv(content: str) -> tuple[int, int]:

    reader = csv.reader(io.StringIO(content))
    try:
        header = next(reader)
    except StopIteration:
        raise MemoryError("CSV header row is empty") from None

    column_count = len(header)
    if column_count == 0:
        raise MemoryError("CSV header row is empty")

    row_count = 0
    for row in reader:
        if len(row) != column_count:
            raise MemoryError(
                f"malformed CSV row {row_count + 1}: expected {column_count} columns, got {len(row)}"
            )
        row_count += 1

    if row_count == 0:
        raise MemoryError("CSV has a header row but no data rows")

    return column_count, row_count


def run(memory: AgentMemory, args: dict) -> str:

    action = args.get("action")

    if action == "write":
        return _handle_write(memory, args)
    if action == "read":
        return _handle_read(memory, args)

    query = args.get("query")
    if isinstance(query, str):
        trimmed = query.strip()
        if trimmed.lower() == "read":
            return _handle_read(memory, args)
        try:
            parsed = json.loads(trimmed)
            if isinstance(parsed, dict) and "action" in parsed:
                return run(memory, parsed)
        except json.JSONDecodeError:
            pass
        return _handle_write(memory, {"code": trimmed})

    raise MemoryError("memory call requires an 'action' field or a 'query' field")


def _handle_write(memory: AgentMemory, args: dict) -> str:
    code = args.get("code")
    if code is None:
        path = args.get("path")
        if path is None:
            raise MemoryError("write requires either a 'code' field or a 'path' field")
        try:
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
        except OSError as e:
            raise MemoryError(f"failed to read file '{path}': {e}") from e

    category = args.get("category", "code")

    csv_shape: Optional[tuple[int, int]] = None
    if category == "dataset":
        try:
            csv_shape = _validate_csv(code)
        except MemoryError as e:
            raise MemoryError(f"invalid CSV for category 'dataset': {e}") from e

    functional_accuracy = float(args.get("functional_accuracy", 0.0) or 0.0)
    rank = args.get("rank")
    rank = float(rank) if rank is not None else None
    agent = args.get("agent")
    algorithm = args.get("algorithm")
    entry_id = args.get("id") or memory.fresh_id()

    cosine_sim = memory.vector_index.remember_and_score(entry_id, code)
    memory.ranked_store.write(
        entry_id, code, cosine_sim, functional_accuracy, rank, category, agent, algorithm
    )

    response = {"status": "ok", "id": entry_id, "cosine_similarity": cosine_sim}
    if csv_shape is not None:
        response["csv_columns"], response["csv_rows"] = csv_shape

    return json.dumps(response)


def _handle_read(memory: AgentMemory, args: dict) -> str:
    top_n = int(args.get("top_n", 1) or 1)
    category = args.get("category", "code")
    agent = args.get("agent")
    algorithm = args.get("algorithm")

    results = memory.ranked_store.recall_top(top_n, category, agent, algorithm)

    if not results:
        return json.dumps({"status": "empty", "message": f"no '{category}' entries remembered yet"})

    return json.dumps({"status": "ok", "results": results})


def remember_csv_file(memory: AgentMemory, path: str, category: str, algorithm: str) -> str:
    return run(
        memory,
        {"action": "write", "path": path, "category": category, "algorithm": algorithm},
    )


def remember_text(
    memory: AgentMemory,
    code: str,
    category: str,
    algorithm: str,
    agent: Optional[str] = None,
) -> str:
    args = {"action": "write", "code": code, "category": category, "algorithm": algorithm}
    if agent is not None:
        args["agent"] = agent
    return run(memory, args)
