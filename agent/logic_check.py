from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Optional

from .llm_client import LLMClient

MAX_LOGICAL_ERRORS = 3

_PROMPT = """You are reviewing a machine learning script for LOGICAL errors only.

The script must define `run(data_path)`, which loads one CSV (features plus a
`target` column), splits it into train and test itself, trains a model, and
reports genuine held-out test-set accuracy.

Count ONLY defects that make the result wrong or misleading, such as:
- evaluating on training data instead of the held-out test split (leakage)
- fitting a scaler/encoder on the whole dataset before splitting (leakage)
- returning a value that is not real test-set accuracy (e.g. hardcoded)
- misaligning features and labels (e.g. shuffling X and y separately)
- computing the metric against the wrong variable

Do NOT count: style, inefficiency, weak hyperparameters, missing type hints,
or anything that is merely "could be better". Those are not logical errors.

Respond with ONLY a JSON object, no markdown fences and no other text:
{{"logical_errors": <integer>, "reasons": ["<short reason>", ...]}}

If the code is logically sound, respond exactly: {{"logical_errors": 0, "reasons": []}}

Code to review:
{code}
"""


def enabled() -> bool:
    """Logical-error checking is on unless `LOGIC_CHECK=0`."""
    return os.environ.get("LOGIC_CHECK", "1").strip() not in ("0", "false", "False")


class LogicChecker:
    """Caches results by code hash, so the same candidate is never
    reviewed twice within one run."""

    def __init__(self, llm_client: Optional[LLMClient]) -> None:
        self.llm_client = llm_client
        self._cache: dict[str, tuple[int, list[str]]] = {}

    async def count(self, code: str) -> tuple[int, list[str]]:
        """Returns `(logical_error_count, reasons)`. Fails open with
        `(0, [])` when checking is disabled, no client is available, or
        the review itself errors."""
        if not code.strip() or not enabled() or self.llm_client is None:
            return 0, []

        key = hashlib.blake2b(code.encode("utf-8"), digest_size=16).hexdigest()
        if key in self._cache:
            return self._cache[key]

        try:
            response = await self.llm_client.generate(
                system_instruction="", history=[{"role": "user", "text": _PROMPT.format(code=code)}]
            )
            count, reasons = _parse(response.text)
        except Exception as e:  # noqa: BLE001 - never penalise a candidate for a failed review
            print(f"[logic_check] review failed ({e}); scoring EL=0 for this candidate")
            count, reasons = 0, []

        result = (min(count, MAX_LOGICAL_ERRORS), reasons)
        self._cache[key] = result
        return result


def _parse(text: str) -> tuple[int, list[str]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return 0, []

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 0, []

    try:
        count = int(parsed.get("logical_errors", 0))
    except (TypeError, ValueError):
        count = 0

    reasons = parsed.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = []
    return max(0, count), [str(r) for r in reasons][:MAX_LOGICAL_ERRORS]
