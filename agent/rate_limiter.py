from __future__ import annotations

import asyncio
import os
import time
from collections import deque


class RateLimiter:

    def __init__(self, max_per_minute: int) -> None:
        self.max_per_minute = max(1, max_per_minute)
        self._window: deque[float] = deque()
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls, default_rpm: int) -> "RateLimiter":

        raw = os.environ.get("LLM_RPM") or os.environ.get("GEMINI_RPM")
        try:
            rpm = int(raw) if raw else default_rpm
        except ValueError:
            rpm = default_rpm

        print(
            f"[rate_limiter] capping LLM requests at {rpm}/minute (check your real quota and "
            "set LLM_RPM to match)"
        )
        return cls(rpm)

    async def acquire(self) -> None:
        while True:
            sleep_for: float | None = None

            async with self._lock:
                now = time.monotonic()
                while self._window and now - self._window[0] >= 60.0:
                    self._window.popleft()

                if len(self._window) < self.max_per_minute:
                    self._window.append(now)
                else:
                    oldest = self._window[0]
                    sleep_for = max(0.0, 60.0 - (now - oldest)) + 0.1

            if sleep_for is None:
                return

            print(f"[rate_limiter] at capacity, waiting {sleep_for:.1f}s before the next request")
            await asyncio.sleep(sleep_for)
