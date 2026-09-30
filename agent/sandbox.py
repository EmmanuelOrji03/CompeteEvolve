
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Optional

import psutil

from .traits import PrimaryTraits


def default_time_limit() -> float:

    try:
        return float(os.environ.get("SANDBOX_TIMEOUT_SECS", "60"))
    except ValueError:
        return 60.0


def _python_bin() -> str:
    override = os.environ.get("SANDBOX_PYTHON")
    if override:
        return override
    return "python" if os.name == "nt" else "python3"


@dataclass
class SandboxResult:
    """Measured primary traits plus the raw stderr, for diagnostics."""

    primary: PrimaryTraits
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.primary.syntax_errors == 0 and self.primary.runtime_errors == 0


_HARNESS_TEMPLATE = '''
import sys, json, time, traceback

{candidate_code}

def _main():
    data_path = sys.argv[1]
    wall_start = time.time()
    try:
        result = run(data_path)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    wall_elapsed = time.time() - wall_start

    # A candidate may return either a rich dict (preferred - lets the
    # harness see TT and TI separately) or a bare accuracy float.
    if isinstance(result, dict):
        accuracy = float(result.get("accuracy", 0.0))
        loss = result.get("loss", None)
        training_time = result.get("training_time", None)
        inference_time = result.get("inference_time", None)
    else:
        accuracy = float(result)
        loss = None
        training_time = None
        inference_time = None

    if training_time is None:
        # No split reported: attribute the whole run to training.
        training_time = wall_elapsed
        inference_time = 0.0
    if inference_time is None:
        inference_time = 0.0
    if loss is None:
        loss = 1.0 - accuracy

    print("__TRAITS__" + json.dumps({{
        "accuracy": accuracy,
        "loss": float(loss),
        "training_time": float(training_time),
        "inference_time": float(inference_time),
        "wall_elapsed": wall_elapsed,
    }}))

if __name__ == "__main__":
    _main()
'''


def _sample_resources(pid: int, stop_event: threading.Event, out: dict) -> None:
    """Background sampler: tracks the child's peak RSS (Mu) and last-seen
    cumulative CPU time (K) until the process exits or `stop_event` is
    set. CPU time is re-read on every tick because it's cumulative and
    unreadable once the process is gone."""
    try:
        proc = psutil.Process(pid)
    except psutil.Error:
        return

    peak_rss = 0.0
    cpu_seconds = 0.0
    while not stop_event.is_set():
        try:
            with proc.oneshot():
                rss = proc.memory_info().rss / (1024.0 * 1024.0)  # MB
                cpu_times = proc.cpu_times()
                cpu_seconds = max(cpu_seconds, cpu_times.user + cpu_times.system)
            # Include children, in case the candidate spawns workers
            # (e.g. n_jobs=-1 in scikit-learn).
            for child in proc.children(recursive=True):
                try:
                    with child.oneshot():
                        rss += child.memory_info().rss / (1024.0 * 1024.0)
                        child_cpu = child.cpu_times()
                        cpu_seconds = max(cpu_seconds, cpu_seconds + child_cpu.user + child_cpu.system)
                except psutil.Error:
                    continue
            peak_rss = max(peak_rss, rss)
        except psutil.Error:
            break
        time.sleep(0.02)

    out["memory_usage"] = peak_rss
    out["compute_cost"] = cpu_seconds


async def run(candidate_code: str, dataset_csv: str, time_limit: Optional[float] = None) -> SandboxResult:
    """Runs `candidate_code` against `dataset_csv` in a fresh temp
    directory and returns its measured primary traits."""
    if time_limit is None:
        time_limit = default_time_limit()

    tmp_dir = tempfile.mkdtemp(prefix="sandbox_")
    try:
        data_path = os.path.join(tmp_dir, "data.csv")
        script_path = os.path.join(tmp_dir, "candidate.py")

        with open(data_path, "w", encoding="utf-8") as f:
            f.write(dataset_csv)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(_HARNESS_TEMPLATE.format(candidate_code=candidate_code))

        python_bin = _python_bin()

        # ---- SE: syntax check, compiling without executing anything ----
        try:
            compile_proc = await asyncio.create_subprocess_exec(
                python_bin, "-m", "py_compile", script_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, compile_stderr = await compile_proc.communicate()
        except FileNotFoundError as e:
            raise RuntimeError(
                f"couldn't launch a Python interpreter ({e}). Install Python 3 and make sure it's "
                "on PATH (as `python3` on Linux/macOS, `python` on Windows), or set SANDBOX_PYTHON "
                "to the interpreter's full path."
            ) from e

        if compile_proc.returncode != 0:
            return SandboxResult(
                primary=PrimaryTraits(syntax_errors=1, loss=1.0),
                stderr=compile_stderr.decode("utf-8", errors="replace"),
            )

        # ---- execute, measuring Mu/K alongside, with a hard timeout ----
        proc = await asyncio.create_subprocess_exec(
            python_bin, script_path, data_path,
            cwd=tmp_dir,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        resource_out: dict = {}
        stop_event = threading.Event()
        sampler = threading.Thread(
            target=_sample_resources, args=(proc.pid, stop_event, resource_out), daemon=True
        )
        sampler.start()

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=time_limit)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stop_event.set()
            sampler.join(timeout=1.0)
            return SandboxResult(
                primary=PrimaryTraits(
                    runtime_errors=1,
                    training_time=time_limit,
                    loss=1.0,
                    memory_usage=resource_out.get("memory_usage", 0.0),
                    compute_cost=resource_out.get("compute_cost", 0.0),
                ),
                stderr=f"candidate timed out after {time_limit:.1f}s",
            )
        finally:
            stop_event.set()
            sampler.join(timeout=1.0)

        memory_usage = resource_out.get("memory_usage", 0.0)
        compute_cost = resource_out.get("compute_cost", 0.0)

        if proc.returncode != 0:
            return SandboxResult(
                primary=PrimaryTraits(
                    runtime_errors=1,
                    loss=1.0,
                    memory_usage=memory_usage,
                    compute_cost=compute_cost,
                ),
                stderr=stderr.decode("utf-8", errors="replace"),
            )

        stdout_text = stdout.decode("utf-8", errors="replace")
        payload = None
        for line in stdout_text.splitlines():
            if line.startswith("__TRAITS__"):
                payload = line[len("__TRAITS__") :]
        if payload is None:
            # Ran to completion but printed nothing parseable — treat as
            # a runtime error rather than raising, so one malformed
            # candidate can't halt the whole evolution loop.
            return SandboxResult(
                primary=PrimaryTraits(
                    runtime_errors=1, loss=1.0, memory_usage=memory_usage, compute_cost=compute_cost
                ),
                stderr=f"candidate produced no parseable trait output; stdout was: {stdout_text}",
            )

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as e:
            return SandboxResult(
                primary=PrimaryTraits(
                    runtime_errors=1, loss=1.0, memory_usage=memory_usage, compute_cost=compute_cost
                ),
                stderr=f"candidate trait output was not valid JSON ({e}); stdout was: {stdout_text}",
            )

        accuracy = max(0.0, min(1.0, float(parsed.get("accuracy", 0.0))))
        loss = float(parsed.get("loss", 1.0 - accuracy))
        if loss != loss or loss < 0.0:  # NaN or negative
            loss = 1.0 - accuracy

        return SandboxResult(
            primary=PrimaryTraits(
                training_time=max(0.0, float(parsed.get("training_time", 0.0))),
                inference_time=max(0.0, float(parsed.get("inference_time", 0.0))),
                syntax_errors=0,
                runtime_errors=0,
                logical_errors=0,  # filled in later by logic_check.py
                loss=loss,
                model_accuracy=accuracy,
                memory_usage=memory_usage,
                compute_cost=compute_cost,
            ),
            stderr="",
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
