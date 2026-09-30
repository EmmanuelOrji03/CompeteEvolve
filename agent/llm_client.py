from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx


@dataclass
class LLMResponse:
    text: str = ""
    function_calls: list = field(default_factory=list)  # [{"id":str, "name":str, "args":dict}]
    raw: Optional[list] = None  # provider-specific passthrough (see GeminiClient below)


class LLMClient:

    provider_name: str = "base"

    async def generate(
        self, system_instruction: str, history: list[dict], tools: Optional[list[dict]] = None
    ) -> LLMResponse:
        raise NotImplementedError

    async def aclose(self) -> None:
        raise NotImplementedError


def create_client(provider: str, api_key: str, model: str, rate_limiter) -> LLMClient:

    provider = provider.strip().lower()
    if provider == "gemini":
        return GeminiClient(api_key=api_key, model=model, rate_limiter=rate_limiter)
    if provider == "openrouter":
        return OpenRouterClient(api_key=api_key, model=model, rate_limiter=rate_limiter)
    raise ValueError(f"unknown LLM provider '{provider}' (expected 'gemini' or 'openrouter')")


_ROLE_DEFAULTS = {
    "gemini": {
        "chat": "gemini-3.1-flash-lite",
        "evolution": "gemini-3.1-flash-lite",
    },
    "openrouter": {
        # OpenRouter slugs; ":free" variants keep a free-tier run viable.
        "chat": "meta-llama/llama-3.3-70b-instruct",
        "evolution": "meta-llama/llama-3.3-70b-instruct",
    },
}


@dataclass
class ModelRoles:


    chat_provider: str
    chat_model: str
    chat_api_key: str
    evolution_provider: str
    evolution_model: str
    evolution_api_key: str

    def same_endpoint(self) -> bool:

        return (
            self.chat_provider == self.evolution_provider
            and self.chat_model == self.evolution_model
            and self.chat_api_key == self.evolution_api_key
        )

    def describe(self) -> str:
        if self.same_endpoint():
            return f"chat + evolution: {self.chat_provider} ({self.chat_model})"
        return (
            f"chat: {self.chat_provider} ({self.chat_model})  |  "
            f"evolution: {self.evolution_provider} ({self.evolution_model})"
        )

    @classmethod
    def from_env_and_choice(cls, chat_provider: str, evolution_provider: str) -> "ModelRoles":

        chat_provider = chat_provider.strip().lower()
        evolution_provider = evolution_provider.strip().lower()

        chat_model = os.environ.get("CHAT_MODEL") or _ROLE_DEFAULTS[chat_provider]["chat"]
        evolution_model = (
            os.environ.get("EVOLUTION_MODEL") or _ROLE_DEFAULTS[evolution_provider]["evolution"]
        )

        chat_key = os.environ.get(api_key_env_var_for(chat_provider), "")
        evolution_key = os.environ.get(api_key_env_var_for(evolution_provider), "")

        return cls(
            chat_provider=chat_provider,
            chat_model=chat_model,
            chat_api_key=chat_key,
            evolution_provider=evolution_provider,
            evolution_model=evolution_model,
            evolution_api_key=evolution_key,
        )

    def missing_keys(self) -> list[str]:
        missing = []
        if not self.chat_api_key:
            missing.append(api_key_env_var_for(self.chat_provider))
        if not self.evolution_api_key:
            name = api_key_env_var_for(self.evolution_provider)
            if name not in missing:
                missing.append(name)
        return missing


def default_model_for(provider: str) -> str:
    provider = provider.strip().lower()
    if provider == "gemini":
        return os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    if provider == "openrouter":        # A slug from OpenRouter's catalog (openrouter.ai/models), not a
        # bare model name — OpenRouter uses "<org>/<model>" slugs since
        # it routes to many providers behind one endpoint.
        return os.environ.get("OPENROUTER_MODEL", "open-inference/fp8")
    raise ValueError(f"unknown LLM provider '{provider}'")


def api_key_env_var_for(provider: str) -> str:
    provider = provider.strip().lower()
    if provider == "gemini":
        return "GEMINI_API_KEY"
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    raise ValueError(f"unknown LLM provider '{provider}'")



class GeminiClient(LLMClient):
    provider_name = "gemini"
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str, rate_limiter, http_client: Optional[httpx.AsyncClient] = None) -> None:
        self.api_key = api_key
        self.model = model
        self.rate_limiter = rate_limiter
        self._owns_client = http_client is None
        self.client = http_client or httpx.AsyncClient(timeout=120.0)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def _to_contents(self, history: list[dict]) -> list[dict]:
        contents = []
        for turn in history:
            role = turn["role"]
            if role == "user":
                contents.append({"role": "user", "parts": [{"text": turn["text"]}]})
            elif role == "assistant":
                raw_parts = turn.get("raw")
                if raw_parts is not None:
                    contents.append({"role": "model", "parts": raw_parts})
                else:
                    parts = []
                    if turn.get("text"):
                        parts.append({"text": turn["text"]})
                    for fc in turn.get("function_calls", []):
                        parts.append({"functionCall": {"name": fc["name"], "args": fc["args"]}})
                    contents.append({"role": "model", "parts": parts})
            elif role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {"functionResponse": {"name": turn["name"], "response": {"result": turn["result"]}}}
                        ],
                    }
                )
            else:
                raise ValueError(f"unknown history role: {role}")
        return contents

    async def generate(
        self, system_instruction: str, history: list[dict], tools: Optional[list[dict]] = None
    ) -> LLMResponse:
        body: dict = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": self._to_contents(history),
        }
        if tools:
            body["tools"] = [{"functionDeclarations": tools}]

        url = f"{self.API_URL}/{self.model}:generateContent?key={self.api_key}"

        max_retries = 5
        attempt = 0
        while True:
            await self.rate_limiter.acquire()
            resp = await self.client.post(url, json=body)
            data = resp.json()

            candidates = data.get("candidates")
            if candidates:
                return self._parse_candidate(candidates[0])

            error = data.get("error", {})
            status = error.get("status", "")
            if status == "RESOURCE_EXHAUSTED" and attempt < max_retries:
                retry_delay_secs = 10.0
                for detail in error.get("details", []):
                    if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                        raw_delay = detail.get("retryDelay", "10s")
                        try:
                            retry_delay_secs = float(raw_delay.rstrip("s"))
                        except ValueError:
                            pass
                attempt += 1
                print(f"[gemini] quota exceeded, retrying in {retry_delay_secs:.1f}s (attempt {attempt}/{max_retries})")
                await asyncio.sleep(retry_delay_secs + 0.5)
                continue

            message = error.get("message")
            suffix = f" ({message})" if message else ""
            raise RuntimeError(f"Gemini request failed{suffix}: {data}")

    def _parse_candidate(self, candidate: dict) -> LLMResponse:
        parts = candidate.get("content", {}).get("parts", [])
        text = ""
        function_calls = []
        for i, part in enumerate(parts):
            if "functionCall" in part:
                fc = part["functionCall"]
                function_calls.append({"id": f"call_{i}", "name": fc.get("name", ""), "args": fc.get("args", {})})
            elif "text" in part:
                text += part["text"]
        return LLMResponse(text=text, function_calls=function_calls, raw=parts)




class OpenRouterClient(LLMClient):


    provider_name = "openrouter"
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str, rate_limiter, http_client: Optional[httpx.AsyncClient] = None) -> None:
        self.api_key = api_key
        self.model = model
        self.rate_limiter = rate_limiter
        self._owns_client = http_client is None
        self.client = http_client or httpx.AsyncClient(timeout=120.0)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def _to_messages(self, system_instruction: str, history: list[dict]) -> list[dict]:
        messages: list[dict] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        for turn in history:
            role = turn["role"]
            if role == "user":
                messages.append({"role": "user", "content": turn["text"]})
            elif role == "assistant":
                msg: dict = {"role": "assistant", "content": turn.get("text") or None}
                function_calls = turn.get("function_calls", [])
                if function_calls:
                    msg["tool_calls"] = [
                        {
                            "id": fc.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {"name": fc["name"], "arguments": json.dumps(fc["args"])},
                        }
                        for i, fc in enumerate(function_calls)
                    ]
                messages.append(msg)
            elif role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": turn.get("id", "call_0"),
                        "content": turn["result"],
                    }
                )
            else:
                raise ValueError(f"unknown history role: {role}")
        return messages

    @staticmethod
    def _to_openai_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    async def generate(
        self, system_instruction: str, history: list[dict], tools: Optional[list[dict]] = None
    ) -> LLMResponse:
        body: dict = {
            "model": self.model,
            "messages": self._to_messages(system_instruction, history),
        }
        if tools:
            body["tools"] = self._to_openai_tools(tools)
            body["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            # Optional attribution headers OpenRouter uses for its
            # leaderboards — harmless to include, never required.
            "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "https://github.com/"),
            "X-Title": os.environ.get("OPENROUTER_APP_NAME", "CompeteEvolve"),
        }

        max_retries = 5
        attempt = 0
        while True:
            await self.rate_limiter.acquire()
            resp = await self.client.post(self.API_URL, json=body, headers=headers)
            data = resp.json()

            choices = data.get("choices")
            if choices:
                return self._parse_choice(choices[0])

            error = data.get("error", {})
            code = error.get("code") or str(resp.status_code)
            if resp.status_code == 429 and attempt < max_retries:
                attempt += 1
                retry_delay_secs = 5.0 * attempt
                print(f"[openrouter] rate limited, retrying in {retry_delay_secs:.1f}s (attempt {attempt}/{max_retries})")
                await asyncio.sleep(retry_delay_secs)
                continue

            message = error.get("message")
            suffix = f" ({message})" if message else ""
            raise RuntimeError(f"OpenRouter request failed [{code}]{suffix}: {data}")

    def _parse_choice(self, choice: dict) -> LLMResponse:
        message = choice.get("message", {})
        text = message.get("content") or ""
        function_calls = []
        for tc in message.get("tool_calls") or []:
            function = tc.get("function", {})
            raw_args = function.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {}
            function_calls.append({"id": tc.get("id", ""), "name": function.get("name", ""), "args": args})
        return LLMResponse(text=text, function_calls=function_calls)
