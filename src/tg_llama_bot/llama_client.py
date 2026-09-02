import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import httpx

from tg_llama_bot.models import ChatMessage, ServerCapabilities


class LlamaError(RuntimeError):
    """Base error for llama-server operations."""


class LlamaConnectionError(LlamaError):
    """A transient server or network problem exhausted its retries."""


class LlamaInputError(LlamaError):
    """The server rejected a non-retryable request."""


class LlamaProtocolError(LlamaError):
    """The server returned a malformed or unusable response."""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts: int = 3
    base_delay: float = 0.25
    max_delay: float = 2.0


DEFAULT_RETRY_POLICY = RetryPolicy()


class LlamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        http_client: httpx.AsyncClient | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http_client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )
        self._retry_policy = retry_policy
        self._sleep = sleep
        self._jitter = jitter

    async def discover(self) -> ServerCapabilities:
        health = self._json(await self._request("GET", "/health"))
        if health.get("status") != "ok":
            raise LlamaConnectionError("llama-server не готов принимать запросы.")

        models_payload = self._json(await self._request("GET", "/v1/models"))
        entries = models_payload.get("data")
        model_key = "id"
        if entries is None:
            entries = models_payload.get("models")
            model_key = "model"
        if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
            raise LlamaProtocolError("llama-server не вернул доступную модель.")
        model_id = entries[0].get(model_key)
        if not isinstance(model_id, str) or not model_id.strip():
            raise LlamaProtocolError("llama-server вернул некорректный ID модели.")

        props = self._json(await self._request("GET", "/props"))
        defaults = props.get("default_generation_settings")
        if not isinstance(defaults, dict):
            raise LlamaProtocolError("В /props отсутствуют настройки генерации.")
        n_ctx = defaults.get("n_ctx")
        if isinstance(n_ctx, bool) or not isinstance(n_ctx, int) or n_ctx <= 0:
            raise LlamaProtocolError("В /props указан некорректный размер контекста.")

        params = defaults.get("params", {})
        if not isinstance(params, dict):
            raise LlamaProtocolError("В /props указаны некорректные параметры генерации.")
        server_limit = self._positive_limit(
            params.get("max_tokens"),
            params.get("n_predict"),
        )
        max_output_tokens = server_limit or min(2048, max(1, n_ctx // 4))

        reasoning = props.get("reasoning_format", params.get("reasoning_format", "none"))
        if not isinstance(reasoning, str):
            reasoning = "none"
        modalities_payload = props.get("modalities", {})
        modalities = ["text"]
        if isinstance(modalities_payload, dict):
            modalities.extend(
                name
                for name in ("vision", "video", "audio")
                if modalities_payload.get(name) is True
            )

        return ServerCapabilities(
            model_id=model_id.strip(),
            n_ctx=n_ctx,
            max_output_tokens=max_output_tokens,
            server_max_output_tokens=server_limit,
            reasoning_format=reasoning,
            modalities=tuple(modalities),
        )

    async def count_tokens(self, content: str) -> int:
        response = await self._request(
            "POST",
            "/tokenize",
            json={"content": content, "add_special": False},
        )
        payload = self._json(response)
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            raise LlamaProtocolError("Некорректный ответ эндпоинта /tokenize.")
        return len(tokens)

    async def complete(
        self,
        model_id: str,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> str:
        response = await self._request(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        payload = self._json(response)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LlamaProtocolError("llama-server не вернул вариант ответа.")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise LlamaProtocolError("llama-server вернул некорректное сообщение.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LlamaProtocolError("llama-server вернул пустой ответ.")
        return content

    async def aclose(self) -> None:
        if not self._http.is_closed:
            await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        attempts = max(1, self._retry_policy.attempts)
        for attempt in range(1, attempts + 1):
            try:
                response = await self._http.request(method, path, json=json)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == attempts:
                    raise LlamaConnectionError(
                        "Не удалось подключиться к llama-server после повторных попыток."
                    ) from exc
                await self._wait_before_retry(attempt)
                continue

            if response.status_code in {408, 429} or response.status_code >= 500:
                if attempt == attempts:
                    raise LlamaConnectionError(
                        f"llama-server временно недоступен (HTTP {response.status_code})."
                    )
                await self._wait_before_retry(attempt)
                continue
            if response.status_code >= 400:
                raise LlamaInputError(
                    f"llama-server отклонил запрос (HTTP {response.status_code})."
                )
            return response

        raise AssertionError("retry loop exhausted unexpectedly")

    async def _wait_before_retry(self, attempt: int) -> None:
        exponential = self._retry_policy.base_delay * 2 ** (attempt - 1)
        delay = min(self._retry_policy.max_delay, exponential)
        delay += self._jitter(0.0, 0.1)
        await self._sleep(delay)

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise LlamaProtocolError("llama-server вернул невалидный JSON.") from exc
        if not isinstance(payload, dict):
            raise LlamaProtocolError("llama-server вернул JSON неверного типа.")
        return payload

    @staticmethod
    def _positive_limit(*values: object) -> int | None:
        for value in values:
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return None
