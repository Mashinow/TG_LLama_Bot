import json

import httpx
import pytest

from tg_llama_bot.llama_client import (
    LlamaClient,
    LlamaConnectionError,
    LlamaInputError,
    LlamaProtocolError,
    RetryPolicy,
)
from tg_llama_bot.models import ChatMessage, ImageAttachment


@pytest.mark.asyncio
async def test_discover_selects_first_model_and_server_properties() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "first.gguf"}, {"id": "second.gguf"}]},
            )
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "default_generation_settings": {
                        "params": {
                            "max_tokens": -1,
                            "n_predict": -1,
                            "reasoning_format": "none",
                        },
                        "n_ctx": 65536,
                    },
                    "modalities": {
                        "vision": False,
                        "audio": False,
                        "video": False,
                    },
                },
            )
        raise AssertionError(request.url.path)

    http = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    client = LlamaClient("http://test", http_client=http)
    capabilities = await client.discover()
    assert capabilities.model_id == "first.gguf"
    assert capabilities.n_ctx == 65536
    assert capabilities.max_output_tokens == 2048
    assert capabilities.server_max_output_tokens is None
    assert capabilities.reasoning_format == "none"
    assert capabilities.modalities == ("text",)
    await client.aclose()


@pytest.mark.asyncio
async def test_discover_accepts_models_shape_and_positive_server_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payloads = {
            "/health": {"status": "ok"},
            "/v1/models": {"models": [{"model": "first.gguf"}]},
            "/props": {
                "default_generation_settings": {
                    "params": {"max_tokens": 512},
                    "n_ctx": 4096,
                },
                "reasoning_format": "deepseek",
                "modalities": {"vision": True, "audio": False, "video": False},
            },
        }
        return httpx.Response(200, json=payloads[request.url.path])

    http = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    client = LlamaClient("http://test", http_client=http)
    capabilities = await client.discover()
    assert capabilities.model_id == "first.gguf"
    assert capabilities.max_output_tokens == 512
    assert capabilities.server_max_output_tokens == 512
    assert capabilities.reasoning_format == "deepseek"
    assert capabilities.modalities == ("text", "vision")
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("models_payload", "props_payload"),
    [
        ({"data": []}, {"default_generation_settings": {"n_ctx": 4096}}),
        (
            {"data": [{"id": "first"}]},
            {"default_generation_settings": {"n_ctx": "bad"}},
        ),
    ],
)
async def test_discover_rejects_missing_model_or_invalid_context(
    models_payload: dict,
    props_payload: dict,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payloads = {
            "/health": {"status": "ok"},
            "/v1/models": models_payload,
            "/props": props_payload,
        }
        return httpx.Response(200, json=payloads[request.url.path])

    http = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    client = LlamaClient("http://test", http_client=http)
    with pytest.raises(LlamaProtocolError):
        await client.discover()
    await client.aclose()


@pytest.mark.asyncio
async def test_count_tokens_sends_llama_server_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tokenize"
        assert json.loads(request.content) == {
            "content": "hello",
            "add_special": False,
        }
        return httpx.Response(200, json={"tokens": [10, 20, 30]})

    http = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    client = LlamaClient("http://test", http_client=http)
    assert await client.count_tokens("hello") == 3
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_sends_chat_contract_and_returns_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content) == {
            "model": "model.gguf",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 128,
            "stream": False,
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "answer"}}]},
        )

    http = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    client = LlamaClient("http://test", http_client=http)
    answer = await client.complete(
        "model.gguf",
        [ChatMessage("user", "hello")],
        128,
    )
    assert answer == "answer"
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_sends_images_as_data_url_content_parts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content) == {
            "model": "model.gguf",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,aW1hZ2UtYnl0ZXM="
                            },
                        },
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
            "max_tokens": 128,
            "stream": False,
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "answer"}}]},
        )

    http = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    client = LlamaClient("http://test", http_client=http)
    answer = await client.complete(
        "model.gguf",
        [
            ChatMessage(
                "user",
                "describe",
                (ImageAttachment("image/jpeg", b"image-bytes"),),
            )
        ],
        128,
    )
    assert answer == "answer"
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_rejects_empty_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": ""}}]},
        )

    http = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    client = LlamaClient("http://test", http_client=http)
    with pytest.raises(LlamaProtocolError):
        await client.complete("model.gguf", [], 32)
    await client.aclose()


@pytest.mark.asyncio
async def test_transient_failure_stops_after_three_total_attempts() -> None:
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, json={"error": "busy"})

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    http = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    client = LlamaClient(
        "http://test",
        http_client=http,
        retry_policy=RetryPolicy(attempts=3, base_delay=0.1, max_delay=1.0),
        sleep=fake_sleep,
        jitter=lambda low, high: 0.0,
    )
    with pytest.raises(LlamaConnectionError):
        await client.count_tokens("hello")
    assert attempts == 3
    assert delays == [0.1, 0.2]
    await client.aclose()


@pytest.mark.asyncio
async def test_http_400_is_not_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"error": "invalid"})

    http = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    client = LlamaClient("http://test", http_client=http)
    with pytest.raises(LlamaInputError):
        await client.count_tokens("x")
    assert attempts == 1
    await client.aclose()
