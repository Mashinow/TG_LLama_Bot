from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest

from scripts.live_smoke import format_smoke_error, smoke
from tg_llama_bot.models import AppConfig, ChatMessage, ServerCapabilities


class FakeLlama:
    def __init__(self, capabilities: ServerCapabilities) -> None:
        self.capabilities = capabilities
        self.completion_call: tuple[str, list[ChatMessage], int] | None = None
        self.exited = False

    async def discover(self) -> ServerCapabilities:
        return self.capabilities

    async def complete(
        self,
        model_id: str,
        messages: list[ChatMessage],
        max_tokens: int,
    ) -> str:
        self.completion_call = (model_id, list(messages), max_tokens)
        return "pong"

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.exited = True


class FakeBot:
    def __init__(self) -> None:
        self.closed = False
        self.session = SimpleNamespace(close=self.close)

    async def get_me(self) -> SimpleNamespace:
        return SimpleNamespace(username="test_bot", id=123)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_smoke_checks_completion_and_telegram_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capabilities = ServerCapabilities(
        "model.gguf",
        4096,
        512,
        512,
        "none",
        ("text",),
    )
    fake_llama = FakeLlama(capabilities)
    fake_bot = FakeBot()
    monkeypatch.setattr(
        "scripts.live_smoke.load_config",
        lambda path: AppConfig("123:SECRET", "http://127.0.0.1:8080", ()),
    )
    monkeypatch.setattr(
        "scripts.live_smoke.make_llama_client",
        lambda url: fake_llama,
    )
    monkeypatch.setattr("scripts.live_smoke.make_bot", lambda token: fake_bot)

    result = await smoke(tmp_path / "config.yaml", "Reply with pong")

    assert result.model_id == "model.gguf"
    assert result.bot_username == "test_bot"
    assert result.completion == "pong"
    assert fake_llama.completion_call == (
        "model.gguf",
        [ChatMessage("user", "Reply with pong")],
        32,
    )
    assert fake_llama.exited
    assert fake_bot.closed


def test_smoke_error_redacts_telegram_token() -> None:
    token = "123:SECRET"
    error = format_smoke_error(
        RuntimeError("https://api.telegram.org/bot123:SECRET/getMe failed"),
        token,
    )
    assert token not in error
    assert "<redacted>" in error
