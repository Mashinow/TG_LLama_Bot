import asyncio
from types import SimpleNamespace
from typing import Self

import pytest

import tg_llama_bot.bot as bot_module
from tg_llama_bot.bot import (
    ACCESS_DENIED_TEXT,
    DEFAULT_IMAGE_PROMPT,
    HELP_TEXT,
    IMAGE_DOWNLOAD_ERROR_TEXT,
    IMAGE_TOO_LARGE_TEXT,
    INPUT_TOO_LONG_TEXT,
    MAX_IMAGE_BYTES,
    RESET_TEXT,
    START_TEXT,
    UNSUPPORTED_TEXT,
    UPSTREAM_ERROR_TEXT,
    VISION_UNAVAILABLE_TEXT,
    BotDependencies,
    BotHandlers,
    ChatService,
    VisionUnavailableError,
    build_dispatcher,
    is_allowed,
    run_bot,
    split_telegram_text,
)
from tg_llama_bot.history import HistoryStore, InputTooLongError
from tg_llama_bot.llama_client import LlamaConnectionError
from tg_llama_bot.models import (
    AppConfig,
    ChatMessage,
    ImageAttachment,
    RuntimeEvent,
    RuntimeState,
    ServerCapabilities,
)

CAPABILITIES = ServerCapabilities(
    model_id="model.gguf",
    n_ctx=4096,
    max_output_tokens=512,
    server_max_output_tokens=512,
    reasoning_format="none",
    modalities=("text",),
)

VISION_CAPABILITIES = ServerCapabilities(
    model_id="vision.gguf",
    n_ctx=4096,
    max_output_tokens=512,
    server_max_output_tokens=512,
    reasoning_format="none",
    modalities=("text", "vision"),
)

IMAGE = ImageAttachment("image/jpeg", b"image-bytes")


async def character_counter(content: str) -> int:
    return len(content)


class FakeLlama:
    def __init__(
        self,
        answer: str = "model answer",
        error: Exception | None = None,
    ) -> None:
        self.answer = answer
        self.error = error
        self.completed_messages: list[list[ChatMessage]] = []
        self.exited = False

    async def count_tokens(self, content: str) -> int:
        return len(content)

    async def complete(
        self,
        model_id: str,
        messages: list[ChatMessage],
        max_tokens: int,
    ) -> str:
        self.completed_messages.append(list(messages))
        if self.error is not None:
            raise self.error
        return self.answer

    async def discover(self) -> ServerCapabilities:
        return CAPABILITIES

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.exited = True


class FakeService:
    def __init__(
        self,
        answer: str = "model answer",
        error: Exception | None = None,
    ) -> None:
        self.answer_text = answer
        self.error = error
        self.calls: list[tuple[int, str, tuple[ImageAttachment, ...]]] = []

    async def answer(
        self,
        chat_id: int,
        text: str,
        images: tuple[ImageAttachment, ...] = (),
    ) -> str:
        self.calls.append((chat_id, text, images))
        if self.error is not None:
            raise self.error
        return self.answer_text


class FakeTelegramDownloader:
    def __init__(self, payload: bytes = b"image-bytes", error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.calls: list[object] = []

    async def download(self, downloadable: object, *, destination) -> object:
        self.calls.append(downloadable)
        if self.error is not None:
            raise self.error
        destination.write(self.payload)
        return destination


class FakeMessage:
    def __init__(
        self,
        user_id: int,
        chat_id: int,
        text: str | None = "hello",
        *,
        caption: str | None = None,
        photo: list[object] | None = None,
        document: object | None = None,
        bot: object | None = None,
    ) -> None:
        self.from_user = SimpleNamespace(id=user_id)
        self.chat = SimpleNamespace(id=chat_id)
        self.text = text
        self.caption = caption
        self.photo = photo
        self.document = document
        self.bot = bot or FakeTelegramDownloader()
        self.answers: list[str] = []

    async def answer(self, text: str) -> None:
        self.answers.append(text)


def test_empty_allowlist_allows_everyone() -> None:
    assert is_allowed(1, ())


def test_populated_allowlist_rejects_unknown_user() -> None:
    assert is_allowed(7, (7, 9))
    assert not is_allowed(8, (7, 9))


def test_split_prefers_paragraph_then_newline_boundaries() -> None:
    text = "first paragraph\n\nsecond paragraph\nline"
    parts = split_telegram_text(text, limit=20)
    assert "".join(parts) == text
    assert all(0 < len(part) <= 20 for part in parts)
    assert parts[0] == "first paragraph\n\n"


def test_split_uses_hard_boundary_for_unbroken_text() -> None:
    assert split_telegram_text("abcdefgh", limit=3) == ["abc", "def", "gh"]


@pytest.mark.asyncio
async def test_chat_service_commits_only_successful_exchange() -> None:
    llama = FakeLlama()
    history = HistoryStore()
    service = ChatService(llama, CAPABILITIES, history)
    result = await service.answer(10, "hello")
    assert result == "model answer"
    next_messages = await history.prepare(10, "next", character_counter, 1000)
    assert next_messages == [
        ChatMessage("user", "hello"),
        ChatMessage("assistant", "model answer"),
        ChatMessage("user", "next"),
    ]


@pytest.mark.asyncio
async def test_chat_service_does_not_commit_failed_exchange() -> None:
    llama = FakeLlama(error=LlamaConnectionError("offline"))
    history = HistoryStore()
    service = ChatService(llama, CAPABILITIES, history)
    with pytest.raises(LlamaConnectionError):
        await service.answer(10, "hello")
    assert await history.prepare(10, "next", character_counter, 1000) == [
        ChatMessage("user", "next")
    ]


@pytest.mark.asyncio
async def test_chat_service_retains_image_for_follow_up() -> None:
    llama = FakeLlama()
    history = HistoryStore()
    service = ChatService(llama, VISION_CAPABILITIES, history)
    await service.answer(10, "describe", images=(IMAGE,))
    await service.answer(10, "what color?")
    assert llama.completed_messages[1] == [
        ChatMessage("user", "describe", (IMAGE,)),
        ChatMessage("assistant", "model answer"),
        ChatMessage("user", "what color?"),
    ]


@pytest.mark.asyncio
async def test_chat_service_rejects_image_when_model_has_no_vision() -> None:
    llama = FakeLlama()
    service = ChatService(llama, CAPABILITIES, HistoryStore())
    with pytest.raises(VisionUnavailableError):
        await service.answer(10, "describe", images=(IMAGE,))
    assert llama.completed_messages == []


@pytest.mark.asyncio
async def test_unauthorized_text_never_reaches_model() -> None:
    service = FakeService()
    handlers = BotHandlers(
        AppConfig("token", allowed_user_ids=(7,)),
        service,
        HistoryStore(),
    )
    message = FakeMessage(user_id=8, chat_id=10)
    await handlers.text(message)
    assert service.calls == []
    assert message.answers == [ACCESS_DENIED_TEXT]


@pytest.mark.asyncio
async def test_reset_clears_only_current_chat() -> None:
    history = HistoryStore()
    history.commit(10, "old", "answer")
    history.commit(11, "keep", "answer")
    handlers = BotHandlers(AppConfig("token"), FakeService(), history)
    message = FakeMessage(user_id=7, chat_id=10, text="/reset")
    await handlers.reset(message)
    assert message.answers == [RESET_TEXT]
    assert await history.prepare(10, "new", character_counter, 1000) == [
        ChatMessage("user", "new")
    ]
    assert len(await history.prepare(11, "new", character_counter, 1000)) == 3


@pytest.mark.asyncio
async def test_long_answer_is_sent_as_multiple_messages() -> None:
    service = FakeService(answer="x" * 5000)
    handlers = BotHandlers(AppConfig("token"), service, HistoryStore())
    message = FakeMessage(user_id=7, chat_id=10)
    await handlers.text(message)
    assert [len(part) for part in message.answers] == [4096, 904]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (InputTooLongError("too long"), INPUT_TOO_LONG_TEXT),
        (LlamaConnectionError("offline"), UPSTREAM_ERROR_TEXT),
    ],
)
async def test_text_handler_maps_model_errors_to_safe_messages(
    error: Exception,
    expected: str,
) -> None:
    handlers = BotHandlers(
        AppConfig("token"),
        FakeService(error=error),
        HistoryStore(),
    )
    message = FakeMessage(user_id=7, chat_id=10)
    await handlers.text(message)
    assert message.answers == [expected]


@pytest.mark.asyncio
async def test_photo_with_caption_reaches_model_as_jpeg() -> None:
    service = FakeService()
    downloader = FakeTelegramDownloader()
    small = SimpleNamespace(file_id="small", file_size=4)
    large = SimpleNamespace(file_id="large", file_size=11)
    message = FakeMessage(
        user_id=7,
        chat_id=10,
        text=None,
        caption="Что изображено?",
        photo=[small, large],
        bot=downloader,
    )
    handlers = BotHandlers(AppConfig("token"), service, HistoryStore())
    await handlers.image(message)
    assert downloader.calls == [large]
    assert service.calls == [
        (10, "Что изображено?", (ImageAttachment("image/jpeg", b"image-bytes"),))
    ]
    assert message.answers == ["model answer"]


@pytest.mark.asyncio
async def test_captionless_photo_uses_default_prompt() -> None:
    service = FakeService()
    message = FakeMessage(
        user_id=7,
        chat_id=10,
        text=None,
        photo=[SimpleNamespace(file_id="photo", file_size=11)],
    )
    handlers = BotHandlers(AppConfig("token"), service, HistoryStore())
    await handlers.image(message)
    assert service.calls[0][1] == DEFAULT_IMAGE_PROMPT


@pytest.mark.asyncio
async def test_image_document_preserves_mime_type() -> None:
    service = FakeService()
    document = SimpleNamespace(
        file_id="document",
        file_size=11,
        mime_type="image/png",
    )
    message = FakeMessage(
        user_id=7,
        chat_id=10,
        text=None,
        caption="analyze",
        document=document,
    )
    handlers = BotHandlers(AppConfig("token"), service, HistoryStore())
    await handlers.image(message)
    assert service.calls == [
        (10, "analyze", (ImageAttachment("image/png", b"image-bytes"),))
    ]


@pytest.mark.asyncio
async def test_reported_oversized_image_is_rejected_before_download() -> None:
    service = FakeService()
    downloader = FakeTelegramDownloader()
    message = FakeMessage(
        user_id=7,
        chat_id=10,
        text=None,
        photo=[SimpleNamespace(file_id="photo", file_size=MAX_IMAGE_BYTES + 1)],
        bot=downloader,
    )
    handlers = BotHandlers(AppConfig("token"), service, HistoryStore())
    await handlers.image(message)
    assert downloader.calls == []
    assert service.calls == []
    assert message.answers == [IMAGE_TOO_LARGE_TEXT]


@pytest.mark.asyncio
async def test_downloaded_oversized_image_is_rejected() -> None:
    service = FakeService()
    downloader = FakeTelegramDownloader(payload=b"x" * (MAX_IMAGE_BYTES + 1))
    message = FakeMessage(
        user_id=7,
        chat_id=10,
        text=None,
        photo=[SimpleNamespace(file_id="photo", file_size=None)],
        bot=downloader,
    )
    handlers = BotHandlers(AppConfig("token"), service, HistoryStore())
    await handlers.image(message)
    assert service.calls == []
    assert message.answers == [IMAGE_TOO_LARGE_TEXT]


@pytest.mark.asyncio
async def test_empty_downloaded_image_is_rejected() -> None:
    service = FakeService()
    downloader = FakeTelegramDownloader(payload=b"")
    message = FakeMessage(
        user_id=7,
        chat_id=10,
        text=None,
        photo=[SimpleNamespace(file_id="photo", file_size=None)],
        bot=downloader,
    )
    handlers = BotHandlers(AppConfig("token"), service, HistoryStore())
    await handlers.image(message)
    assert service.calls == []
    assert message.answers == [IMAGE_DOWNLOAD_ERROR_TEXT]


@pytest.mark.asyncio
async def test_non_image_document_is_unsupported() -> None:
    service = FakeService()
    message = FakeMessage(
        user_id=7,
        chat_id=10,
        text=None,
        document=SimpleNamespace(
            file_id="document",
            file_size=10,
            mime_type="application/pdf",
        ),
    )
    handlers = BotHandlers(AppConfig("token"), service, HistoryStore())
    await handlers.image(message)
    assert service.calls == []
    assert message.answers == [UNSUPPORTED_TEXT]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (VisionUnavailableError(), VISION_UNAVAILABLE_TEXT),
        (LlamaConnectionError("offline"), UPSTREAM_ERROR_TEXT),
    ],
)
async def test_image_handler_maps_model_errors_to_safe_messages(
    error: Exception,
    expected: str,
) -> None:
    service = FakeService(error=error)
    message = FakeMessage(
        user_id=7,
        chat_id=10,
        text=None,
        photo=[SimpleNamespace(file_id="photo", file_size=11)],
    )
    handlers = BotHandlers(AppConfig("token"), service, HistoryStore())
    await handlers.image(message)
    assert message.answers == [expected]


@pytest.mark.asyncio
async def test_image_download_failure_returns_safe_message() -> None:
    downloader = FakeTelegramDownloader(error=OSError("read failed"))
    message = FakeMessage(
        user_id=7,
        chat_id=10,
        text=None,
        photo=[SimpleNamespace(file_id="photo", file_size=11)],
        bot=downloader,
    )
    handlers = BotHandlers(AppConfig("token"), FakeService(), HistoryStore())
    await handlers.image(message)
    assert message.answers == [IMAGE_DOWNLOAD_ERROR_TEXT]


def test_dispatcher_registers_image_before_text_and_fallback() -> None:
    dispatcher = build_dispatcher(
        BotDependencies(AppConfig("token"), VISION_CAPABILITIES, FakeLlama(), HistoryStore())
    )
    callback_names = [handler.callback.__name__ for handler in dispatcher.message.handlers]
    assert callback_names[-3:] == ["image", "text", "unsupported"]


@pytest.mark.asyncio
async def test_command_and_unsupported_handlers_return_helpful_text() -> None:
    handlers = BotHandlers(AppConfig("token"), FakeService(), HistoryStore())
    start_message = FakeMessage(user_id=7, chat_id=10, text="/start")
    help_message = FakeMessage(user_id=7, chat_id=10, text="/help")
    media_message = FakeMessage(user_id=7, chat_id=10, text=None)
    await handlers.start(start_message)
    await handlers.help(help_message)
    await handlers.unsupported(media_message)
    assert start_message.answers == [START_TEXT]
    assert help_message.answers == [HELP_TEXT]
    assert media_message.answers == [UNSUPPORTED_TEXT]


class FakeBot:
    def __init__(self) -> None:
        self.validated = False
        self.webhook_delete_calls: list[bool] = []
        self.closed = False
        self.session = SimpleNamespace(close=self.close)

    async def get_me(self) -> SimpleNamespace:
        self.validated = True
        return SimpleNamespace(username="test_bot")

    async def delete_webhook(self, *, drop_pending_updates: bool) -> bool:
        self.webhook_delete_calls.append(drop_pending_updates)
        return True

    async def close(self) -> None:
        self.closed = True


class FakeDispatcher:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.released = asyncio.Event()

    async def start_polling(self, bot, **kwargs) -> None:
        self.started.set()
        await self.released.wait()

    async def stop_polling(self) -> None:
        self.released.set()


@pytest.mark.asyncio
async def test_run_bot_discovers_validates_polls_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llama = FakeLlama()
    telegram_bot = FakeBot()
    dispatcher = FakeDispatcher()
    events: list[RuntimeEvent] = []
    stop_event = asyncio.Event()

    monkeypatch.setattr(bot_module, "make_llama_client", lambda url: llama)
    monkeypatch.setattr(bot_module, "make_bot", lambda token: telegram_bot)
    monkeypatch.setattr(bot_module, "build_dispatcher", lambda dependencies: dispatcher)

    task = asyncio.create_task(run_bot(AppConfig("token"), stop_event, events.append))
    await asyncio.wait_for(dispatcher.started.wait(), timeout=1)
    stop_event.set()
    await asyncio.wait_for(task, timeout=1)

    assert telegram_bot.validated
    assert telegram_bot.webhook_delete_calls == [False]
    assert telegram_bot.closed
    assert llama.exited
    assert RuntimeEvent("capabilities", CAPABILITIES) in events
    assert RuntimeEvent("state", RuntimeState.RUNNING) in events
