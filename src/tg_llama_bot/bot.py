import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from io import BytesIO

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.backoff import BackoffConfig

from tg_llama_bot.history import HistoryStore, InputTooLongError
from tg_llama_bot.llama_client import LlamaClient, LlamaError
from tg_llama_bot.models import (
    AppConfig,
    EventSink,
    ImageAttachment,
    RuntimeEvent,
    RuntimeState,
    ServerCapabilities,
)


class VisionUnavailableError(ValueError):
    """The selected model does not advertise image input support."""

START_TEXT = "Бот запущен. Отправьте текст или изображение, чтобы обратиться к модели."
HELP_TEXT = (
    "Доступны текст, фотографии и изображения-документы. "
    "Команда /reset очищает историю."
)
RESET_TEXT = "История этого чата очищена."
ACCESS_DENIED_TEXT = "Доступ к боту запрещён."
UNSUPPORTED_TEXT = "Поддерживаются текст, фотографии и изображения-документы."
INPUT_TOO_LONG_TEXT = "Сообщение слишком длинное для контекста модели."
UPSTREAM_ERROR_TEXT = "Модель временно недоступна. Попробуйте позже."
DEFAULT_IMAGE_PROMPT = "Опиши изображение."
VISION_UNAVAILABLE_TEXT = "Текущая модель не поддерживает анализ изображений."
IMAGE_TOO_LARGE_TEXT = "Изображение слишком большое. Максимальный размер — 10 МиБ."
IMAGE_DOWNLOAD_ERROR_TEXT = "Не удалось загрузить изображение. Попробуйте отправить его снова."
MAX_IMAGE_BYTES = 10 * 1024 * 1024
PENDING_TEXT = "..."
STREAM_EDIT_INTERVAL_TOKENS = 150

TokenSink = Callable[[str], Awaitable[None]]


def is_allowed(user_id: int | None, allowed_user_ids: tuple[int, ...]) -> bool:
    if user_id is None:
        return False
    return not allowed_user_ids or user_id in allowed_user_ids


class ChatService:
    def __init__(
        self,
        llama: LlamaClient,
        capabilities: ServerCapabilities,
        history: HistoryStore,
    ) -> None:
        self._llama = llama
        self._capabilities = capabilities
        self._history = history

    async def answer(
        self,
        chat_id: int,
        text: str,
        images: tuple[ImageAttachment, ...] = (),
        on_token: TokenSink | None = None,
    ) -> str:
        if images and "vision" not in self._capabilities.modalities:
            raise VisionUnavailableError
        async with self._history.lock_for(chat_id):
            prompt_budget = (
                self._capabilities.n_ctx
                - self._capabilities.max_output_tokens
                - 128
            )
            messages = await self._history.prepare(
                chat_id,
                text,
                self._llama.count_tokens,
                prompt_budget,
                images,
            )
            chunks: list[str] = []
            async for token in self._llama.stream_complete(
                self._capabilities.model_id,
                messages,
                self._capabilities.max_output_tokens,
            ):
                chunks.append(token)
                if on_token is not None:
                    await on_token(token)
            answer = "".join(chunks)
            self._history.commit(chat_id, text, answer, images)
            return answer


class BotHandlers:
    def __init__(
        self,
        config: AppConfig,
        service: ChatService,
        history: HistoryStore,
    ) -> None:
        self._config = config
        self._service = service
        self._history = history

    async def start(self, message: Message) -> None:
        if await self._authorize(message):
            await message.answer(START_TEXT)

    async def help(self, message: Message) -> None:
        if await self._authorize(message):
            await message.answer(HELP_TEXT)

    async def reset(self, message: Message) -> None:
        if not await self._authorize(message):
            return
        await self._history.reset(message.chat.id)
        await message.answer(RESET_TEXT)

    async def text(self, message: Message) -> None:
        if not await self._authorize(message):
            return
        if not isinstance(message.text, str):
            await message.answer(UNSUPPORTED_TEXT)
            return
        await self._answer_with_progress(message, message.text)

    async def image(self, message: Message) -> None:
        if not await self._authorize(message):
            return
        pending = await message.answer(PENDING_TEXT)

        downloadable: object | None = None
        media_type: str | None = None
        if message.photo:
            downloadable = message.photo[-1]
            media_type = "image/jpeg"
        elif message.document is not None:
            document_media_type = message.document.mime_type
            if isinstance(document_media_type, str) and document_media_type.startswith(
                "image/"
            ):
                downloadable = message.document
                media_type = document_media_type

        if downloadable is None or media_type is None:
            await pending.edit_text(UNSUPPORTED_TEXT)
            return

        reported_size = getattr(downloadable, "file_size", None)
        if isinstance(reported_size, int) and reported_size > MAX_IMAGE_BYTES:
            await pending.edit_text(IMAGE_TOO_LARGE_TEXT)
            return

        destination = BytesIO()
        try:
            await message.bot.download(downloadable, destination=destination)
        except (TelegramAPIError, OSError):
            await pending.edit_text(IMAGE_DOWNLOAD_ERROR_TEXT)
            return

        image_data = destination.getvalue()
        if len(image_data) > MAX_IMAGE_BYTES:
            await pending.edit_text(IMAGE_TOO_LARGE_TEXT)
            return
        if not image_data:
            await pending.edit_text(IMAGE_DOWNLOAD_ERROR_TEXT)
            return

        prompt = (
            message.caption
            if isinstance(message.caption, str) and message.caption.strip()
            else DEFAULT_IMAGE_PROMPT
        )
        await self._answer_with_progress(
            message,
            prompt,
            images=(ImageAttachment(media_type, image_data),),
            pending=pending,
        )

    async def unsupported(self, message: Message) -> None:
        if await self._authorize(message):
            await message.answer(UNSUPPORTED_TEXT)

    async def _answer_with_progress(
        self,
        message: Message,
        text: str,
        images: tuple[ImageAttachment, ...] = (),
        pending: Message | None = None,
    ) -> None:
        if pending is None:
            pending = await message.answer(PENDING_TEXT)
        chunks: list[str] = []
        token_count = 0
        rendered_text = PENDING_TEXT

        async def on_token(token: str) -> None:
            nonlocal token_count, rendered_text
            chunks.append(token)
            token_count += 1
            if token_count % STREAM_EDIT_INTERVAL_TOKENS == 0:
                current_text = "".join(chunks)
                if current_text != rendered_text:
                    try:
                        await pending.edit_text(current_text)
                    except TelegramAPIError:
                        return
                    rendered_text = current_text

        try:
            answer = await self._service.answer(
                message.chat.id,
                text,
                images=images,
                on_token=on_token,
            )
        except VisionUnavailableError:
            await pending.edit_text(VISION_UNAVAILABLE_TEXT)
            return
        except InputTooLongError:
            await pending.edit_text(INPUT_TOO_LONG_TEXT)
            return
        except LlamaError:
            await pending.edit_text(UPSTREAM_ERROR_TEXT)
            return

        if answer != rendered_text:
            await pending.edit_text(answer)

    async def _authorize(self, message: Message) -> bool:
        user_id = message.from_user.id if message.from_user is not None else None
        allowed = is_allowed(user_id, self._config.allowed_user_ids)
        if not allowed:
            await message.answer(ACCESS_DENIED_TEXT)
        return allowed


@dataclass(frozen=True, slots=True)
class BotDependencies:
    config: AppConfig
    capabilities: ServerCapabilities
    llama: LlamaClient
    history: HistoryStore


def build_dispatcher(dependencies: BotDependencies) -> Dispatcher:
    dispatcher = Dispatcher()
    service = ChatService(
        dependencies.llama,
        dependencies.capabilities,
        dependencies.history,
    )
    handlers = BotHandlers(dependencies.config, service, dependencies.history)
    dispatcher.message.register(handlers.start, Command("start"))
    dispatcher.message.register(handlers.help, Command("help"))
    dispatcher.message.register(handlers.reset, Command("reset"))
    dispatcher.message.register(handlers.image, F.photo | F.document)
    dispatcher.message.register(handlers.text, F.text)
    dispatcher.message.register(handlers.unsupported)
    return dispatcher


def make_llama_client(base_url: str) -> LlamaClient:
    return LlamaClient(base_url)


def make_bot(token: str) -> Bot:
    return Bot(token=token)


async def run_bot(
    config: AppConfig,
    stop_event: asyncio.Event,
    emit: EventSink,
) -> None:
    async with make_llama_client(config.llama_base_url) as llama:
        capabilities = await llama.discover()
        telegram_bot = make_bot(config.telegram_token)
        history = HistoryStore(
            warning_callback=lambda text: emit(RuntimeEvent("log", text))
        )
        polling_task: asyncio.Task[None] | None = None
        stop_waiter: asyncio.Task[bool] | None = None
        try:
            await telegram_bot.get_me()
            await telegram_bot.delete_webhook(drop_pending_updates=False)
            dispatcher = build_dispatcher(
                BotDependencies(config, capabilities, llama, history)
            )
            emit(RuntimeEvent("capabilities", capabilities))
            emit(RuntimeEvent("state", RuntimeState.RUNNING))
            polling_task = asyncio.create_task(
                dispatcher.start_polling(
                    telegram_bot,
                    handle_signals=False,
                    close_bot_session=False,
                    backoff_config=BackoffConfig(
                        min_delay=1.0,
                        max_delay=30.0,
                        factor=1.5,
                        jitter=0.1,
                    ),
                )
            )
            stop_waiter = asyncio.create_task(stop_event.wait())
            await asyncio.sleep(0)
            done, _ = await asyncio.wait(
                {polling_task, stop_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if polling_task in done:
                await polling_task
            else:
                await dispatcher.stop_polling()
                await polling_task
        finally:
            if stop_waiter is not None:
                stop_waiter.cancel()
                await asyncio.gather(stop_waiter, return_exceptions=True)
            if polling_task is not None and not polling_task.done():
                polling_task.cancel()
                await asyncio.gather(polling_task, return_exceptions=True)
            history.clear()
            await telegram_bot.session.close()
