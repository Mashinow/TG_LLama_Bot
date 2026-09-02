import asyncio
from dataclasses import dataclass

from aiogram import Bot, Dispatcher, F
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

START_TEXT = "Бот запущен. Отправьте текстовое сообщение, чтобы обратиться к модели."
HELP_TEXT = "Доступны текстовые сообщения и команда /reset для очистки истории."
RESET_TEXT = "История этого чата очищена."
ACCESS_DENIED_TEXT = "Доступ к боту запрещён."
UNSUPPORTED_TEXT = "Поддерживаются только текстовые сообщения."
INPUT_TOO_LONG_TEXT = "Сообщение слишком длинное для контекста модели."
UPSTREAM_ERROR_TEXT = "Модель временно недоступна. Попробуйте позже."


def is_allowed(user_id: int | None, allowed_user_ids: tuple[int, ...]) -> bool:
    if user_id is None:
        return False
    return not allowed_user_ids or user_id in allowed_user_ids


def split_telegram_text(text: str, limit: int = 4096) -> list[str]:
    if limit <= 0:
        raise ValueError("Telegram message limit must be positive.")
    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n\n", 0, limit + 1)
        if split_at >= 0:
            split_at += 2
        else:
            split_at = remaining.rfind("\n", 0, limit + 1)
            if split_at >= 0:
                split_at += 1
        if split_at <= 0:
            split_at = limit
        parts.append(remaining[:split_at])
        remaining = remaining[split_at:]
    if remaining:
        parts.append(remaining)
    return parts


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
            answer = await self._llama.complete(
                self._capabilities.model_id,
                messages,
                self._capabilities.max_output_tokens,
            )
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
        try:
            answer = await self._service.answer(message.chat.id, message.text)
        except InputTooLongError:
            await message.answer(INPUT_TOO_LONG_TEXT)
            return
        except LlamaError:
            await message.answer(UPSTREAM_ERROR_TEXT)
            return
        for part in split_telegram_text(answer):
            await message.answer(part)

    async def unsupported(self, message: Message) -> None:
        if await self._authorize(message):
            await message.answer(UNSUPPORTED_TEXT)

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
