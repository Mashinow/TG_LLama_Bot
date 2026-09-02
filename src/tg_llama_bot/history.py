import asyncio
import math
from collections.abc import Awaitable, Callable

from tg_llama_bot.llama_client import LlamaConnectionError, LlamaProtocolError
from tg_llama_bot.models import ChatMessage, ImageAttachment

TokenCounter = Callable[[str], Awaitable[int]]
IMAGE_TOKEN_ESTIMATE = 2048


class InputTooLongError(ValueError):
    """The newest user message cannot fit the prompt budget."""


class HistoryStore:
    def __init__(
        self,
        warning_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._warning_callback = warning_callback
        self._histories: dict[int, list[ChatMessage]] = {}
        self._locks: dict[int, asyncio.Lock] = {}

    def lock_for(self, chat_id: int) -> asyncio.Lock:
        return self._locks.setdefault(chat_id, asyncio.Lock())

    async def prepare(
        self,
        chat_id: int,
        user_text: str,
        token_counter: TokenCounter,
        prompt_budget: int,
        images: tuple[ImageAttachment, ...] = (),
    ) -> list[ChatMessage]:
        candidate = [
            *self._histories.get(chat_id, ()),
            ChatMessage("user", user_text, images),
        ]
        use_fallback = False

        async def count(messages: list[ChatMessage]) -> int:
            nonlocal use_fallback
            serialized = "".join(
                f"{message.role}\n{message.content}\n" for message in messages
            )
            if use_fallback:
                text_tokens = conservative_token_estimate(serialized)
                return text_tokens + sum(
                    len(message.images) * IMAGE_TOKEN_ESTIMATE for message in messages
                )
            try:
                text_tokens = await token_counter(serialized)
            except (LlamaConnectionError, LlamaProtocolError):
                use_fallback = True
                if self._warning_callback is not None:
                    self._warning_callback(
                        "Tokenization unavailable; using conservative estimate."
                    )
                text_tokens = conservative_token_estimate(serialized)
            return text_tokens + sum(
                len(message.images) * IMAGE_TOKEN_ESTIMATE for message in messages
            )

        newest = [candidate[-1]]
        if prompt_budget <= 0 or await count(newest) > prompt_budget:
            raise InputTooLongError("Новое сообщение не помещается в контекст модели.")

        while len(candidate) > 1 and await count(candidate) > prompt_budget:
            candidate = candidate[2:]
        return candidate

    def commit(
        self,
        chat_id: int,
        user_text: str,
        assistant_text: str,
        images: tuple[ImageAttachment, ...] = (),
    ) -> None:
        history = self._histories.setdefault(chat_id, [])
        history.extend(
            (
                ChatMessage("user", user_text, images),
                ChatMessage("assistant", assistant_text),
            )
        )

    async def reset(self, chat_id: int) -> None:
        async with self.lock_for(chat_id):
            self._histories.pop(chat_id, None)

    def clear(self) -> None:
        self._histories.clear()
        self._locks.clear()


def conservative_token_estimate(serialized_text: str) -> int:
    return max(1, math.ceil(len(serialized_text) / 3))
