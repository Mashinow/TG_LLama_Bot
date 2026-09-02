import pytest

from tg_llama_bot.history import HistoryStore, InputTooLongError
from tg_llama_bot.llama_client import LlamaConnectionError
from tg_llama_bot.models import ChatMessage


async def character_counter(content: str) -> int:
    return len(content)


@pytest.mark.asyncio
async def test_histories_are_isolated_by_chat() -> None:
    store = HistoryStore()
    store.commit(1, "one", "answer one")
    store.commit(2, "two", "answer two")
    first = await store.prepare(1, "next", character_counter, 100)
    assert first == [
        ChatMessage("user", "one"),
        ChatMessage("assistant", "answer one"),
        ChatMessage("user", "next"),
    ]


@pytest.mark.asyncio
async def test_prepare_trims_oldest_complete_exchange() -> None:
    store = HistoryStore()
    store.commit(1, "12345", "67890")
    messages = await store.prepare(1, "abc", character_counter, 9)
    assert messages == [ChatMessage("user", "abc")]


@pytest.mark.asyncio
async def test_oversized_new_message_is_rejected() -> None:
    store = HistoryStore()
    with pytest.raises(InputTooLongError):
        await store.prepare(1, "12345", character_counter, 10)


@pytest.mark.asyncio
async def test_reset_affects_only_one_chat() -> None:
    store = HistoryStore()
    store.commit(1, "one", "a")
    store.commit(2, "two", "b")
    await store.reset(1)
    assert await store.prepare(1, "new", character_counter, 100) == [
        ChatMessage("user", "new")
    ]
    assert len(await store.prepare(2, "new", character_counter, 100)) == 3


def test_each_chat_has_a_stable_independent_lock() -> None:
    store = HistoryStore()
    assert store.lock_for(1) is store.lock_for(1)
    assert store.lock_for(1) is not store.lock_for(2)


@pytest.mark.asyncio
async def test_tokenizer_failure_uses_conservative_estimate_and_warns_once() -> None:
    warnings: list[str] = []
    store = HistoryStore(warning_callback=warnings.append)

    async def unavailable_counter(content: str) -> int:
        raise LlamaConnectionError("tokenizer unavailable")

    messages = await store.prepare(1, "abc", unavailable_counter, 3)
    assert messages == [ChatMessage("user", "abc")]
    assert warnings == ["Tokenization unavailable; using conservative estimate."]


@pytest.mark.asyncio
async def test_clear_removes_all_conversations() -> None:
    store = HistoryStore()
    store.commit(1, "one", "answer")
    store.commit(2, "two", "answer")
    store.clear()
    assert await store.prepare(1, "new", character_counter, 100) == [
        ChatMessage("user", "new")
    ]
    assert await store.prepare(2, "new", character_counter, 100) == [
        ChatMessage("user", "new")
    ]
