# Telegram Bot for llama.cpp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python 3.12 desktop application that configures, starts, and stops a resilient aiogram Telegram bot backed by the locally running llama.cpp server.

**Architecture:** Keep tkinter in the main thread and run aiogram plus all HTTP work in a dedicated asyncio worker thread. Discover the first model and its capabilities from llama-server on every start, keep bounded per-chat histories in memory, and move events from the worker to the GUI through a thread-safe queue.

**Tech Stack:** Python 3.12, tkinter, aiogram 3.x, httpx, PyYAML, pytest, pytest-asyncio, Ruff

**Spec:** `docs/superpowers/specs/2026-09-03-telegram-llama-bot-design.md`

## Global Constraints

- Use the existing virtual environment with Python 3.12 and declare `>=3.12,<3.13`.
- Use a native tkinter GUI; all widget access remains on the main thread.
- Run aiogram and llama-server I/O on one dedicated asyncio worker thread.
- The only editable settings are `telegram_token`, `llama_base_url`, and optional `allowed_user_ids`.
- Default `llama_base_url` to `http://127.0.0.1:8080`.
- Select the first model returned by `/v1/models`; obtain context and model modes from `/props`.
- Keep history per Telegram chat in memory, support `/reset`, and clear history on Stop.
- Retry only connection failures, timeouts, HTTP 408, 429, and 5xx; allow three total attempts with exponential delay and jitter.
- Never place the Telegram token in tracked files, log entries, exception text, test output, or Bot API URLs shown to the user.
- Use the supplied token only for the requested live validation, then recommend rotating it through BotFather.

## Planned file structure

```text
.
├── .gitignore
├── config.example.yaml
├── pyproject.toml
├── README.md
├── scripts/
│   ├── __init__.py
│   └── live_smoke.py
├── src/tg_llama_bot/
│   ├── __init__.py
│   ├── app.py
│   ├── bot.py
│   ├── config.py
│   ├── gui.py
│   ├── history.py
│   ├── llama_client.py
│   ├── models.py
│   └── runtime.py
└── tests/
    ├── test_bot.py
    ├── test_config.py
    ├── test_gui.py
    ├── test_history.py
    ├── test_llama_client.py
    ├── test_runtime.py
    └── test_smoke_script.py
```

### Task 1: Project foundation and YAML configuration

**Files:**
- Create: `.gitignore`
- Create: `config.example.yaml`
- Create: `pyproject.toml`
- Create: `src/tg_llama_bot/__init__.py`
- Create: `src/tg_llama_bot/models.py`
- Create: `src/tg_llama_bot/config.py`
- Create: `tests/test_config.py`
- Include: the approved spec and this plan in the initial repository commit
- Keep untracked: `codex.md`, because it currently contains the exposed test credential

**Interfaces:**
- Produces: `AppConfig(telegram_token: str, llama_base_url: str, allowed_user_ids: tuple[int, ...])`
- Produces: `ChatMessage(role: Literal["user", "assistant"], content: str)`
- Produces: `ServerCapabilities(model_id: str, n_ctx: int, max_output_tokens: int, server_max_output_tokens: int | None, reasoning_format: str, modalities: tuple[str, ...])`
- Produces: `RuntimeState` enum and `RuntimeEvent(kind, payload)`
- Produces: `EventSink = Callable[[RuntimeEvent], None]`
- Produces: `load_config(path: Path) -> AppConfig`
- Produces: `save_config(path: Path, config: AppConfig) -> None`
- Produces: `parse_allowed_user_ids(raw: str) -> tuple[int, ...]`
- Produces: `format_allowed_user_ids(ids: tuple[int, ...]) -> str`

- [ ] **Step 1: Initialize source control and project metadata**

Run:

```powershell
.venv/Scripts/python.exe --version
```

Create `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "tg-llama-bot"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "aiogram>=3.0,<4.0",
  "httpx>=0.27,<1.0",
  "PyYAML>=6.0,<7.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-asyncio>=0.23",
  "ruff>=0.9",
]

[project.scripts]
tg-llama-bot = "tg_llama_bot.app:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py312"
line-length = 100
```

Install the editable package:

```powershell
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

Expected: installation succeeds under Python 3.12.

- [ ] **Step 2: Write failing configuration tests**

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from tg_llama_bot.config import (
    ConfigError,
    format_allowed_user_ids,
    load_config,
    parse_allowed_user_ids,
    save_config,
)
from tg_llama_bot.models import AppConfig


def test_missing_config_uses_safe_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path / "config.yaml")
    assert config == AppConfig(
        telegram_token="",
        llama_base_url="http://127.0.0.1:8080",
        allowed_user_ids=(),
    )


def test_config_round_trip_normalizes_url_and_ids(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    original = AppConfig("secret", "http://127.0.0.1:8080/", (22, 11, 22))
    save_config(path, original)
    assert load_config(path) == AppConfig(
        telegram_token="secret",
        llama_base_url="http://127.0.0.1:8080",
        allowed_user_ids=(11, 22),
    )


def test_allowed_user_ids_parse_and_format() -> None:
    assert parse_allowed_user_ids("42, 7,42") == (7, 42)
    assert format_allowed_user_ids((7, 42)) == "7, 42"


@pytest.mark.parametrize("raw", ["zero", "0", "-3", "12.5"])
def test_allowed_user_ids_reject_invalid_values(raw: str) -> None:
    with pytest.raises(ConfigError):
        parse_allowed_user_ids(raw)


def test_config_rejects_non_http_url(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("llama_base_url: ftp://localhost\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="http"):
        load_config(path)


def test_save_finishes_with_atomic_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    replacements: list[tuple[Path, Path]] = []
    real_replace = __import__("os").replace

    def recording_replace(source, destination) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr("tg_llama_bot.config.os.replace", recording_replace)
    save_config(path, AppConfig("secret"))
    assert len(replacements) == 1
    assert replacements[0][1] == path
    assert replacements[0][0].parent == path.parent
```

- [ ] **Step 3: Run the tests and verify the expected failure**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_config.py -q
```

Expected: collection fails because `tg_llama_bot.config` and its interfaces do not exist.

- [ ] **Step 4: Implement typed models and configuration persistence**

Create immutable dataclasses in `models.py` with these exact fields:

```python
class RuntimeState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class AppConfig:
    telegram_token: str = ""
    llama_base_url: str = "http://127.0.0.1:8080"
    allowed_user_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class ServerCapabilities:
    model_id: str
    n_ctx: int
    max_output_tokens: int
    server_max_output_tokens: int | None
    reasoning_format: str
    modalities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    kind: Literal["state", "log", "capabilities", "error"]
    payload: object


EventSink: TypeAlias = Callable[[RuntimeEvent], None]
```

In `config.py`, use `yaml.safe_load` and `yaml.safe_dump(sort_keys=False)`. Normalize URLs with `rstrip("/")`, accept only `http://` or `https://`, deduplicate/sort positive user IDs, and save atomically by writing a UTF-8 temporary file in the destination directory before `os.replace(temp_path, path)`. A missing file returns defaults; malformed YAML and invalid field types raise `ConfigError` with no configuration dump.

Create `.gitignore`:

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
config.yaml
codex.md
*.log
```

Create `config.example.yaml`:

```yaml
telegram_token: ""
llama_base_url: "http://127.0.0.1:8080"
allowed_user_ids: []
```

- [ ] **Step 5: Run configuration tests and lint**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_config.py -q
.venv/Scripts/python.exe -m ruff check src/tg_llama_bot/models.py src/tg_llama_bot/config.py tests/test_config.py
```

Expected: all configuration tests pass and Ruff reports no errors.

- [ ] **Step 6: Commit the foundation**

Run:

```powershell
git add .
git commit -m "chore: scaffold telegram llama bot"
```

Expected: the initial commit includes project metadata, approved documentation, configuration code, and passing tests; `config.yaml` and secret-bearing `codex.md` remain ignored.

### Task 2: llama-server discovery, completion, tokenization, and retries

**Files:**
- Create: `src/tg_llama_bot/llama_client.py`
- Create: `tests/test_llama_client.py`

**Interfaces:**
- Consumes: `ChatMessage` and `ServerCapabilities` from Task 1
- Produces: `RetryPolicy(attempts: int = 3, base_delay: float = 0.25, max_delay: float = 2.0)`
- Produces: `LlamaClient(base_url, http_client=None, retry_policy=RetryPolicy(), sleep=asyncio.sleep, jitter=random.uniform)`
- Produces: `async discover() -> ServerCapabilities`
- Produces: `async count_tokens(content: str) -> int`
- Produces: `async complete(model_id: str, messages: Sequence[ChatMessage], max_tokens: int) -> str`
- Produces: `async aclose() -> None`
- Produces: asynchronous context-manager methods returning and closing the same `LlamaClient`
- Produces exceptions: `LlamaConnectionError`, `LlamaProtocolError`, and `LlamaInputError`

- [ ] **Step 1: Write failing discovery and retry tests**

Create `tests/test_llama_client.py` with an `httpx.MockTransport` handler and these core cases:

```python
import httpx
import pytest

from tg_llama_bot.llama_client import (
    LlamaClient,
    LlamaConnectionError,
    LlamaInputError,
    LlamaProtocolError,
    RetryPolicy,
)


@pytest.mark.asyncio
async def test_discover_selects_first_model_and_props() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "first"}, {"id": "second"}]})
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "default_generation_settings": {
                        "params": {"max_tokens": -1, "n_predict": -1},
                        "n_ctx": 65536,
                    },
                    "reasoning_format": "none",
                    "modalities": {"vision": False, "audio": False, "video": False},
                },
            )
        raise AssertionError(request.url.path)

    http = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    client = LlamaClient("http://test", http_client=http)
    capabilities = await client.discover()
    assert capabilities.model_id == "first"
    assert capabilities.n_ctx == 65536
    assert capabilities.max_output_tokens == 2048
    assert capabilities.server_max_output_tokens is None
    assert capabilities.modalities == ("text",)
    await client.aclose()


@pytest.mark.asyncio
async def test_tokenize_returns_number_of_server_tokens() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tokenize"
        return httpx.Response(200, json={"tokens": [10, 20, 30]})

    http = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    client = LlamaClient("http://test", http_client=http)
    assert await client.count_tokens("hello") == 3
    await client.aclose()


@pytest.mark.asyncio
async def test_transient_failure_retries_only_three_total_attempts() -> None:
    attempts = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
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
```

Append these explicit edge-case tests, using the same `httpx.MockTransport` construction:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "models_payload",
    [
        {"data": [{"id": "first"}]},
        {"models": [{"model": "first"}]},
    ],
)
async def test_discover_accepts_both_model_list_shapes(models_payload) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=models_payload)
        return httpx.Response(
            200,
            json={
                "default_generation_settings": {
                    "n_ctx": 4096,
                    "params": {"max_tokens": 512},
                }
            },
        )

    http = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    capabilities = await LlamaClient("http://test", http_client=http).discover()
    assert capabilities.model_id == "first"
    assert capabilities.max_output_tokens == 512
    assert capabilities.server_max_output_tokens == 512
    await http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("models_payload", "props_payload"),
    [
        ({"data": []}, {"default_generation_settings": {"n_ctx": 4096}}),
        ({"data": [{"id": "first"}]}, {"default_generation_settings": {"n_ctx": "bad"}}),
    ],
)
async def test_discover_rejects_missing_model_or_invalid_context(
    models_payload,
    props_payload,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
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
    with pytest.raises(LlamaProtocolError):
        await LlamaClient("http://test", http_client=http).discover()
    await http.aclose()


@pytest.mark.asyncio
async def test_http_400_is_not_retried() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"error": "invalid"})

    http = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LlamaInputError):
        await LlamaClient("http://test", http_client=http).count_tokens("x")
    assert attempts == 1
    await http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(("content", "raises"), [("answer", False), ("", True)])
async def test_completion_requires_non_empty_content(content: str, raises: bool) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    http = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    client = LlamaClient("http://test", http_client=http)
    if raises:
        with pytest.raises(LlamaProtocolError):
            await client.complete("model", [], 32)
    else:
        assert await client.complete("model", [], 32) == "answer"
    await client.aclose()
```

- [ ] **Step 2: Run the client tests and verify failure**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_llama_client.py -q
```

Expected: collection fails because `tg_llama_bot.llama_client` does not exist.

- [ ] **Step 3: Implement the bounded HTTP client**

Implement one shared `httpx.AsyncClient` with explicit `Timeout(connect=5, read=120, write=10, pool=5)`. If a client is injected, retain it and close it from `aclose()` because ownership is transferred to `LlamaClient`.

Use these exact request bodies:

```python
token_body = {"content": content, "add_special": False}
completion_body = {
    "model": model_id,
    "messages": [
        {"role": message.role, "content": message.content}
        for message in messages
    ],
    "max_tokens": max_tokens,
    "stream": False,
}
```

Implement a private `_request(method, path, json=None) -> httpx.Response` loop. Retry `httpx.TransportError`, `httpx.TimeoutException`, and statuses `408`, `429`, or `>=500`; compute delay as `min(max_delay, base_delay * 2 ** (attempt - 1)) + jitter(0, 0.1)`. Raise a sanitized domain exception after the third total attempt. For other 4xx responses, raise `LlamaInputError` immediately. Validate JSON shapes explicitly and raise `LlamaProtocolError` without echoing raw response bodies.

Discovery rules:

```python
model_id = first_entry["id"] if "data" is present else first_entry["model"]
n_ctx = props["default_generation_settings"]["n_ctx"]
reported = params.get("max_tokens", params.get("n_predict"))
server_limit = reported if isinstance(reported, int) and reported > 0 else None
effective_limit = server_limit or min(2048, max(1, n_ctx // 4))
```

Always include `"text"` in modalities and append each truthy server modality. Read `reasoning_format` from the top level, falling back to `default_generation_settings.params.reasoning_format`, then `"none"`.

- [ ] **Step 4: Run focused and combined tests**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_llama_client.py tests/test_config.py -q
.venv/Scripts/python.exe -m ruff check src tests
```

Expected: all tests pass; no lint errors.

- [ ] **Step 5: Commit the llama client**

Run:

```powershell
git add src/tg_llama_bot/llama_client.py tests/test_llama_client.py
git commit -m "feat: add resilient llama server client"
```

### Task 3: Per-chat history, locking, and context trimming

**Files:**
- Create: `src/tg_llama_bot/history.py`
- Create: `tests/test_history.py`

**Interfaces:**
- Consumes: `ChatMessage`
- Produces: `InputTooLongError`
- Produces: `HistoryStore(warning_callback: Callable[[str], None] | None = None)`
- Produces: `HistoryStore.lock_for(chat_id: int) -> asyncio.Lock`
- Produces: `async HistoryStore.prepare(chat_id: int, user_text: str, token_counter: Callable[[str], Awaitable[int]], prompt_budget: int) -> list[ChatMessage]`
- Produces: `HistoryStore.commit(chat_id: int, user_text: str, assistant_text: str) -> None`
- Produces: `async HistoryStore.reset(chat_id: int) -> None`
- Produces: `HistoryStore.clear() -> None`

- [ ] **Step 1: Write failing history tests**

Create `tests/test_history.py`:

```python
import pytest

from tg_llama_bot.history import HistoryStore, InputTooLongError
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
    messages = await store.prepare(1, "abc", character_counter, 7)
    assert messages == [ChatMessage("user", "abc")]


@pytest.mark.asyncio
async def test_oversized_new_message_is_rejected() -> None:
    store = HistoryStore()
    with pytest.raises(InputTooLongError):
        await store.prepare(1, "12345", character_counter, 4)


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
```

Append this fallback-token test:

```python
@pytest.mark.asyncio
async def test_tokenizer_failure_uses_conservative_estimate_and_warns_once() -> None:
    warnings: list[str] = []
    store = HistoryStore(warning_callback=warnings.append)

    async def unavailable_counter(content: str) -> int:
        raise LlamaConnectionError("tokenizer unavailable")

    messages = await store.prepare(1, "abc", unavailable_counter, 3)
    assert messages == [ChatMessage("user", "abc")]
    assert warnings == ["Tokenization unavailable; using conservative estimate."]
```

- [ ] **Step 2: Run the history tests and verify failure**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_history.py -q
```

Expected: collection fails because `tg_llama_bot.history` does not exist.

- [ ] **Step 3: Implement non-mutating preparation and explicit commit**

Store complete exchanges as a flat list of alternating `ChatMessage("user", ...)` and `ChatMessage("assistant", ...)`. `prepare` must append the candidate user message to a copy, count messages serialized as `"{role}\n{content}\n"`, and remove the first two messages repeatedly until the total fits. It must never mutate stored history. `commit` appends both user and assistant messages only after a successful completion.

Use this fallback only when the server token counter raises `LlamaConnectionError` or `LlamaProtocolError`:

```python
def conservative_token_estimate(serialized_text: str) -> int:
    return max(1, math.ceil(len(serialized_text) / 3))
```

Do not hide `InputTooLongError` or unrelated programming errors behind the fallback.

- [ ] **Step 4: Run history and regression tests**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_history.py tests/test_llama_client.py -q
.venv/Scripts/python.exe -m ruff check src tests
```

Expected: all tests pass and lint is clean.

- [ ] **Step 5: Commit conversation history**

Run:

```powershell
git add src/tg_llama_bot/history.py tests/test_history.py
git commit -m "feat: add bounded per-chat history"
```

### Task 4: Telegram chat service and aiogram handlers

**Files:**
- Create: `src/tg_llama_bot/bot.py`
- Create: `tests/test_bot.py`

**Interfaces:**
- Consumes: `AppConfig`, `ServerCapabilities`, `LlamaClient`, and `HistoryStore`
- Produces: `is_allowed(user_id: int, allowed_user_ids: tuple[int, ...]) -> bool`
- Produces: `split_telegram_text(text: str, limit: int = 4096) -> list[str]`
- Produces: `ChatService.answer(chat_id: int, text: str) -> str`
- Produces: `BotHandlers(config: AppConfig, service: ChatService, history: HistoryStore)` methods `start`, `help`, `reset`, `text`, and `unsupported`
- Produces: `build_dispatcher(dependencies: BotDependencies) -> aiogram.Dispatcher`
- Produces: `async run_bot(config: AppConfig, stop_event: asyncio.Event, emit: EventSink) -> None`

- [ ] **Step 1: Write failing service and formatting tests**

Create `tests/test_bot.py` with:

```python
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tg_llama_bot.bot import (
    ACCESS_DENIED_TEXT,
    HELP_TEXT,
    INPUT_TOO_LONG_TEXT,
    RESET_TEXT,
    START_TEXT,
    UNSUPPORTED_TEXT,
    BotHandlers,
    ChatService,
    is_allowed,
    split_telegram_text,
)
from tg_llama_bot.history import HistoryStore, InputTooLongError
from tg_llama_bot.models import AppConfig, ChatMessage, ServerCapabilities


CAPABILITIES = ServerCapabilities(
    model_id="model.gguf",
    n_ctx=100,
    max_output_tokens=20,
    server_max_output_tokens=None,
    reasoning_format="none",
    modalities=("text",),
)


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
    llama = SimpleNamespace(
        count_tokens=AsyncMock(side_effect=lambda value: len(value)),
        complete=AsyncMock(return_value="model answer"),
    )
    history = HistoryStore()
    service = ChatService(llama, CAPABILITIES, history)
    result = await service.answer(10, "hello")
    assert result == "model answer"
    next_messages = await history.prepare(10, "next", llama.count_tokens, 80)
    assert [message.content for message in next_messages] == [
        "hello",
        "model answer",
        "next",
    ]


@pytest.mark.asyncio
async def test_chat_service_does_not_commit_failed_exchange() -> None:
    llama = SimpleNamespace(
        count_tokens=AsyncMock(side_effect=lambda value: len(value)),
        complete=AsyncMock(side_effect=RuntimeError("failure")),
    )
    history = HistoryStore()
    service = ChatService(llama, CAPABILITIES, history)
    with pytest.raises(RuntimeError):
        await service.answer(10, "hello")
    assert await history.prepare(10, "next", llama.count_tokens, 80) == [
        ChatMessage("user", "next")
    ]
```

Append these handler tests:

```python
async def character_counter(content: str) -> int:
    return len(content)


class FakeMessage:
    def __init__(self, user_id: int, chat_id: int, text: str = "hello") -> None:
        self.from_user = SimpleNamespace(id=user_id)
        self.chat = SimpleNamespace(id=chat_id)
        self.text = text
        self.answer = AsyncMock()


@pytest.mark.asyncio
async def test_unauthorized_text_never_reaches_model() -> None:
    service = SimpleNamespace(answer=AsyncMock())
    handlers = BotHandlers(
        AppConfig("token", allowed_user_ids=(7,)),
        service,
        HistoryStore(),
    )
    message = FakeMessage(user_id=8, chat_id=10)
    await handlers.text(message)
    service.answer.assert_not_awaited()
    message.answer.assert_awaited_once_with(ACCESS_DENIED_TEXT)


@pytest.mark.asyncio
async def test_reset_clears_only_current_chat() -> None:
    history = HistoryStore()
    history.commit(10, "old", "answer")
    history.commit(11, "keep", "answer")
    handlers = BotHandlers(
        AppConfig("token"),
        SimpleNamespace(answer=AsyncMock()),
        history,
    )
    message = FakeMessage(user_id=7, chat_id=10, text="/reset")
    await handlers.reset(message)
    message.answer.assert_awaited_once_with(RESET_TEXT)
    assert await history.prepare(10, "new", character_counter, 100) == [
        ChatMessage("user", "new")
    ]
    assert len(await history.prepare(11, "new", character_counter, 100)) == 3


@pytest.mark.asyncio
async def test_long_answer_is_sent_as_multiple_messages() -> None:
    service = SimpleNamespace(answer=AsyncMock(return_value="x" * 5000))
    handlers = BotHandlers(AppConfig("token"), service, HistoryStore())
    message = FakeMessage(user_id=7, chat_id=10)
    await handlers.text(message)
    assert message.answer.await_count == 2


@pytest.mark.asyncio
async def test_oversized_input_gets_specific_message() -> None:
    service = SimpleNamespace(
        answer=AsyncMock(side_effect=InputTooLongError("too long"))
    )
    handlers = BotHandlers(AppConfig("token"), service, HistoryStore())
    message = FakeMessage(user_id=7, chat_id=10)
    await handlers.text(message)
    message.answer.assert_awaited_once_with(INPUT_TOO_LONG_TEXT)


@pytest.mark.asyncio
async def test_start_and_unsupported_handlers_return_fixed_helpful_text() -> None:
    handlers = BotHandlers(
        AppConfig("token"),
        SimpleNamespace(answer=AsyncMock()),
        HistoryStore(),
    )
    start_message = FakeMessage(user_id=7, chat_id=10, text="/start")
    help_message = FakeMessage(user_id=7, chat_id=10, text="/help")
    media_message = FakeMessage(user_id=7, chat_id=10, text="")
    await handlers.start(start_message)
    await handlers.help(help_message)
    await handlers.unsupported(media_message)
    start_message.answer.assert_awaited_once_with(START_TEXT)
    help_message.answer.assert_awaited_once_with(HELP_TEXT)
    media_message.answer.assert_awaited_once_with(UNSUPPORTED_TEXT)
```

- [ ] **Step 2: Run the bot tests and verify failure**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_bot.py -q
```

Expected: collection fails because `tg_llama_bot.bot` does not exist.

- [ ] **Step 3: Implement ChatService and pure helpers**

`ChatService.answer` acquires `history.lock_for(chat_id)`, calculates `prompt_budget = capabilities.n_ctx - capabilities.max_output_tokens - 128`, calls `history.prepare`, awaits `llama.complete`, validates non-empty output, then commits the exchange. Different chat IDs therefore remain concurrent.

Implement splitting with this deterministic order for each remaining chunk:

1. Return immediately if it fits.
2. Search backward for `"\n\n"` within the limit.
3. Search backward for `"\n"`.
4. Split exactly at `limit`.

Retain the chosen delimiter in the preceding part so concatenating the parts reconstructs the original response exactly.

- [ ] **Step 4: Implement aiogram handlers and polling runner**

Create `BotDependencies` as a frozen dataclass containing config, capabilities, llama client, and history store. `BotHandlers` checks `message.from_user.id` against the allowlist before every command or content handler.

Register handlers in this precedence order:

```python
router.message.register(handlers.start, Command("start"))
router.message.register(handlers.help, Command("help"))
router.message.register(handlers.reset, Command("reset"))
router.message.register(handlers.text, F.text)
router.message.register(handlers.unsupported)
```

`run_bot` must:

```python
async with LlamaClient(config.llama_base_url) as llama:
    capabilities = await llama.discover()
    bot = Bot(token=config.telegram_token)
    history = HistoryStore(warning_callback=emit_log)
    try:
        await bot.get_me()
        dispatcher = build_dispatcher(
            BotDependencies(config, capabilities, llama, history)
        )
        emit(RuntimeEvent("capabilities", capabilities))
        emit(RuntimeEvent("state", RuntimeState.RUNNING))
        polling = asyncio.create_task(
            dispatcher.start_polling(
                bot,
                handle_signals=False,
                backoff_config=BackoffConfig(
                    min_delay=1.0,
                    max_delay=30.0,
                    factor=1.5,
                    jitter=0.1,
                ),
            )
        )
        await stop_event.wait()
        await dispatcher.stop_polling()
        await polling
    finally:
        history.clear()
        await bot.session.close()
```

Initialize `history` before the `try` so cleanup remains valid if `get_me` fails. If polling finishes before `stop_event`, propagate its exception instead of waiting forever. Map final generation errors to one generic Russian user message; log only the domain exception type and sanitized summary.

- [ ] **Step 5: Run bot, history, and client tests**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_bot.py tests/test_history.py tests/test_llama_client.py -q
.venv/Scripts/python.exe -m ruff check src tests
```

Expected: all tests pass and lint is clean.

- [ ] **Step 6: Commit Telegram behavior**

Run:

```powershell
git add src/tg_llama_bot/bot.py tests/test_bot.py
git commit -m "feat: add telegram chat handlers"
```

### Task 5: Threaded runtime controller and safe shutdown

**Files:**
- Create: `src/tg_llama_bot/runtime.py`
- Create: `tests/test_runtime.py`

**Interfaces:**
- Consumes: `AppConfig`, `RuntimeEvent`, `RuntimeState`, and `run_bot`
- Produces: `RuntimeController(event_queue: queue.Queue[RuntimeEvent], runner=run_bot)`
- Produces: `start(config: AppConfig) -> bool`
- Produces: `request_stop() -> bool`
- Produces: `wait_stopped(timeout: float) -> bool`
- Produces: read-only `state: RuntimeState`
- Produces: `sanitize_exception(exc: BaseException, token: str) -> str`

- [ ] **Step 1: Write failing lifecycle and redaction tests**

Create `tests/test_runtime.py`:

```python
import queue
import threading

from tg_llama_bot.models import AppConfig, RuntimeEvent, RuntimeState
from tg_llama_bot.runtime import RuntimeController, sanitize_exception


CONFIG = AppConfig("123:SECRET", "http://127.0.0.1:8080", ())


def test_start_and_stop_run_async_runner_in_worker_thread() -> None:
    entered = threading.Event()

    async def runner(config, stop_event, emit) -> None:
        entered.set()
        emit(RuntimeEvent("state", RuntimeState.RUNNING))
        await stop_event.wait()

    events: queue.Queue[RuntimeEvent] = queue.Queue()
    controller = RuntimeController(events, runner=runner)
    assert controller.start(CONFIG)
    assert entered.wait(1.0)
    assert not controller.start(CONFIG)
    assert controller.request_stop()
    assert controller.wait_stopped(2.0)
    assert controller.state is RuntimeState.STOPPED


def test_duplicate_stop_is_rejected() -> None:
    controller = RuntimeController(queue.Queue(), runner=lambda *args: None)
    assert not controller.request_stop()


def test_runner_failure_emits_sanitized_error_and_stops() -> None:
    async def runner(config, stop_event, emit) -> None:
        raise RuntimeError("request to /bot123:SECRET/getMe failed")

    events: queue.Queue[RuntimeEvent] = queue.Queue()
    controller = RuntimeController(events, runner=runner)
    assert controller.start(CONFIG)
    assert controller.wait_stopped(2.0)
    payloads = [events.get_nowait().payload for _ in range(events.qsize())]
    assert all("123:SECRET" not in str(payload) for payload in payloads)
    assert controller.state is RuntimeState.STOPPED


def test_sanitize_exception_removes_token_and_bot_url_segment() -> None:
    text = sanitize_exception(
        RuntimeError("https://api.telegram.org/bot123:SECRET/getMe"),
        "123:SECRET",
    )
    assert "123:SECRET" not in text
    assert "<redacted>" in text
```

- [ ] **Step 2: Run runtime tests and verify failure**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_runtime.py -q
```

Expected: collection fails because `tg_llama_bot.runtime` does not exist.

- [ ] **Step 3: Implement the state machine and worker-thread bridge**

Guard `_state`, `_thread`, `_loop`, and `_stop_event` with one `threading.Lock`. Allowed transitions are:

```text
STOPPED -> STARTING -> RUNNING -> STOPPING -> STOPPED
                    \-> ERROR -> STOPPED
STARTING -> STOPPING -> STOPPED
```

`start` changes state to STARTING before creating a non-daemon worker thread. The thread runs `asyncio.run(_run(config))`; `_run` creates its `asyncio.Event`, invokes the injected runner, catches exceptions, emits a sanitized error, and always publishes STOPPED in `finally`.

Pass the runner a controller-owned `_emit(event)` callback. When it receives a `RuntimeEvent("state", RuntimeState.RUNNING)`, update the guarded controller state before putting the event on the GUI queue. Reject a late RUNNING event if Stop was already requested, preserving STOPPING. All other events go directly to the queue.

`request_stop` changes STARTING or RUNNING to STOPPING and uses:

```python
loop.call_soon_threadsafe(stop_event.set)
```

If the loop or event is not yet installed, store `_stop_requested = True`; `_run` sets the event immediately after creating it. `request_stop` never joins the worker and therefore never blocks tkinter. `wait_stopped` joins only when explicitly called by tests or the window-close polling path.

Sanitize error text by replacing the exact token and any `/bot<token>/` fragment with `<redacted>`. Emit exception class plus sanitized text, never `repr(config)`.

- [ ] **Step 4: Run runtime regression and race tests**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_runtime.py -q
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe -m ruff check src tests
```

Expected: lifecycle tests complete without hanging; the full suite passes.

- [ ] **Step 5: Commit runtime control**

Run:

```powershell
git add src/tg_llama_bot/runtime.py tests/test_runtime.py
git commit -m "feat: add threaded bot runtime"
```

### Task 6: Native tkinter GUI and application entry point

**Files:**
- Create: `src/tg_llama_bot/gui.py`
- Create: `src/tg_llama_bot/app.py`
- Create: `tests/test_gui.py`

**Interfaces:**
- Consumes: configuration functions, `RuntimeController`, `RuntimeEvent`, `RuntimeState`, and `ServerCapabilities`
- Produces: `ControlState(start_enabled: bool, stop_enabled: bool, fields_enabled: bool)`
- Produces: `controls_for_state(state: RuntimeState) -> ControlState`
- Produces: `BotWindow(root: tkinter.Tk, config_path: Path, controller: RuntimeController, event_queue: queue.Queue[RuntimeEvent])`
- Produces: `main(config_path: Path = Path("config.yaml")) -> int`

- [ ] **Step 1: Write failing GUI state tests without opening a window**

Create `tests/test_gui.py`:

```python
import pytest

from pathlib import Path
from unittest.mock import Mock

import tg_llama_bot.app as app_module
from tg_llama_bot.gui import controls_for_state
from tg_llama_bot.models import RuntimeState


@pytest.mark.parametrize(
    ("state", "start", "stop", "fields"),
    [
        (RuntimeState.STOPPED, True, False, True),
        (RuntimeState.STARTING, False, True, False),
        (RuntimeState.RUNNING, False, True, False),
        (RuntimeState.STOPPING, False, False, False),
        (RuntimeState.ERROR, True, False, True),
    ],
)
def test_controls_follow_runtime_state(state, start, stop, fields) -> None:
    controls = controls_for_state(state)
    assert controls.start_enabled is start
    assert controls.stop_enabled is stop
    assert controls.fields_enabled is fields


def test_main_constructs_window_and_enters_mainloop(monkeypatch) -> None:
    root = Mock()
    controller = object()
    captured = {}

    class FakeWindow:
        def __init__(self, root_arg, config_path, controller_arg, event_queue) -> None:
            captured["args"] = (root_arg, config_path, controller_arg, event_queue)

    monkeypatch.setattr(app_module.tk, "Tk", lambda: root)
    monkeypatch.setattr(
        app_module,
        "RuntimeController",
        lambda event_queue: controller,
    )
    monkeypatch.setattr(app_module, "BotWindow", FakeWindow)

    assert app_module.main() == 0
    root.mainloop.assert_called_once_with()
    assert captured["args"][0:3] == (root, Path("config.yaml"), controller)
```

- [ ] **Step 2: Run GUI tests and verify failure**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_gui.py -q
```

Expected: collection fails because `tg_llama_bot.gui` does not exist.

- [ ] **Step 3: Implement the window layout and event draining**

Build one compact window with:

- masked Telegram-token entry;
- server URL entry;
- optional allowlist entry;
- Save, Start, and Stop buttons;
- status label;
- read-only labels for model, context, output limit, reasoning format, and modalities;
- read-only scrolling log text area.

Do not add editable generation controls or a model selector. Use `ttk` widgets and `grid` with column weight on the value column.

`BotWindow` loads `config.yaml` on construction. Save parses IDs, validates through `save_config`, and shows validation errors with `messagebox.showerror`. Start calls Save first, then `controller.start(config)`. Stop only calls `controller.request_stop()`.

Schedule event draining every 100 ms:

```python
def _drain_events(self) -> None:
    while True:
        try:
            event = self._event_queue.get_nowait()
        except queue.Empty:
            break
        self._apply_event(event)
    self.root.after(100, self._drain_events)
```

Apply state events through `controls_for_state`; render capability events into read-only labels; append log and error events without dumping configuration objects.

- [ ] **Step 4: Implement non-blocking window close and CLI entry**

The WM_DELETE_WINDOW handler requests Stop, then uses `root.after(100, _poll_closed)` until `controller.wait_stopped(0)` succeeds. Destroy immediately when already stopped. After 5 seconds, ask whether to continue waiting; never terminate the worker thread forcibly.

`app.main` creates `queue.Queue[RuntimeEvent]`, `RuntimeController`, `tkinter.Tk`, and `BotWindow`, then enters `mainloop` and returns 0. Add:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run GUI tests and the full automated suite**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_gui.py -q
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m compileall -q src
```

Expected: all tests pass, Ruff is clean, and compilation succeeds.

- [ ] **Step 6: Perform a manual GUI-only smoke check**

Run:

```powershell
.venv/Scripts/python.exe -m tg_llama_bot.app
```

Expected: the window opens; the URL defaults to `http://127.0.0.1:8080`; token characters are masked; Save writes ignored `config.yaml`; Start/Stop controls change state without freezing the window. Close the window after the check.

- [ ] **Step 7: Commit the desktop interface**

Run:

```powershell
git add src/tg_llama_bot/gui.py src/tg_llama_bot/app.py tests/test_gui.py
git commit -m "feat: add desktop bot controller"
```

### Task 7: Live smoke command, documentation, and end-to-end verification

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/live_smoke.py`
- Create: `tests/test_smoke_script.py`
- Create: `README.md`

**Interfaces:**
- Consumes: `load_config` and `LlamaClient`
- Produces: `async smoke(config_path: Path, prompt: str) -> SmokeResult`
- Produces: `format_smoke_error(exc: BaseException, token: str) -> str`
- Produces: CLI exit code 0 on successful llama-server discovery, short completion, and Telegram `getMe`; nonzero on a sanitized failure

- [ ] **Step 1: Write failing smoke-script tests**

Create `tests/test_smoke_script.py`:

```python
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from scripts.live_smoke import format_smoke_error, smoke
from tg_llama_bot.models import AppConfig, ServerCapabilities


class FakeLlama:
    def __init__(self, capabilities: ServerCapabilities) -> None:
        self.discover = AsyncMock(return_value=capabilities)
        self.complete = AsyncMock(return_value="pong")
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.exited = True


@pytest.mark.asyncio
async def test_smoke_checks_llama_completion_and_telegram_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capabilities = ServerCapabilities(
        "model.gguf", 4096, 512, 512, "none", ("text",)
    )
    fake_llama = FakeLlama(capabilities)
    fake_bot = SimpleNamespace(
        get_me=AsyncMock(return_value=SimpleNamespace(username="test_bot")),
        session=SimpleNamespace(close=AsyncMock()),
    )
    monkeypatch.setattr(
        "scripts.live_smoke.load_config",
        lambda path: AppConfig("123:SECRET", "http://127.0.0.1:8080", ()),
    )
    monkeypatch.setattr("scripts.live_smoke.make_llama_client", lambda url: fake_llama)
    monkeypatch.setattr("scripts.live_smoke.make_bot", lambda token: fake_bot)
    result = await smoke(tmp_path / "config.yaml", "Reply with pong")
    assert result.model_id == "model.gguf"
    assert result.bot_username == "test_bot"
    assert result.completion == "pong"
    assert fake_llama.exited
    fake_bot.session.close.assert_awaited_once()


def test_smoke_error_redacts_telegram_token() -> None:
    token = "123:SECRET"
    error = format_smoke_error(
        RuntimeError("https://api.telegram.org/bot123:SECRET/getMe failed"),
        token,
    )
    assert token not in error
    assert "<redacted>" in error
```

- [ ] **Step 2: Run the smoke-script tests and verify failure**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_smoke_script.py -q
```

Expected: import fails because `scripts/live_smoke.py` does not exist.

- [ ] **Step 3: Implement the non-interactive live smoke script**

The script accepts:

```text
--config PATH   default: config.yaml
--prompt TEXT   default: Reply with exactly: pong
```

`smoke` loads config, rejects an empty token, discovers capabilities, sends one completion using `[ChatMessage("user", prompt)]`, validates non-empty output, calls Telegram `getMe`, closes both clients, and returns a frozen `SmokeResult(model_id, bot_username, completion)`. `format_smoke_error` delegates token removal to `runtime.sanitize_exception`. Print model ID, bot username, and at most the first 200 completion characters. Never print the token, config object, Bot API URL, or raw exception.

- [ ] **Step 4: Write operating documentation**

Create `README.md` with exact commands for:

```powershell
.venv/Scripts/python.exe --version
.venv/Scripts/python.exe -m pip install -e ".[dev]"
Copy-Item config.example.yaml config.yaml
.venv/Scripts/python.exe -m tg_llama_bot.app
.venv/Scripts/python.exe scripts/live_smoke.py --config config.yaml
.venv/Scripts/python.exe -m pytest -q
```

Document the three fields, default server URL, automatic first-model selection, server-derived limits, Start/Stop behavior, `/start`, `/help`, `/reset`, in-memory history, allowlist semantics, retry policy, and troubleshooting for unavailable server, invalid bot token, missing tkinter, and inputs exceeding context. State explicitly that `config.yaml` contains a secret and must remain untracked.

- [ ] **Step 5: Run all automated verification**

Run:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests scripts
.venv/Scripts/python.exe -m compileall -q src scripts
git check-ignore config.yaml
```

Expected: all tests pass; Ruff and compileall succeed; `git check-ignore` prints `config.yaml`.

- [ ] **Step 6: Run the live service smoke test**

Populate local ignored `config.yaml` with the supplied test token, then run:

```powershell
.venv/Scripts/python.exe scripts/live_smoke.py --config config.yaml
```

Expected: the script reports the detected llama model, a non-empty short completion, and the Telegram bot username without revealing the token.

- [ ] **Step 7: Verify a real Telegram conversation and shutdown**

Run:

```powershell
.venv/Scripts/python.exe -m tg_llama_bot.app
```

In Telegram, send `/start`, a short Russian prompt, `/reset`, and another short prompt from an allowed user. Expected: both prompts receive non-empty model responses, `/reset` confirms only that chat was cleared, the GUI remains responsive, and Stop returns to STOPPED with no polling or HTTP-session warnings.

- [ ] **Step 8: Hand off credential rotation**

Tell the user that the plaintext token in `codex.md` must be revoked through BotFather and that its replacement belongs only in ignored `config.yaml`. Do not revoke or replace the credential automatically, and do not add either token to a commit.

- [ ] **Step 9: Commit documentation and verified smoke tooling**

Run:

```powershell
git add README.md scripts/__init__.py scripts/live_smoke.py tests/test_smoke_script.py
git commit -m "docs: add setup and live verification"
git status --short
```

Expected: the working tree is clean except for any explicitly retained local-only files; `config.yaml` and secrets are absent from the commit.
