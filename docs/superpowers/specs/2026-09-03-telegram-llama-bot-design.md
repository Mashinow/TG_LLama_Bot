# Telegram Bot for llama.cpp — Design Specification

## Goal

Build a Python 3.12 desktop application that runs a resilient Telegram bot against an already-running OpenAI-compatible `llama-server`, with minimal YAML configuration and a native GUI for editing settings and starting or stopping the bot.

## Confirmed environment

- The project currently contains only `codex.md`; there is no existing application structure or Git repository.
- The local server is reachable at `http://127.0.0.1:8080`.
- `GET /health` returns a healthy status.
- `GET /v1/models` returns at least one model and includes its context metadata.
- `GET /props` returns the active model alias, `n_ctx`, generation defaults, reasoning format, modalities, and chat-template capabilities.
- The implementation must use the existing Python 3.12 virtual environment and `aiogram`.

## User-visible behavior

The desktop window contains only three editable settings:

1. Telegram bot token.
2. Llama server URL, defaulting to `http://127.0.0.1:8080`.
3. Optional comma-separated Telegram user IDs. An empty value allows all users; a populated value acts as an allowlist.

The window also contains Save, Start, and Stop buttons; a status indicator; a compact log view; and read-only server information showing the detected model, context size, generation limit, reasoning mode, and supported modalities.

Save validates the editable values and writes them to `config.yaml`. Start saves valid values, discovers the server capabilities, and starts Telegram polling. Stop shuts down polling and network clients without closing the GUI. Closing the window performs the same graceful shutdown before exiting.

## Configuration

`config.yaml` has the following schema:

```yaml
telegram_token: ""
llama_base_url: "http://127.0.0.1:8080"
allowed_user_ids: []
```

Rules:

- `telegram_token` is required to start but may be empty in the example file.
- `llama_base_url` is normalized by removing a trailing slash and must use `http` or `https`.
- `allowed_user_ids` is a list of positive integers. The GUI accepts comma-separated IDs and converts them to this list.
- Model name, context size, generation settings, and model modes are not stored in YAML because they are discovered from the server on every Start.
- `config.yaml` is ignored by version control. `config.example.yaml` contains no secret.

## Architecture

The application is a single process with two execution domains:

- The main thread owns `tkinter` and all widget access.
- A worker thread owns a dedicated asyncio event loop containing `aiogram`, HTTP clients, chat state, and shutdown coordination.

Communication from the worker to the GUI uses a thread-safe event queue. The GUI drains the queue with `tkinter.after()`. GUI actions invoke a narrow runtime controller API; the worker never calls widgets directly.

### Modules

- `app.py`: application entry point and dependency assembly.
- `config.py`: YAML loading, normalization, validation, and atomic saving.
- `models.py`: typed configuration, server-capability, GUI-event, and runtime-state models.
- `gui.py`: widgets, field conversion, state transitions, log presentation, and close handling.
- `runtime.py`: worker-thread lifecycle, asyncio loop ownership, Start/Stop synchronization, and GUI event publication.
- `llama_client.py`: health check, server discovery, token accounting, chat-completion requests, timeouts, and bounded retries.
- `history.py`: per-chat histories, per-chat locks, reset, and context-budget trimming.
- `bot.py`: `aiogram` dispatcher, access control, commands, text-message handling, and Telegram message splitting.

## Server discovery and generation policy

Start performs these requests before polling begins:

1. `GET /health` must report a usable server.
2. `GET /v1/models` supplies the available models. The first model ID is selected.
3. `GET /props` supplies the active context size and generation defaults.

The application records:

- selected model ID;
- `n_ctx` context capacity;
- server `max_tokens` or `n_predict` when it is a positive finite value;
- reasoning format and chat-template capabilities;
- text, vision, audio, and video modalities.

Sampling parameters are omitted from `/v1/chat/completions`, allowing the server's discovered defaults to remain authoritative. If the server reports a positive output-token limit, it is passed as `max_tokens`. If the reported value is `-1` or otherwise unbounded, the client uses a non-configurable safety ceiling of 2,048 output tokens while still respecting `n_ctx`.

The first advertised model is selected automatically. A failure to discover any model prevents Start and is displayed in the GUI.

## Telegram behavior

The bot supports:

- `/start`: confirms availability and explains the basic commands.
- `/help`: describes text chat, `/reset`, and access behavior.
- `/reset`: clears only the current Telegram chat's history.
- Plain text: sends the chat history plus the new user message to `/v1/chat/completions`.

Non-text messages receive a short text-only capability notice. Unauthorized users receive a generic access-denied response and their messages are never sent to the model.

Responses longer than Telegram's message limit are split at safe text boundaries, preferring paragraph and newline boundaries and falling back to a hard character boundary. Empty model responses are treated as upstream errors rather than sent to Telegram.

## Conversation history and concurrency

- History is keyed by Telegram chat ID and exists only in memory.
- Each history contains ordered `user` and `assistant` messages; failed requests do not append an assistant entry.
- Each chat has an asyncio lock, so messages in one chat are handled sequentially while different chats can generate concurrently when the server allows it.
- `/reset` acquires the same chat lock before clearing history.
- Histories are discarded when the bot stops or the application exits.

Before each completion, the client reserves the discovered output budget and trims the oldest complete user/assistant exchange pairs until the prompt fits the remaining `n_ctx` budget. Token counts use the llama-server tokenization endpoint when available. If tokenization is unavailable, a conservative character-based estimate is used and an informational warning is logged. The newest user message is retained; if it alone cannot fit, the user receives a message explaining that the input is too long.

## Start and Stop lifecycle

### Start

1. Reject Start if the runtime is already starting, running, or stopping.
2. Read and validate GUI values.
3. Save them atomically to YAML.
4. Disable editable controls and publish a starting state.
5. Create the worker thread and its asyncio event loop.
6. Discover llama-server capabilities.
7. Validate the Telegram token through the Bot API.
8. Start `aiogram` long polling.
9. Publish discovered server information and a running state.

Any failure during Start closes created clients, stops the worker loop, publishes a sanitized error, and returns the GUI to the stopped state.

### Stop

1. Reject duplicate Stop requests while already stopped or stopping.
2. Publish a stopping state.
3. Signal the asyncio runtime through its thread-safe shutdown mechanism.
4. Stop polling and cancel in-flight completion tasks.
5. Close Telegram and llama-server HTTP sessions.
6. Wait for the worker thread with a bounded timeout.
7. Clear in-memory histories and publish the stopped state.

Start after Stop creates a new event loop, new clients, and empty histories.

## Failure handling

Telegram polling uses `aiogram`'s built-in backoff and reconnection behavior. Llama requests use one shared asynchronous HTTP client with explicit connection, read, write, and pool timeouts.

Only transient failures are retried:

- connection failures;
- timeouts;
- HTTP 408;
- HTTP 429;
- HTTP 5xx.

The retry policy allows at most three total attempts with exponential delay and random jitter. Authentication failures, invalid configuration, malformed responses, and other HTTP 4xx responses fail immediately. A final generation failure produces one short Telegram error message and a sanitized GUI log entry. The client does not retry a completed response and therefore does not knowingly duplicate model output.

Logs must never contain the Telegram token, full Bot API URLs containing it, or raw configuration dumps. Worker exceptions are converted to typed GUI events before crossing the thread boundary.

## Testing strategy

### Unit tests

- YAML defaults, round-trip saving, normalization, atomic replacement, and validation failures.
- Allowlist parsing and authorization decisions.
- Per-chat history isolation, reset, concurrency locking, exchange-pair trimming, and oversized newest messages.
- Retry classification, attempt count, backoff invocation, and non-retryable failures.
- Telegram response splitting at paragraph, newline, and hard boundaries.
- Runtime state transitions and rejection of duplicate Start/Stop operations.

### Mocked integration tests

- Successful `/health`, `/v1/models`, and `/props` discovery.
- Empty model list and malformed property responses.
- Completion requests use the selected model and server-derived limits.
- Transient llama-server failure followed by recovery.
- Final upstream failure leaves history consistent.
- Authorized and unauthorized Telegram message flows through an `aiogram` dispatcher.

### Live smoke tests

- Verify the running local llama-server endpoints.
- Send a short completion request and validate non-empty output.
- Validate the configured Telegram token using `getMe` without logging it.
- Start polling through the application and manually confirm one real Telegram message receives a model response.
- Stop the bot and verify polling and the worker thread terminate cleanly.

## Packaging and operational files

- `pyproject.toml` declares Python `>=3.12,<3.13`, runtime dependencies, and pytest configuration.
- `README.md` documents environment creation, installation, YAML configuration, starting the llama-server prerequisite, launching the GUI, commands, and troubleshooting.
- `.gitignore` excludes `config.yaml`, virtual environments, caches, test artifacts, and logs.
- The first delivery is run from source; executable packaging is outside the current scope.

## Acceptance criteria

- The application launches under Python 3.12 with a responsive native GUI.
- Only the token, server URL, and optional allowlist are editable configuration fields.
- Start discovers and displays the first server model and its capabilities without requiring model configuration.
- An authorized Telegram text message receives a response generated by the local llama-server.
- Histories remain isolated per chat and `/reset` clears only the requesting chat.
- Temporary Telegram and llama-server outages are retried according to the bounded policy.
- Stop and window close release polling, HTTP sessions, tasks, and the worker thread.
- Automated tests pass, the live server smoke request succeeds, and the Telegram token validation succeeds.
- Secrets do not appear in tracked configuration examples, logs, or test output.

## Security note

The test token currently appears in plaintext in `codex.md`. It may be used only for the requested live validation, must not be copied into source files or documentation, and should be revoked and replaced through BotFather after testing.
