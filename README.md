# Telegram Bot for a Local llama.cpp Server

<img width="464" height="847" alt="ex" src="https://github.com/user-attachments/assets/0640982d-4d4a-44b6-8ff1-ce0b64524826" />

This desktop application runs an aiogram-based Telegram bot and forwards text and images to an already running OpenAI-compatible llama-server. Settings are edited in a native tkinter interface, and the Start and Stop buttons control the bot without closing the GUI.

## Features

- Long polling; no public webhook or inbound port is required.
- Text conversations with separate in-memory history for each Telegram chat.
- Telegram photos and image documents for vision-capable models.
- Streaming responses through edits to a single Telegram message.
- Automatic llama-server model and capability discovery.
- Configurable Telegram user allowlist.
- Retry and backoff handling for temporary network and server failures.

## Requirements

To run from source, use any Python interpreter on which the dependencies declared in `pyproject.toml` install and work correctly. The interpreter must also provide a working tkinter/Tcl/Tk installation for the desktop interface.

You also need:

- a running OpenAI-compatible llama-server, available at `http://127.0.0.1:8080` by default;
- a Telegram bot token issued by BotFather.

Python is not required when using the packaged executable from a release.

## Run the Release Executable

1. Download `tg-llama-bot.exe` from the Releases page of this repository.
2. Start `tg-llama-bot.exe`.
3. Enter the Telegram token and other settings in the desktop interface.
4. Click **Save**, then click **Start** and wait for the running status.

The executable creates its local configuration automatically. All configuration changes are made through the desktop interface.

## Run from Source

You may use any supported environment manager or an existing Python environment. Install the project dependencies and start the application:

```text
python -m pip install -e .
python -m tg_llama_bot.app
```

Enter the Telegram token and other settings in the desktop interface, click **Save**, and then start the bot.

## Build the One-File Executable

Install the development dependencies and run PyInstaller with the tracked specification:

```text
python -m pip install -e ".[dev]"
python -m PyInstaller --noconfirm --clean tg-llama-bot.spec
```

The build output is `dist/tg-llama-bot.exe`. The executable bundles the Python runtime and project dependencies; users running it do not need a separate Python installation.

## Configuration

Configuration is created and edited through the desktop interface. It contains three settings:

- `telegram_token` is required before the bot can start.
- `llama_base_url` defaults to the local llama-server address.
- An empty `allowed_user_ids` list allows everyone to use the bot. A list such as `[123456789]` restricts access to those Telegram user IDs.

Click **Save** after making changes. The application stores the settings locally and loads them on the next launch. Never put the Telegram token in source code, documentation, logs, or error messages.

## Desktop Interface

1. Enter the Telegram bot token.
2. Change the llama-server URL or allowlist if needed.
3. Click **Save**.
4. Click **Start** and wait for the running status.
5. Click **Stop** or close the window to shut down the bot.

The discovered model and server capabilities are displayed as read-only information. The GUI remains on the main thread, while Telegram polling and HTTP requests run on a separate asyncio thread.

On each start, the application checks `/health`, selects the first model returned by `/v1/models`, and reads context size, generation limits, reasoning format, and modalities from `/props`. If the server reports unlimited generation with `-1`, the bot applies a safety limit of 2,048 generated tokens.

The bot uses Telegram long polling. On startup it calls `deleteWebhook` while preserving pending updates, because Telegram does not allow webhook delivery and long polling at the same time. If the same bot is connected to another webhook integration, coordinate the delivery-mode switch before starting this application.

## Telegram Commands

- `/start` — show a short introduction.
- `/help` — show the available features.
- `/reset` — clear the history of the current chat.
- Plain text — continue the conversation with the model.
- Photo or image document — send an image to a vision-capable model.

## Image Support

The bot accepts Telegram photos and documents with an `image/*` MIME type. The maximum image size is 10 MiB. The image caption is used as the prompt; without a caption, the bot asks the model to describe the image.

Images are sent only when llama-server reports the `vision` modality through `/props`. If a text-only model is loaded, the bot rejects the image and explains that image analysis is unavailable. Other attachment types are not sent to the model.

As in the llama-server UI, an image remains attached to its message and is sent again with later questions in the same conversation. `/reset` removes the image together with the chat history. Older user/assistant pairs and their images are also removed when the history approaches the model's `n_ctx` limit.

## Streaming Responses

When a request starts, the bot immediately sends `...`. It consumes the llama-server response as a stream and edits that same Telegram message after every 150 generated tokens. When generation finishes, the message is updated with the complete response. The bot does not create extra messages for response chunks.

The complete response must fit in one Telegram message. If it exceeds Telegram's message-size limit, shorten the request, reduce the generation limit, or reset the conversation.

## Reliability and History

Telegram polling uses aiogram backoff. Requests to llama-server are retried up to three times after network failures, timeouts, HTTP 408, HTTP 429, or 5xx responses. Other 4xx responses are not retried.

Messages from the same chat are processed sequentially so concurrent requests cannot mix their history. History exists only in memory, is isolated per chat, and is cleared when the bot stops.

## Development Checks

Install the development dependencies before running the checks:

```text
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests scripts
python -m compileall -q src scripts
```

The live smoke test checks model discovery, a short generation, and Telegram `getMe`:

```text
python scripts/live_smoke.py --config config.yaml
```

It prints the model ID, bot username, and a short response excerpt, but never prints the Telegram token.

## Troubleshooting

- **llama-server is unavailable:** check `http://127.0.0.1:8080/health` and the configured `llama_base_url`.
- **The model list is empty:** make sure a model is loaded and `/v1/models` returns at least one item.
- **The Telegram token is invalid:** issue a token through BotFather, enter it in the desktop interface, and click **Save**.
- **tkinter is unavailable:** use a Python distribution that includes a working tkinter/Tcl/Tk installation.
- **The response does not fit:** shorten the request, reduce the generation limit, or use `/reset`.
- **The model cannot analyze images:** verify that `/props` reports the `vision` modality; a text-only model will not receive the attachment.
- **The image is rejected:** send a photo or an `image/*` document no larger than 10 MiB.
- **Access is denied:** check the Telegram user ID in `allowed_user_ids`; an empty list disables the restriction.

Revoke any token that has ever been exposed in plain text through BotFather, then enter the replacement token in the desktop interface and click **Save**.
