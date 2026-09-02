# Image Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native Telegram image input whose images remain available to later messages until reset or history trimming.

**Architecture:** Store immutable image bytes alongside user text in `ChatMessage`, convert them to OpenAI-compatible base64 `image_url` parts only at the llama HTTP boundary, and keep history budgeting independent of binary serialization. A dedicated Telegram handler downloads bounded image data into memory and delegates to the existing serialized per-chat service.

**Tech Stack:** Python 3.12, aiogram 3, httpx, pytest, pytest-asyncio, Ruff

**Spec:** `docs/superpowers/specs/2026-09-03-image-input-design.md`

## Global Constraints

- Use the existing `.venv` with Python 3.12.
- Keep images only in memory; never write Telegram uploads to disk.
- Limit each downloaded image to 10 MiB.
- Preserve text-only API behavior and existing public constructors.
- Preserve the user's unrelated tracked deletion of `main.py`.
- Implement every production behavior through a failing test first.

---

### Task 1: Multimodal llama request contract

**Files:**
- Modify: `src/tg_llama_bot/models.py`
- Modify: `src/tg_llama_bot/llama_client.py`
- Modify: `tests/test_llama_client.py`

**Interfaces:**
- Produces: `ImageAttachment(media_type: str, data: bytes)`
- Produces: `ChatMessage(role: Literal["user", "assistant"], content: str, images: tuple[ImageAttachment, ...] = ())`
- Consumes: `LlamaClient.complete(model_id, messages, max_tokens)`

- [ ] **Step 1: Write the failing multimodal request test**

Add a test that calls `complete` with JPEG bytes `b"image-bytes"` and asserts the literal request payload contains:

```python
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
```

The existing text-only contract test must remain unchanged and passing.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_llama_client.py -q
```

Expected: failure because `ImageAttachment` and image-aware serialization do not exist.

- [ ] **Step 3: Implement the minimal model and serializer**

Add the frozen slotted value object and the optional `images` field. In `LlamaClient.complete`, keep a string for messages without images; otherwise build image parts followed by a text part when `content` is non-empty. Encode bytes with standard base64 and preserve the attachment MIME type in the data URL.

- [ ] **Step 4: Run the client tests and verify GREEN**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_llama_client.py -q
.venv/Scripts/python.exe -m ruff check src/tg_llama_bot/models.py src/tg_llama_bot/llama_client.py tests/test_llama_client.py
```

Expected: all focused tests pass and Ruff is clean.

- [ ] **Step 5: Commit the request contract**

```powershell
git add src/tg_llama_bot/models.py src/tg_llama_bot/llama_client.py tests/test_llama_client.py
git commit -m "feat: add multimodal llama messages"
```

### Task 2: Image-aware retained history

**Files:**
- Modify: `src/tg_llama_bot/history.py`
- Modify: `src/tg_llama_bot/bot.py`
- Modify: `tests/test_history.py`
- Modify: `tests/test_bot.py`

**Interfaces:**
- Extends: `HistoryStore.prepare(..., images: tuple[ImageAttachment, ...] = ())`
- Extends: `HistoryStore.commit(..., images: tuple[ImageAttachment, ...] = ())`
- Extends: `ChatService.answer(chat_id, text, images=())`
- Produces: `VisionUnavailableError`

- [ ] **Step 1: Write failing retention and trimming tests**

Add a history test that commits a message with an image, prepares the next text turn, and asserts the exact earlier `ChatMessage` still contains that image. Add a second test with a small prompt budget proving the oldest complete exchange, including its image, is removed as one unit.

Add service tests proving a successful image exchange is committed and appears in the next completion request, while a model without `vision` raises `VisionUnavailableError` and never calls completion.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_history.py tests/test_bot.py -q
```

Expected: failures because history and service do not accept images or enforce vision capabilities.

- [ ] **Step 3: Implement image retention and conservative budgeting**

Pass the image tuple into candidate and committed user messages. Serialize only role and text for `/tokenize`, then add a fixed conservative allowance of 2048 tokens for each image in the candidate. Preserve the existing oldest-complete-exchange trimming loop.

Extend `ChatService.answer` with `images=()`. Before preparing history, raise `VisionUnavailableError` when images are present and `vision` is absent from capabilities. Commit images only after a successful completion.

- [ ] **Step 4: Run focused and regression tests**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_history.py tests/test_bot.py -q
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe -m ruff check src tests
```

Expected: all tests pass and Ruff is clean.

- [ ] **Step 5: Commit retained image history**

```powershell
git add src/tg_llama_bot/history.py src/tg_llama_bot/bot.py tests/test_history.py tests/test_bot.py
git commit -m "feat: retain images in chat history"
```

### Task 3: Telegram image handling and documentation

**Files:**
- Modify: `src/tg_llama_bot/bot.py`
- Modify: `tests/test_bot.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `BotHandlers.image(message: Message) -> None`
- Registers: image handler before the text and unsupported fallbacks
- Uses: `message.bot.download(downloadable, destination=io.BytesIO())`

- [ ] **Step 1: Write failing Telegram handler tests**

Use complete fake Telegram photo/document structures and a fake downloader. Add independent tests proving:

- a photo with a caption reaches `ChatService.answer` with `image/jpeg` bytes;
- a captionless photo uses the literal default `Опиши изображение.`;
- an image document preserves its MIME type;
- a reported or downloaded payload over 10 MiB is rejected before model completion;
- a non-image document remains unsupported;
- `VisionUnavailableError`, Telegram download errors, and model errors map to safe user-facing messages;
- dispatcher registration places the media handler before the fallback.

- [ ] **Step 2: Run the bot tests and verify RED**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_bot.py -q
```

Expected: failures because the image handler and messages are missing.

- [ ] **Step 3: Implement bounded in-memory downloads**

Add constants for the 10 MiB limit, default prompt, and safe error text. Select the largest Telegram photo variant or an `image/*` document, check reported size, download into `BytesIO`, check actual size and non-emptiness, then call `ChatService.answer` with one `ImageAttachment`. Reuse response splitting and authorization behavior. Register this handler before `F.text` and the unsupported fallback.

- [ ] **Step 4: Update user documentation**

Document photo and image-document input, captions, the default description prompt, 10 MiB limit, vision capability requirement, in-memory retention across follow-ups, and `/reset` cleanup.

- [ ] **Step 5: Run full verification**

Run:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests scripts
.venv/Scripts/python.exe -m compileall -q src scripts
```

Then run a local llama-server multimodal smoke request with a generated tiny PNG and verify a non-empty response without writing the image to the repository.

- [ ] **Step 6: Commit Telegram image support**

```powershell
git add src/tg_llama_bot/bot.py tests/test_bot.py README.md docs/superpowers/specs/2026-09-03-image-input-design.md docs/superpowers/plans/2026-09-03-image-input.md
git commit -m "feat: accept telegram images"
```

