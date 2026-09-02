# Image Input Design

## Goal

Add native vision input to the Telegram bot with conversation semantics matching the bundled `llama-server` UI.

## Confirmed llama-ui behavior

- An image is stored as part of the user message.
- The OpenAI-compatible request represents it as an `image_url` content part containing a base64 `data:` URL.
- Later requests resend the complete retained conversation, including earlier images.
- Clearing the conversation or trimming its oldest exchanges removes the associated images.
- Images are sent directly to the vision model; OCR is not used.

## Telegram behavior

- Accept Telegram photos and image documents whose MIME type starts with `image/`.
- Accept one image per Telegram message and use its caption as the prompt.
- When no caption is supplied, use the Russian default prompt `Опиши изображение.`.
- Reject files larger than 10 MiB before download when Telegram reports their size, and verify the downloaded size again.
- If the selected llama model does not advertise the `vision` modality, return a clear user-facing error without calling completion.
- Keep image bytes only in the in-memory per-chat history. `/reset`, application shutdown, and history trimming remove them.

## Request and history representation

Introduce an immutable `ImageAttachment(media_type: str, data: bytes)` value and extend `ChatMessage` with an `images` tuple. Text-only construction remains backward compatible.

For messages with images, `LlamaClient.complete` sends content in the same order as llama-ui:

```json
[
  {
    "type": "image_url",
    "image_url": {
      "url": "data:image/jpeg;base64,..."
    }
  },
  {
    "type": "text",
    "text": "Опиши изображение."
  }
]
```

Text-only messages continue to use a plain string `content`.

History budgeting never serializes raw image bytes into `/tokenize`. It counts the textual conversation normally and reserves a conservative fixed allowance per retained image. Oldest complete user/assistant exchanges continue to be removed first.

## Error handling and scope

- Telegram download failures produce a generic safe attachment error.
- Empty or oversized downloaded data is rejected before storing it.
- Existing authorization, retry, response splitting, and polling behavior remain unchanged.
- Album aggregation, video, audio, OCR, disk persistence, and image generation are outside this change.

