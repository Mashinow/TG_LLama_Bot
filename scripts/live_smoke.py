import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tg_llama_bot.bot import make_bot, make_llama_client
from tg_llama_bot.config import ConfigError, load_config
from tg_llama_bot.models import ChatMessage
from tg_llama_bot.runtime import sanitize_exception


@dataclass(frozen=True, slots=True)
class SmokeResult:
    model_id: str
    bot_username: str
    completion: str


async def smoke(config_path: Path, prompt: str) -> SmokeResult:
    config = load_config(config_path)
    if not config.telegram_token:
        raise ConfigError("В конфигурации отсутствует Telegram bot token.")
    if not prompt.strip():
        raise ConfigError("Smoke prompt не может быть пустым.")

    async with make_llama_client(config.llama_base_url) as llama:
        capabilities = await llama.discover()
        completion = await llama.complete(
            capabilities.model_id,
            [ChatMessage("user", prompt)],
            min(32, capabilities.max_output_tokens),
        )
        telegram_bot = make_bot(config.telegram_token)
        try:
            identity = await telegram_bot.get_me()
        finally:
            await telegram_bot.session.close()

    username = identity.username or str(identity.id)
    return SmokeResult(capabilities.model_id, username, completion)


def format_smoke_error(exc: BaseException, token: str) -> str:
    return sanitize_exception(exc, token)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check llama-server completion and Telegram bot identity."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML configuration (default: config.yaml).",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: pong",
        help="Short prompt sent to llama-server.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = ""
    try:
        token = load_config(args.config).telegram_token
        result = asyncio.run(smoke(args.config, args.prompt))
    # This is a CLI boundary; errors must be sanitized before reaching stderr.
    except Exception as exc:  # noqa: BLE001
        print(f"Smoke failed: {format_smoke_error(exc, token)}", file=sys.stderr)
        return 1

    preview = result.completion.replace("\r", " ").replace("\n", " ")[:200]
    print(f"Model: {result.model_id}")
    print(f"Telegram bot: @{result.bot_username}")
    print(f"Completion: {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
