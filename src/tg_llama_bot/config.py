import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import yaml

from tg_llama_bot.models import AppConfig

DEFAULT_LLAMA_BASE_URL = "http://127.0.0.1:8080"


class ConfigError(ValueError):
    """Configuration cannot be parsed or validated."""


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        return AppConfig()

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError("Не удалось прочитать YAML-конфигурацию.") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("Корень YAML-конфигурации должен быть объектом.")

    token = raw.get("telegram_token", "")
    base_url = raw.get("llama_base_url", DEFAULT_LLAMA_BASE_URL)
    allowed_ids = raw.get("allowed_user_ids", [])
    if not isinstance(token, str):
        raise ConfigError("telegram_token должен быть строкой.")
    if not isinstance(base_url, str):
        raise ConfigError("llama_base_url должен быть строкой.")
    if not isinstance(allowed_ids, list):
        raise ConfigError("allowed_user_ids должен быть списком.")

    return _normalize_config(
        AppConfig(
            telegram_token=token,
            llama_base_url=base_url,
            allowed_user_ids=_normalize_ids(allowed_ids),
        )
    )


def save_config(path: Path, config: AppConfig) -> None:
    normalized = _normalize_config(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "telegram_token": normalized.telegram_token,
        "llama_base_url": normalized.llama_base_url,
        "allowed_user_ids": list(normalized.allowed_user_ids),
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            yaml.safe_dump(payload, temporary, sort_keys=False, allow_unicode=True)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    except (OSError, yaml.YAMLError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ConfigError("Не удалось сохранить YAML-конфигурацию.") from exc


def parse_allowed_user_ids(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    try:
        values = [int(part.strip()) for part in raw.split(",")]
    except ValueError as exc:
        raise ConfigError("Telegram user ID должны быть целыми числами.") from exc
    return _normalize_ids(values)


def format_allowed_user_ids(ids: tuple[int, ...]) -> str:
    return ", ".join(str(value) for value in _normalize_ids(ids))


def _normalize_config(config: AppConfig) -> AppConfig:
    base_url = config.llama_base_url.strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError("llama_base_url должен быть корректным http/https URL.")
    return AppConfig(
        telegram_token=config.telegram_token.strip(),
        llama_base_url=base_url,
        allowed_user_ids=_normalize_ids(config.allowed_user_ids),
    )


def _normalize_ids(values: list[object] | tuple[int, ...]) -> tuple[int, ...]:
    normalized: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError("Telegram user ID должны быть положительными целыми числами.")
        normalized.add(value)
    return tuple(sorted(normalized))
