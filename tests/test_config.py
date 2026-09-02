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


def test_partial_config_uses_default_server_url(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("telegram_token: test-token\n", encoding="utf-8")
    assert load_config(path).llama_base_url == "http://127.0.0.1:8080"


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

    def recording_replace(source: str | bytes, destination: str | bytes) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr("tg_llama_bot.config.os.replace", recording_replace)
    save_config(path, AppConfig("secret"))
    assert len(replacements) == 1
    assert replacements[0][1] == path
    assert replacements[0][0].parent == path.parent
