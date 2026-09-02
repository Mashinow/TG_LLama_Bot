from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppConfig:
    telegram_token: str = ""
    llama_base_url: str = "http://127.0.0.1:8080"
    allowed_user_ids: tuple[int, ...] = ()
