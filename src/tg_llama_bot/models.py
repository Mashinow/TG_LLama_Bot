from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal


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


class RuntimeState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    kind: Literal["state", "log", "capabilities", "error"]
    payload: object


type EventSink = Callable[[RuntimeEvent], None]
