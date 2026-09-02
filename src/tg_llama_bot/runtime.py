import asyncio
import queue
import threading
from collections.abc import Awaitable, Callable

from tg_llama_bot.bot import run_bot
from tg_llama_bot.models import AppConfig, EventSink, RuntimeEvent, RuntimeState

type AsyncRunner = Callable[[AppConfig, asyncio.Event, EventSink], Awaitable[None]]


def sanitize_exception(exc: BaseException, token: str) -> str:
    message = str(exc) or "Ошибка без дополнительных сведений."
    if token:
        message = message.replace(f"/bot{token}/", "/bot<redacted>/")
        message = message.replace(token, "<redacted>")
    return f"{type(exc).__name__}: {message}"


class RuntimeController:
    def __init__(
        self,
        event_queue: queue.Queue[RuntimeEvent],
        runner: AsyncRunner = run_bot,
    ) -> None:
        self._events = event_queue
        self._runner = runner
        self._lock = threading.Lock()
        self._state = RuntimeState.STOPPED
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._stop_requested = False
        self._active_token = ""

    @property
    def state(self) -> RuntimeState:
        with self._lock:
            return self._state

    def start(self, config: AppConfig) -> bool:
        with self._lock:
            if self._state is not RuntimeState.STOPPED:
                return False
            self._state = RuntimeState.STARTING
            self._stop_requested = False
            self._active_token = config.telegram_token
            thread = threading.Thread(
                target=self._thread_main,
                args=(config,),
                name="telegram-bot-runtime",
                daemon=False,
            )
            self._thread = thread
        self._events.put(RuntimeEvent("state", RuntimeState.STARTING))
        thread.start()
        return True

    def request_stop(self) -> bool:
        with self._lock:
            if self._state not in {RuntimeState.STARTING, RuntimeState.RUNNING}:
                return False
            self._state = RuntimeState.STOPPING
            self._stop_requested = True
            loop = self._loop
            stop_event = self._stop_event
        self._events.put(RuntimeEvent("state", RuntimeState.STOPPING))
        if loop is not None and stop_event is not None:
            loop.call_soon_threadsafe(stop_event.set)
        return True

    def wait_stopped(self, timeout: float) -> bool:
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        with self._lock:
            return self._state is RuntimeState.STOPPED and (
                self._thread is None or not self._thread.is_alive()
            )

    def _thread_main(self, config: AppConfig) -> None:
        asyncio.run(self._run(config))

    async def _run(self, config: AppConfig) -> None:
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        with self._lock:
            self._loop = loop
            self._stop_event = stop_event
            stop_requested = self._stop_requested
        if stop_requested:
            stop_event.set()

        try:
            await self._runner(config, stop_event, self._emit)
        # This is the worker-thread boundary; every runner failure must reach the GUI.
        except Exception as exc:  # noqa: BLE001
            self._events.put(
                RuntimeEvent(
                    "error",
                    sanitize_exception(exc, config.telegram_token),
                )
            )
            self._set_state(RuntimeState.ERROR)
        finally:
            with self._lock:
                self._loop = None
                self._stop_event = None
                self._active_token = ""
            self._set_state(RuntimeState.STOPPED)

    def _emit(self, event: RuntimeEvent) -> None:
        if event.kind == "state" and event.payload is RuntimeState.RUNNING:
            with self._lock:
                if self._state is RuntimeState.STOPPING:
                    return
                if self._state is not RuntimeState.STARTING:
                    return
                self._state = RuntimeState.RUNNING
        self._events.put(event)

    def _set_state(self, state: RuntimeState) -> None:
        with self._lock:
            self._state = state
        self._events.put(RuntimeEvent("state", state))
