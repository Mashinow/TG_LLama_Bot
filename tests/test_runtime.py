import queue
import threading

from tg_llama_bot.models import AppConfig, RuntimeEvent, RuntimeState
from tg_llama_bot.runtime import RuntimeController, sanitize_exception

CONFIG = AppConfig("123:SECRET", "http://127.0.0.1:8080", ())


def drain(events: queue.Queue[RuntimeEvent]) -> list[RuntimeEvent]:
    drained: list[RuntimeEvent] = []
    while not events.empty():
        drained.append(events.get_nowait())
    return drained


def test_start_and_stop_run_async_runner_in_worker_thread() -> None:
    entered = threading.Event()
    runner_thread_ids: list[int] = []

    async def runner(config, stop_event, emit) -> None:
        runner_thread_ids.append(threading.get_ident())
        entered.set()
        emit(RuntimeEvent("state", RuntimeState.RUNNING))
        await stop_event.wait()

    events: queue.Queue[RuntimeEvent] = queue.Queue()
    controller = RuntimeController(events, runner=runner)
    assert controller.start(CONFIG)
    assert entered.wait(1.0)
    assert runner_thread_ids != [threading.get_ident()]
    assert not controller.start(CONFIG)
    assert controller.request_stop()
    assert controller.wait_stopped(2.0)
    assert controller.state is RuntimeState.STOPPED
    states = [
        event.payload
        for event in drain(events)
        if event.kind == "state"
    ]
    assert states == [
        RuntimeState.STARTING,
        RuntimeState.RUNNING,
        RuntimeState.STOPPING,
        RuntimeState.STOPPED,
    ]


def test_stop_requested_before_async_event_exists_is_not_lost() -> None:
    release_runner_start = threading.Event()

    async def runner(config, stop_event, emit) -> None:
        release_runner_start.set()
        await stop_event.wait()

    events: queue.Queue[RuntimeEvent] = queue.Queue()
    controller = RuntimeController(events, runner=runner)
    assert controller.start(CONFIG)
    assert controller.request_stop()
    assert release_runner_start.wait(1.0)
    assert controller.wait_stopped(2.0)


def test_duplicate_stop_is_rejected_while_stopped() -> None:
    controller = RuntimeController(queue.Queue())
    assert not controller.request_stop()


def test_runner_failure_emits_sanitized_error_and_stops() -> None:
    async def runner(config, stop_event, emit) -> None:
        raise RuntimeError("request to /bot123:SECRET/getMe failed")

    events: queue.Queue[RuntimeEvent] = queue.Queue()
    controller = RuntimeController(events, runner=runner)
    assert controller.start(CONFIG)
    assert controller.wait_stopped(2.0)
    emitted = drain(events)
    payloads = [str(event.payload) for event in emitted]
    assert all("123:SECRET" not in payload for payload in payloads)
    assert any("<redacted>" in payload for payload in payloads)
    assert RuntimeEvent("state", RuntimeState.ERROR) in emitted
    assert RuntimeEvent("state", RuntimeState.STOPPED) in emitted
    assert controller.state is RuntimeState.STOPPED


def test_sanitize_exception_removes_token_and_bot_url_segment() -> None:
    text = sanitize_exception(
        RuntimeError("https://api.telegram.org/bot123:SECRET/getMe"),
        "123:SECRET",
    )
    assert "123:SECRET" not in text
    assert "<redacted>" in text
