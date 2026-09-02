from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tg_llama_bot.app as app_module
from tg_llama_bot.gui import controls_for_state, handle_windows_entry_shortcut
from tg_llama_bot.models import RuntimeState


@pytest.mark.parametrize(
    ("state", "start", "stop", "fields"),
    [
        (RuntimeState.STOPPED, True, False, True),
        (RuntimeState.STARTING, False, True, False),
        (RuntimeState.RUNNING, False, True, False),
        (RuntimeState.STOPPING, False, False, False),
        (RuntimeState.ERROR, True, False, True),
    ],
)
def test_controls_follow_runtime_state(
    state: RuntimeState,
    start: bool,
    stop: bool,
    fields: bool,
) -> None:
    controls = controls_for_state(state)
    assert controls.start_enabled is start
    assert controls.stop_enabled is stop
    assert controls.fields_enabled is fields


def test_physical_ctrl_v_generates_one_paste_event() -> None:
    widget = Mock()
    event = SimpleNamespace(keycode=86, widget=widget)
    assert handle_windows_entry_shortcut(event) == "break"
    widget.event_generate.assert_called_once_with("<<Paste>>")


def test_other_ctrl_key_is_left_to_tkinter() -> None:
    widget = Mock()
    event = SimpleNamespace(keycode=66, widget=widget)
    assert handle_windows_entry_shortcut(event) is None
    widget.event_generate.assert_not_called()


def test_main_constructs_window_and_enters_mainloop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Mock()
    controller = object()
    captured: dict[str, tuple] = {}

    class FakeWindow:
        def __init__(
            self,
            root_arg,
            config_path,
            controller_arg,
            event_queue,
        ) -> None:
            captured["args"] = (
                root_arg,
                config_path,
                controller_arg,
                event_queue,
            )

    monkeypatch.setattr(app_module.tk, "Tk", lambda: root)
    monkeypatch.setattr(
        app_module,
        "RuntimeController",
        lambda event_queue: controller,
    )
    monkeypatch.setattr(app_module, "BotWindow", FakeWindow)

    assert app_module.main() == 0
    root.mainloop.assert_called_once_with()
    assert captured["args"][0:3] == (root, Path("config.yaml"), controller)
