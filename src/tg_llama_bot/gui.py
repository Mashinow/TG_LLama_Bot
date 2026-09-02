import queue
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from tg_llama_bot.config import (
    ConfigError,
    format_allowed_user_ids,
    load_config,
    parse_allowed_user_ids,
    save_config,
)
from tg_llama_bot.models import (
    AppConfig,
    RuntimeEvent,
    RuntimeState,
    ServerCapabilities,
)
from tg_llama_bot.runtime import RuntimeController


@dataclass(frozen=True, slots=True)
class ControlState:
    start_enabled: bool
    stop_enabled: bool
    fields_enabled: bool


def handle_windows_entry_shortcut(event: tk.Event) -> str | None:
    if event.keycode != 0x56:
        return None
    event.widget.event_generate("<<Paste>>")
    return "break"


def controls_for_state(state: RuntimeState) -> ControlState:
    if state in {RuntimeState.STOPPED, RuntimeState.ERROR}:
        return ControlState(True, False, True)
    if state in {RuntimeState.STARTING, RuntimeState.RUNNING}:
        return ControlState(False, True, False)
    return ControlState(False, False, False)


class BotWindow:
    def __init__(
        self,
        root: tk.Tk,
        config_path: Path,
        controller: RuntimeController,
        event_queue: queue.Queue[RuntimeEvent],
    ) -> None:
        self.root = root
        self._config_path = config_path
        self._controller = controller
        self._event_queue = event_queue
        self._closing = False
        self._close_started = 0.0

        self.token_var = tk.StringVar()
        self.url_var = tk.StringVar()
        self.allowed_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Остановлен")
        self.model_var = tk.StringVar(value="—")
        self.context_var = tk.StringVar(value="—")
        self.output_var = tk.StringVar(value="—")
        self.reasoning_var = tk.StringVar(value="—")
        self.modalities_var = tk.StringVar(value="—")

        self._build_layout()
        self._load_initial_config()
        self._apply_state(RuntimeState.STOPPED)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._drain_events)

    def _build_layout(self) -> None:
        self.root.title("Telegram llama.cpp Bot")
        self.root.minsize(660, 540)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        settings = ttk.LabelFrame(self.root, text="Настройки", padding=10)
        settings.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="Telegram token").grid(
            row=0, column=0, padx=(0, 8), pady=4, sticky="w"
        )
        self.token_entry = ttk.Entry(settings, textvariable=self.token_var, show="*")
        self.token_entry.grid(row=0, column=1, pady=4, sticky="ew")

        ttk.Label(settings, text="Llama server URL").grid(
            row=1, column=0, padx=(0, 8), pady=4, sticky="w"
        )
        self.url_entry = ttk.Entry(settings, textvariable=self.url_var)
        self.url_entry.grid(row=1, column=1, pady=4, sticky="ew")

        ttk.Label(settings, text="Allowed user IDs").grid(
            row=2, column=0, padx=(0, 8), pady=4, sticky="w"
        )
        self.allowed_entry = ttk.Entry(settings, textvariable=self.allowed_var)
        self.allowed_entry.grid(row=2, column=1, pady=4, sticky="ew")
        if sys.platform == "win32":
            for entry in (self.token_entry, self.url_entry, self.allowed_entry):
                entry.bind(
                    "<Control-KeyPress>",
                    handle_windows_entry_shortcut,
                    add="+",
                )

        buttons = ttk.Frame(self.root, padding=(10, 5))
        buttons.grid(row=1, column=0, sticky="ew")
        buttons.columnconfigure(4, weight=1)
        self.save_button = ttk.Button(buttons, text="Save", command=self._save)
        self.save_button.grid(row=0, column=0, padx=(0, 6))
        self.start_button = ttk.Button(buttons, text="Start", command=self._start)
        self.start_button.grid(row=0, column=1, padx=6)
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop)
        self.stop_button.grid(row=0, column=2, padx=6)
        ttk.Label(buttons, text="Статус:").grid(row=0, column=3, padx=(18, 4))
        ttk.Label(buttons, textvariable=self.status_var).grid(row=0, column=4, sticky="w")

        capabilities = ttk.LabelFrame(
            self.root,
            text="Обнаруженные параметры сервера",
            padding=10,
        )
        capabilities.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        capabilities.columnconfigure(1, weight=1)
        rows = (
            ("Модель", self.model_var),
            ("Контекст", self.context_var),
            ("Лимит ответа", self.output_var),
            ("Reasoning", self.reasoning_var),
            ("Модальности", self.modalities_var),
        )
        for row, (label, variable) in enumerate(rows):
            ttk.Label(capabilities, text=label).grid(
                row=row, column=0, padx=(0, 8), pady=2, sticky="w"
            )
            ttk.Label(capabilities, textvariable=variable).grid(
                row=row, column=1, pady=2, sticky="w"
            )

        log_frame = ttk.LabelFrame(self.root, text="Журнал", padding=10)
        log_frame.grid(row=3, column=0, padx=10, pady=(5, 10), sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = scrolledtext.ScrolledText(log_frame, height=10, state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")

    def _load_initial_config(self) -> None:
        try:
            config = load_config(self._config_path)
        except ConfigError as exc:
            config = AppConfig()
            self._append_log(f"Ошибка конфигурации: {exc}")
        self._set_form(config)

    def _set_form(self, config: AppConfig) -> None:
        self.token_var.set(config.telegram_token)
        self.url_var.set(config.llama_base_url)
        self.allowed_var.set(format_allowed_user_ids(config.allowed_user_ids))

    def _read_form(self) -> AppConfig:
        return AppConfig(
            telegram_token=self.token_var.get(),
            llama_base_url=self.url_var.get(),
            allowed_user_ids=parse_allowed_user_ids(self.allowed_var.get()),
        )

    def _save(self, *, show_message: bool = True) -> AppConfig | None:
        try:
            save_config(self._config_path, self._read_form())
            config = load_config(self._config_path)
        except ConfigError as exc:
            messagebox.showerror("Ошибка конфигурации", str(exc), parent=self.root)
            return None
        self._set_form(config)
        self._append_log("Конфигурация сохранена.")
        if show_message:
            self.status_var.set("Настройки сохранены")
        return config

    def _start(self) -> None:
        config = self._save(show_message=False)
        if config is None:
            return
        if not config.telegram_token:
            messagebox.showerror(
                "Не задан токен",
                "Введите Telegram bot token.",
                parent=self.root,
            )
            return
        if self._controller.start(config):
            self._append_log("Запуск Telegram-бота…")

    def _stop(self) -> None:
        if self._controller.request_stop():
            self._append_log("Остановка Telegram-бота…")

    def _drain_events(self) -> None:
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break
            self._apply_event(event)
        self.root.after(100, self._drain_events)

    def _apply_event(self, event: RuntimeEvent) -> None:
        if event.kind == "state" and isinstance(event.payload, RuntimeState):
            self._apply_state(event.payload)
        elif event.kind == "capabilities" and isinstance(
            event.payload, ServerCapabilities
        ):
            self._apply_capabilities(event.payload)
        elif event.kind == "log":
            self._append_log(str(event.payload))
        elif event.kind == "error":
            self._append_log(f"Ошибка: {event.payload}")
            messagebox.showerror("Ошибка бота", str(event.payload), parent=self.root)

    def _apply_state(self, state: RuntimeState) -> None:
        labels = {
            RuntimeState.STOPPED: "Остановлен",
            RuntimeState.STARTING: "Запуск…",
            RuntimeState.RUNNING: "Запущен",
            RuntimeState.STOPPING: "Остановка…",
            RuntimeState.ERROR: "Ошибка",
        }
        self.status_var.set(labels[state])
        controls = controls_for_state(state)
        field_state = "normal" if controls.fields_enabled else "disabled"
        for entry in (self.token_entry, self.url_entry, self.allowed_entry):
            entry.configure(state=field_state)
        self.save_button.configure(state=field_state)
        self.start_button.configure(
            state="normal" if controls.start_enabled else "disabled"
        )
        self.stop_button.configure(
            state="normal" if controls.stop_enabled else "disabled"
        )

    def _apply_capabilities(self, capabilities: ServerCapabilities) -> None:
        self.model_var.set(capabilities.model_id)
        self.context_var.set(str(capabilities.n_ctx))
        suffix = "" if capabilities.server_max_output_tokens else " (safety)"
        self.output_var.set(f"{capabilities.max_output_tokens}{suffix}")
        self.reasoning_var.set(capabilities.reasoning_format)
        self.modalities_var.set(", ".join(capabilities.modalities))

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"{text}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_close(self) -> None:
        if self._controller.state is RuntimeState.STOPPED:
            self.root.destroy()
            return
        if self._closing:
            return
        self._closing = True
        self._close_started = time.monotonic()
        self._controller.request_stop()
        self._poll_closed()

    def _poll_closed(self) -> None:
        if self._controller.wait_stopped(0):
            self.root.destroy()
            return
        if time.monotonic() - self._close_started >= 5.0:
            keep_waiting = messagebox.askretrycancel(
                "Остановка",
                "Бот ещё завершает сетевые операции. Продолжить ожидание?",
                parent=self.root,
            )
            if not keep_waiting:
                self._closing = False
                return
            self._close_started = time.monotonic()
        self.root.after(100, self._poll_closed)
