import queue
import tkinter as tk
from pathlib import Path

from tg_llama_bot.gui import BotWindow
from tg_llama_bot.models import RuntimeEvent
from tg_llama_bot.runtime import RuntimeController


def main(config_path: Path = Path("config.yaml")) -> int:
    event_queue: queue.Queue[RuntimeEvent] = queue.Queue()
    controller = RuntimeController(event_queue)
    root = tk.Tk()
    BotWindow(root, config_path, controller, event_queue)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
