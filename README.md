# Telegram-бот для локального llama.cpp

Десктопное приложение запускает Telegram-бота на базе aiogram и направляет текстовые сообщения в уже работающий OpenAI-совместимый llama-server. Настройки редактируются в нативном интерфейсе tkinter; кнопки Start и Stop управляют ботом без остановки GUI.

## Требования

- Windows и Python 3.12;
- виртуальное окружение .venv;
- запущенный llama-server, по умолчанию на http://127.0.0.1:8080;
- Telegram bot token от BotFather.

Приложение при каждом Start проверяет /health, получает первую модель из /v1/models, а размер контекста, лимит генерации, reasoning format и модальности — из /props. Если сервер сообщает неограниченную генерацию (-1), используется safety-лимит до 2048 токенов.

## Установка

```powershell
.venv/Scripts/python.exe --version
.venv/Scripts/python.exe -m pip install -e ".[dev]"
Copy-Item config.example.yaml config.yaml
```

Версия первой команды должна начинаться с Python 3.12.

## Конфигурация

config.yaml содержит только три поля:

```yaml
telegram_token: "TOKEN_FROM_BOTFATHER"
llama_base_url: "http://127.0.0.1:8080"
allowed_user_ids: []
```

- telegram_token обязателен для Start.
- llama_base_url уже имеет локальное значение по умолчанию.
- Пустой allowed_user_ids разрешает доступ всем. Список вроде [123456789] ограничивает бота указанными Telegram user ID.

config.yaml исключён из Git, поскольку содержит секрет. Не добавляйте токен в исходный код, README, логи или сообщения об ошибках.

## Запуск

```powershell
.venv/Scripts/python.exe -m tg_llama_bot.app
```

В окне:

1. Введите Telegram token.
2. При необходимости измените URL или allowlist.
3. Нажмите Save.
4. Нажмите Start и дождитесь статуса «Запущен».
5. Для завершения нажмите Stop или закройте окно.

Обнаруженная модель и серверные параметры отображаются только для чтения. GUI остаётся в главном потоке, а polling и HTTP-запросы выполняются в отдельном asyncio-потоке.

При Start приложение вызывает Telegram deleteWebhook с сохранением pending updates, поскольку Bot API не разрешает одновременно использовать webhook и long polling. Если этот бот обслуживает другую webhook-интеграцию, не запускайте polling до согласованного переключения режима доставки.

## Команды Telegram

- /start — краткая инструкция;
- /help — доступные возможности;
- /reset — очистить историю только текущего чата;
- обычный текст — продолжить диалог с моделью.

История хранится отдельно для каждого чата только в памяти и очищается после Stop. Старые пары «пользователь/ассистент» удаляются, когда история приближается к n_ctx. Нетекстовые сообщения не отправляются модели.

## Отказоустойчивость

Telegram polling использует backoff aiogram. Запросы к llama-server повторяются максимум три раза при сетевом сбое, timeout, HTTP 408, 429 или 5xx. Другие 4xx не повторяются. Один чат обрабатывается последовательно, поэтому параллельные сообщения не перемешивают его историю.

## Проверки

Автоматические:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests scripts
.venv/Scripts/python.exe -m compileall -q src scripts
```

Live smoke проверяет discovery модели, короткую генерацию и Telegram getMe:

```powershell
.venv/Scripts/python.exe scripts/live_smoke.py --config config.yaml
```

Команда выводит ID модели, username бота и короткий фрагмент ответа, но никогда не печатает токен.

## Диагностика

- **llama-server недоступен:** проверьте http://127.0.0.1:8080/health и значение llama_base_url.
- **Список моделей пуст:** убедитесь, что модель загружена и /v1/models возвращает хотя бы одну запись.
- **Неверный Telegram token:** выпустите токен через BotFather и сохраните его только в config.yaml.
- **Нет tkinter:** переустановите Python 3.12 с компонентом Tcl/Tk.
- **Сообщение не помещается:** сократите новый запрос или выполните /reset.
- **Нет доступа:** проверьте Telegram user ID в allowed_user_ids; пустой список отключает ограничение.

Токен, который когда-либо публиковался открытым текстом, следует отозвать через BotFather после завершения проверки и заменить новым в локальном config.yaml.
