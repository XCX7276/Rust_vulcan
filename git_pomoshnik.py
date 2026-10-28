# -*- coding: utf-8 -*-
"""
Гит-помощник — программа, которая помогает работать с git простыми словами.

Главные правила программы:
  * Перед каждым действием она объясняет, ЧТО будет сделано, и показывает команды.
  * Ничего не меняется, пока пользователь не нажмёт «Да, выполнить».
    (Без спроса программа только ЧИТАЕТ состояние проекта командой
     `git status`, чтобы показывать напоминания. Это ничего не меняет.)
  * Все тексты и напоминания — только в окне программы. Никаких системных
    уведомлений, звуков, писем и т.п.

Как устроен файл (сверху вниз):
  1. Настройки и тексты (словарик, подсказки по ошибкам).
  2. «Движок» — функции без окон: запуск git, разбор `git status`,
     перевод названия задачи в имя ветки.
  3. Step и Plan — описание действия: список команд + объяснение.
  4. Класс App — окно: кнопки, консоль, напоминания, подтверждения.

Запуск:  python git_pomoshnik.py
Нужно:   Python 3.8+ (tkinter входит в стандартную поставку) и установленный git.
"""

import os
import re
import json
import sys
import time
import queue
import random
import shlex
import shutil
import threading
import subprocess
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog


# =====================================================================
# 1. НАСТРОЙКИ И ТЕКСТЫ
# =====================================================================

APP_TITLE = "Гит-помощник"
SANDBOX_DIR = Path.home() / "GitTrainer"          # папка учебной песочницы
SETTINGS_FILE = Path.home() / ".git_pomoshnik.json"  # настройки программы (режим)
STATUS_REFRESH_MS = 15000                          # как часто обновлять напоминания
SAVE_REMINDER_MIN = 30                             # через сколько минут напомнить «сохранись»
NETWORK_TIMEOUT = 300                              # сек. на одну команду
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # чтобы не мигала консоль в Windows

# Коды из `git status`, которые означают конфликт
CONFLICT_CODES = {"UU", "AA", "DD", "AU", "UA", "DU", "UD"}

COLLEAGUE_TASKS = [
    "поправить опечатку на главной странице",
    "добавить кнопку «Назад»",
    "обновить телефон в контактах",
    "ускорить загрузку списка",
    "переименовать пункт меню «Профиль»",
    "добавить проверку пустого поля",
    "исправить дату в отчёте",
]

GLOSSARY = """☰ СЛОВАРИК — git простыми словами

• Репозиторий (проект) — папка с твоим кодом, за которой следит git и помнит все её версии.
• Коммит (сохранение) — снимок проекта в какой-то момент с подписью «что сделано». К любому снимку можно вернуться.
• Ветка — отдельная «копия-черновик» проекта для одной задачи. Ты меняешь свою ветку и никому не мешаешь.
• main (или master) — главная ветка. Там лежит рабочая версия, которую видят все. Напрямую в ней обычно не работают.
• Сервер (remote, origin) — общее хранилище проекта (GitHub, GitLab, сервер компании). Через него вы обмениваетесь работой.
• pull («забрать») — скачать с сервера изменения коллег и добавить к себе.
• push («отправить») — выложить свои сохранения на сервер.
• merge («слить», «влить») — объединить изменения из одной ветки в другую.
• Конфликт — ты и коллега поменяли одно и то же место в файле, и git просит человека решить, что оставить.
• stash («отложить») — убрать несохранённые изменения в «карман», чтобы позже вернуть.
• Pull Request / Merge Request — на сайте: «просьба влить мою ветку в главную». Коллега смотрит код и одобряет.
"""

WELCOME = """Привет! Я помогу работать с git так, как это делают в компании.

Как это устроено:
• Слева — кнопки обычного рабочего дня, по порядку: 1 → 7.
• Перед любым действием я простыми словами объясню, что будет, и покажу команды.
• Ничего не выполняется, пока ты не нажмёшь «Да, выполнить».
• Внизу — консоль: в ней видно, какие команды git на самом деле запускаются и что git ответил.
• Сверху — напоминания: что не забыть сделать.

С чего начать:
• «Песочница» — тренировочный проект прямо на твоём компьютере: свой «сервер» и «коллега Ваня». Сломать ничего нельзя.
• «Открыть проект…» — твой настоящий проект.
• «Словарик» — что значат слова вроде «коммит» и «ветка»."""

# Подсказки по ошибкам git: (кусочки текста ошибки, объяснение по-русски).
# Проверяются по порядку, срабатывает первая подходящая.
ERROR_HINTS = [
    (["CONFLICT", "Automatic merge failed", "fix conflicts", "unmerged files", "Unmerged paths"],
     "Конфликт: ты и кто-то ещё изменили одно и то же место в файле, и git не может сам решить, "
     "какая версия правильная.\n\n"
     "Что делать:\n"
     "1. Нажми «Открыть папку» и открой файл(ы) из списка в напоминаниях.\n"
     "2. Найди кусок между строками <<<<<<< и >>>>>>>. Над ======= — твоя версия, под ним — чужая.\n"
     "3. Оставь нужный текст (можно оба варианта), а строки <<<<<<<, ======= и >>>>>>> удали.\n"
     "4. Сохрани файл и нажми «4. Сохранить работу».\n\n"
     "Передумал? Кнопка «Отменить слияние» вернёт всё как было до него."),
    (["would be overwritten", "Please commit your changes or stash them", "Your local changes"],
     "Git остановился, чтобы не затереть твои несохранённые изменения.\n\n"
     "Сначала сохрани их («4. Сохранить работу») или отложи («Отложить изменения»), потом повтори."),
    (["Please tell me who you are", "Author identity unknown", "empty ident", "unable to auto-detect email"],
     "Git не знает, как тебя зовут: каждое сохранение подписывается именем автора.\n\n"
     "Нажми «☺ Сменить аккаунт…» (под строкой состояния) и введи имя и почту. Это делается один раз."),
    (["nothing to commit", "nothing added to commit"],
     "Сохранять нечего — изменений нет."),
    (["[rejected]", "fetch first", "non-fast-forward", "Updates were rejected"],
     "Сервер не принял отправку: там уже есть изменения коллег, которых нет у тебя.\n\n"
     "Нажми «1. Начать день», чтобы забрать их, а потом снова «5. Отправить»."),
    (["Authentication failed", "could not read Username", "terminal prompts disabled",
      "Permission denied (publickey)"],
     "Сервер не пустил: нужен вход (логин или токен).\n\n"
     "Один раз отправь изменения из обычной консоли или настрой вход через Git Credential Manager — "
     "после этого помощник сможет отправлять сам."),
    (["Repository not found", "does not exist or you do not have"],
     "Сервер не нашёл такой репозиторий. Проверь адрес — или у твоего аккаунта нет к нему доступа "
     "(возможно, ты вошёл под другим аккаунтом)."),
    (["not fully merged"],
     "Git не стал удалять ветку: в ней есть сохранения, которых нет в главной ветке — они бы пропали.\n\n"
     "Сначала влей её («6. Задача готова») или, если работа точно не нужна, удали через «✂ Удалить ветку…» "
     "и выбери «всё равно удалить»."),
    (["remote ref does not exist", "unable to delete"],
     "На сервере этой ветки уже нет — удалять там нечего."),
    (["already exists and is not an empty directory"],
     "Папка с таким именем уже есть и в ней что-то лежит. Выбери другое место или другое имя."),
    (["Could not resolve host", "unable to access", "Failed to connect", "Connection timed out",
      "Could not read from remote"],
     "Не удалось связаться с сервером. Проверь интернет и адрес сервера — возможно, сайт недоступен из твоей сети."),
    (["No configured push destination", "does not appear to be a git repository", "No such remote"],
     "У проекта не настроен сервер (remote) — отправлять и забирать некуда."),
    (["couldn't find remote ref"],
     "На сервере нет такой ветки (например, главную ветку туда ещё ни разу не отправляли)."),
    (["refusing to merge unrelated histories"],
     "Git отказался объединять два совсем не связанных проекта. Обычно это значит, что выбрана не та папка или не тот сервер."),
    (["already exists"],
     "Ветка (или папка) с таким именем уже есть. Придумай другое название."),
    (["problem with the editor"],
     "Эта команда хотела открыть текстовый редактор, а помощник так не умеет.\n"
     "Для сохранения используй кнопку «4. Сохранить работу» или добавь к команде -m \"текст\"."),
    (["No stash entries"], "Отложенных изменений нет."),
    (["There is no merge to abort", "MERGE_HEAD missing"], "Сейчас нет незаконченного слияния — отменять нечего."),
    (["has no upstream branch"],
     "Эта ветка ещё не связана с сервером. Нажми «5. Отправить» — помощник сам её свяжет."),
    (["is not a git command"], "У git нет такой команды. Проверь, правильно ли написано."),
    (["not a valid branch name"], "Недопустимое имя ветки."),
    (["did not match any", "unknown revision", "invalid reference"], "Такой ветки или файла нет."),
    (["not a git repository"], "Эта папка не является проектом git."),
    (["слишком долго"], "Команда работала слишком долго и была остановлена. Возможно, сервер не отвечает."),
]


def explain_error(output):
    """Переводит ответ git с ошибкой в понятное объяснение."""
    for patterns, hint in ERROR_HINTS:
        for p in patterns:
            if p.lower() in output.lower():
                return hint
    return ("Git сообщил об ошибке. Его ответ — в консоли ниже (обычно на английском).\n"
            "Если непонятно, скопируй текст ошибки и спроси у кого-нибудь.")


# =====================================================================
# 2. «ДВИЖОК» — работа с git без окон
# =====================================================================

def git_env():
    """Окружение для git: запрещаем ему что-либо спрашивать в невидимой консоли."""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"     # не спрашивать логин в консоли (окно входа Windows работает)
    env["GIT_MERGE_AUTOEDIT"] = "no"     # слияние не открывает редактор
    env["GIT_EDITOR"] = "false"          # любая попытка открыть редактор сразу даёт ошибку, а не зависание
    env["GIT_PAGER"] = "cat"             # без постраничного просмотра
    return env


def run_git(args, cwd, timeout=NETWORK_TIMEOUT, extra_env=None):
    """Запускает git с аргументами args в папке cwd. Возвращает (код, текст ответа)."""
    cmd = ["git", "-c", "core.quotepath=false"] + list(args)   # quotepath=false — русские имена файлов как есть
    env = git_env()
    env.update(extra_env or {})
    try:
        p = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=env,
            stdin=subprocess.DEVNULL, timeout=timeout, creationflags=CREATE_NO_WINDOW,
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, "Команда выполнялась слишком долго и была остановлена."
    except FileNotFoundError:
        return -1, "Git не найден на компьютере (или папка не существует)."
    except Exception as e:  # noqa
        return -1, f"Не удалось запустить git: {e}"


def parse_status(out):
    """Разбирает вывод `git status --porcelain=v1 -b` в словарь."""
    st = dict(branch="?", upstream=None, ahead=0, behind=0, files=[], conflicts=[],
              unborn=False, detached=False, upstream_gone=False)
    lines = out.splitlines()
    if lines and lines[0].startswith("## "):
        head = lines[0][3:]
        lines = lines[1:]
        for prefix in ("No commits yet on ", "Initial commit on "):
            if head.startswith(prefix):
                st["unborn"] = True
                st["branch"] = head[len(prefix):].strip()
                break
        else:
            if head.startswith("HEAD (no branch)"):
                st["detached"] = True
                st["branch"] = "(без ветки)"
            else:
                m = re.match(r"^(\S+?)(?:\.\.\.(\S+))?(?: \[(.*)\])?$", head)
                if m:
                    st["branch"] = m.group(1)
                    st["upstream"] = m.group(2)
                    info = m.group(3) or ""
                    a = re.search(r"ahead (\d+)", info)
                    b = re.search(r"behind (\d+)", info)
                    st["ahead"] = int(a.group(1)) if a else 0
                    st["behind"] = int(b.group(1)) if b else 0
                    if "gone" in info:
                        st["upstream_gone"] = True
                        st["upstream"] = None
    for ln in lines:
        if len(ln) < 4:
            continue
        xy, path = ln[:2], ln[3:]
        if " -> " in path:                      # переименование: берём новое имя
            path = path.split(" -> ", 1)[1]
        path = path.strip('"')
        st["files"].append((xy, path))
        if xy in CONFLICT_CODES:
            st["conflicts"].append(path)
    return st


# Готовые наборы правил для .gitignore — какие файлы git не должен сохранять
GITIGNORE_TEMPLATES = [
    ("Мусор Windows", ["Thumbs.db", "desktop.ini", "$RECYCLE.BIN/"]),
    ("Python", ["__pycache__/", "*.pyc", ".venv/", "venv/", "build/", "dist/", "*.egg-info/"]),
    ("Unity", ["[Ll]ibrary/", "[Tt]emp/", "[Oo]bj/", "[Bb]uild/", "[Bb]uilds/", "[Ll]ogs/",
               "[Uu]ser[Ss]ettings/", "[Mm]emoryCaptures/", ".vs/", "*.csproj", "*.sln", "*.user",
               "*.pidb", "*.booproj", "*.svd", "*.pdb", "*.mdb", "*.opendb", "*.VC.db", "crashlytics-build.properties"]),
    ("Visual Studio / C#", ["bin/", "obj/", ".vs/", "*.user", "*.suo"]),
    ("JavaScript / Node", ["node_modules/", "npm-debug.log*", "dist/"]),
    ("PHP", ["vendor/", ".env"]),
    ("Rust", ["target/"]),
    ("Редакторы (VS Code, IDEA)", [".vscode/", ".idea/"]),
    ("Логи и временные файлы", ["*.log", "*.tmp", "*.bak"]),
]

SCOPE_NAMES = {
    "local": "только для этого проекта",
    "worktree": "только для этого проекта",
    "global": "для всех проектов на компьютере",
    "system": "системная настройка",
    "command": "задано при запуске",
}


def read_identity(repo):
    """Кем подписываются сохранения: имя, почта и откуда взята настройка."""
    ident = {}
    for key in ("name", "email"):
        code, out = run_git(["config", "--show-scope", "--get", f"user.{key}"], repo, 15)
        if code == 0 and "\t" in out:
            scope, value = out.rstrip("\n").split("\t", 1)
            ident[key], ident[key + "_scope"] = value.strip(), scope
        else:
            ident[key], ident[key + "_scope"] = "", ""
    return ident


def parse_remote_url(url):
    """Разбирает адрес сервера: на каком сайте, чей репозиторий, указан ли логин в адресе.
    Примеры адресов:
        https://github.com/dan/proekt.git          — вход по логину/токену
        https://dan@github.com/dan/proekt.git      — логин прямо в адресе
        git@github.com:dan/proekt.git               — вход по SSH-ключу
        C:/Users/dan/GitTrainer/server.git          — папка на компьютере (песочница)
    """
    url = url.strip()
    info = dict(url=url, kind="other", host="", owner="", repo="", login="")
    m = re.match(r"^(https?)://(?:([^@/]+)@)?([^/:]+)(?::\d+)?/(.+?)/?$", url)
    if m:
        info.update(kind="https", host=m.group(3), login=(m.group(2) or "").split(":")[0])
        path = m.group(4)
    else:
        m = (re.match(r"^ssh://(?:([^@/]+)@)?([^/:]+)(?::\d+)?/(.+?)/?$", url) or
             re.match(r"^(?:([^@/]+)@)?([^/:]{2,}):(?!/)(.+?)/?$", url))   # git@host:owner/repo
        if not m:
            info["kind"] = "folder"
            return info
        info.update(kind="ssh", host=m.group(2))
        path = m.group(3)
    parts = [x for x in path.split("/") if x]
    if parts:
        info["repo"] = re.sub(r"\.git$", "", parts[-1])
        info["owner"] = "/".join(parts[:-1])
    return info


def gcm_github_accounts(cwd):
    """Какие аккаунты GitHub сохранены в Git Credential Manager (стандартно ставится с Git для Windows).
    Только читает список, ничего не спрашивает и окон входа не открывает."""
    if sys.platform != "win32":
        return []
    code, out = run_git(["credential-manager", "github", "list"], cwd, 10,
                        extra_env={"GCM_INTERACTIVE": "never"})
    if code != 0:
        return []
    return [ln.strip() for ln in out.splitlines()
            if ln.strip() and " " not in ln.strip() and len(ln.strip()) < 60]


TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def make_branch_slug(text):
    """«Исправить кнопку входа» -> «ispravit-knopku-vhoda» (имена веток лучше латиницей)."""
    out = []
    for ch in text.strip().lower():
        if ch in TRANSLIT:
            out.append(TRANSLIT[ch])
        elif ch.isascii() and ch.isalnum():
            out.append(ch)
        else:
            out.append("-")
    slug = re.sub(r"-+", "-", "".join(out)).strip("-")
    return slug[:40].strip("-")


def files_with_conflict_marks(repo, paths):
    """Какие из файлов ещё содержат метки конфликта <<<<<<< (значит, конфликт не разобран)."""
    bad = []
    for p in paths:
        try:
            text = (Path(repo) / p).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if re.search(r"^(<<<<<<< |>>>>>>> )", text, re.M):
            bad.append(p)
    return bad


def q(arg):
    """Показать аргумент так, чтобы команду можно было вставить в консоль как есть."""
    if not arg or re.search(r'[\s"&|<>^%()]', arg):
        return '"' + arg.replace('"', '\\"') + '"'
    return arg


def load_settings():
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(data):
    try:
        SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


# =====================================================================
# 3. ШАГ И ПЛАН
# =====================================================================

class Step:
    """Один шаг: либо команда git (args), либо действие на Python (func) с подписью show."""

    def __init__(self, args=None, cwd=None, func=None, show="", manual=None, timeout=NETWORK_TIMEOUT):
        self.timeout = timeout  # сколько секунд можно ждать (скачивание большого проекта — дольше)
        self.args = args
        self.cwd = cwd
        self.func = func
        self.show = show
        self.manual = manual    # команда консоли вместо func для ручного режима (None — «сделай руками»)

    def display(self):
        if self.args is not None:
            return "git " + " ".join(q(a) for a in self.args)
        return f"({self.show})"

    def shell(self):
        """Команда для ручного ввода. None — шаг нужно сделать руками (написан в show)."""
        if self.args is not None:
            return self.display()
        return self.manual


class Plan:
    """Действие целиком: заголовок, объяснение простыми словами, шаги и что сказать в конце.

    safe=True — действие только смотрит и ничего не меняет (тогда хватает нажатия кнопки).
    on_success(output) может вернуть текст, который заменит обычное сообщение об успехе.
    """

    def __init__(self, title, explain, steps, safe=False, success="", on_success=None, on_fail=None,
                 on_copy=None, manual_after="", preview=None):
        self.preview = preview            # preview(модель) — «проиграть» действие на копии графа, вернуть пояснения
        self.on_fail = on_fail
        self.on_copy = on_copy            # что сделать, когда показали команды в ручном режиме
        self.manual_after = manual_after  # подсказка «что сделать после ручного выполнения»
        self.title = title
        self.explain = explain
        self.steps = steps
        self.safe = safe
        self.success = success
        self.on_success = on_success


# =====================================================================
# 4. ТЁМНАЯ ТЕМА, ПОДСКАЗКИ, ПРОКРУТКА
# =====================================================================

# Тёмная тема. Различия — по яркости, а не по цвету (оттенки серого),
# чтобы всё читалось при любом цветовосприятии.
C = dict(
    bg="#1e1e1e",        # фон окна
    panel="#262626",     # фон поля «Что сейчас происходит»
    console="#121212",   # фон консоли
    field="#2d2d2d",     # поля ввода, списки
    fg="#e6e6e6",        # обычный текст
    dim="#8c8c8c",       # бледный текст (недоступные кнопки)
    btn="#353535",       # кнопка
    btn_hover="#4a4a4a", # кнопка под мышкой
    btn_press="#5c5c5c", # нажатая кнопка
    btn_off="#262626",   # недоступная кнопка
    border="#4a4a4a",    # рамки
    sel="#5e5e5e",       # выделенный текст
    tip_bg="#3a3a3a",    # фон всплывающей подсказки
    tip_border="#b0b0b0",
)


def dark_titlebar(win):
    """Тёмный заголовок окна в Windows 10/11 (на других системах ничего не делает)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        win.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        on = ctypes.c_int(1)
        for attr in (20, 19):    # 20 — Windows 11 и новые 10, 19 — старые сборки Windows 10
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(on), 4) == 0:
                break
    except Exception:  # noqa
        pass


class Tooltip:
    """Всплывающая подсказка: показывается, если задержать мышку на элементе."""

    DELAY_MS = 400

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        self.job = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _=None):
        self._cancel()
        self.job = self.widget.after(self.DELAY_MS, self._show)

    def _cancel(self):
        if self.job:
            self.widget.after_cancel(self.job)
            self.job = None

    def _show(self):
        self.job = None
        text = self.text() if callable(self.text) else self.text   # текст может вычисляться в момент показа
        if self.tip or not text:
            return
        x = self.widget.winfo_pointerx() + 16
        y = self.widget.winfo_pointery() + 14
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)            # без рамки и заголовка
        try:
            tip.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        frame = tk.Frame(tip, bg=C["tip_border"], padx=1, pady=1)   # светлая рамка вокруг
        frame.pack()
        tk.Label(frame, text=text, justify="left", bg=C["tip_bg"], fg=C["fg"],
                 font=("Segoe UI", 10), wraplength=380, padx=8, pady=5).pack()
        tip.update_idletasks()
        # не даём подсказке вылезти за край экрана
        w, h = tip.winfo_reqwidth(), tip.winfo_reqheight()
        x = min(x, tip.winfo_screenwidth() - w - 8)
        y = min(y, tip.winfo_screenheight() - h - 8)
        tip.wm_geometry(f"+{x}+{y}")
        self.tip = tip

    def _hide(self, _=None):
        self._cancel()
        if self.tip:
            self.tip.destroy()
            self.tip = None


class ScrollFrame(ttk.Frame):
    """Рамка с вертикальной прокруткой. Элементы кладутся в self.inner.
    Полоса прокрутки появляется, только когда всё не помещается."""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.canvas = tk.Canvas(self, bg=C["bg"], highlightthickness=0, bd=0)
        self.sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.sb.set)
        self.inner = ttk.Frame(self.canvas)
        self.win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.grid(row=0, column=0, sticky="ns")
        self.sb.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        self.rowconfigure(0, weight=1)
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", lambda e: self._update_bar())
        # колесо мыши: Windows/macOS — <MouseWheel>, Linux — кнопки 4/5
        self.bind_all("<MouseWheel>", self._on_wheel, add="+")
        self.bind_all("<Button-4>", self._on_wheel, add="+")
        self.bind_all("<Button-5>", self._on_wheel, add="+")

    def _on_inner(self, _=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=self.inner.winfo_reqwidth())
        self._update_bar()

    def _needs_scroll(self):
        return self.inner.winfo_reqheight() > self.canvas.winfo_height() + 1

    def _update_bar(self):
        if self._needs_scroll():
            self.sb.grid()
        else:
            self.sb.grid_remove()
            self.canvas.yview_moveto(0)

    def _on_wheel(self, e):
        # крутим, только если мышка над этой панелью (над кнопками внутри неё тоже)
        w = self.winfo_containing(e.x_root, e.y_root)
        me = str(self)
        if w is None or not (str(w) == me or str(w).startswith(me + ".")) or not self._needs_scroll():
            return None
        if getattr(e, "num", None) == 4:
            step = -1
        elif getattr(e, "num", None) == 5:
            step = 1
        else:
            step = -1 if e.delta > 0 else 1
        self.canvas.yview_scroll(step * 2, "units")
        return "break"


# Что делает каждая кнопка — показывается во всплывающей подсказке
BUTTON_HELP = {
    "start": "Утром, перед работой: забрать с сервера всё, что сделали коллеги (git pull).",
    "new": "Начать задачу в отдельной ветке-черновике, чтобы не мешать главной ветке (git switch -c).",
    "changed": "Посмотреть, какие файлы ты изменил с последнего сохранения. Ничего не меняет (git status).",
    "save": "Сделать снимок проекта с подписью «что сделано» — коммит (git add + git commit).",
    "push": "Выложить твои сохранения на сервер, чтобы их увидели коллеги (git push).",
    "done": "Задача закончена: перенести ветку задачи в главную ветку и отправить на сервер (git merge).",
    "end": "Перед уходом: сохранить всё несохранённое и отправить на сервер.",
    "history": "Список последних сохранений и схема веток. Ничего не меняет (git log).",
    "lines": "Показать изменения построчно: «+» — добавленная строка, «-» — удалённая. Ничего не меняет (git diff).",
    "show": "Выбрать сохранение из списка и посмотреть, что именно в нём поменялось. Ничего не меняет (git show).",
    "fetch": "Узнать, что нового на сервере, НЕ трогая твои файлы. После этого напоминания покажут, "
             "отправили ли коллеги что-то новое (git fetch).",
    "remote": "Подключить проект к серверу (GitHub, GitLab…) или поменять адрес сервера (git remote).",
    "delbranch": "Удалить ненужную ветку — у себя и, если хочешь, на сервере (git branch -d, git push --delete).",
    "restore": "Выбрать файлы и вернуть их как в последнем сохранении. Несохранённые изменения в них пропадут "
               "(git restore; новые файлы — git clean).",
    "uncommit": "Отменить последнее сохранение, если оно ещё НЕ отправлено. Изменения не пропадут — "
                "вернутся несохранёнными (git reset --soft).",
    "revert": "Отменить уже отправленное сохранение: git сделает новое сохранение «наоборот». "
              "История не переписывается — так безопасно для коллег (git revert).",
    "gitignore": "Указать, какие файлы git не должен сохранять: временные, сборку, личные настройки (.gitignore).",
    "clone": "Скачать проект с сервера (GitHub, GitLab…) в папку на компьютере и сразу открыть его (git clone).",
    "switch": "Перейти в другую ветку — файлы в папке станут как в ней (git switch).",
    "stash": "Убрать несохранённые изменения «в карман», чтобы проект стал чистым (git stash).",
    "unstash": "Достать из «кармана» отложенные изменения обратно (git stash pop).",
    "abort": "Прервать незаконченное слияние или отмену сохранения с конфликтом — всё вернётся как было "
             "(git merge --abort / git revert --abort).",
    "identity": "Указать или сменить аккаунт: имя и почту, которыми подписываются сохранения — "
                "для этого проекта или для всех (git config).",
    "colleague": "Только в песочнице: «коллега Ваня» меняет файл и отправляет на сервер — "
                 "чтобы потренироваться забирать чужие изменения и разбирать конфликты.",
}


# =====================================================================
# 5. ГРАФ ВЕТОК (как на learngitbranching.js.org)
# =====================================================================

class GitGraph:
    """Модель для рисования: сохранения (коммиты), ветки у тебя и ярлыки веток на сервере.
    Умеет «проигрывать» действия на копии — так строится предпросмотр «что будет»."""

    LIMIT = 150     # сколько последних сохранений загружать

    def __init__(self):
        self.commits = {}       # код -> {"parents": [...], "subject", "author", "date"}
        self.order = []         # от старых к новым (родители раньше детей)
        self.branches = {}      # твои ветки: имя -> код сохранения
        self.remotes = {}       # ветки на сервере: "origin/main" -> код
        self.head = None        # имя текущей ветки (None — «без ветки»)
        self.head_commit = None
        self.ghosts = set()     # сохранения, которых ещё нет (появятся после действия)
        self._n = 0

    @classmethod
    def load(cls, repo):
        g = cls()
        code, _ = run_git(["rev-parse", "--verify", "--quiet", "HEAD"], repo, 15)
        revs = (["HEAD"] if code == 0 else []) + ["--branches", "--remotes"]
        code, out = run_git(["log", *revs, "--topo-order", f"-n{cls.LIMIT}", "--date=format:%d.%m.%Y %H:%M",
                             "--format=%H%x09%P%x09%an%x09%ad%x09%s"], repo, 20)
        if code == 0:
            for ln in reversed(out.splitlines()):
                parts = ln.split("\t", 4)
                if len(parts) == 5:
                    h, parents, author, date, subj = parts
                    g.commits[h] = dict(parents=parents.split(), subject=subj, author=author, date=date)
                    g.order.append(h)
        _, refs = run_git(["for-each-ref", "--format=%(refname)%09%(objectname)",
                           "refs/heads", "refs/remotes"], repo, 15)
        for ln in refs.splitlines():
            if "\t" not in ln:
                continue
            name, h = ln.split("\t", 1)
            if name.startswith("refs/heads/"):
                g.branches[name[11:]] = h
            elif name.startswith("refs/remotes/") and not name.endswith("/HEAD"):
                g.remotes[name[13:]] = h
        code, br = run_git(["symbolic-ref", "-q", "--short", "HEAD"], repo, 15)
        g.head = br.strip() if code == 0 else None
        code, h = run_git(["rev-parse", "--verify", "--quiet", "HEAD"], repo, 15)
        g.head_commit = h.strip() if code == 0 else None
        return g

    def signature(self):
        return (tuple(self.order[-5:]), len(self.order), tuple(sorted(self.branches.items())),
                tuple(sorted(self.remotes.items())), self.head, self.head_commit)

    def copy(self):
        g = GitGraph()
        g.commits = {h: dict(c, parents=list(c["parents"])) for h, c in self.commits.items()}
        g.order = list(self.order)
        g.branches, g.remotes = dict(self.branches), dict(self.remotes)
        g.head, g.head_commit, g.ghosts, g._n = self.head, self.head_commit, set(self.ghosts), self._n
        return g

    # ---------------------------------------------- вопросы к модели
    def ancestors(self, h):
        seen, stack = set(), [h]
        while stack:
            c = stack.pop()
            if c in seen or c not in self.commits:
                continue
            seen.add(c)
            stack.extend(self.commits[c]["parents"])
        return seen

    def is_ancestor(self, a, b):
        """Есть ли сохранение a в истории b (то есть b уже «содержит» a)."""
        return a == b or a in self.ancestors(b)

    def reachable(self):
        res = set()
        for h in list(self.branches.values()) + list(self.remotes.values()) + [self.head_commit]:
            if h:
                res |= self.ancestors(h)
        return res

    def short(self, h):
        return "новое" if h in self.ghosts else h[:7]

    # ---------------------------------------------- «проигрывание» действий
    def ghost(self, parents, subject):
        self._n += 1
        h = f"~будет{self._n}"
        self.commits[h] = dict(parents=[p for p in parents if p], subject=subject, author="", date="")
        self.order.append(h)
        self.ghosts.add(h)
        return h

    def _move_head(self, h):
        self.head_commit = h
        if self.head:
            self.branches[self.head] = h

    def commit(self, subject, extra_parent=None):
        self._move_head(self.ghost([self.head_commit, extra_parent], subject))

    def merge_into_head(self, other, what):
        """Слияние other в текущую ветку. Возвращает пояснение для подписи."""
        cur = self.head_commit
        if not other or other not in self.commits:
            return f"{what}: по последним данным ничего нового."
        if not cur:
            self._move_head(other)
            return f"{what}: ветка просто встанет на то же сохранение."
        if self.is_ancestor(other, cur):
            return f"{what}: уже есть у тебя — ничего не изменится."
        if self.is_ancestor(cur, other):
            self._move_head(other)
            return f"{what}: «перемотка вперёд» — новых сохранений не будет, ярлык ветки просто сдвинется."
        self.commit(f"Слияние ({what})", other)
        return f"{what}: появится сохранение-слияние с двумя «родителями»."

    def switch(self, name):
        self.head = name
        self.head_commit = self.branches.get(name)

    def new_branch(self, name):
        self.branches[name] = self.head_commit
        self.head = name

    def push(self, remote, branch):
        if self.branches.get(branch):
            self.remotes[f"{remote}/{branch}"] = self.branches[branch]

    def delete(self, name):
        self.branches.pop(name, None)

    def delete_remote(self, name):
        self.remotes.pop(name, None)

    def reset_head(self, to):
        self._move_head(to)


class GraphView(ttk.Frame):
    """Рисует GitGraph на холсте. Всё различается яркостью, пунктиром и подписями — не цветом."""

    DX, DY, R = 66, 100, 13    # шаг по горизонтали, по вертикали, радиус кружка
    LEFT, TOP = 46, 56

    LEGEND = ("Что значат значки на схеме:\n"
              "● — сохранение (коммит). Наведи на кружок мышку — увидишь подпись, автора и дату.\n"
              "◎ двойное кольцо и ★ — ты сейчас здесь.\n"
              "Толстая светлая линия — история твоей текущей ветки.\n"
              "Светлый ярлык — твоя ветка;  ⇅ — ветка на сервере;  ⇅ рядом с именем — «совпадает с сервером».\n"
              "Слева — старое, справа — новое.\n\n"
              "Предпросмотр (когда нажал действие, но ещё не подтвердил):\n"
              "◌ + серый полый кружок пунктиром — новое сохранение, которое появится;\n"
              "серый ярлык пунктиром «встанет сюда» — куда сдвинется ветка;\n"
              "✕ — сохранение или ветка уйдёт.\n\n"
              "Колесо мыши — вверх/вниз, Shift + колесо — влево/вправо.")

    def __init__(self, parent, on_big=None):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        top = tk.Frame(self, bg=C["panel"])
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        top.columnconfigure(0, weight=1)
        self.banner = tk.Label(top, text="", justify="left", anchor="w", bg=C["panel"], fg=C["fg"],
                               font=("Segoe UI", 10), padx=8, pady=5)
        self.banner.grid(row=0, column=0, sticky="ew")
        legend = tk.Label(top, text="ⓘ Значки", bg=C["panel"], fg=C["fg"], font=("Segoe UI", 10, "underline"),
                          padx=6, cursor="question_arrow")
        legend.grid(row=0, column=1, sticky="ne", pady=6)
        Tooltip(legend, lambda: self.LEGEND)
        if on_big:
            big = ttk.Button(top, text="⛶ Большое окно", command=on_big)
            big.grid(row=0, column=2, sticky="ne", padx=6, pady=4)
            Tooltip(big, "Открыть схему веток в отдельном большом окне — его можно развернуть на весь экран. "
                         "Предпросмотр действий показывается и там.")
        self.bind("<Configure>", lambda e: self.banner.configure(wraplength=max(300, e.width - 250)))
        self.cv = tk.Canvas(self, bg=C["console"], highlightthickness=0)
        self.cv.grid(row=1, column=0, sticky="nsew")
        ys = ttk.Scrollbar(self, orient="vertical", command=self.cv.yview)
        xs = ttk.Scrollbar(self, orient="horizontal", command=self.cv.xview)
        ys.grid(row=1, column=1, sticky="ns")
        xs.grid(row=2, column=0, sticky="ew")
        self.cv.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        # колесо — вверх/вниз, Shift + колесо — влево/вправо
        self.cv.bind("<MouseWheel>", lambda e: self.cv.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.cv.bind("<Shift-MouseWheel>", lambda e: self.cv.xview_scroll(-3 if e.delta > 0 else 3, "units"))
        self.cv.bind("<Button-4>", lambda e: self.cv.yview_scroll(-1, "units"))
        self.cv.bind("<Button-5>", lambda e: self.cv.yview_scroll(1, "units"))
        self.cv.bind("<Shift-Button-4>", lambda e: self.cv.xview_scroll(-3, "units"))
        self.cv.bind("<Shift-Button-5>", lambda e: self.cv.xview_scroll(3, "units"))
        self.font = ("Segoe UI", 9)
        self.font_b = ("Segoe UI", 9, "bold")
        self.remote_name = "origin"
        self.model = None
        self._pos = {}

    # ---------------------------------------------- раскладка
    def layout(self, m, main):
        """Где рисовать каждое сохранение: x — «поколение» (дальше от начала — правее),
        y — дорожка ветки (главная ветка — верхняя)."""
        gen = {}
        for h in m.order:
            ps = [p for p in m.commits[h]["parents"] if p in gen]
            gen[h] = max(gen[p] for p in ps) + 1 if ps else 0
        lane_of, spans = {}, []

        def alloc(x0, x1):
            for i, sp in enumerate(spans):
                if all(x1 + 1 < a or x0 - 1 > b for a, b in sp):
                    sp.append((x0, x1))
                    return i
            spans.append([(x0, x1)])
            return len(spans) - 1

        def chain(tip):
            seq, c = [], tip
            while c and c in m.commits and c not in lane_of and c not in seq:
                seq.append(c)
                ps = m.commits[c]["parents"]
                c = ps[0] if ps else None
            if not seq:
                return
            xs = [gen[x] for x in seq]
            x0 = min(xs) - (1 if (c and c in gen) else 0)    # место под изгиб к ветке-родителю
            lane = alloc(x0, max(xs))
            for x in seq:
                lane_of[x] = lane

        tips = []
        if main in m.branches:
            tips.append(m.branches[main])
        if f"{self.remote_name}/{main}" in m.remotes:
            tips.append(m.remotes[f"{self.remote_name}/{main}"])
        tips.append(m.head_commit)
        tips += [m.branches[b] for b in sorted(m.branches)] + [m.remotes[r] for r in sorted(m.remotes)]
        for t in tips:
            chain(t)
        for h in reversed(m.order):
            chain(h)
        return {h: (self.LEFT + gen[h] * self.DX, self.TOP + lane_of[h] * self.DY) for h in m.order}

    # ---------------------------------------------- отрисовка
    def show(self, model, main="main", remote="origin", preview=None, title="", notes=()):
        self.model = model
        self.remote_name = remote or "origin"
        cv = self.cv
        cv.delete("all")
        if model is None or not model.order:
            cv.create_text(20, 20, anchor="nw", fill=C["fg"], font=("Segoe UI", 11),
                           text="Сохранений пока нет — граф появится после первого сохранения.")
            self.banner.configure(text="Граф веток: кружки — сохранения, ярлыки — ветки." if model else
                                  "Открой проект, чтобы увидеть граф.")
            cv.configure(scrollregion=(0, 0, 10, 10))
            return
        m = preview or model
        pos = self._pos = self.layout(m, main)
        reach_now = model.reachable()
        reach_after = preview.reachable() if preview else reach_now
        dim = preview is not None
        # «дорожка» текущей ветки: от сохранения, где ты сейчас, назад по первым родителям — рисуем толще
        cur_line, c = set(), model.head_commit
        while c and c in model.commits and c not in cur_line:
            cur_line.add(c)
            ps = model.commits[c]["parents"]
            c = ps[0] if ps else None
        GHOST = "#8e8e8e"       # серый — то, что только БУДЕТ после действия

        # линии от сохранения к «родителям»
        for h in m.order:
            cx, cy = pos[h]
            for i, p in enumerate(m.commits[h]["parents"]):
                ghost = h in m.ghosts
                mine = (not ghost) and i == 0 and h in cur_line and p in cur_line
                kw = dict(width=2 if ghost else (4 if mine else 2),
                          fill=GHOST if ghost else ("#f0f0f0" if mine else "#8c8c8c"),
                          arrow="last", arrowshape=(8, 9, 4))
                if ghost:
                    kw["dash"] = (5, 3)
                if p not in pos:     # родитель за краем загруженной истории
                    cv.create_line(cx - self.R, cy, cx - self.DX * 0.6, cy, **kw)
                    continue
                px, py = pos[p]
                if py == cy:
                    pts = [cx - self.R, cy, px + self.R, py]
                elif i == 0:         # ветка «отошла» от родителя — изгиб у родителя
                    pts = [cx - self.R, cy, px + self.DX * 0.55, cy, px + self.R * 0.7, py]
                else:                # слияние — изгиб у сохранения-слияния
                    pts = [cx - self.R * 0.7, cy, cx - self.DX * 0.55, py, px + self.R, py]
                cv.create_line(*pts, smooth=True, **kw)

        # кружки-сохранения
        for h in m.order:
            x, y = pos[h]
            r = self.R
            tag = f"c{len(h)}_{h[-12:]}"
            if h in m.ghosts:
                # будущее сохранение: полый серый кружок пунктиром с «+»
                cv.create_oval(x - r - 2, y - r - 2, x + r + 2, y + r + 2, outline=GHOST, width=2,
                               dash=(4, 3), fill=C["console"], tags=tag)
                cv.create_text(x, y, text="+", fill="#b4b4b4", font=("Segoe UI", 12, "bold"), tags=tag)
            elif dim and h in reach_now and h not in reach_after:
                # уйдёт из истории ветки
                cv.create_oval(x - r, y - r, x + r, y + r, outline="#9a9a9a", width=2, dash=(3, 3),
                               fill=C["console"], tags=tag)
                cv.create_text(x, y, text="✕", fill="#e0e0e0", font=self.font_b, tags=tag)
            else:
                is_head = h == model.head_commit
                if is_head:     # где ты сейчас — двойное кольцо
                    cv.create_oval(x - r - 5, y - r - 5, x + r + 5, y + r + 5, outline="#ffffff", width=2, tags=tag)
                cv.create_oval(x - r, y - r, x + r, y + r, outline="#ffffff" if is_head else "#d0d0d0",
                               width=2, fill="#ececec" if (is_head or h in cur_line) else "#9e9e9e", tags=tag)
            cv.tag_bind(tag, "<Enter>", lambda e, h=h, m=m: self._tip(h, m))
            cv.tag_bind(tag, "<Leave>", lambda e: cv.delete("tip"))

        # ярлыки веток
        placed = []     # рамки уже нарисованных ярлыков — новые не должны на них налезать
        circles = [(px - self.R - 1, py - self.R - 1, px + self.R + 1, py + self.R + 1) for px, py in pos.values()]

        def hits(b):
            return any(b[0] < o[2] + 3 and b[2] + 3 > o[0] and b[1] < o[3] + 2 and b[3] + 2 > o[1]
                       for o in placed + circles)

        def label(h, text, style, side):
            if h not in pos:
                return
            x, y = pos[h]
            ty = y - self.R - 14 if side == "up" else y + self.R + 14
            fnt = self.font_b if style in ("cur", "new", "newcur") else self.font
            fg = {"cur": "#111111", "local": "#111111", "remote": "#d8d8d8", "dim": "#bdbdbd",
                  "new": "#a8a8a8", "newcur": "#b8b8b8", "old": "#9a9a9a", "gone": "#e0e0e0"}[style]
            t = cv.create_text(x, ty, text=text, fill=fg, font=fnt)
            x0, y0, x1, y1 = cv.bbox(t)
            box = dict(cur=dict(fill="#ffffff", outline="#ffffff", width=2),
                       local=dict(fill="#b8b8b8", outline="#b8b8b8", width=1),
                       remote=dict(fill=C["console"], outline="#9a9a9a", width=1),
                       dim=dict(fill="#3a3a3a", outline="#5a5a5a", width=1),
                       new=dict(fill=C["console"], outline=GHOST, width=1, dash=(4, 2)),
                       newcur=dict(fill=C["console"], outline=GHOST, width=2, dash=(4, 2)),
                       old=dict(fill=C["console"], outline="#5a5a5a", width=1, dash=(2, 3)),
                       gone=dict(fill=C["console"], outline="#d0d0d0", width=1, dash=(2, 3)))[style]
            # если место занято — ищем ближайшее свободное: рядом сбоку, потом на ступеньку дальше.
            # Если ярлык пришлось сдвинуть, рисуем к кружку тонкую «ножку», чтобы было видно, чей он.
            step = -21 if side == "up" else 21
            bw, bh = x1 - x0, y1 - y0
            chosen = None
            for k in (0, 1, 2):
                for dx in (0, 1, -1):
                    shift = dx * (bw / 2 + 22) if dx else 0
                    cand = (x0 - 6 + shift, y0 - 2 + k * step, x1 + 6 + shift, y1 + 2 + k * step)
                    if not hits(cand):
                        chosen = (shift, k * step)
                        break
                if chosen:
                    break
            if not chosen:       # совсем тесно — ставим ещё дальше по вертикали
                k = 3
                while hits((x0 - 6, y0 - 2 + k * step, x1 + 6, y1 + 2 + k * step)) and k < 10:
                    k += 1
                chosen = (0, k * step)
            cv.move(t, *chosen)
            x0, y0, x1, y1 = cv.bbox(t)
            b = (x0 - 6, y0 - 2, x1 + 6, y1 + 2)
            placed.append(b)
            rect = cv.create_rectangle(*b, **box)
            cv.tag_lower(rect, t)
            if chosen != (0, 0):
                lx = min(max(x, b[0] + 6), b[2] - 6)        # ближайшая к кружку точка ярлыка
                ly = b[3] if side == "up" else b[1]
                cy0 = y - self.R - 1 if side == "up" else y + self.R + 1
                leg = cv.create_line(x, cy0, lx, ly, fill=box["outline"], width=1, dash=box.get("dash"))
                cv.tag_lower(leg, rect)

        cur_after = m.head
        rp = self.remote_name + "/"
        if not dim:
            for name, h in sorted(m.branches.items(), key=lambda kv: kv[0] != main):
                synced = m.remotes.get(rp + name) == h
                star = "★ " if name == m.head else ""
                label(h, f"{star}{name}" + ("  ⇅" if synced else ""), "cur" if name == m.head else "local", "up")
            for name, h in sorted(m.remotes.items()):
                local = name[len(rp):] if name.startswith(rp) else None
                if local and m.branches.get(local) == h:
                    continue        # совпадает с твоей веткой — уже показано значком ⇅
                label(h, f"⇅ {name}", "remote", "down")
            if m.head is None and m.head_commit:
                label(m.head_commit, "★ без ветки", "cur", "up")
        else:
            for kind, before, after, side in (("b", model.branches, preview.branches, "up"),
                                              ("r", model.remotes, preview.remotes, "down")):
                for name in sorted(set(before) | set(after), key=lambda n: n != main):
                    b, a = before.get(name), after.get(name)
                    plain = name if kind == "b" else f"⇅ {name}"
                    now_style = ("cur" if name == model.head else "local") if kind == "b" else "remote"
                    now_text = ("★ " if kind == "b" and name == model.head else "") + plain
                    if a and a == b:
                        label(a, now_text, now_style, side)          # ярлык остаётся на месте
                        if kind == "b" and name == cur_after and model.head != cur_after:
                            label(a, "★ будешь здесь", "newcur", side)
                    elif a:
                        if b:
                            label(b, now_text, now_style, side)      # где ярлык сейчас
                        star = "★ " if kind == "b" and name == cur_after else ""
                        label(a, f"{star}{plain}  ({'новая — будет' if not b else 'встанет сюда'})",
                              "newcur" if star else "new", side)
                    elif b:
                        label(b, f"✕ {plain} (удалится)", "gone", side)

        x0, y0, x1, y1 = cv.bbox("all")
        sx0, sy0, sx1, sy1 = min(0, x0 - 20), min(0, y0 - 20), x1 + 40, y1 + 30
        cv.configure(scrollregion=(sx0, sy0, sx1, sy1))
        # прокручиваем так, чтобы было видно, где ты сейчас (в предпросмотре — новое сохранение)
        gone = [h for h in m.order if dim and h in reach_now and h not in reach_after]
        focus = (next((h for h in reversed(m.order) if h in m.ghosts), None) or (gone[-1] if gone else None)
                 or m.head_commit or model.head_commit)
        fx, fy = pos.get(focus, (x1, y0))

        def scroll_to():
            w, hgt = max(cv.winfo_width(), 50), max(cv.winfo_height(), 50)
            cv.xview_moveto(max(0.0, (fx - w * 0.65 - sx0) / max(1, sx1 - sx0)))
            cv.yview_moveto(max(0.0, (fy - hgt * 0.45 - sy0) / max(1, sy1 - sy0)))
        self.after_idle(scroll_to)

        if dim:
            text = (f"ПРЕДПРОСМОТР «{title}»:  обычное — как сейчас,  СЕРОЕ ПУНКТИРОМ — что будет,  ✕ — уйдёт.")
            if notes:
                text += "\n" + "\n".join("• " + n for n in notes)
        else:
            text = ("Схема веток: ★ — ты здесь, толстая линия — твоя ветка.  "
                    "Нажми действие — серым пунктиром появится, что будет.")
            if notes:
                text += "\n" + "\n".join("• " + n for n in notes)
        self.banner.configure(text=text)

    def _tip(self, h, m):
        """Подсказка у кружка: подпись сохранения, автор, дата."""
        self.cv.delete("tip")
        c = m.commits.get(h)
        if not c:
            return
        if h not in self._pos:
            return
        x, y = self._pos[h]
        text = ("Будет новое сохранение:\n" + c["subject"]) if h in m.ghosts else \
            f"{h[:7]}  ·  {c['date']}  ·  {c['author']}\n{c['subject']}"
        t = self.cv.create_text(x + 18, y + 18, text=text, anchor="nw", fill=C["fg"], font=self.font,
                                width=360, tags="tip")
        bx = self.cv.bbox(t)
        r = self.cv.create_rectangle(bx[0] - 6, bx[1] - 4, bx[2] + 6, bx[3] + 4, fill=C["tip_bg"],
                                     outline=C["tip_border"], tags="tip")
        self.cv.tag_lower(r, t)


# =====================================================================
# 6. ОКНО ПРОГРАММЫ
# =====================================================================

class App:
    def __init__(self, root):
        self.root = root
        self.repo = None            # Path к открытому проекту
        self.git_dir = None         # Path к папке .git
        self.is_sandbox = False
        self.day_started = False    # нажимал ли «Начать день» с момента открытия проекта
        self.busy = False           # сейчас выполняются команды
        self.pending = None         # план, ждущий подтверждения
        self.state = {}             # последнее прочитанное состояние проекта
        self.q = queue.Queue()      # сообщения из фонового потока в окно
        self.git_ok = shutil.which("git") is not None
        self.gcm_cache = {}         # аккаунты из менеджера паролей Windows (читаются один раз)
        self.gmodel = None          # граф веток (последнее прочитанное состояние)
        self.settings = load_settings()
        self.manual_var = tk.BooleanVar(value=bool(self.settings.get("manual")))

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(80, self._poll_queue)
        self.root.after(STATUS_REFRESH_MS, self._auto_refresh)

        if self.git_ok:
            code, ver = run_git(["--version"], Path.home(), 15)
            self.console_note(ver.strip() if code == 0 else "git найден, но не отвечает")
            self.set_explain(WELCOME)
        else:
            self.set_explain(
                "✖ Git не установлен (или не найден).\n\n"
                "Помощнику нужен сам git — он только подсказывает и запускает его команды.\n"
                "Установи Git для Windows (файл вида Git-…-64-bit.exe) или в консоли выполни:\n"
                "    winget install Git.Git\n"
                "После установки перезапусти помощник.")
        self.refresh_status()

    # ---------------------------------------------------------------- UI
    def _apply_dark_theme(self, ui_font):
        """Тёмная тема для всех элементов окна. Тема clam — единственная встроенная,
        у которой можно поменять цвета кнопок и в Windows."""
        r = self.root
        r.configure(bg=C["bg"])
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=C["bg"], foreground=C["fg"], fieldbackground=C["field"],
                        bordercolor=C["border"], lightcolor=C["border"], darkcolor=C["border"],
                        troughcolor=C["bg"], selectbackground=C["sel"], selectforeground=C["fg"],
                        insertcolor=C["fg"], focuscolor=C["fg"], font=ui_font)
        style.map(".", foreground=[("disabled", C["dim"])])
        btn_map = dict(background=[("disabled", C["btn_off"]), ("pressed", C["btn_press"]),
                                   ("active", C["btn_hover"])],
                       foreground=[("disabled", C["dim"])],
                       bordercolor=[("disabled", C["btn_off"]), ("active", C["fg"])])
        style.configure("TButton", background=C["btn"], foreground=C["fg"], padding=(8, 5), focusthickness=1)
        style.map("TButton", **btn_map)
        style.configure("Act.TButton", background=C["btn"], padding=(10, 7), anchor="w")
        style.map("Act.TButton", **btn_map)
        # «Да, выполнить» — светлая кнопка с тёмным текстом: видна сразу, по яркости, а не по цвету
        style.configure("Yes.TButton", background="#d8d8d8", foreground="#111111",
                        font=(ui_font[0], ui_font[1], "bold"), padding=(10, 5))
        style.map("Yes.TButton", background=[("disabled", C["btn_off"]), ("pressed", "#a8a8a8"),
                                             ("active", "#f2f2f2")],
                  foreground=[("disabled", C["dim"])])
        style.configure("TLabel", background=C["bg"], foreground=C["fg"], font=ui_font)
        style.configure("Status.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("TFrame", background=C["bg"])
        style.configure("TLabelframe", background=C["bg"], bordercolor=C["border"])
        style.configure("TLabelframe.Label", background=C["bg"], foreground=C["fg"],
                        font=("Segoe UI", 11, "bold"))
        style.configure("TCheckbutton", background=C["bg"], foreground=C["fg"],
                        indicatorbackground=C["field"], indicatorforeground=C["fg"])
        style.map("TCheckbutton", background=[("active", C["bg"])],
                  indicatorbackground=[("pressed", C["btn_press"]), ("active", C["btn_hover"])])
        style.configure("TEntry", fieldbackground=C["field"], foreground=C["fg"], insertcolor=C["fg"])
        # вкладки: выбранная — светлая с тёмным текстом (видно по яркости)
        style.configure("TNotebook", background=C["bg"], bordercolor=C["border"], tabmargins=(0, 2, 0, 0))
        style.configure("TNotebook.Tab", background=C["btn"], foreground=C["fg"], padding=(12, 4),
                        bordercolor=C["border"], font=(ui_font[0], ui_font[1]))
        style.map("TNotebook.Tab", background=[("selected", "#d8d8d8"), ("active", C["btn_hover"])],
                  foreground=[("selected", "#111111")], expand=[("selected", (1, 2, 1, 0))])
        style.configure("Vertical.TScrollbar", background=C["btn"], troughcolor=C["bg"],
                        arrowcolor=C["fg"], bordercolor=C["border"])
        style.map("Vertical.TScrollbar", background=[("active", C["btn_hover"])])
        style.configure("Horizontal.TScrollbar", background=C["btn"], troughcolor=C["bg"],
                        arrowcolor=C["fg"], bordercolor=C["border"])
        style.map("Horizontal.TScrollbar", background=[("active", C["btn_hover"])])
        # разделитель между объяснением и графом — потянуть мышкой
        style.configure("TPanedwindow", background=C["bg"])
        style.configure("Sash", sashthickness=8, gripcount=12, background=C["btn"], lightcolor=C["border"],
                        bordercolor=C["border"])
        # обычные (не ttk) элементы: окна ввода текста, меню, списки
        for key, val in (("*Background", C["bg"]), ("*Foreground", C["fg"]),
                         ("*activeBackground", C["btn_hover"]), ("*activeForeground", C["fg"]),
                         ("*selectBackground", C["sel"]), ("*selectForeground", C["fg"]),
                         ("*insertBackground", C["fg"]), ("*highlightBackground", C["bg"]),
                         ("*Entry.Background", C["field"]), ("*Listbox.Background", C["field"]),
                         ("*Button.Background", C["btn"]), ("*Menu.Background", C["field"]),
                         ("*Menu.activeBackground", C["sel"])):
            r.option_add(key, val)

    def _build_ui(self):
        r = self.root
        r.title(APP_TITLE)
        # размер окна — под экран: не больше 1200×800 и не больше самого экрана (с запасом на панель задач)
        w = min(1200, r.winfo_screenwidth() - 40)
        h = min(800, r.winfo_screenheight() - 90)
        r.geometry(f"{w}x{h}+{max(0, (r.winfo_screenwidth() - w) // 2)}+10")
        r.minsize(min(900, w), min(560, h))

        ui_font = ("Segoe UI", 11)
        self.mono = ("Consolas", 11)
        self._apply_dark_theme(ui_font)

        r.columnconfigure(0, weight=1)
        r.rowconfigure(3, weight=1)

        # --- верхняя строка: проект и кнопки
        top = ttk.Frame(r, padding=(10, 8, 10, 0))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        self.repo_label = ttk.Label(top, text="Проект: не выбран", font=("Segoe UI", 11, "bold"))
        self.repo_label.grid(row=0, column=0, sticky="w")
        # кнопки — отдельной строкой под названием проекта, чтобы длинный путь их не выталкивал за край
        bar = ttk.Frame(top)
        bar.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.btn_open = ttk.Button(bar, text="Открыть проект…", command=self.choose_repo)
        self.btn_clone = ttk.Button(bar, text="⤓ Скачать с сервера…", command=self.act_clone)
        self.btn_sandbox = ttk.Button(bar, text="Песочница (тренировка)", command=self.act_sandbox)
        self.btn_folder = ttk.Button(bar, text="Открыть папку", command=self.open_folder)
        self.btn_gloss = ttk.Button(bar, text="Словарик", command=lambda: self.set_explain(GLOSSARY))
        self.btn_refresh = ttk.Button(bar, text="↻ Обновить", command=self.manual_refresh)
        for i, b in enumerate((self.btn_open, self.btn_clone, self.btn_sandbox, self.btn_folder,
                               self.btn_refresh, self.btn_gloss), start=1):
            b.grid(row=0, column=i, padx=(0 if i == 1 else 6, 0))
        self.chk_manual = ttk.Checkbutton(
            top, text="✋ Только подсказывать — я сам ввожу команды (программа ничего не выполняет)",
            variable=self.manual_var, command=self.on_mode_change)
        self.chk_manual.grid(row=2, column=0, sticky="w", pady=(6, 0))
        Tooltip(self.repo_label, lambda: str(self.repo) if self.repo else "Проект не выбран")
        for w, t in ((self.btn_open, "Выбрать папку своего проекта на компьютере."),
                     (self.btn_clone, BUTTON_HELP["clone"]),
                     (self.btn_sandbox, "Тренировочный проект со своим «сервером» и коллегой Ваней. "
                                        "Сломать ничего нельзя."),
                     (self.btn_folder, "Открыть папку проекта в Проводнике."),
                     (self.btn_refresh, "Перечитать состояние проекта и обновить напоминания "
                                        "(например, после ручного ввода команд)."),
                     (self.btn_gloss, "Что значат слова git: коммит, ветка, pull, push и т.д."),
                     (self.chk_manual, "Включено: кнопки ничего не выполняют, а только показывают команды "
                                       "для ручного ввода с кнопками «Скопировать».\n"
                                       "Выключено: команды выполняются после «Да, выполнить».")):
            Tooltip(w, t)

        # --- строка состояния
        info = ttk.Frame(r, padding=(10, 6, 10, 2))
        info.grid(row=1, column=0, sticky="ew")
        info.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(info, text="", style="Status.TLabel")
        self.status_label.grid(row=0, column=0, columnspan=2, sticky="w")
        # к какому аккаунту привязана работа: кто подписывает сохранения и куда/под кем идёт отправка
        self.account_label = ttk.Label(info, text="", justify="left")
        self.account_label.grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.btn_account = ttk.Button(info, text="☺ Сменить аккаунт…", command=self.act_identity)
        self.btn_account.grid(row=1, column=1, sticky="ne", padx=(8, 0), pady=(3, 0))
        info.bind("<Configure>", lambda e: self.account_label.configure(wraplength=max(300, e.width - 220)))
        Tooltip(self.account_label,
                "Подпись — имя и почта, которыми git подписывает каждое сохранение (коммит). "
                "GitHub/GitLab по почте понимают, чьё это сохранение, поэтому почта должна совпадать "
                "с почтой в аккаунте на сайте.\n\n"
                "Сервер — куда уходит «Отправить»: сайт, чей это репозиторий и под каким логином "
                "ты входишь. Логин хранится отдельно от подписи: у git это две разные вещи.")
        Tooltip(self.btn_account, "Поменять имя и почту, которыми подписываются сохранения — "
                                  "для этого проекта или для всех проектов на компьютере.")

        # --- напоминания
        rem = ttk.LabelFrame(r, text=" Напоминания ", padding=(8, 4))
        rem.grid(row=2, column=0, sticky="ew", padx=10, pady=(2, 6))
        rem.columnconfigure(0, weight=1)
        self.rem_label = tk.Label(rem, text="", justify="left", anchor="w", font=("Segoe UI", 11),
                                  bg=C["bg"], fg=C["fg"])
        self.rem_label.grid(row=0, column=0, sticky="ew")
        rem.bind("<Configure>", lambda e: self.rem_label.configure(wraplength=max(200, e.width - 30)))

        # --- основная часть: слева кнопки, справа объяснение и консоль
        main = ttk.Frame(r, padding=(10, 0, 10, 10))
        main.grid(row=3, column=0, sticky="nsew")
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # левая панель прокручивается колесом мыши, если кнопки не помещаются по высоте
        left_scroll = self.left_scroll = ScrollFrame(main)
        left_scroll.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        left = left_scroll.inner

        day = ttk.LabelFrame(left, text=" Рабочий день ", padding=6)
        day.pack(fill="x")
        self.repo_buttons = []

        def add(parent, text, cmd, help_key):
            b = ttk.Button(parent, text=text, style="Act.TButton", command=cmd, width=30)
            b.pack(fill="x", pady=2)
            # подсказка при наведении: полное название кнопки + что она делает
            Tooltip(b, f"{text}\n\n{BUTTON_HELP[help_key]}")
            self.repo_buttons.append(b)
            return b

        add(day, "1. ☀ Начать день (забрать свежее)", self.act_start_day, "start")
        add(day, "2. ✚ Новая задача (своя ветка)", self.act_new_task, "new")
        add(day, "3. ? Что я изменил", self.act_what_changed, "changed")
        add(day, "4. ✎ Сохранить работу", self.act_save, "save")
        add(day, "5. ⇧ Отправить на сервер", self.act_push, "push")
        add(day, "6. ✔ Задача готова (влить в main)", self.act_task_done, "done")
        add(day, "7. ☾ Закончить день", self.act_end_day, "end")

        def group(title):
            g = ttk.LabelFrame(left, text=f" {title} ", padding=6)
            g.pack(fill="x", pady=(8, 0))
            return g

        look = group("Посмотреть")
        add(look, "☰ История сохранений", self.act_history, "history")
        add(look, "± Изменения построчно", self.act_diff_lines, "lines")
        add(look, "◎ Подробно о сохранении…", self.act_show_commit, "show")

        srv = group("Сервер")
        add(srv, "☁ Проверить сервер (без скачивания)", self.act_fetch, "fetch")
        add(srv, "⇅ Подключить / сменить сервер…", self.act_remote, "remote")

        br = group("Ветки")
        add(br, "⇄ Перейти в другую ветку", self.act_switch_branch, "switch")
        add(br, "✂ Удалить ветку…", self.act_delete_branch, "delbranch")

        undo = group("Отменить")
        add(undo, "↺ Вернуть файл как было…", self.act_restore, "restore")
        add(undo, "↶ Отменить последнее сохранение", self.act_uncommit, "uncommit")
        add(undo, "⟲ Отменить отправленное сохранение…", self.act_revert, "revert")
        add(undo, "⨯ Прервать слияние / отмену", self.act_merge_abort, "abort")
        add(undo, "⤓ Отложить изменения", self.act_stash, "stash")
        add(undo, "⤒ Вернуть отложенное", self.act_unstash, "unstash")

        conf = group("Настройки")
        add(conf, "☺ Аккаунт: кто я (имя и почта)", self.act_identity, "identity")
        add(conf, "⊘ Не сохранять файлы (.gitignore)…", self.act_gitignore, "gitignore")

        sand = ttk.LabelFrame(left, text=" Песочница ", padding=6)
        sand.pack(fill="x", pady=(8, 0))
        self.btn_colleague = ttk.Button(sand, text="☻ Коллега Ваня поработал", style="Act.TButton",
                                        command=self.act_colleague, width=30)
        self.btn_colleague.pack(fill="x", pady=2)
        Tooltip(self.btn_colleague, "☻ Коллега Ваня поработал\n\n" + BUTTON_HELP["colleague"])

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        # объяснение сверху, граф/консоль снизу; границу между ними можно тянуть мышкой
        paned = ttk.PanedWindow(right, orient="vertical")
        paned.grid(row=0, column=0, sticky="nsew")
        self.paned = paned

        def place_sash(_=None):
            # объяснению — около 40% высоты, но не меньше 150 точек, чтобы его всегда было видно
            hgt = paned.winfo_height()
            if hgt > 200 and not getattr(self, "_sash_placed", False):
                paned.sashpos(0, max(150, int(hgt * 0.4)))
                self._sash_placed = True
        paned.bind("<Configure>", place_sash, add="+")

        # объяснение + подтверждение
        exp = ttk.LabelFrame(paned, text=" Что сейчас происходит ", padding=6)
        paned.add(exp, weight=2)
        exp.columnconfigure(0, weight=1)
        exp.rowconfigure(0, weight=1)
        self.explain = tk.Text(exp, height=6, width=30, wrap="word", font=("Segoe UI", 11),
                               relief="flat", padx=6, pady=4, cursor="arrow",
                               bg=C["panel"], fg=C["fg"], insertbackground=C["fg"],
                               selectbackground=C["sel"], selectforeground=C["fg"],
                               highlightthickness=0)
        self.explain.grid(row=0, column=0, sticky="nsew")
        exp_sb = ttk.Scrollbar(exp, command=self.explain.yview)
        exp_sb.grid(row=0, column=1, sticky="ns")
        self.explain.configure(yscrollcommand=exp_sb.set)
        self.explain.tag_configure("b", font=("Segoe UI", 11, "bold"))

        cf = ttk.Frame(exp)
        cf.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.yes_btn = ttk.Button(cf, text="✔ Да, выполнить", command=self.confirm_yes, style="Yes.TButton")
        self.no_btn = ttk.Button(cf, text="✖ Отмена", command=self.confirm_no)
        self.yes_btn.pack(side="left")
        self.no_btn.pack(side="left", padx=(8, 0))
        self.yes_btn.state(["disabled"])
        self.no_btn.state(["disabled"])
        self.copy_frame = ttk.Frame(cf)          # сюда в ручном режиме встают кнопки «Скопировать»
        self.copy_frame.pack(side="left", padx=(16, 0))
        self.copy_msg = ttk.Label(exp, text="", font=("Segoe UI", 10, "bold"))
        self.copy_msg.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # вкладки: граф веток и консоль
        self.tabs = ttk.Notebook(paned)
        paned.add(self.tabs, weight=3)
        self.graph = GraphView(self.tabs, on_big=self.open_graph_window)
        self.graph_views = [self.graph]      # главная схема + большое окно, если открыто
        self._graph_args = ((None,), {})
        self.tabs.add(self.graph, text="◎ Граф веток")
        con = ttk.Frame(self.tabs, padding=4)
        self.tabs.add(con, text="▤ Консоль (команды git)")
        con.columnconfigure(0, weight=1)
        con.rowconfigure(0, weight=1)
        self.console = tk.Text(con, width=30, wrap="word", font=self.mono, bg=C["console"], fg=C["fg"],
                               insertbackground=C["fg"], selectbackground=C["sel"],
                               selectforeground="#ffffff", highlightthickness=0,
                               padx=8, pady=6, state="disabled")
        self.console.grid(row=0, column=0, sticky="nsew")
        con_sb = ttk.Scrollbar(con, command=self.console.yview)
        con_sb.grid(row=0, column=1, sticky="ns")
        self.console.configure(yscrollcommand=con_sb.set)
        # Различаем строки не цветом, а оформлением: команда — жирная после «>», комментарии — с «#»
        self.console.tag_configure("cmd", font=(self.mono[0], self.mono[1], "bold"), foreground="#ffffff")
        self.console.tag_configure("note", font=(self.mono[0], self.mono[1], "italic"), foreground="#bbbbbb")

        own = ttk.Frame(right)       # поле «Своя команда» — под вкладками, видно всегда
        own.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        own.columnconfigure(1, weight=1)
        ttk.Label(own, text="Своя команда:  git").grid(row=0, column=0)
        self.own_entry = ttk.Entry(own, font=self.mono)
        self.own_entry.grid(row=0, column=1, sticky="ew", padx=6)
        self.own_entry.bind("<Return>", lambda e: self.act_own_command())
        self.btn_own = ttk.Button(own, text="Выполнить…", command=self.act_own_command)
        self.btn_own.grid(row=0, column=2)
        self.repo_buttons.append(self.btn_own)
        self.repo_buttons.append(self.btn_account)
        ttk.Button(own, text="Очистить", command=self.console_clear).grid(row=0, column=3, padx=(6, 0))

        for w in (self.explain, self.console):
            self._make_copyable(w)

    # --------------------------------------------------------- копирование
    def _make_copyable(self, w):
        """Выделение + Ctrl+C (в любой раскладке), меню по правой кнопке, двойной щелчок по команде."""
        menu = tk.Menu(w, tearoff=0)
        menu.add_command(label="Копировать выделенное", command=lambda: self._copy_selection(w))
        menu.add_command(label="Копировать всё", command=lambda: self.copy_text(w.get("1.0", "end").strip(),
                                                                                 "Скопирован весь текст"))
        w.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))
        # 67 — код клавиши C в Windows: так Ctrl+C работает и в русской раскладке (Ctrl+С)
        w.bind("<Control-Key>", lambda e: self._copy_selection(w) if (e.keycode == 67 or
                                                                      e.keysym.lower() == "c") else None)
        w.bind("<Double-Button-1>", lambda e: self._copy_line(w, e))

    def _copy_selection(self, w):
        try:
            text = w.get("sel.first", "sel.last")
        except tk.TclError:
            return "break"
        if text.strip():
            self.copy_text(text, "Скопировано выделенное")
        return "break"

    def _copy_line(self, w, event):
        """Двойной щелчок по строке с командой — копирует её целиком (без «C:\\...>» перед ней)."""
        idx = w.index(f"@{event.x},{event.y}")
        line = w.get(f"{idx} linestart", f"{idx} lineend").strip()
        line = re.sub(r"^.*?> (?=git )", "", line)
        if not re.match(r"^(git|cd|mkdir) ", line):
            return None                      # не команда — обычное выделение слова
        self.copy_text(line, f"Скопирована команда: {line[:70]}")
        return "break"

    def copy_text(self, text, what="Скопировано"):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()                   # чтобы буфер обмена не пропал после закрытия окна
        self.copy_msg.configure(text=f"⧉ {what}. Вставь в консоль: Ctrl+V или правый щелчок.")

    def _clear_copy_buttons(self):
        for child in self.copy_frame.winfo_children():
            child.destroy()
        self.copy_msg.configure(text="")

    # ----------------------------------------------------- вывод на экран
    def set_explain(self, text):
        self.explain.configure(state="normal")
        self.explain.delete("1.0", "end")
        first, _, rest = text.partition("\n")
        self.explain.insert("end", first, "b")
        if rest:
            self.explain.insert("end", "\n" + rest)
        self.explain.configure(state="disabled")
        self.explain.yview_moveto(0)

    def _console_write(self, text, tag=None):
        self.console.configure(state="normal")
        self.console.insert("end", text + "\n", tag)
        self.console.see("end")
        self.console.configure(state="disabled")

    def console_cmd(self, cwd, cmd):
        self._console_write("")
        self._console_write(f"{cwd}> {cmd}", "cmd")

    def console_out(self, text):
        self._console_write(text)

    def console_note(self, text):
        self._console_write("# " + text, "note")

    def console_clear(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    # --------------------------------------------------- состояние проекта
    def read_status(self):
        """Читает состояние проекта. Ничего не меняет (--no-optional-locks)."""
        if not (self.git_ok and self.repo):
            return {"repo": False}
        code, out = run_git(["--no-optional-locks", "status", "--porcelain=v1", "-b"], self.repo, 15)
        if code != 0:
            return {"repo": False, "error": out}
        st = parse_status(out)
        st["repo"] = True

        _, rem = run_git(["remote"], self.repo, 15)
        st["remotes"] = [x.strip() for x in rem.splitlines() if x.strip()]
        st["remote"] = "origin" if "origin" in st["remotes"] else (st["remotes"][0] if st["remotes"] else None)

        _, refs = run_git(["for-each-ref", "--format=%(refname)",
                           "refs/heads/main", "refs/heads/master", "refs/remotes", "refs/stash"], self.repo, 15)
        refs = set(refs.split())
        st["has_stash"] = "refs/stash" in refs
        main = None
        for name in ("main", "master"):
            if f"refs/heads/{name}" in refs:
                main = name
                break
        if not main and st["remote"]:
            for name in ("main", "master"):
                if f"refs/remotes/{st['remote']}/{name}" in refs:
                    main = name
                    break
        if not main:
            main = st["branch"] if st["unborn"] else "main"
        st["main"] = main

        st["last_commit"] = None
        if not st["unborn"]:
            code, ts = run_git(["log", "-1", "--format=%ct"], self.repo, 15)
            if code == 0 and ts.strip().isdigit():
                st["last_commit"] = int(ts.strip())

        st["reverting"] = bool(self.git_dir and (self.git_dir / "REVERT_HEAD").exists())
        st["merging"] = bool(self.git_dir and (self.git_dir / "MERGE_HEAD").exists()) or st["reverting"]
        # когда последний раз проверяли сервер (git сам обновляет этот файл при fetch и pull)
        st["last_fetch"] = None
        if self.git_dir and (self.git_dir / "FETCH_HEAD").exists():
            st["last_fetch"] = (self.git_dir / "FETCH_HEAD").stat().st_mtime
        # сколько новых сохранений в главной ветке на сервере, которых нет в твоей ветке
        st["main_behind"] = 0
        if st["remote"] and not st["unborn"] and st["branch"] != st["main"] and \
                f"refs/remotes/{st['remote']}/{st['main']}" in refs:
            code, n = run_git(["rev-list", "--count", f"HEAD..{st['remote']}/{st['main']}"], self.repo, 15)
            if code == 0 and n.strip().isdigit():
                st["main_behind"] = int(n.strip())
        st["ident"] = read_identity(self.repo)
        st["server"] = self.read_server(st["remote"]) if st["remote"] else None
        return st

    def read_server(self, remote):
        """Сведения о сервере и логине. Список аккаунтов из менеджера паролей читаем один раз на сайт."""
        code, url = run_git(["remote", "get-url", remote], self.repo, 15)
        if code != 0:
            return None
        srv = parse_remote_url(url)
        srv["login_from"] = "url" if srv["login"] else ""
        if srv["kind"] == "https" and not srv["login"]:
            code, user = run_git(["config", "--get-urlmatch", "credential.username", srv["url"]], self.repo, 15)
            if code == 0 and user.strip():
                srv["login"], srv["login_from"] = user.strip(), "config"
            elif srv["host"].lower() == "github.com":
                if "github" not in self.gcm_cache:
                    self.gcm_cache["github"] = gcm_github_accounts(self.repo)
                accounts = self.gcm_cache["github"]
                if accounts:
                    srv["login"], srv["login_from"] = ", ".join(accounts), "gcm"
        return srv

    def account_text(self, s):
        """Текст строки «Аккаунт» под строкой состояния."""
        ident = s.get("ident") or {}
        if ident.get("name") and ident.get("email"):
            scope = SCOPE_NAMES.get(ident["email_scope"] or ident["name_scope"], "")
            who = f"☺ Подпись: {ident['name']} <{ident['email']}>" + (f"  ({scope})" if scope else "")
        elif ident.get("name") or ident.get("email"):
            who = f"☺ Подпись: {ident.get('name') or '?'} <{ident.get('email') or '?'}>  — ⚠ указано не всё"
        else:
            who = "☺ Подпись: ⚠ не указана — git не знает, кто ты. Нажми «Сменить аккаунт…»"
        return who + "\n" + self.server_text(s.get("server"))

    @staticmethod
    def server_text(srv):
        if not srv:
            return "⇅ Сервер: нет — работа только на этом компьютере"
        if srv["kind"] == "folder":
            return f"⇅ Сервер: папка на компьютере ({srv['url']}) — вход не нужен"
        repo = f"{srv['owner']}/{srv['repo']}" if srv["owner"] else srv["repo"]
        text = f"⇅ Сервер: {srv['host']}   ·   репозиторий: {repo}"
        if srv["kind"] == "ssh":
            return text + "   ·   вход: по SSH-ключу (какой аккаунт — решает ключ)"
        if srv["login"]:
            where = {"url": "указан в адресе сервера", "config": "из настроек git",
                     "gcm": "сохранён в Windows"}.get(srv["login_from"], "")
            many = "ы" if "," in srv["login"] else ""
            return text + f"   ·   логин{many}: {srv['login']}" + (f" ({where})" if where else "")
        return text + "   ·   логин: не сохранён — спросят при первой отправке"

    def refresh_status(self):
        if self.busy:
            return
        self.state = self.read_status()
        self._update_graph()
        self._update_status_line()
        self._update_reminders()
        self.set_actions_enabled(self.pending is None)

    def _update_graph(self, force=False):
        """Перечитать граф и перерисовать, если что-то поменялось (и сейчас не показан предпросмотр)."""
        if not self.state.get("repo"):
            self.gmodel = None
            self._graph_show(None)
            return
        m = GitGraph.load(self.repo)
        changed = force or self.gmodel is None or m.signature() != self.gmodel.signature()
        self.gmodel = m
        if changed and self.pending is None:
            self._graph_show(m, self.state["main"], self.state["remote"])

    def _graph_show(self, *args, **kw):
        """Нарисовать схему во всех открытых окнах (основное + большое)."""
        self._graph_args = (args, kw)
        for v in list(self.graph_views):
            try:
                v.show(*args, **kw)
            except tk.TclError:          # окно уже закрыли
                self.graph_views.remove(v)

    def open_graph_window(self):
        """Схема веток в отдельном большом окне."""
        for v in self.graph_views[1:]:
            try:
                v.winfo_toplevel().lift()
                return
            except tk.TclError:
                pass
        win = tk.Toplevel(self.root)
        win.title(f"{APP_TITLE} — схема веток")
        win.configure(bg=C["bg"])
        w = min(1400, win.winfo_screenwidth() - 60)
        h = min(900, win.winfo_screenheight() - 100)
        win.geometry(f"{w}x{h}+20+20")
        dark_titlebar(win)
        view = GraphView(win)
        view.pack(fill="both", expand=True, padx=6, pady=6)
        self.graph_views.append(view)

        def close():
            if view in self.graph_views:
                self.graph_views.remove(view)
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", close)
        args, kw = self._graph_args
        view.show(*args, **kw)

    def show_preview(self, plan):
        """Предпросмотр на графе: копируем модель, «проигрываем» на ней действие."""
        if not self.gmodel:
            return
        if plan.preview is None:
            self._graph_show(self.gmodel, self.state["main"], self.state["remote"],
                             notes=["Это действие не меняет сохранения и ветки — граф останется таким же."])
        else:
            pm = self.gmodel.copy()
            try:
                notes = plan.preview(pm) or []
            except Exception as e:  # noqa — предпросмотр не должен ломать программу
                notes, pm = [f"Предпросмотр не получился: {e}"], None
            if pm is None:
                self._graph_show(self.gmodel, self.state["main"], self.state["remote"], notes=notes)
            else:
                self._graph_show(self.gmodel, self.state["main"], self.state["remote"], preview=pm,
                                 title=plan.title, notes=notes)
        self.tabs.select(self.graph)

    def _auto_refresh(self):
        if not self.busy and self.pending is None:
            self.refresh_status()
        self.root.after(STATUS_REFRESH_MS, self._auto_refresh)

    def _update_status_line(self):
        s = self.state
        if not s.get("repo"):
            self.status_label.configure(text="")
            self.account_label.configure(text="")
            return
        self.account_label.configure(text=self.account_text(s))
        parts = [f"Ветка: {s['branch']}",
                 f"Несохранённых файлов: {len(s['files'])}",
                 f"Не отправлено: {s['ahead']}" if s.get("upstream") else "Ветка не на сервере",
                 f"Сервер: {s['remote']}" if s["remote"] else "Сервера нет"]
        if self.is_sandbox:
            parts.append("ПЕСОЧНИЦА")
        if self.manual_var.get():
            parts.append("✋ ТОЛЬКО ПОДСКАЗКИ")
        self.status_label.configure(text="   ·   ".join(parts))

    def _update_reminders(self):
        """Собирает список напоминаний. Показываются только в этом окне."""
        s = self.state
        r = []
        if not self.git_ok:
            r.append("✖ Git не установлен — помощник не может работать.")
        elif not self.repo:
            r.append("ℹ Выбери проект («Открыть проект…») или начни с «Песочницы».")
        elif not s.get("repo"):
            r.append("✖ Не удаётся прочитать проект. Возможно, папку удалили или переместили.")
        else:
            if s["conflicts"]:
                r.append("⚠ КОНФЛИКТ в файлах: " + ", ".join(s["conflicts"][:5]) +
                         ". Исправь их и нажми «4. Сохранить работу» (или «Прервать слияние / отмену»).")
            elif s["reverting"]:
                r.append("⚠ Отмена сохранения не закончена: нажми «4. Сохранить работу», чтобы её завершить.")
            elif s["merging"]:
                r.append("⚠ Слияние не закончено: нажми «4. Сохранить работу», чтобы его завершить.")
            if not self.day_started and s["remote"]:
                r.append("☀ Перед началом работы нажми «1. Начать день» — забери свежие изменения коллег.")
            if s["behind"]:
                r.append(f"⇩ На сервере {s['behind']} новых сохранений — забери их («1. Начать день»).")
            if s["main_behind"]:
                r.append(f"⇩ В главной ветке ({s['main']}) на сервере {s['main_behind']} сохранений, которых нет "
                         "в твоей ветке. «1. Начать день» подтянет их к тебе.")
            if s["remote"] and self.day_started:
                hours = (time.time() - s["last_fetch"]) / 3600 if s["last_fetch"] else None
                if hours is None or hours > 2:
                    when = "ни разу не проверялся" if hours is None else f"не проверялся {int(hours)} ч"
                    r.append(f"☁ Сервер {when} — коллеги могли что-то отправить. "
                             "Нажми «☁ Проверить сервер» (твои файлы не изменятся).")
            if s["files"] and s["branch"] == s["main"] and not s["merging"]:
                r.append("⚠ Ты меняешь файлы прямо в главной ветке. В компании так не принято — "
                         "создай ветку («2. Новая задача»), изменения переедут с тобой.")
            if s["files"] and not s["conflicts"]:
                minutes = (time.time() - s["last_commit"]) / 60 if s["last_commit"] else 0
                if minutes > SAVE_REMINDER_MIN:
                    r.append(f"✎ Ты не сохранялся уже {int(minutes)} мин. Не забудь «4. Сохранить работу».")
                else:
                    r.append(f"✎ Есть несохранённые изменения ({len(s['files'])} файл.). "
                             "Закончил кусочек работы — сохрани.")
            if s["remote"] and s["upstream"] and s["ahead"]:
                r.append(f"⇧ {s['ahead']} сохранений ещё не отправлены на сервер («5. Отправить»).")
            if s["remote"] and not s["upstream"] and not s["unborn"] and not s["detached"]:
                r.append("⇧ Эта ветка ещё ни разу не отправлялась на сервер.")
            if not s["remote"]:
                r.append("ℹ У проекта нет сервера — вся работа только на этом компьютере.")
            if s["has_stash"]:
                r.append("⤓ У тебя есть отложенные изменения — не забудь их вернуть, если нужны.")
            ident = s.get("ident") or {}
            if not (ident.get("name") and ident.get("email")):
                r.append("☺ Git не знает, кто ты: сохранения не получатся. Нажми «☺ Сменить аккаунт…» "
                         "и укажи имя и почту (почту — ту же, что в аккаунте на GitHub/GitLab).")
            if s["detached"]:
                r.append("⚠ Ты не в ветке (смотришь старую версию). Перейди в ветку: «Перейти в другую ветку».")
            if not r:
                r.append("✔ Всё в порядке: всё сохранено и отправлено.")
        self.rem_label.configure(text="\n".join(r))

    def set_actions_enabled(self, on):
        repo_ok = on and self.git_ok and bool(self.state.get("repo"))
        for b in self.repo_buttons:
            b.state(["!disabled"] if repo_ok else ["disabled"])
        for b in (self.btn_open, self.btn_sandbox, self.btn_clone):
            b.state(["!disabled"] if (on and self.git_ok) else ["disabled"])
        self.btn_colleague.state(["!disabled"] if (repo_ok and self.is_sandbox) else ["disabled"])

    # ------------------------------------------- подтверждение и запуск
    def propose(self, plan):
        """Показывает план. Изменяющие действия ждут «Да, выполнить»."""
        if self.busy or self.pending:
            return
        self._clear_copy_buttons()
        if not plan.safe:
            self.show_preview(plan)
        if self.manual_var.get():
            self._show_manual(plan)
            return
        if plan.safe:
            self.tabs.select(1)          # «посмотреть» — ответ git будет в консоли
            self.set_explain(f"{plan.title}\n\n{plan.explain}")
            self._execute(plan)
            return
        self.pending = plan
        lines = [f"▶ {plan.title}", "", plan.explain, "", "Будут выполнены команды:"]
        lines += ["    " + st.display() for st in plan.steps]
        lines += ["", "Ничего не произойдёт, пока ты не нажмёшь «Да, выполнить»."]
        self.set_explain("\n".join(lines))
        self.set_actions_enabled(False)
        self.yes_btn.state(["!disabled"])
        self.no_btn.state(["!disabled"])
        self.yes_btn.focus_set()

    # ------------------------------------------------ ручной режим
    def manual_blocks(self, plan):
        """Делит план на блоки команд и шаги «сделай руками». Каждый блок начинается с cd."""
        blocks, cmds, cur_dir = [], [], None
        for st in plan.steps:
            cmd = st.shell()
            if cmd is None:
                if cmds:
                    blocks.append(("cmds", cmds))
                    cmds = []
                blocks.append(("hand", st.show))
                cur_dir = None
                continue
            cwd = st.cwd or self.repo
            if cwd is not None and Path(cwd) != cur_dir:
                cmds.append(f'cd "{cwd}"')
                cur_dir = Path(cwd)
            cmds.append(cmd)
        if cmds:
            blocks.append(("cmds", cmds))
        return blocks

    def _show_manual(self, plan):
        blocks = self.manual_blocks(plan)
        n_cmd = sum(1 for k, _ in blocks if k == "cmds")
        lines = [f"✋ {plan.title} — команды для ручного ввода", "", plan.explain, "",
                 "Режим «Только подсказывать»: я ничего не выполняю. Скопируй команды и вставь в консоль.", ""]
        self.console_note(f"{plan.title} — команды для ручного ввода (программа их НЕ выполняла):")
        i = 0
        for kind, item in blocks:
            if kind == "cmds":
                i += 1
                if n_cmd > 1:
                    lines.append(f"Блок {i}:")
                lines += ["    " + c for c in item]
                lines.append("")
                for c in item:
                    self._console_write(c, "cmd")
                text = "\n".join(item)
                label = f"⧉ Скопировать блок {i}" if n_cmd > 1 else "⧉ Скопировать команды"
                ttk.Button(self.copy_frame, text=label,
                           command=lambda t=text, k=i: self.copy_text(
                               t, f"Скопирован блок {k}" if n_cmd > 1 else "Команды скопированы")
                           ).pack(side="left", padx=(0, 6))
            else:
                lines += [f"✋ Сделай сам: {item}", ""]
                self.console_note(f"сделай сам: {item}")
        lines.append("Двойной щелчок по строке с командой (здесь или в консоли) — скопировать только её.")
        lines.append(plan.manual_after or "Когда выполнишь — нажми «↻ Обновить», и напоминания подстроятся.")
        self.set_explain("\n".join(lines))
        if plan.on_copy:
            plan.on_copy()

    def on_mode_change(self):
        manual = self.manual_var.get()
        self.settings["manual"] = manual
        save_settings(self.settings)
        if self.pending:
            self.confirm_no()
        self._clear_copy_buttons()
        if self.state.get("repo"):
            self._update_graph(force=True)
        if manual:
            self.set_explain("✋ Включён режим «Только подсказывать»\n\n"
                             "Теперь кнопки ничего не выполняют. Они объясняют, что нужно сделать, и показывают "
                             "готовые команды — их можно скопировать кнопкой «Скопировать», двойным щелчком "
                             "по строке или выделением + Ctrl+C.\n\n"
                             "Каждый блок начинается с cd в папку проекта, так что вставлять можно в любую "
                             "консоль (cmd, PowerShell или Git Bash).\n\n"
                             "Программа по-прежнему только читает состояние проекта (git status), "
                             "чтобы показывать напоминания. После ручного ввода нажми «↻ Обновить».")
        else:
            self.set_explain("Обычный режим\n\nКнопки снова выполняют команды — после твоего «Да, выполнить».")
        self.refresh_status()

    def manual_refresh(self):
        self.refresh_status()
        self._update_graph(force=True)
        self.copy_msg.configure(text="↻ Состояние проекта обновлено.")

    def confirm_yes(self):
        plan, self.pending = self.pending, None
        if plan:
            self._execute(plan)

    def confirm_no(self):
        self.pending = None
        self.yes_btn.state(["disabled"])
        self.no_btn.state(["disabled"])
        self.set_explain("Отменено.\n\nНичего не выполнялось, проект не изменился.")
        self._update_graph(force=True)
        self.set_actions_enabled(True)

    def _execute(self, plan):
        self.busy = True
        self.set_actions_enabled(False)
        self.yes_btn.state(["disabled"])
        self.no_btn.state(["disabled"])
        self.console_note(plan.title)
        threading.Thread(target=self._worker, args=(plan,), daemon=True).start()

    def _worker(self, plan):
        """Выполняется в фоне, чтобы окно не зависало. Останавливается на первой ошибке."""
        outputs, ok = [], True
        for st in plan.steps:
            cwd = st.cwd or self.repo
            self.q.put(("cmd", str(cwd), st.display()))
            if st.func:
                try:
                    good, text = st.func()
                except Exception as e:  # noqa
                    good, text = False, f"Ошибка: {e}"
            else:
                code, text = run_git(st.args, cwd, st.timeout)
                good = code == 0
            if text.strip():
                self.q.put(("out", text.rstrip()))
            outputs.append(text)
            if not good:
                ok = False
                break
        self.q.put(("done", plan, ok, "\n".join(outputs)))

    def _poll_queue(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "cmd":
                    self.console_cmd(item[1], item[2])
                elif item[0] == "out":
                    self.console_out(item[1])
                elif item[0] == "done":
                    self._finish(*item[1:])
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _finish(self, plan, ok, output):
        self.busy = False
        if ok:
            self.console_note("✔ готово")
            msg = plan.success or "Готово."
            if plan.on_success:
                extra = plan.on_success(output)
                if extra:
                    msg = extra
            self.set_explain(f"✔ {plan.title} — готово\n\n{msg}")
        else:
            self.console_note("✖ остановлено: git сообщил об ошибке")
            if plan.on_fail:
                plan.on_fail(output)
            self.set_explain(f"✖ {plan.title} — не получилось\n\n{explain_error(output)}")
            self.tabs.select(1)          # при ошибке полезно видеть ответ git
        self.gmodel = None               # граф перерисуется в любом случае
        self.refresh_status()

    # ------------------------------------------------- открытие проекта
    def choose_repo(self):
        d = filedialog.askdirectory(title="Выбери папку проекта", parent=self.root)
        if d:
            self.open_repo(Path(d))

    def open_repo(self, path):
        code, out = run_git(["rev-parse", "--show-toplevel"], path, 15)
        if code != 0:
            self.propose(Plan(
                "Сделать папку проектом git",
                f"Папка {path} пока не проект git — git за ней не следит.\n\n"
                "Можно «подключить» git: он создаст в папке скрытую служебную папку .git "
                "и начнёт запоминать версии. Твои файлы не изменятся.",
                [Step(["init"], cwd=path), Step(["symbolic-ref", "HEAD", "refs/heads/main"], cwd=path)],
                on_success=lambda o: (self.open_repo(path),
                                      "Готово! Теперь это проект git с главной веткой main.\n\n"
                                      "Дальше: сделай первое сохранение («4. Сохранить работу»).")[1],
                manual_after="Когда выполнишь — снова выбери эту папку через «Открыть проект…»."))
            return
        top = Path(out.strip())
        self.repo = top
        code, gd = run_git(["rev-parse", "--absolute-git-dir"], top, 15)
        self.git_dir = Path(gd.strip()) if code == 0 else top / ".git"
        try:
            self.is_sandbox = top.resolve() == (SANDBOX_DIR / "moy_proekt").resolve()
        except OSError:
            self.is_sandbox = False
        self.day_started = False
        self.repo_label.configure(text=f"Проект: {top}")
        self.console_note(f"открыт проект {top}")
        self.refresh_status()
        if not self.busy and not self.pending:
            self.set_explain(
                f"Проект открыт: {top.name}\n\n"
                "Обычный рабочий день в компании выглядит так:\n"
                "1. Утром забираешь свежие изменения коллег.\n"
                "2. Для задачи создаёшь свою ветку.\n"
                "3–4. Работаешь и время от времени сохраняешься (коммит).\n"
                "5. Отправляешь сохранения на сервер.\n"
                "6. Когда задача готова — вливаешь её в главную ветку.\n"
                "7. Вечером сохраняешь и отправляешь всё, что не успел.\n\n"
                "Подсказки, что сделать сейчас, — в блоке «Напоминания» сверху.")

    def open_folder(self):
        target = self.repo or (SANDBOX_DIR if SANDBOX_DIR.exists() else None)
        if not target:
            self.set_explain("Нечего открывать\n\nСначала выбери проект или создай песочницу.")
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(target))  # noqa
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as e:  # noqa
            self.set_explain(f"Не удалось открыть папку\n\n{e}")

    # ----------------------------------------------------- помощники
    def fresh(self):
        """Перечитать проект прямо перед действием, чтобы решать по актуальному состоянию."""
        self.refresh_status()
        return self.state

    def info(self, title, text):
        """Сообщение без действий (когда делать нечего или нельзя)."""
        self.set_explain(f"{title}\n\n{text}")

    def ask(self, title, prompt, initial=""):
        return simpledialog.askstring(title, prompt, parent=self.root, initialvalue=initial)

    # =============================================== ДЕЙСТВИЯ: рабочий день
    def act_start_day(self):
        s = self.fresh()
        if not s["remote"]:
            self.day_started = True
            self.refresh_status()
            self.info("☀ Начинаем день",
                      "У этого проекта нет сервера, поэтому забирать нечего — вся работа только у тебя.\n"
                      "Можно сразу работать: «2. Новая задача».")
            return
        if s["detached"]:
            self.info("☀ Начать день", "Ты сейчас не в ветке. Сначала перейди в ветку («Перейти в другую ветку»).")
            return
        remote, main, br = s["remote"], s["main"], s["branch"]
        steps, text = [], ["Забираем с сервера то, что успели сделать коллеги, чтобы ты работал "
                           "с самой свежей версией проекта.",
                           "",
                           "git pull — это «скачай с сервера и добавь к себе»."]
        if s["upstream"]:
            steps.append(Step(["pull", "--no-rebase"]))
        if br != main or not s["upstream"]:
            steps.append(Step(["pull", "--no-rebase", remote, main]))
            if br != main:
                text += ["", f"Ты в ветке задачи «{br}», поэтому ещё подтягиваем в неё свежую главную ветку "
                             f"({main}). Так ты заранее увидишь, не мешают ли твоей задаче чужие изменения."]
        if s["files"]:
            text += ["", "⚠ У тебя есть несохранённые изменения. Обычно это не мешает, но если коллега менял "
                         "те же файлы, git остановится и попросит сначала сохраниться."]

        def done(out):
            self.day_started = True
            if re.search(r"Already up.to.date", out) and "Fast-forward" not in out and "Merge made" not in out:
                return "Новых изменений не было — у тебя и так самая свежая версия.\n\nМожно работать!"
            return ("Свежие изменения коллег теперь у тебя. Что пришло — видно в консоли.\n\n"
                    "Дальше: начинаешь новую задачу — «2. Новая задача»; продолжаешь старую — просто работай.")

        def failed(out):
            self.day_started = True     # день начат, просто с проблемой — подсказка будет про неё

        upstream = s["upstream"]

        def preview(m):
            notes = []
            if upstream:
                notes.append(m.merge_into_head(m.remotes.get(upstream), f"с сервера ({upstream})"))
            if br != main or not upstream:
                notes.append(m.merge_into_head(m.remotes.get(f"{remote}/{main}"), f"главная ветка с сервера"))
            return notes + ["Предпросмотр — по последней проверке сервера. Если коллеги отправили что-то позже, появится и это."]

        self.propose(Plan("Начать день: забрать свежие изменения", "\n".join(text), steps,
                          on_success=done, on_fail=failed, on_copy=lambda: failed(""), preview=preview))

    def act_new_task(self):
        s = self.fresh()
        if s["merging"] or s["conflicts"]:
            self.info("Новая задача", "Сначала закончи или отмени слияние (см. напоминания).")
            return
        name = self.ask("Новая задача", "Коротко опиши задачу\n(например: исправить кнопку входа):")
        if not name:
            return
        slug = make_branch_slug(name)
        if not slug:
            self.info("Новая задача", "Не получилось сделать имя ветки из этого текста. Попробуй словами.")
            return
        branch = f"zadacha/{slug}"
        main, remote = s["main"], s["remote"]
        text = ["В компании каждую задачу делают в отдельной ветке — это как черновик-копия проекта.",
                "Главная ветка остаётся рабочей, а ты спокойно меняешь своё и никому не мешаешь.",
                "", f"Имя ветки: {branch}  (латиницей — так принято, чтобы не было проблем с кодировками)."]
        steps = []
        if s["files"] or s["unborn"]:
            steps.append(Step(["switch", "-c", branch]))
            if s["files"]:
                text += ["", "У тебя есть несохранённые изменения — они переедут в новую ветку вместе с тобой."]
        else:
            if s["branch"] != main:
                steps.append(Step(["switch", main]))
                text += ["", f"Сначала возвращаемся в главную ветку ({main}) — новая задача начинается от неё."]
            if remote:
                steps.append(Step(["pull", "--no-rebase", remote, main]))
                text += ["", "Забираем свежую главную ветку с сервера, чтобы начать с актуальной версии."]
            steps.append(Step(["switch", "-c", branch]))
        from_here = bool(s["files"] or s["unborn"])

        def preview(m):
            notes = []
            if not from_here:
                if m.head != main and main in m.branches:
                    m.switch(main)
                if remote:
                    notes.append(m.merge_into_head(m.remotes.get(f"{remote}/{main}"), "главная ветка с сервера"))
            m.new_branch(branch)
            notes.append(f"Появится ярлык «{branch}» на том же сохранении, ★ переедет на него. "
                         "Новые сохранения пойдут уже в эту ветку.")
            return notes

        self.propose(Plan(f"Новая задача: {name}", "\n".join(text), steps, preview=preview,
                          success=f"Ты в новой ветке {branch}. Можно работать!\n\n"
                                  "Меняй файлы, а закончив кусочек — «4. Сохранить работу»."))

    def act_what_changed(self):
        def done(out):
            self.refresh_status()
            return self.changes_summary()
        self.propose(Plan("Что я изменил",
                          "Только смотрим — ничего не меняется.\n"
                          "git status — список изменённых файлов, git diff --stat — сколько строк поменялось.",
                          [Step(["status"]), Step(["diff", "--stat"])], safe=True, on_success=done))

    def changes_summary(self):
        files = self.state.get("files", [])
        if not files:
            return "Изменений нет — всё уже сохранено."
        groups = {"Конфликт (нужно разобрать)": [], "Новые файлы": [], "Изменённые": [], "Удалённые": []}
        for xy, p in files:
            if xy in CONFLICT_CODES:
                groups["Конфликт (нужно разобрать)"].append(p)
            elif xy == "??" or "A" in xy:
                groups["Новые файлы"].append(p)
            elif "D" in xy:
                groups["Удалённые"].append(p)
            else:
                groups["Изменённые"].append(p)
        lines = [f"Несохранённых файлов: {len(files)}", ""]
        for title, items in groups.items():
            if items:
                lines.append(f"{title}:")
                lines += [f"   • {p}" for p in items[:30]]
                if len(items) > 30:
                    lines.append(f"   … и ещё {len(items) - 30}")
        lines += ["", "Подробности (какие строки) — в консоли. Чтобы сохранить — «4. Сохранить работу»."]
        return "\n".join(lines)

    def act_save(self):
        s = self.fresh()
        if not s["files"] and not s["merging"]:
            self.info("Сохранить работу", "Сохранять нечего — изменений нет.")
            return
        marked = files_with_conflict_marks(self.repo, s["conflicts"])
        if marked:
            self.info("Сохранить работу — рано",
                      "В этих файлах ещё остались метки конфликта (<<<<<<< / >>>>>>>):\n   • " +
                      "\n   • ".join(marked) +
                      "\n\nОткрой их, оставь нужный текст, удали строки с метками и сохрани файл. Потом нажми снова.")
            return
        default = ("Отмена сохранения: разобрал конфликт" if s["reverting"] else
                   "Слияние: разобрал конфликт" if s["merging"] else "")
        msg = self.ask("Сохранить работу", "Что ты сделал? Напиши коротко\n(например: добавил проверку пароля):",
                       default)
        if not msg or not msg.strip():
            return
        text = ["Делаем сохранение (коммит) — снимок проекта с подписью. К нему всегда можно будет вернуться.",
                "", "git add -A — отметить все изменённые файлы для сохранения;",
                "git commit -m \"…\" — сделать сам снимок с твоей подписью.",
                "", "Сохранение остаётся только на твоём компьютере. Чтобы его увидели коллеги — «5. Отправить»."]
        ident = s.get("ident") or {}
        if ident.get("name") and ident.get("email"):
            text += ["", f"☺ Сохранение будет подписано: {ident['name']} <{ident['email']}>"]
        else:
            text += ["", "⚠ Git не знает твоё имя и почту — сохранение не получится. "
                         "Сначала нажми «☺ Сменить аккаунт…»."]
        if s["branch"] == s["main"] and not s["merging"]:
            text += ["", "⚠ Ты в главной ветке. В компании обычно сохраняются в своей ветке задачи "
                         "(«2. Новая задача»). Для личного проекта это нормально."]
        merge_parent = None
        if s["merging"] and not s["reverting"] and self.git_dir:
            try:
                merge_parent = (self.git_dir / "MERGE_HEAD").read_text().split()[0]
            except (OSError, IndexError):
                pass

        def preview(m):
            m.commit(msg.strip(), merge_parent)
            return [f"Появится новое сохранение «{msg.strip()}», ярлык ветки «{m.head}» сдвинется на него."
                    if m.head else "Появится новое сохранение."] + \
                   (["Это завершит слияние: у сохранения будет два «родителя»."] if merge_parent else [])

        self.propose(Plan("Сохранить работу", "\n".join(text),
                          [Step(["add", "-A"]), Step(["commit", "-m", msg.strip()])], preview=preview,
                          success="Работа сохранена.\n\nКогда будет пара сохранений (или в конце дня) — "
                                  "«5. Отправить на сервер»."))

    def act_push(self):
        s = self.fresh()
        if not s["remote"]:
            self.info("Отправить", "У проекта нет сервера — отправлять некуда.\n\n"
                                   "Создай пустой репозиторий на GitHub/GitLab и подключи его кнопкой "
                                   "«⇅ Подключить / сменить сервер…» (блок «Сервер»).")
            return
        if s["unborn"]:
            self.info("Отправить", "Отправлять пока нечего — сначала сохрани работу хотя бы раз.")
            return
        if s["detached"]:
            self.info("Отправить", "Ты не в ветке. Сначала перейди в ветку.")
            return
        if s["upstream"] and not s["ahead"]:
            extra = "\n\nНо есть несохранённые изменения — сначала «4. Сохранить работу»." if s["files"] else ""
            self.info("Отправить", "Отправлять нечего — все сохранения уже на сервере." + extra)
            return
        br, remote = s["branch"], s["remote"]
        text = ["Отправляем твои сохранения на сервер. После этого их увидят коллеги, "
                "и работа не пропадёт, даже если с компьютером что-то случится.",
                "", "git push — «выложи мои сохранения на сервер»."]
        if s["upstream"]:
            steps = [Step(["push"])]
        else:
            steps = [Step(["push", "-u", remote, br])]
            text += ["", f"Ветка «{br}» отправляется впервые, поэтому добавляем -u — "
                         "это связывает её с сервером, дальше будет просто git push."]
        text += ["", "Куда и под каким аккаунтом:", "    " + self.server_text(s.get("server"))]
        if s["files"]:
            text += ["", "⚠ Несохранённые изменения не отправятся — уйдут только сделанные сохранения."]
        def preview(m):
            m.push(remote, br)
            return [f"Ярлык сервера «{remote}/{br}» встанет туда же, где твоя ветка — сервер «догонит» тебя."]

        self.propose(Plan("Отправить на сервер", "\n".join(text), steps, preview=preview,
                          success="Сохранения на сервере. Коллеги могут их забрать."))

    def act_task_done(self):
        s = self.fresh()
        br, main, remote = s["branch"], s["main"], s["remote"]
        if br == main:
            self.info("Задача готова",
                      f"Ты сейчас в главной ветке ({main}). Эта кнопка вливает ветку задачи в главную — "
                      "сначала перейди в ветку задачи («Перейти в другую ветку»).")
            return
        if s["detached"] or s["unborn"]:
            self.info("Задача готова", "Здесь пока нечего вливать.")
            return
        if s["files"] or s["merging"]:
            self.info("Задача готова", "Сначала сохрани всю работу («4. Сохранить работу»).")
            return
        text = [f"Переносим всё из ветки задачи «{br}» в главную ветку ({main}).",
                "",
                "Как это бывает в компании: ты отправляешь ветку на сервер и на сайте (GitHub, GitLab) "
                "создаёшь Pull Request — «просьбу влить». Коллега смотрит код, и если всё хорошо, нажимает «влить».",
                "Здесь мы делаем то же самое сами, на компьютере:",
                f"  1) переходим в главную ветку,"]
        steps = [Step(["switch", main])]
        if remote:
            steps.append(Step(["pull", "--no-rebase", remote, main]))
            text.append("  2) забираем её свежую версию с сервера,")
        steps.append(Step(["merge", "--no-edit", br]))
        text.append(f"  {len(steps)}) вливаем в неё твою задачу (git merge),")
        if remote:
            steps.append(Step(["push", "-u", remote, main]))
            text.append(f"  {len(steps)}) отправляем обновлённую главную ветку на сервер.")
        delete = self.choose(
            "Задача готова",
            f"После слияния ветка задачи «{br}» больше не нужна. В компании её обычно удаляют — "
            "и у себя, и на сервере, чтобы ветки не копились.\n\n"
            "Вся работа из неё уже будет в главной ветке, так что ничего не пропадёт.",
            ["Влить и удалить ветку задачи (как в компании)", "Только влить, ветку оставить"])
        if delete is None:
            return
        del_remote = bool(delete == 0 and remote and self.remote_branch_exists(remote, br))
        if delete == 0:
            # -D, а не -d: обычный -d сверяет ветку с её копией на сервере и откажется, если последние
            # сохранения задачи туда не отправлялись. Но предыдущий шаг уже влил всё в главную ветку,
            # а если он не получится — до удаления дело не дойдёт.
            steps.append(Step(["branch", "-D", br]))
            text.append(f"  {len(steps)}) удаляем ветку задачи у себя (git branch -D — вся работа к этому "
                        f"моменту уже в {main}),")
            if del_remote:
                steps.append(Step(["push", remote, "--delete", br]))
                text.append(f"  {len(steps)}) удаляем её на сервере (git push --delete).")
        text += ["", "Если кто-то менял те же строки — будет конфликт, я подскажу, что делать."]
        def preview(m):
            tip = m.branches.get(br)
            m.switch(main)
            notes = []
            if remote:
                notes.append(m.merge_into_head(m.remotes.get(f"{remote}/{main}"), "главная ветка с сервера"))
            notes.append(m.merge_into_head(tip, f"задача «{br}»"))
            if remote:
                m.push(remote, main)
            if delete == 0:
                m.delete(br)
                if del_remote:
                    m.delete_remote(f"{remote}/{br}")
                notes.append(f"Ярлык «{br}» удалится — все её сохранения уже будут в {main}.")
            return notes

        self.propose(Plan("Задача готова: влить в главную ветку", "\n".join(text), steps, preview=preview,
                          success=f"Задача «{br}» влита в {main}"
                                  + (" и ветка удалена" if delete == 0 else "") + ". Ты сейчас в главной ветке.\n\n"
                                  "Для следующей задачи — снова «2. Новая задача»."))

    def remote_branch_exists(self, remote, branch):
        code, _ = run_git(["rev-parse", "--verify", "--quiet", f"refs/remotes/{remote}/{branch}"], self.repo, 15)
        return code == 0

    def act_end_day(self):
        s = self.fresh()
        remote = s["remote"]
        steps = []
        text = ["Перед уходом сохраняем и отправляем всё на сервер: работа не потеряется, "
                "а коллеги увидят, на чём ты остановился. Незаконченная работа в своей ветке — это нормально."]
        if s["conflicts"] or s["merging"]:
            self.info("Закончить день", "Сначала закончи или отмени слияние (см. напоминания).")
            return
        if s["files"]:
            msg = self.ask("Закончить день", "Подпиши сохранение:", "Незаконченная работа (конец дня)")
            if not msg or not msg.strip():
                return
            steps += [Step(["add", "-A"]), Step(["commit", "-m", msg.strip()])]
            text += ["", "1) Сохраняем несохранённое (git add + git commit)."]
        if remote and not s["detached"] and (s["files"] or s["ahead"] or (not s["upstream"] and not s["unborn"])):
            steps.append(Step(["push"] if s["upstream"] else ["push", "-u", remote, s["branch"]]))
            text += [f"{'2' if s['files'] else '1'}) Отправляем на сервер (git push)."]
        if not steps:
            self.info("☾ Закончить день", "Всё сохранено и отправлено. Хорошего вечера!")
            return
        if s["branch"] == s["main"]:
            text += ["", "⚠ Ты в главной ветке — незаконченная работа попадёт прямо в неё. "
                         "Если это рабочий проект, лучше сначала «2. Новая задача»."]
        will_commit = bool(s["files"])
        will_push = any(st.args and st.args[0] == "push" for st in steps)
        end_msg = msg.strip() if will_commit else ""
        cur = s["branch"]

        def preview(m):
            notes = []
            if will_commit:
                m.commit(end_msg)
                notes.append("Появится сохранение с незаконченной работой.")
            if will_push:
                m.push(remote, cur)
                notes.append(f"Сервер «догонит» твою ветку ({remote}/{cur}).")
            return notes

        self.propose(Plan("Закончить день", "\n".join(text), steps, preview=preview,
                          success="Всё сохранено и отправлено. Хорошего вечера! ☾"))

    # ====================================================== ДЕЙСТВИЯ: ещё
    def act_history(self):
        self.fresh()
        if self.state["unborn"]:
            self.info("История", "Сохранений пока нет.")
            return
        self.propose(Plan("История сохранений",
                          "Только смотрим — ничего не меняется.\n"
                          "Последние 20 сохранений: слева код снимка, справа подпись. "
                          "Линии * | / показывают, как ветки расходились и сливались. "
                          "В скобках — где сейчас ветки (HEAD — это ты).",
                          [Step(["log", "--oneline", "--graph", "--decorate", "-n", "20"])], safe=True,
                          success="История — в консоли ниже."))

    def act_switch_branch(self):
        self.fresh()
        code, out = run_git(["branch", "--format=%(refname:short)"], self.repo, 15)
        branches = [b.strip() for b in out.splitlines() if b.strip()] if code == 0 else []
        if not branches:
            self.info("Ветки", "Веток пока нет (нужно хотя бы одно сохранение).")
            return
        picked = self.pick_list("Перейти в ветку", "Выбери ветку (➜ — где ты сейчас):",
                                [("➜ " if b == self.state["branch"] else "   ") + b for b in branches])
        if not picked:
            return
        b = branches[picked[0]]
        if b == self.state["branch"]:
            return
        text = [f"Переходим в ветку «{b}». Файлы в папке станут такими, какие они в этой ветке.",
                "Ничего не теряется: твоя текущая ветка останется как есть."]
        if self.state["files"]:
            text += ["", "⚠ Есть несохранённые изменения. Git либо перенесёт их с собой, либо откажется, "
                         "если они мешают. Надёжнее сначала сохраниться или «Отложить изменения»."]
        def preview(m):
            m.switch(b)
            return [f"★ переедет на ветку «{b}». Сохранения не меняются — меняются только файлы в папке."]

        self.propose(Plan(f"Перейти в ветку {b}", "\n".join(text), [Step(["switch", b])], preview=preview,
                          success=f"Ты в ветке «{b}»."))

    def act_stash(self):
        self.fresh()
        if not self.state["files"]:
            self.info("Отложить", "Откладывать нечего — несохранённых изменений нет.")
            return
        label = time.strftime("Отложено %d.%m %H:%M")
        self.propose(Plan("Отложить изменения",
                          "Убираем твои несохранённые изменения в «карман» — проект станет чистым, как после "
                          "последнего сохранения. Это нужно, например, чтобы срочно переключиться на другую задачу.\n\n"
                          "Потом их можно вернуть кнопкой «Вернуть отложенное».",
                          [Step(["stash", "push", "-u", "-m", label])],
                          success="Изменения отложены. Проект чистый.\nВернуть: «Вернуть отложенное»."))

    def act_unstash(self):
        self.fresh()
        if not self.state["has_stash"]:
            self.info("Вернуть отложенное", "Отложенных изменений нет.")
            return
        self.propose(Plan("Вернуть отложенное",
                          "Достаём последние отложенные изменения из «кармана» обратно в папку проекта.\n\n"
                          "Если с тех пор ты менял те же места, может возникнуть конфликт — я подскажу.",
                          [Step(["stash", "pop"])],
                          success="Отложенные изменения вернулись в проект."))

    def act_merge_abort(self):
        self.fresh()
        if self.state["reverting"]:
            self.propose(Plan("Прервать отмену сохранения",
                              "Прерываем незаконченную отмену сохранения: всё вернётся как было до неё, "
                              "конфликт исчезнет.",
                              [Step(["revert", "--abort"])], success="Отмена прервана, всё как было."))
            return
        if not self.state["merging"]:
            self.info("Прервать слияние / отмену", "Сейчас нет незаконченного слияния или отмены — прерывать нечего.")
            return
        self.propose(Plan("Отменить слияние",
                          "Отменяем незаконченное слияние: всё вернётся как было до него, "
                          "конфликт исчезнет. Чужие изменения можно будет забрать позже ещё раз.",
                          [Step(["merge", "--abort"])],
                          preview=lambda m: ["Незаконченное слияние исчезнет. Сохранения и ветки останутся как сейчас."],
                          success="Слияние отменено, всё как было."))

    def choose(self, title, text, options):
        """Маленькое тёмное окно с выбором из нескольких кнопок. Возвращает номер варианта или None."""
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg=C["bg"])
        win.transient(self.root)
        win.resizable(False, False)
        dark_titlebar(win)
        ttk.Label(win, text=text, justify="left", wraplength=460, padding=(14, 12, 14, 6)).pack(anchor="w")
        chosen = {}
        bf = ttk.Frame(win, padding=(14, 6, 14, 14))
        bf.pack(fill="x")
        for i, opt in enumerate(options):
            ttk.Button(bf, text=opt, command=lambda k=i: (chosen.setdefault("i", k), win.destroy())
                       ).pack(fill="x", pady=2)
        ttk.Button(bf, text="Отмена", command=win.destroy).pack(fill="x", pady=(8, 0))
        win.update_idletasks()      # ставим окно по центру главного окна
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_reqwidth()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_reqheight()) // 3
        win.geometry(f"+{max(0, x)}+{max(0, y)}")
        win.grab_set()
        self.root.wait_window(win)
        return chosen.get("i")

    def act_identity(self):
        s = self.fresh()
        ident = s.get("ident") or {}
        now = (f"Сейчас: {ident['name']} <{ident['email']}> "
               f"({SCOPE_NAMES.get(ident['email_scope'], ident['email_scope'])})"
               if ident.get("name") and ident.get("email") else "Сейчас подпись не указана.")
        where = self.choose(
            "Аккаунт для сохранений",
            f"{now}\n\nИменем и почтой git подписывает каждое сохранение. Где их задать?\n\n"
            "• Для всех проектов — обычный вариант, делается один раз.\n"
            "• Только для этого проекта — если здесь нужен другой аккаунт "
            "(например, рабочий вместо личного). Он важнее общего.",
            ["Для всех проектов на компьютере", "Только для этого проекта"])
        if where is None:
            return
        scope = "global" if where == 0 else "local"
        cur_name = ident.get("name", "") if ident.get("name_scope") == scope or not ident.get("name") else ""
        cur_mail = ident.get("email", "") if ident.get("email_scope") == scope or not ident.get("email") else ""
        name = self.ask("Аккаунт", "Твоё имя (будет подписью у сохранений):", cur_name or ident.get("name", ""))
        if not name or not name.strip():
            return
        mail = self.ask("Аккаунт", "Твоя почта — та же, что в аккаунте на GitHub/GitLab\n"
                                   "(по ней сайт поймёт, что сохранения твои):", cur_mail or ident.get("email", ""))
        if not mail or not mail.strip():
            return
        flag = "--global" if scope == "global" else "--local"
        explain = ("Записываем в настройки git твоё имя и почту. Ими подписывается каждое сохранение, "
                   "чтобы в команде было видно, кто что сделал.\n\n")
        if scope == "global":
            explain += "--global — для всех проектов на этом компьютере."
            if ident.get("email_scope") == "local":
                explain += ("\n\n⚠ В этом проекте задан свой аккаунт — он важнее общего, поэтому здесь "
                            "подпись не поменяется. Чтобы поменять и тут — выбери «Только для этого проекта».")
        else:
            explain += f"--local — только для проекта {self.repo.name}; в остальных проектах подпись прежняя."
        self.propose(Plan("Сменить аккаунт", explain,
                          [Step(["config", flag, "user.name", name.strip()]),
                           Step(["config", flag, "user.email", mail.strip()])],
                          success=f"Готово. Сохранения подписываются: {name.strip()} <{mail.strip()}>"))

    def act_own_command(self):
        text = self.own_entry.get().strip()
        if text.lower().startswith("git "):
            text = text[4:].strip()
        if not text:
            self.info("Своя команда", "Напиши команду в поле внизу (без слова git), например: log --oneline")
            return
        try:
            args = shlex.split(text, posix=(sys.platform != "win32"))
            args = [a[1:-1] if len(a) >= 2 and a[0] == a[-1] == '"' else a for a in args]
        except ValueError:
            self.info("Своя команда", "Не получилось разобрать команду — проверь кавычки.")
            return
        self.propose(Plan(f"Своя команда: git {text}",
                          "Это твоя команда — я выполню её как есть, без проверок. "
                          "Если не уверен, что она делает, лучше отмени.",
                          [Step(args)], success="Команда выполнена. Ответ git — в консоли.",
                          preview=lambda m: ["Своя команда — что она изменит, заранее не предсказать. "
                                             "После выполнения граф обновится."]))
        self.own_entry.delete(0, "end")

    # ========================================== общие окна выбора
    def pick_list(self, title, prompt, items, multi=False, width=70):
        """Тёмное окно со списком. Возвращает список номеров выбранных строк или None."""
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg=C["bg"])
        win.transient(self.root)
        dark_titlebar(win)
        ttk.Label(win, text=prompt + ("\n(можно выбрать несколько: Ctrl или Shift + щелчок)" if multi else ""),
                  padding=(10, 8), justify="left").pack(anchor="w")
        frame = ttk.Frame(win, padding=(10, 0))
        frame.pack(fill="both", expand=True)
        lb = tk.Listbox(frame, font=self.mono, height=min(18, max(3, len(items))), width=width,
                        selectmode="extended" if multi else "browse", activestyle="none",
                        bg=C["field"], fg=C["fg"], selectbackground=C["sel"], selectforeground="#ffffff",
                        highlightthickness=1, highlightbackground=C["border"], highlightcolor=C["fg"])
        sb = ttk.Scrollbar(frame, command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        lb.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        for it in items:
            lb.insert("end", it)
        chosen = {}

        def ok(_=None):
            sel = list(lb.curselection())
            if sel:
                chosen["v"] = sel
                win.destroy()

        lb.bind("<Double-Button-1>", ok)
        lb.bind("<Return>", ok)
        bf = ttk.Frame(win, padding=10)
        bf.pack(fill="x")
        ttk.Button(bf, text="Выбрать", command=ok).pack(side="left")
        ttk.Button(bf, text="Отмена", command=win.destroy).pack(side="left", padx=6)
        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_reqwidth()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_reqheight()) // 3
        win.geometry(f"+{max(0, x)}+{max(0, y)}")
        win.grab_set()
        lb.focus_set()
        self.root.wait_window(win)
        return chosen.get("v")

    def list_commits(self, n=40):
        """Последние сохранения текущей ветки: [(код, строка для списка)]."""
        code, out = run_git(["log", f"-n{n}", "--date=format:%d.%m.%Y %H:%M",
                             "--format=%h%x09%ad%x09%an%x09%s"], self.repo, 15)
        if code != 0:
            return []
        # какие сохранения ещё НЕ на сервере (остальные — уже там)
        has_remote = bool(self.state.get("remote"))
        _, local_only = run_git(["log", "--format=%h", "HEAD", "--not", "--remotes"], self.repo, 15)
        local_only = set(local_only.split())
        res = []
        for ln in out.splitlines():
            parts = ln.split("\t", 3)
            if len(parts) == 4:
                h, date, author, subj = parts
                mark = "⇧" if (has_remote and h not in local_only) else "·"
                res.append((h, f"{mark} {h}  {date}  {author[:14]:14}  {subj}"))
        return res

    def pick_commit(self, title, prompt):
        commits = self.list_commits()
        if not commits:
            self.info(title, "Сохранений пока нет.")
            return None
        picked = self.pick_list(title, prompt + "\n⇧ — уже на сервере,  · — только у тебя", [c[1] for c in commits],
                                width=90)
        return commits[picked[0]][0] if picked else None

    # ========================================== ДЕЙСТВИЯ: посмотреть
    def act_diff_lines(self):
        s = self.fresh()
        if not s["files"]:
            self.info("Изменения построчно", "Изменений нет — всё уже сохранено.")
            return
        args = ["diff"] if s["unborn"] else ["diff", "HEAD"]
        untracked = [p for xy, p in s["files"] if xy == "??"]
        extra = ""
        if untracked:
            extra = ("\n\nНовые файлы, которые git ещё ни разу не сохранял, здесь не показываются целиком — "
                     "вот они: " + ", ".join(untracked[:10]) + ("…" if len(untracked) > 10 else ""))
        self.propose(Plan("Изменения построчно",
                          "Только смотрим — ничего не меняется.\n"
                          "В консоли — что поменялось с последнего сохранения:\n"
                          "• строка «diff --git a/файл b/файл» — начало нового файла;\n"
                          "• «@@ -10,4 +10,6 @@» — номер строки, где начинается кусок;\n"
                          "• строка с «+» в начале — добавлена, с «-» — удалена, без знака — не менялась." + extra,
                          [Step(args)], safe=True, success="Изменения — в консоли ниже." + extra))

    def act_show_commit(self):
        self.fresh()
        h = self.pick_commit("Подробно о сохранении", "Выбери сохранение:")
        if not h:
            return
        self.propose(Plan(f"Подробно о сохранении {h}",
                          "Только смотрим — ничего не меняется.\n"
                          "В консоли: кто и когда сохранил, подпись, список файлов, а ниже — сами изменения "
                          "(«+» — добавленная строка, «-» — удалённая).",
                          [Step(["show", "--stat", "-p", h])], safe=True, success="Подробности — в консоли ниже."))

    # ========================================== ДЕЙСТВИЯ: сервер
    def act_fetch(self):
        s = self.fresh()
        if not s["remote"]:
            self.info("Проверить сервер", "У проекта нет сервера. Подключить: «⇅ Подключить / сменить сервер…».")
            return

        def done(out):
            self.refresh_status()
            st = self.state
            lines = []
            if st["behind"]:
                lines.append(f"⇩ В твоей ветке на сервере {st['behind']} новых сохранений — забери: «1. Начать день».")
            if st["main_behind"]:
                lines.append(f"⇩ В главной ветке ({st['main']}) {st['main_behind']} сохранений, которых нет в твоей "
                             "ветке — «1. Начать день» подтянет их.")
            if "[deleted]" in out:
                lines.append("✂ Ветки, которые удалили на сервере, убраны из списка.")
            if not lines:
                lines.append("✔ Ничего нового: у тебя всё самое свежее.")
            return "\n".join(lines) + "\n\nТвои файлы не менялись. Подробности — в консоли."

        self.propose(Plan("Проверить сервер",
                          "Спрашиваем у сервера, что нового, но НЕ трогаем твои файлы и ветки — "
                          "только узнаём. Это git fetch.\n\n"
                          "--prune — заодно забыть ветки, которые на сервере уже удалили.\n\n"
                          "После этого напоминания покажут, прислали ли коллеги что-то новое.",
                          [Step(["fetch", "--prune", s["remote"]])], on_success=done,
                          preview=lambda m: ["Твои ветки не сдвинутся. Сдвинуться могут только ярлыки сервера (⇅) — "
                                             "что прислали коллеги, заранее неизвестно, граф покажет после проверки."]))

    def act_remote(self):
        s = self.fresh()
        cur = s["server"]["url"] if s.get("server") else ""
        url = self.ask("Сервер", "Адрес репозитория на сервере.\n"
                                 "На GitHub: зелёная кнопка «Code» → вкладка HTTPS → скопировать.\n"
                                 "Пример: https://github.com/имя/проект.git", cur)
        if not url or not url.strip() or url.strip() == cur:
            return
        url = url.strip()
        info = parse_remote_url(url)
        where = (f"сайт {info['host']}, репозиторий {info['owner']}/{info['repo']}" if info["host"]
                 else "папка на компьютере")
        self.gcm_cache.clear()
        if s["remote"]:
            self.propose(Plan("Сменить адрес сервера",
                              f"Меняем адрес сервера «{s['remote']}».\nБыло: {cur}\nСтанет: {url}  ({where})\n\n"
                              "Твои файлы и сохранения не меняются — git просто будет отправлять и забирать "
                              "по новому адресу.",
                              [Step(["remote", "set-url", s["remote"], url])],
                              success="Адрес сервера изменён. Проверь его: «☁ Проверить сервер»."))
        else:
            self.propose(Plan("Подключить сервер",
                              f"Подключаем сервер: {url}  ({where}).\n\n"
                              "origin — обычное название главного сервера проекта.\n"
                              "Твои файлы не меняются. Чтобы выложить туда проект — потом «5. Отправить».",
                              [Step(["remote", "add", "origin", url])],
                              success="Сервер подключён. Теперь можно отправлять: «5. Отправить на сервер»."))

    def act_clone(self):
        url = self.ask("Скачать проект", "Адрес репозитория на сервере.\n"
                                         "На GitHub: зелёная кнопка «Code» → вкладка HTTPS → скопировать.\n"
                                         "Пример: https://github.com/имя/проект.git")
        if not url or not url.strip():
            return
        url = url.strip()
        info = parse_remote_url(url)
        name = info["repo"] or re.sub(r"\.git$", "", Path(url.rstrip("/\\")).name) or "proekt"
        parent = filedialog.askdirectory(title="В какую папку положить проект?", parent=self.root)
        if not parent:
            return
        parent = Path(parent)
        name = self.ask("Скачать проект", f"Имя папки для проекта (будет создана в {parent}):", name)
        if not name or not name.strip():
            return
        target = parent / name.strip()
        if target.exists() and any(target.iterdir()):
            self.info("Скачать проект", f"Папка {target} уже есть и в ней что-то лежит. Выбери другое имя.")
            return
        where = f"{info['host']}, репозиторий {info['owner']}/{info['repo']}" if info["host"] else url

        def done(out):
            self.open_repo(target)
            return (f"Проект скачан в {target} и открыт.\n\n"
                    "Сервер уже подключён (origin). Дальше — как обычно: «2. Новая задача».")

        self.propose(Plan("Скачать проект с сервера",
                          f"Скачиваем проект ({where}) в папку {target}.\n\n"
                          "git clone копирует весь проект со всей историей и сразу подключает сервер, "
                          "так что «Начать день» и «Отправить» будут работать.\n\n"
                          "Если репозиторий закрытый, при первом скачивании git может открыть окно входа "
                          "(для GitHub — в браузере). Большой проект может качаться несколько минут.",
                          [Step(["clone", url, name.strip()], cwd=parent, timeout=1800)], on_success=done,
                          manual_after=f"Когда скачается — «Открыть проект…» и выбери папку {target}."))

    # ========================================== ДЕЙСТВИЯ: ветки
    def act_delete_branch(self):
        s = self.fresh()
        code, out = run_git(["branch", "--format=%(refname:short)"], self.repo, 15)
        branches = [b.strip() for b in out.splitlines()
                    if b.strip() and b.strip() not in (s["branch"], s["main"])] if code == 0 else []
        if not branches:
            self.info("Удалить ветку", f"Удалять нечего: кроме главной ({s['main']}) и той, где ты сейчас, "
                                       "веток нет.\n\nСвою текущую ветку удалить нельзя — сначала перейди в другую.")
            return
        picked = self.pick_list("Удалить ветку", "Какую ветку удалить?", branches)
        if not picked:
            return
        b = branches[picked[0]]
        merged = run_git(["merge-base", "--is-ancestor", b, s["main"]], self.repo, 15)[0] == 0
        flag, text = "-d", [f"Удаляем ветку «{b}»."]
        if merged:
            # влита в главную — проверено выше; -D, потому что -d сверяет с копией ветки на сервере
            flag = "-D"
            text.append(f"Вся работа из неё уже есть в главной ветке ({s['main']}), так что ничего не пропадёт.")
        else:
            ok = self.choose("Удалить ветку",
                             f"⚠ В ветке «{b}» есть сохранения, которых НЕТ в главной ветке ({s['main']}).\n"
                             "Если удалить её — эта работа пропадёт.",
                             ["Всё равно удалить (работа пропадёт)"])
            if ok is None:
                return
            flag = "-D"
            text.append("⚠ В ней есть сохранения, которых нет в главной ветке, — они пропадут. "
                        "-D — «удалить, даже если не влита».")
        steps = [Step(["branch", flag, b])]
        if s["remote"] and self.remote_branch_exists(s["remote"], b):
            both = self.choose("Удалить ветку", f"Ветка «{b}» есть и на сервере. Удалить и там?",
                               ["Удалить у себя и на сервере", "Только у себя"])
            if both is None:
                return
            if both == 0:
                steps.append(Step(["push", s["remote"], "--delete", b]))
                text.append("На сервере тоже удаляем (git push --delete) — коллеги перестанут её видеть.")
        del_remote = any(st.args[:1] == ["push"] for st in steps)

        def preview(m):
            m.delete(b)
            if del_remote:
                m.delete_remote(f"{s['remote']}/{b}")
            return [f"Ярлык «{b}» исчезнет." + ("" if merged else
                    " Сохранения, которые были только в ней, станут ни к чему не привязаны (✕).")]

        self.propose(Plan(f"Удалить ветку {b}", "\n".join(text), steps, preview=preview,
                          success=f"Ветка «{b}» удалена."))

    # ========================================== ДЕЙСТВИЯ: отменить
    def act_restore(self):
        s = self.fresh()
        kinds = []
        for xy, p in s["files"]:
            if xy in CONFLICT_CODES or "R" in xy or "C" in xy:
                continue
            if xy == "??":
                kinds.append(("new", p, f"новый     {p}"))
            elif xy[0] == "A" or s["unborn"]:
                kinds.append(("added", p, f"новый     {p}"))
            elif "D" in xy:
                kinds.append(("tracked", p, f"удалён    {p}"))
            else:
                kinds.append(("tracked", p, f"изменён   {p}"))
        if not kinds:
            self.info("Вернуть файл как было", "Нечего возвращать — несохранённых изменений нет"
                      + (" (файлы с конфликтом возвращаются кнопкой «Прервать слияние / отмену»)."
                         if s["conflicts"] else "."))
            return
        picked = self.pick_list("Вернуть файл как было", "Какие файлы вернуть как в последнем сохранении?",
                                [k[2] for k in kinds], multi=True)
        if not picked:
            return
        sel = [kinds[i] for i in picked]
        tracked = [p for k, p, _ in sel if k == "tracked"]
        added = [p for k, p, _ in sel if k == "added"]
        new = [p for k, p, _ in sel if k == "new"]
        steps, text = [], ["Возвращаем выбранные файлы к последнему сохранению.", ""]
        if tracked:
            steps.append(Step(["restore", "--source=HEAD", "--staged", "--worktree", "--"] + tracked))
            text.append("• Изменённые и удалённые файлы станут как в последнем сохранении (git restore).")
        if added:
            steps.append(Step(["rm", "-r", "-f", "--quiet", "--"] + added))
            text.append("• Новые файлы, уже отмеченные для сохранения, будут УДАЛЕНЫ с диска (git rm -f).")
        if new:
            steps.append(Step(["clean", "-f", "-d", "--"] + new))
            text.append("• Новые файлы, которых git ещё не знает, будут УДАЛЕНЫ с диска (git clean).")
        text += ["", "⚠ Несохранённые изменения в этих файлах пропадут НАВСЕГДА — git их не запоминал, "
                     "вернуть будет нельзя.",
                 "Если не уверен — лучше «⤓ Отложить изменения»: так их можно будет вернуть."]
        self.propose(Plan("Вернуть файлы как было", "\n".join(text), steps,
                          success="Готово, выбранные файлы — как в последнем сохранении."))

    def act_uncommit(self):
        s = self.fresh()
        if s["unborn"]:
            self.info("Отменить последнее сохранение", "Сохранений пока нет.")
            return
        if s["merging"]:
            self.info("Отменить последнее сохранение", "Сначала закончи или прерви слияние / отмену.")
            return
        if run_git(["rev-parse", "--verify", "--quiet", "HEAD~1"], self.repo, 15)[0] != 0:
            self.info("Отменить последнее сохранение",
                      "Это самое первое сохранение в проекте — эта кнопка его не отменяет.")
            return
        code, on_server = run_git(["branch", "-r", "--contains", "HEAD"], self.repo, 15)
        if code == 0 and on_server.strip():
            self.info("Отменить последнее сохранение — нельзя",
                      "Это сохранение уже отправлено на сервер, коллеги могли его забрать. "
                      "Если «вырезать» его из истории, у них всё сломается.\n\n"
                      "Для отправленных сохранений есть безопасный способ: «⟲ Отменить отправленное сохранение…» — "
                      "git сделает новое сохранение «наоборот».")
            return
        _, subj = run_git(["log", "-1", "--format=%h  %s"], self.repo, 15)
        self.propose(Plan("Отменить последнее сохранение",
                          f"Отменяем последнее сохранение: {subj.strip()}\n\n"
                          "Изменения из него НЕ пропадут — они вернутся в файлы как несохранённые. "
                          "Можно поправить и сохранить заново (например, если ошибся в подписи или забыл файл).\n\n"
                          "git reset --soft HEAD~1 — «сдвинуть ветку на одно сохранение назад, файлы не трогать».\n"
                          "Так можно только с сохранениями, которые ещё не отправлены на сервер.",
                          [Step(["reset", "--soft", "HEAD~1"])],
                          preview=lambda m: (m.reset_head(m.commits[m.head_commit]["parents"][0]),
                                             ["Ярлык ветки шагнёт на одно сохранение назад. Отменённое сохранение "
                                              "(✕) уйдёт из ветки, а его изменения вернутся в файлы."])[1],
                          success="Сохранение отменено, изменения вернулись несохранёнными. "
                                  "Поправь и нажми «4. Сохранить работу»."))

    def act_revert(self):
        s = self.fresh()
        if s["files"] or s["merging"]:
            self.info("Отменить отправленное сохранение",
                      "Сначала сохрани («4. Сохранить работу») или отложи («⤓ Отложить изменения») текущие изменения "
                      "— или закончи слияние.")
            return
        h = self.pick_commit("Отменить сохранение", "Какое сохранение отменить?")
        if not h:
            return
        _, subj = run_git(["log", "-1", "--format=%s", h], self.repo, 15)
        _, parents = run_git(["rev-list", "--parents", "-n", "1", h], self.repo, 15)
        args = ["revert", "--no-edit"]
        text = [f"Отменяем сохранение {h}: «{subj.strip()}».", "",
                "git revert не стирает его из истории, а делает НОВОЕ сохранение, которое всё возвращает назад "
                "(добавленное — убирает, удалённое — возвращает). Поэтому так безопасно отменять даже то, "
                "что уже на сервере и что коллеги забрали.", "",
                "Если после этого сохранения те же строки меняли ещё — будет конфликт, я подскажу, что делать.", "",
                "Новое сохранение останется у тебя — чтобы отмену увидели коллеги, потом «5. Отправить»."]
        if len(parents.split()) > 2:
            args += ["-m", "1"]
            text += ["", "Это сохранение-слияние, поэтому добавляем -m 1: «отменить то, что было влито»."]
        def preview(m):
            m.commit(f"Отмена «{subj.strip()}»")
            return [f"Старое сохранение {h} останется на месте — появится НОВОЕ, которое его отменяет."]

        self.propose(Plan("Отменить сохранение", "\n".join(text), [Step(args + [h])], preview=preview,
                          success="Готово: появилось новое сохранение, которое отменяет выбранное.\n"
                                  "Отправь его коллегам: «5. Отправить на сервер»."))

    # ========================================== ДЕЙСТВИЯ: настройки
    def act_gitignore(self):
        self.fresh()
        path = self.repo / ".gitignore"
        try:
            existing = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            existing = []
        have = {ln.strip() for ln in existing}
        options = [f"{name}  ({len([x for x in rules if x not in have])} новых правил)"
                   for name, rules in GITIGNORE_TEMPLATES] + ["Своё правило…"]
        now = (f"Сейчас в .gitignore {len([x for x in existing if x.strip() and not x.startswith('#')])} правил."
               if existing else "Файла .gitignore пока нет — создам.")
        picked = self.pick_list(".gitignore — что не сохранять",
                                f"{now}\nGit не будет сохранять файлы, подходящие под эти правила "
                                "(временные файлы, сборку, личные настройки).\nВыбери набор:", options, multi=True,
                                width=50)
        if not picked:
            return
        add_lines, names = [], []
        for i in picked:
            if i == len(GITIGNORE_TEMPLATES):
                rule = self.ask(".gitignore", "Правило. Примеры:\n"
                                              "  *.log — все файлы с расширением .log\n"
                                              "  build/ — вся папка build\n"
                                              "  secret.txt — один файл")
                if rule and rule.strip():
                    add_lines.append(rule.strip())
                    names.append("своё правило")
            else:
                name, rules = GITIGNORE_TEMPLATES[i]
                add_lines += rules
                names.append(name)
        add_lines = [x for i, x in enumerate(add_lines) if x not in have and x not in add_lines[:i]]
        if not add_lines:
            self.info(".gitignore", "Все эти правила уже есть в .gitignore.")
            return

        def write():
            with open(path, "a", encoding="utf-8") as f:
                if existing and existing[-1].strip():
                    f.write("\n")
                f.write(f"# {', '.join(names)}\n" + "\n".join(add_lines) + "\n")
            return True, f"В .gitignore дописано правил: {len(add_lines)}"

        def done(out):
            code, tracked = run_git(["ls-files", "-ci", "--exclude-standard"], self.repo, 15)
            tracked = [t for t in tracked.splitlines() if t.strip()] if code == 0 else []
            msg = ("Готово. Не забудь сохранить сам файл .gitignore («4. Сохранить работу») — "
                   "тогда правила будут и у коллег.")
            if tracked:
                msg += ("\n\n⚠ Эти файлы git уже хранит, на них правила не действуют:\n   • " +
                        "\n   • ".join(tracked[:15]) + ("\n   …" if len(tracked) > 15 else "") +
                        "\nЧтобы git перестал их хранить (на диске они останутся), введи в поле «Своя команда»:\n"
                        "   rm -r --cached " + " ".join(q(t) for t in tracked[:15]))
            return msg

        self.propose(Plan(".gitignore: не сохранять файлы",
                          f"Дописываем в файл .gitignore (папка проекта) правила «{', '.join(names)}»:\n    " +
                          "\n    ".join(add_lines) + "\n\n"
                          "Файлы, подходящие под эти правила, git перестанет замечать и сохранять. "
                          "На диске они остаются, просто не попадают в сохранения и на сервер.",
                          [Step(func=write, cwd=self.repo,
                                show=f"открой (или создай) файл .gitignore в папке {self.repo} и допиши в конец "
                                     "строки:\n    " + "\n    ".join(add_lines))],
                          on_success=done))

    # ================================================== ПЕСОЧНИЦА
    def act_sandbox(self):
        moy = SANDBOX_DIR / "moy_proekt"
        server = SANDBOX_DIR / "server.git"
        kol = SANDBOX_DIR / "kollega"
        if (moy / ".git").exists() and (kol / ".git").exists():
            self.open_repo(moy)
            self.info("Песочница открыта",
                      f"Папка: {SANDBOX_DIR}\n\n"
                      "• moy_proekt — твой проект (с ним работают кнопки слева);\n"
                      "• server.git — «сервер компании»;\n"
                      "• kollega — компьютер коллеги Вани.\n\n"
                      "Меняй файлы в moy_proekt («Открыть папку»), жми кнопки по порядку, "
                      "а «Коллега Ваня поработал» — чтобы на сервере появились чужие изменения.\n\n"
                      "Чтобы начать заново, закрой программу и удали папку GitTrainer.")
            return
        if SANDBOX_DIR.exists() and any(SANDBOX_DIR.iterdir()):
            self.info("Песочница", f"Папка {SANDBOX_DIR} уже есть, но песочница в ней неполная.\n\n"
                                   "Удали эту папку вручную и нажми «Песочница» снова.")
            return

        def mk():
            SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
            return True, f"Создана папка {SANDBOX_DIR}"

        def files():
            (moy / "README.md").write_text(
                "# Учебный проект\n\nЭто песочница Гит-помощника. Меняй файлы как хочешь — это тренировка.\n",
                encoding="utf-8")
            (moy / "список_дел.txt").write_text(
                "Список дел команды\n==================\n- настроить проект\n", encoding="utf-8")
            return True, "Созданы файлы README.md и список_дел.txt"

        steps = [
            Step(func=mk, show=f"создаём папку {SANDBOX_DIR}", cwd=Path.home(),
                 manual=f'mkdir "{SANDBOX_DIR}"'),
            Step(["init", "--bare", "server.git"], cwd=SANDBOX_DIR),
            Step(["symbolic-ref", "HEAD", "refs/heads/main"], cwd=server),
            Step(["clone", "server.git", "moy_proekt"], cwd=SANDBOX_DIR),
            Step(["symbolic-ref", "HEAD", "refs/heads/main"], cwd=moy),
            Step(["config", "user.name", "Я (тренировка)"], cwd=moy),
            Step(["config", "user.email", "ya@trenirovka.local"], cwd=moy),
            Step(func=files, cwd=moy,
                 show=f"создай в папке {moy} два файла с любым текстом: README.md и список_дел.txt"),
            Step(["add", "-A"], cwd=moy),
            Step(["commit", "-m", "Начало проекта"], cwd=moy),
            Step(["push", "-u", "origin", "main"], cwd=moy),
            Step(["clone", "server.git", "kollega"], cwd=SANDBOX_DIR),
            Step(["config", "user.name", "Коллега Ваня"], cwd=kol),
            Step(["config", "user.email", "vanya@trenirovka.local"], cwd=kol),
        ]

        def done(out):
            self.open_repo(moy)
            return ("Песочница готова!\n\n"
                    "• moy_proekt — твой проект (с ним работают кнопки слева);\n"
                    "• server.git — «сервер компании» (как GitHub, только в папке);\n"
                    "• kollega — компьютер коллеги Вани.\n\n"
                    "Попробуй так: 1. Начать день → 2. Новая задача → поменяй список_дел.txt "
                    "(«Открыть папку») → 4. Сохранить → 5. Отправить → 6. Задача готова.\n"
                    "А потом нажми «Коллега Ваня поработал» и снова «1. Начать день».")

        self.propose(Plan("Создать песочницу (тренировочный проект)",
                          f"Создаю в папке {SANDBOX_DIR} учебную «компанию»:\n"
                          "• свой «сервер» (папка server.git — работает без интернета),\n"
                          "• твой проект moy_proekt,\n"
                          "• компьютер коллеги Вани (папка kollega).\n\n"
                          "Имена авторов записываются только внутри песочницы, твои общие настройки git не меняются.",
                          steps, on_success=done,
                          manual_after="Когда выполнишь все блоки — нажми «Песочница» ещё раз, она откроется."))

    def act_colleague(self):
        kol = SANDBOX_DIR / "kollega"
        if not (kol / ".git").exists():
            self.info("Коллега", "Папки коллеги нет — пересоздай песочницу.")
            return
        task = random.choice(COLLEAGUE_TASKS)
        line = f"- [Ваня] {task}"

        def edit():
            with open(kol / "список_дел.txt", "a", encoding="utf-8") as f:
                f.write(line + "\n")
            return True, f"Ваня дописал в конец список_дел.txt строку: {line}"

        steps = [
            Step(["pull", "--no-rebase"], cwd=kol),
            Step(func=edit, cwd=kol,
                 show=f"за Ваню: открой {kol / 'список_дел.txt'}, допиши в конец строку «{line}» и сохрани"),
            Step(["add", "-A"], cwd=kol),
            Step(["commit", "-m", f"Добавил задачу: {task}"], cwd=kol),
            Step(["push"], cwd=kol),
        ]

        def done(out):
            self.day_started = False
            return ("Ваня отправил свои изменения на сервер. Теперь там есть то, чего нет у тебя.\n\n"
                    "Забери их: «1. Начать день».\n"
                    "Если ты тоже дописывал что-то в конец список_дел.txt — получишь учебный конфликт. "
                    "Это нормально: разбери его по подсказкам.")

        self.propose(Plan("Коллега Ваня поработал",
                          "Имитируем работу коллеги: Ваня у себя (папка kollega) забирает свежую версию, "
                          "дописывает строку в список_дел.txt, сохраняет и отправляет на сервер.\n"
                          "Команды выполняются в папке Вани — твой проект не трогается.",
                          steps, on_success=done, on_copy=lambda: setattr(self, "day_started", False),
                          preview=self._preview_colleague(task)))

    def _preview_colleague(self, task):
        remote, main = self.state.get("remote") or "origin", self.state.get("main") or "main"

        def preview(m):
            base = m.remotes.get(f"{remote}/{main}")
            if not base:
                return []
            m.remotes[f"{remote}/{main}"] = m.ghost([base], f"Ваня: {task}")
            return [f"У Вани появится новое сохранение, и он отправит его на сервер: сдвинется ярлык ⇅ {remote}/{main}. "
                    "Твои ветки не изменятся — у тебя этого сохранения пока нет."]
        return preview

    # ===================================================== выход
    def on_close(self):
        if self.busy and not messagebox.askyesno(
                APP_TITLE, "Сейчас выполняется команда git. Закрыть всё равно?", parent=self.root):
            return
        s = self.state if self.state.get("repo") else {}
        warn = []
        if s.get("files"):
            warn.append(f"несохранённые изменения ({len(s['files'])} файл.)")
        if s.get("ahead") or (s.get("remote") and not s.get("upstream") and not s.get("unborn")
                              and not s.get("detached") and s.get("branch") != s.get("main")):
            warn.append("сохранения, которых нет на сервере")
        if warn and not messagebox.askyesno(
                "Перед уходом",
                "У тебя есть " + " и ".join(warn) + ".\n\n"
                "В компании перед уходом обычно сохраняют и отправляют работу "
                "(кнопка «7. Закончить день»).\n\nВсё равно закрыть?", parent=self.root):
            return
        self.root.destroy()


def main():
    if sys.platform == "win32":
        try:  # чёткий текст на мониторах с масштабированием
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa
            pass
    root = tk.Tk()
    App(root)
    dark_titlebar(root)
    root.mainloop()


if __name__ == "__main__":
    main()
