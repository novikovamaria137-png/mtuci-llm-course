"""Вызов функций и цикл агента. Урок 2.4.

Ключевая мысль урока, дословно из документации GigaChat:
«Модели не исполняют функции самостоятельно, а принимают решения о работе
с ними, опираясь на имеющиеся знания, текущий разговор и описание функций
из запроса».

То есть модель возвращает не результат, а намерение: имя функции и аргументы.
Выполняет — ваша программа. Отсюда всё остальное содержание модуля: раз
выполняете вы, то вы и отвечаете за то, что именно будет выполнено.
"""
import json
import re
from dataclasses import dataclass, field

NAME_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


class ToolError(Exception):
    """Проблема с инструментом: его нет, аргументы не те, вызов запрещён."""


@dataclass
class Tool:
    """Описание инструмента, доступного модели.

    parameters — JSON-схема аргументов, тот же формат, что в уроке 2.3.
    writes — признак того, что инструмент меняет состояние мира. Отделён
    от читающих намеренно: неверный читающий вызов стоит времени,
    неверный изменяющий может стоить гораздо большего.
    """
    name: str
    description: str
    parameters: dict
    fn: callable
    writes: bool = False

    def __post_init__(self):
        if not NAME_RE.fullmatch(self.name):
            raise ToolError(
                f"недопустимое имя инструмента {self.name!r}: "
                "только латиница, цифры и подчёркивание")
        if not self.description.strip():
            raise ToolError(
                f"{self.name}: пустое описание. Модель выбирает инструмент "
                "по описанию — без него выбор будет случайным")
        if self.parameters.get("type") != "object":
            raise ToolError(f"{self.name}: parameters должен быть объектом JSON-схемы")

    def describe(self):
        """Описание в том виде, в каком его ждёт API."""
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters}


class Registry:
    """Белый список инструментов.

    Модель может назвать что угодно, в том числе несуществующее. Реестр —
    единственное место, где решается, будет ли что-то выполнено.
    """

    def __init__(self, tools=()):
        self._tools = {}
        for t in tools:
            self.add(t)

    def add(self, tool):
        if tool.name in self._tools:
            raise ToolError(f"инструмент {tool.name} уже зарегистрирован")
        self._tools[tool.name] = tool
        return tool

    def names(self):
        return sorted(self._tools)

    def describe(self):
        """Массив описаний для передачи в запрос."""
        return [self._tools[n].describe() for n in self.names()]

    # ── проверка аргументов ────────────────────────────────────────
    _TYPES = {"string": str, "integer": int, "number": (int, float),
              "boolean": bool, "array": list, "object": dict}

    def _validate(self, tool, args):
        problems = []
        if not isinstance(args, dict):
            return [f"аргументы должны быть объектом, получен {type(args).__name__}"]
        props = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])

        for name in required:
            if name not in args:
                problems.append(f"нет обязательного аргумента {name!r}")

        for name, value in args.items():
            if name not in props:
                problems.append(
                    f"аргумент {name!r} не описан в схеме. "
                    "Модель могла его выдумать")
                continue
            spec = props[name]
            expected = self._TYPES.get(spec.get("type"))
            if expected and not isinstance(value, expected):
                problems.append(
                    f"аргумент {name!r}: ожидался {spec['type']}, "
                    f"получен {type(value).__name__}")
                continue
            if "enum" in spec and value not in spec["enum"]:
                problems.append(
                    f"аргумент {name!r}: значение {value!r} не входит "
                    f"в допустимые {spec['enum']}")
        return problems

    # ── вызов ──────────────────────────────────────────────────────
    def call(self, name, args, allow_writes=False):
        """Выполняет инструмент после всех проверок.

        allow_writes выключен по умолчанию. Изменяющие операции требуют
        явного разрешения на стороне вызывающего кода — чтобы включение
        такой возможности было осознанным решением, а не умолчанием.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(
                f"инструмент {name!r} не зарегистрирован. "
                f"Доступны: {self.names()}")
        if tool.writes and not allow_writes:
            raise ToolError(
                f"инструмент {name!r} изменяет состояние и запрещён "
                "в этом режиме. Разрешите явно, если это то, что нужно")
        problems = self._validate(tool, args)
        if problems:
            raise ToolError(f"{name}: " + "; ".join(problems))
        return tool.fn(**args)


# ── цикл агента ────────────────────────────────────────────────────
@dataclass
class Step:
    n: int
    kind: str            # "call" | "answer" | "error"
    tool: str = ""
    args: dict = field(default_factory=dict)
    result: object = None
    error: str = ""


def run_agent(decide, registry, task, max_steps=5, allow_writes=False):
    """Цикл агента.

    decide — функция, которой передаётся задача и журнал уже сделанного;
    она возвращает либо {"tool": имя, "args": {...}}, либо {"answer": текст}.
    В настоящем приложении её роль играет модель.

    Агент — это не магия, а цикл с ограничением. Ограничений здесь три,
    и каждое закрывает свой способ никогда не остановиться:
      max_steps          — общий предел числа шагов;
      повтор вызова      — тот же инструмент с теми же аргументами подряд;
      ошибка инструмента — возвращается модели, но шаг всё равно потрачен.
    """
    if max_steps < 1:
        raise ValueError("max_steps должно быть не меньше 1")

    trace, last_call = [], None
    for n in range(1, max_steps + 1):
        decision = decide(task, trace)

        if "answer" in decision:
            trace.append(Step(n=n, kind="answer", result=decision["answer"]))
            return {"answer": decision["answer"], "trace": trace, "stopped": "ответ"}

        name = decision.get("tool")
        args = decision.get("args", {})

        signature = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if signature == last_call:
            trace.append(Step(n=n, kind="error", tool=name, args=args,
                              error="повтор того же вызова с теми же аргументами"))
            return {"answer": None, "trace": trace, "stopped": "зацикливание"}
        last_call = signature

        try:
            result = registry.call(name, args, allow_writes=allow_writes)
            trace.append(Step(n=n, kind="call", tool=name, args=args, result=result))
        except ToolError as e:
            trace.append(Step(n=n, kind="error", tool=name, args=args, error=str(e)))

    return {"answer": None, "trace": trace, "stopped": "исчерпан лимит шагов"}
