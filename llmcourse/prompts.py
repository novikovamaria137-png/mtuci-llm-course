"""Промпт как часть программы. Урок 2.2.

Промпт хранится отдельно от кода вызова: у него есть имя, версия и явный
список параметров. Это позволяет менять формулировку, не трогая логику,
и видеть в журнале, какой версией промпта получен ответ.

Почему не str.format. Во-первых, .format умеет обращаться к атрибутам объекта
("{x.__class__}"), и на шаблоне, пришедшем извне, это дыра в безопасности.
Во-вторых, нам нужна подстановка и ничего больше — а всё лишнее в инструменте
рано или поздно кто-нибудь применит.

Правила шаблона те же, что в Python, чтобы знание переносилось:
    {name}  — параметр
    {{      — литеральная открывающая скобка
    }}      — литеральная закрывающая скобка
Одиночная } без пары — ошибка. Это не придирка: чаще всего она означает
незакрытый или неверно записанный параметр.
"""
import re
from dataclasses import dataclass

# Имя параметра — любой допустимый идентификатор, включая кириллицу: Python это
# разрешает. Но в примерах курса имена латинские — так принято, и так их видно
# в чужом коде без сюрпризов с раскладкой.
_NAME = re.compile(r"[^\W\d]\w*")


class PromptError(ValueError):
    """Ошибка в шаблоне промпта или в его заполнении."""


def parse(text):
    """Разбирает шаблон в список кусков: ("lit", строка) или ("field", имя)."""
    out, buf, i, n = [], [], 0, len(text)

    def flush():
        if buf:
            out.append(("lit", "".join(buf)))
            buf.clear()

    while i < n:
        ch = text[i]
        if ch == "{":
            if i + 1 < n and text[i + 1] == "{":
                buf.append("{"); i += 2; continue
            j = text.find("}", i + 1)
            if j == -1:
                raise PromptError("незакрытая « { » в шаблоне")
            name = text[i + 1:j]
            if not _NAME.fullmatch(name):
                raise PromptError(
                    f"недопустимое имя параметра: {{{name}}}. "
                    "Для литеральной скобки используйте {{ и }}")
            flush()
            out.append(("field", name))
            i = j + 1
            continue
        if ch == "}":
            if i + 1 < n and text[i + 1] == "}":
                buf.append("}"); i += 2; continue
            raise PromptError(
                "одиночная « } » в шаблоне. Для литеральной скобки пишите }}")
        buf.append(ch); i += 1
    flush()
    return out


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    template: str
    system: str = ""

    @property
    def fields(self):
        """Имена параметров, которые нужно передать при заполнении."""
        parts = parse(self.template) + parse(self.system)
        return sorted({v for kind, v in parts if kind == "field"})

    def render(self, **values):
        """Заполняет шаблон. Молча ничего не проглатывает."""
        need, got = set(self.fields), set(values)
        if need - got:
            raise PromptError(
                f"{self.label()}: не переданы параметры {sorted(need - got)}")
        if got - need:
            raise PromptError(
                f"{self.label()}: лишние параметры {sorted(got - need)}. "
                f"Ожидались {self.fields}. Опечатка в имени — частая причина")

        def fill(text):
            return "".join(
                v if kind == "lit" else str(values[v]) for kind, v in parse(text))

        return {"system": fill(self.system), "user": fill(self.template)}

    def label(self):
        """Метка для журнала: по ней потом видно, чем получен ответ."""
        return f"{self.name}@{self.version}"


class Registry:
    """Все промпты проекта в одном месте."""

    def __init__(self):
        self._items = {}

    def add(self, prompt):
        old = self._items.get(prompt.name)
        if old is not None and old.version == prompt.version:
            raise PromptError(
                f"промпт {prompt.name} версии {prompt.version} уже зарегистрирован. "
                "Меняете формулировку — поднимите версию")
        self._items[prompt.name] = prompt
        return prompt

    def get(self, name):
        if name not in self._items:
            raise PromptError(f"промпт {name} не найден. Есть: {sorted(self._items)}")
        return self._items[name]

    def names(self):
        return sorted(self._items)
