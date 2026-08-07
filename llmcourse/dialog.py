"""Многошаговый диалог. Урок 2.2.

Модель не помнит предыдущие сообщения. Память диалога — это то, что вы
сами прикладываете к каждому запросу. Отсюда два следствия, которые
и составляют содержание модуля:

1. История растёт, и вместе с ней растёт стоимость каждого следующего хода.
2. История рано или поздно упрётся в размер контекста модели.

Поэтому историю приходится обрезать, и делать это осознанно.

Ограничение поставщика, которое надо учитывать. В GigaChat системный промпт
должен быть ровно один и первым сообщением; иначе API отвечает ошибкой 422
«system message must be the first message» (см. OpenAPI-спецификацию, поле
messages). Наш класс держит это правило по умолчанию — так код остаётся
переносимым.
"""
from dataclasses import dataclass, field


def approx_tokens(text):
    """ОЦЕНКА числа токенов, не замер. См. урок 2.1."""
    return max(1, len(text) // 3)


@dataclass
class Dialog:
    system: str = ""
    budget_tokens: int = 4000          # сколько токенов истории готовы отправлять
    _turns: list = field(default_factory=list)   # [(role, content), ...] без system

    # ── наполнение ────────────────────────────────────────────────────
    def add(self, role, content):
        if role == "system":
            raise ValueError(
                "системное сообщение задаётся один раз при создании диалога: "
                "многие поставщики принимают ровно один system и только первым")
        if role not in ("user", "assistant"):
            raise ValueError(f"неизвестная роль: {role}")
        self._turns.append((role, content))
        return self

    def user(self, content):
        return self.add("user", content)

    def assistant(self, content):
        return self.add("assistant", content)

    # ── подсчёт ───────────────────────────────────────────────────────
    def tokens(self, turns=None):
        turns = self._turns if turns is None else turns
        total = approx_tokens(self.system) if self.system else 0
        return total + sum(approx_tokens(c) for _, c in turns)

    # ── обрезка ───────────────────────────────────────────────────────
    def fit(self):
        """Возвращает историю, укладывающуюся в бюджет.

        Системное сообщение не выбрасывается никогда: в нём правила работы,
        без него модель меняет поведение целиком. Обрезаются самые старые
        ходы. Последний ход пользователя сохраняется всегда — иначе запрос
        теряет смысл.
        """
        base = approx_tokens(self.system) if self.system else 0
        if base >= self.budget_tokens and self._turns:
            raise ValueError(
                "системное сообщение само не помещается в бюджет: "
                f"{base} токенов при бюджете {self.budget_tokens}")

        kept, total = [], base
        for role, content in reversed(self._turns):
            need = approx_tokens(content)
            if total + need > self.budget_tokens and kept:
                break
            total += need
            kept.append((role, content))
        return list(reversed(kept))

    def messages(self):
        """Готовый массив messages для запроса к модели."""
        out = [{"role": "system", "content": self.system}] if self.system else []
        out += [{"role": r, "content": c} for r, c in self.fit()]
        return out

    def dropped(self):
        """Сколько старых ходов не вошло в запрос."""
        return len(self._turns) - len(self.fit())
