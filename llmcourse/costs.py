"""Расчёт стоимости эксплуатации. Урок 2.2.

Нужен для итогового проекта модуля: там требуется расчёт стоимости
на 1000 запросов. Все цены задаются вызывающей стороной — модуль не знает
и не выдумывает тарифов.

Две схемы оплаты, встречающиеся у поставщиков:
  раздельная — своя цена за входные и за выходные токены;
  пакетная   — единая цена за токен (тогда price_in == price_out).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    """Цена в рублях за 1000 токенов."""
    per_1k_in: float
    per_1k_out: float

    @classmethod
    def flat(cls, per_1k):
        """Единая ставка: пакетная схема оплаты."""
        return cls(per_1k, per_1k)


def one_call(tokens_in, tokens_out, price):
    return tokens_in / 1000 * price.per_1k_in + tokens_out / 1000 * price.per_1k_out


def run(n_requests, tokens_in, tokens_out, price, retry_factor=1.0):
    """Стоимость серии одинаковых запросов.

    retry_factor — во сколько раз больше обращений реально уходит из-за
    повторов. 1.0 означает «повторов нет». Значение берётся из наблюдений,
    а не из головы: посчитайте долю повторов на пилотном прогоне.
    """
    if retry_factor < 1:
        raise ValueError("retry_factor не может быть меньше 1")
    calls = n_requests * retry_factor
    return {
        "calls": calls,
        "tokens_in": calls * tokens_in,
        "tokens_out": calls * tokens_out,
        "cost": calls * one_call(tokens_in, tokens_out, price),
    }


def conversation(n_turns, user_tokens, answer_tokens, price, system_tokens=0):
    """Стоимость диалога из n_turns ходов БЕЗ обрезки истории.

    На каждом ходу заново отправляется вся предыдущая переписка. Поэтому
    суммарный расход растёт не линейно, а квадратично по числу ходов.
    Возвращает список по ходам — чтобы это было видно, а не постулировалось.
    """
    rows, history, total = [], system_tokens, 0.0
    for turn in range(1, n_turns + 1):
        history += user_tokens
        c = one_call(history, answer_tokens, price)
        total += c
        history += answer_tokens
        rows.append({"turn": turn, "tokens_in": history - answer_tokens,
                     "cost": c, "total": total})
    return rows
