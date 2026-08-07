"""Клиент для работы с языковой моделью. Уроки 2.1–2.2.

Написан на OpenAI-совместимый интерфейс: работает с российскими API
и с локальными рантаймами. Смена поставщика — правка .env, не кода.
Без ключа работает в автономном режиме на заглушке.
"""
import time, random, hashlib
from dataclasses import dataclass
from . import config


@dataclass
class Usage:
    """Накопительный счётчик расхода."""
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    price_in: float = 0.0     # рублей за 1000 входных токенов
    price_out: float = 0.0    # рублей за 1000 выходных

    @property
    def cost(self):
        return (self.tokens_in / 1000 * self.price_in +
                self.tokens_out / 1000 * self.price_out)

    def report(self):
        return (f"обращений: {self.calls}   "
                f"токенов: {self.tokens_in} вход / {self.tokens_out} выход   "
                f"стоимость: {self.cost:.4f} руб.")


def approx_tokens(text):
    """Грубая ОЦЕНКА числа токенов до вызова API.

    Это оценка, а не замер: точное число даёт токенизатор конкретной
    модели (см. урок 1.2). Нужна, чтобы прикинуть стоимость заранее.
    """
    return max(1, len(text) // 3)


class LLM:
    def __init__(self, price_in=0.0, price_out=0.0, max_retries=4, timeout=60):
        cfg = config.settings()
        self.offline = cfg["OFFLINE"]
        self.model = cfg["MODEL"]
        self.base_url = cfg["LLM_BASE_URL"]
        self.max_retries = max_retries
        self.usage = Usage(price_in=price_in, price_out=price_out)
        self._client = None
        if not self.offline:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url,
                                  api_key=cfg["LLM_API_KEY"],
                                  timeout=timeout)

    def _offline_answer(self, messages):
        """Детерминированный ответ: одинаковый запрос — одинаковый ответ."""
        text = " ".join(m["content"] for m in messages)
        h = hashlib.sha256(text.encode()).hexdigest()[:6]
        return (f"[автономный режим] Ответ-заглушка {h}. "
                f"Получено сообщений: {len(messages)}, символов: {len(text)}. "
                f"Подставьте ключ, чтобы обратиться к модели.")

    @staticmethod
    def _is_retryable(e):
        """Повторяем только то, что имеет шанс пройти со второго раза."""
        if type(e).__name__ in ("RateLimitError", "APITimeoutError",
                                "APIConnectionError", "InternalServerError",
                                "TimeoutError", "ConnectionError"):
            return True
        return getattr(e, "status_code", None) in (408, 429, 500, 502, 503, 504)

    def _with_retry(self, fn):
        """Экспоненциальная задержка со случайной добавкой."""
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as e:
                if not self._is_retryable(e) or attempt == self.max_retries - 1:
                    raise
                time.sleep(0.5 * (2 ** attempt) + random.uniform(0, 0.3))

    def ask(self, prompt, system=None, temperature=0.2, max_tokens=None):
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        self.usage.calls += 1

        if self.offline:
            answer = self._offline_answer(messages)
            self.usage.tokens_in += approx_tokens(" ".join(m["content"] for m in messages))
            self.usage.tokens_out += approx_tokens(answer)
            return answer

        def call():
            kw = dict(model=self.model, messages=messages, temperature=temperature)
            if max_tokens:
                kw["max_tokens"] = max_tokens
            return self._client.chat.completions.create(**kw)

        resp = self._with_retry(call)
        u = getattr(resp, "usage", None)
        if u:
            self.usage.tokens_in += getattr(u, "prompt_tokens", 0)
            self.usage.tokens_out += getattr(u, "completion_tokens", 0)
        return resp.choices[0].message.content
