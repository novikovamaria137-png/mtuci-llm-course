"""Единая точка настройки. Урок 2.1.

Ключи НИКОГДА не пишутся в коде. Порядок поиска:
  1. Colab Secrets  (значок ключа слева в Colab)
  2. переменные окружения
  3. файл .env рядом с проектом
Если ключа нет — включается автономный режим на заглушке.
"""
import os
from pathlib import Path

ENV_KEYS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")


def _from_colab(name):
    try:
        from google.colab import userdata          # есть только в Colab
        return userdata.get(name)
    except Exception:
        return None


def _from_dotenv(name, path=".env"):
    p = Path(path)
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip('"').strip("'")
    return None


def get(name, default=None):
    """Достаёт значение из Colab Secrets, окружения или .env."""
    return _from_colab(name) or os.environ.get(name) or _from_dotenv(name) or default


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def settings():
    """Возвращает конфигурацию и признак автономного режима."""
    cfg = {k: get(k) for k in ENV_KEYS}
    cfg["OFFLINE"] = not bool(cfg["LLM_API_KEY"])
    cfg["MODEL"] = cfg["LLM_MODEL"] or "demo-model"
    return cfg


def describe():
    """Человекочитаемый отчёт об окружении."""
    cfg = settings()
    where = "Google Colab" if in_colab() else "локальная среда"
    return "\n".join([
        f"Среда:            {where}",
        f"Модель:           {cfg['MODEL']}",
        f"Базовый адрес:    {cfg['LLM_BASE_URL'] or 'не задан'}",
        f"Ключ:             {'найден' if not cfg['OFFLINE'] else 'НЕ найден'}",
        f"Режим:            {'автономный (заглушка)' if cfg['OFFLINE'] else 'обращение к API'}",
    ])
