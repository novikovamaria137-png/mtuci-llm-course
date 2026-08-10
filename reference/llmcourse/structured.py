"""Структурированный вывод. Урок 2.3.

Модель возвращает текст. Даже когда вы просите JSON и даже когда поставщик
поддерживает строгую схему, на вход разбора приходит строка — и в ней, кроме
самого JSON, регулярно оказывается лишнее:

    ```json ... ```      обрамление markdown
    Вот результат: {...}  вежливая преамбула
    {...} Надеюсь, помог  постамбула
    {"a": 1,              обрыв по max_tokens

Поэтому разбор должен быть устойчивым, а проверка — строгой. Модуль делает
ровно две вещи: достаёт JSON из шумного текста и проверяет его на соответствие
ожидаемой форме.
"""
import json
import re

FENCE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.S)


class StructureError(ValueError):
    """Ответ модели не удалось привести к ожидаемой форме."""


def strip_fences(text):
    """Снимает обрамление ```...```, если оно есть."""
    m = FENCE.search(text)
    return m.group(1) if m else text


def find_json(text):
    """Находит первый сбалансированный JSON-объект или массив.

    Сканер посимвольный, а не регулярное выражение: скобки встречаются внутри
    строк, и регулярное выражение на этом ломается. Пример, на котором ломаются
    почти все самодельные варианты:

        {"note": "закрывающая } внутри строки"}
    """
    src = strip_fences(text)
    start = None
    for i, ch in enumerate(src):
        if ch in "{[":
            start = i
            break
    if start is None:
        raise StructureError(
            "в ответе модели нет ни { ни [ — JSON отсутствует. "
            f"Начало ответа: {src[:80]!r}")

    opener = src[start]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False

    for i in range(start, len(src)):
        ch = src[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return src[start:i + 1]

    raise StructureError(
        "JSON начался, но не закончился. Самая частая причина — ответ обрезан "
        "по max_tokens: проверьте finish_reason, при значении 'length' "
        "поднимите потолок")


def parse(text):
    """Достаёт JSON из ответа модели и разбирает его."""
    raw = find_json(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise StructureError(
            f"JSON найден, но не разобран: {e.msg} (позиция {e.pos}). "
            f"Фрагмент: {raw[max(0, e.pos - 30):e.pos + 30]!r}")


# ── проверка формы ────────────────────────────────────────────────────
def check(data, required=(), types=None):
    """Проверяет разобранные данные на соответствие ожиданиям.

    required — имена обязательных полей;
    types    — словарь «поле: тип» для полей, тип которых важен.

    Возвращает список претензий. Пустой список означает, что всё в порядке.
    Список, а не первая ошибка: за один проход видно всё, что не так, —
    это заметно быстрее при отладке промпта.
    """
    problems = []
    if not isinstance(data, dict):
        return [f"ожидался объект, получен {type(data).__name__}"]
    for f in required:
        if f not in data:
            problems.append(f"нет обязательного поля {f!r}")
        elif data[f] is None:
            problems.append(f"поле {f!r} пустое (null)")
    for f, t in (types or {}).items():
        if f in data and data[f] is not None and not isinstance(data[f], t):
            problems.append(
                f"поле {f!r}: ожидался {t.__name__}, получен {type(data[f]).__name__}")
    return problems


def parse_checked(text, required=(), types=None):
    """Разбор с проверкой. Бросает StructureError со всеми претензиями сразу."""
    data = parse(text)
    problems = check(data, required, types)
    if problems:
        raise StructureError("ответ не соответствует ожиданиям: " + "; ".join(problems))
    return data


# ── цикл починки ──────────────────────────────────────────────────────
def ask_structured(ask, prompt, required=(), types=None, attempts=3, log=None):
    """Запрашивает структурированный ответ, при неудаче показывает модели её ошибку.

    ask — функция, принимающая текст и возвращающая ответ модели.

    Ограничение числа попыток обязательно. Без него неудачный промпт
    превращается в бесконечный платный цикл — ровно та ошибка, которую
    разбирали в уроке 2.1.
    """
    if attempts < 1:
        raise ValueError("attempts должно быть не меньше 1")
    log = log if log is not None else []
    text = prompt

    for n in range(1, attempts + 1):
        answer = ask(text)
        try:
            data = parse_checked(answer, required, types)
            log.append({"attempt": n, "ok": True})
            return data
        except StructureError as e:
            log.append({"attempt": n, "ok": False, "error": str(e)})
            if n == attempts:
                raise StructureError(
                    f"не удалось получить корректный ответ за {attempts} попыт(ки). "
                    f"Последняя ошибка: {e}")
            text = (
                f"{prompt}\n\n"
                f"Предыдущий ответ не подошёл: {e}\n"
                "Верни только JSON-объект, без пояснений и без обрамления markdown."
            )
    raise AssertionError("недостижимо")
