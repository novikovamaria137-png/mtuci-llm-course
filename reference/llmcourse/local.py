"""Расчёт требований к памяти для локального запуска. Урок 2.5.

Модуль считает арифметику, а не гадает. Что именно он считает:

  веса      = число параметров × байт на параметр          (точно)
  KV-кэш    = 2 × слои × размерность × контекст × батч × байт  (точно,
              если известна архитектура модели)
  накладные = доля сверху на активации и служебные буферы   (ОЦЕНКА)

Первые две величины выводятся из чисел, третья — эмпирическая поправка.
Она помечена явно, потому что в этом курсе оценку не выдают за расчёт.
"""
from dataclasses import dataclass

# Байт на один параметр при разной разрядности.
BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

# Доля сверх весов и кэша на активации, буферы и фрагментацию.
# Это ОЦЕНКА по практике, а не выведенная величина.
OVERHEAD = 0.15

GB = 1024 ** 3


@dataclass(frozen=True)
class Arch:
    """Параметры архитектуры модели, нужные для расчёта KV-кэша."""
    layers: int
    hidden: int
    kv_heads: int = 0        # 0 — считать как обычное внимание
    heads: int = 0

    def kv_factor(self):
        """Во сколько раз KV-кэш меньше из-за группового внимания."""
        if self.kv_heads and self.heads:
            return self.kv_heads / self.heads
        return 1.0


def weights_gb(params_b, quant="int4"):
    """Память под веса. params_b — миллиарды параметров."""
    if quant not in BYTES_PER_PARAM:
        raise ValueError(f"неизвестная разрядность {quant!r}; "
                         f"доступны: {sorted(BYTES_PER_PARAM)}")
    if params_b <= 0:
        raise ValueError("число параметров должно быть больше нуля")
    return params_b * 1e9 * BYTES_PER_PARAM[quant] / GB


def kv_cache_gb(arch, context, batch=1, bytes_per=2):
    """Память под KV-кэш при заданной длине контекста.

    Формула: 2 (ключи и значения) × слои × размерность × контекст × батч.
    Умножается на поправку группового внимания, если она задана.
    """
    if context <= 0 or batch <= 0:
        raise ValueError("контекст и батч должны быть больше нуля")
    raw = 2 * arch.layers * arch.hidden * context * batch * bytes_per
    return raw * arch.kv_factor() / GB


def total_gb(params_b, quant="int4", arch=None, context=0, batch=1):
    """Суммарная потребность в памяти."""
    w = weights_gb(params_b, quant)
    kv = kv_cache_gb(arch, context, batch) if (arch and context) else 0.0
    return {
        "weights": w,
        "kv_cache": kv,
        "overhead": (w + kv) * OVERHEAD,
        "total": (w + kv) * (1 + OVERHEAD),
    }


def verdict(available_gb, params_b, quant="int4", arch=None, context=0, batch=1):
    """Влезет ли. Возвращает словарь с числами и словесным выводом."""
    r = total_gb(params_b, quant, arch, context, batch)
    need = r["total"]
    r["available"] = available_gb
    r["fits"] = need <= available_gb
    margin = available_gb - need
    if margin >= available_gb * 0.25:
        r["verdict"] = "влезает с запасом"
    elif margin >= 0:
        r["verdict"] = "влезает впритык — при росте контекста упрётесь"
    else:
        r["verdict"] = f"не влезает: не хватает {-margin:.1f} ГБ"
    return r


def suggest_quant(available_gb, params_b, arch=None, context=0, batch=1):
    """Наибольшая разрядность, при которой модель ещё помещается."""
    for q in ("fp16", "int8", "int4"):
        if verdict(available_gb, params_b, q, arch, context, batch)["fits"]:
            return q
    return None
