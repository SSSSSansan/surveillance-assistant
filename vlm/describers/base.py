"""Общий контракт ответа (STAGE3_PLAN.md, п. 4.1) и словарь.

СЛОВАРЬ ТОТ ЖЕ, что в vlm/describe_snapshot.py (13 цветов + unknown, 11 типов + unknown).
Если его там поменяют, тест eval/test_cv_methods.py упадёт: он сверяет эти списки с файлом Панды.

Контракт ответа использует ключи `color` / `type` (как в плане). В базе и в snapshots.json
те же значения называются clothing_color / clothing_type: перевод делает eval/export_snapshots_json.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

UNKNOWN = "unknown"

COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
          "black", "white", "gray", "beige", "multicolor", "unknown")
TYPES = ("sweater", "hoodie", "t-shirt", "shirt", "jacket", "coat", "dress",
         "pants_only", "vest", "uniform", "other", "unknown")

# То, что цветовые методы (M1, M2, M3) вообще могут предсказать.
# multicolor / unknown не предсказываем: "нет доминирующего цвета" без нейросети не определить.
PREDICT_COLORS = tuple(c for c in COLORS if c not in ("multicolor", "unknown"))
# Для zero-shot по типам: pants_only/other/unknown - не "вещь", описывать их текстом бессмысленно.
PREDICT_TYPES = tuple(t for t in TYPES if t not in ("pants_only", "other", "unknown"))


def make_answer(color=UNKNOWN, color_2=None, type=UNKNOWN, confidence=0.0,
                description="", method="", mode="crop", prompt_variant="A",
                **extras) -> dict:
    """Один ответ в формате общего контракта. Лишние поля уходят в extras."""
    ans = {
        "color": color,
        "color_2": color_2,
        "type": type,
        "confidence": None if confidence is None else round(float(confidence), 4),
        "description": description,
        "method": method,
        "mode": mode,
        "prompt_variant": prompt_variant,
    }
    if extras:
        ans["extras"] = extras
    return ans


class Describer(ABC):
    """Любой метод (гистограмма, SigLIP, PAR, VLM) наследует этот класс."""

    method = "base"
    mode = "crop"
    prompt_variant = "A"
    needs_full_body = False  # True: методу нужен кроп человека целиком, а не торса

    @abstractmethod
    def describe(self, img_bgr) -> dict:
        """img_bgr: numpy-массив BGR (как отдаёт cv2.imread)."""

    def describe_many(self, imgs):
        return [self.describe(i) for i in imgs]

    def params(self) -> dict:
        """Параметры для таблицы runs (JSON)."""
        return {}

    def weights(self) -> str:
        return ""
