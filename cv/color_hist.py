"""T17. M1: цвет одежды по пикселям (без нейросетей). Словарь - как у Панды (vlm/describe_snapshot.py):
светло-синий, бирюзовый, тёмно-синий -> blue, бордовый -> red, оттенки хаки -> beige.

Шаги: (1) мягкий баланс белого (gray-world, сила wb_alpha), (2) берём центр кропа
(овал), чтобы отсечь фон по краям, (3) выкидываем кожу (YCrCb), (4) k-means (k=3)
в Lab, (5) каждый кластер -> один из цветов словаря по правилам HSV.

Тип одежды этот метод НЕ определяет: type="unknown" (это нормально, в отчёте M1
сравнивается только по цвету).

Использование из кода:
    from cv.color_hist import HistColorDescriber
    ans = HistColorDescriber().describe(cv2.imread("crop.jpg"))
"""
from __future__ import annotations

import cv2
import numpy as np

from vlm.describers.base import PREDICT_COLORS, UNKNOWN, Describer, make_answer


def classify_hsv(h: float, s: float, v: float, dark_sat: float = 0.0,
                 neutral_black_v: float = 0.30) -> str:
    """h в градусах 0-360, s и v в 0-1. Порядок проверок важен.
    dark_sat: тёмные пиксели (v<0.35) с насыщенностью НИЖЕ этого порога считаем чёрными, а не цветными
              (оттенок у тёмного ненадёжен: чёрная куртка с синеватым отливом камеры -> blue). 0 = выключено.
    neutral_black_v: ниже этой яркости серый без цвета называем black (выше - gray). По умолчанию 0.30."""
    if v < 0.18:
        return "black"
    if v < 0.35 and s < dark_sat:
        return "black"
    if s < 0.12:
        if v > 0.80:
            return "white"
        if v < neutral_black_v:
            return "black"
        return "gray"
    if 15 <= h < 55 and s < 0.38 and v >= 0.55:
        return "beige"
    if 8 <= h < 45 and v < 0.55:
        return "brown"
    if h >= 345 or h < 12:
        if s < 0.55 and v > 0.65:
            return "pink"
        return "red"  # бордовый тоже red (как в словаре этапа 2)
    if h < 40:
        return "orange"
    if h < 70:
        return "yellow"
    if h < 165:
        return "green"
    if h < 255:
        return "blue"  # голубой, бирюзовый, тёмно-синий: как в правилах Панды (все -> blue)
    if h < 300:
        return "purple"
    return "pink"


def bgr_to_hsv_deg(bgr) -> tuple[float, float, float]:
    px = np.array([[bgr]], dtype=np.float32) / 255.0
    h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
    return float(h), float(s), float(v)


def gray_world(img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Частичная коррекция баланса белого. alpha=0 выключает.
    Полный gray-world портит кроп, где одна красная кофта занимает весь кадр,
    поэтому по умолчанию только половина коррекции."""
    if alpha <= 0:
        return img
    means = img.reshape(-1, 3).mean(0) + 1e-6
    gain = means.mean() / means
    gain = 1 + alpha * (gain - 1)
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def skin_mask(img: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    return (cr >= 137) & (cr <= 173) & (cb >= 85) & (cb <= 127) & (y > 60)


def center_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (max(1, int(w * 0.42)), max(1, int(h * 0.48))),
                0, 0, 360, 1, -1)
    return m.astype(bool)


class HistColorDescriber(Describer):
    method = "cv:hist"

    def __init__(self, wb_alpha=0.5, use_skin_filter=True, k=3, min_pixels=60,
                 second_share=0.20, max_pixels=4000, seed=0, dark_sat=0.0, neutral_black_v=0.30):
        self.wb_alpha, self.use_skin_filter, self.k = wb_alpha, use_skin_filter, k
        self.dark_sat, self.neutral_black_v = dark_sat, neutral_black_v
        self.min_pixels, self.second_share = min_pixels, second_share
        self.max_pixels, self.seed = max_pixels, seed

    def params(self):
        return {"wb_alpha": self.wb_alpha, "skin_filter": self.use_skin_filter, "k": self.k,
                "dark_sat": self.dark_sat, "neutral_black_v": self.neutral_black_v}

    def describe(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0:
            return make_answer(method=self.method, description="empty crop")
        img = gray_world(img_bgr, self.wb_alpha)
        h, w = img.shape[:2]
        mask = center_mask(h, w)
        if self.use_skin_filter:
            sk = skin_mask(img)
            # если "кожей" оказалось больше половины - скорее всего это розовая/бежевая
            # одежда, а не кожа: фильтр не применяем
            if sk[mask].mean() < 0.5:
                mask = mask & ~sk
        pix = img[mask]
        if len(pix) < self.min_pixels:
            return make_answer(method=self.method, description="too few pixels")

        rng = np.random.default_rng(self.seed)
        if len(pix) > self.max_pixels:
            pix = pix[rng.choice(len(pix), self.max_pixels, replace=False)]

        lab = cv2.cvtColor((pix.astype(np.float32) / 255.0).reshape(-1, 1, 3),
                           cv2.COLOR_BGR2Lab).reshape(-1, 3)
        k = min(self.k, len(pix))
        cv2.setRNGSeed(self.seed)
        _, labels, _ = cv2.kmeans(lab, k, None,
                                  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0),
                                  3, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()

        shares: dict[str, float] = {}
        for ci in range(k):
            sel = pix[labels == ci]
            if len(sel) == 0:
                continue
            med = np.median(sel, axis=0)  # представитель кластера = медиана BGR
            name = classify_hsv(*bgr_to_hsv_deg(med), self.dark_sat, self.neutral_black_v)
            shares[name] = shares.get(name, 0.0) + len(sel) / len(pix)

        ranked = sorted(shares.items(), key=lambda kv: -kv[1])
        color, conf = ranked[0]
        color_2 = ranked[1][0] if len(ranked) > 1 and ranked[1][1] >= self.second_share else None
        return make_answer(color=color, color_2=color_2, type=UNKNOWN, confidence=conf,
                           description=f"{color}" + (f"/{color_2}" if color_2 else ""),
                           method=self.method)
