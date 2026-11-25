"""T24a. M2: SigLIP zero-shot. Сравниваем кроп с текстами "a photo of a person wearing {color} clothes."
и "a photo of a person wearing a {type}." и берём лучшие совпадения. Обучения нет.

Установка (один раз):
    pip install transformers pillow sentencepiece protobuf     (torch у вас уже есть для YOLO)

Windows + кириллица в пути: sentencepiece не открывает файлы из C:\\Users\\<кириллица>\\.cache.
Решение: перед запуском  $env:HF_HOME = "C:\\hf_cache"   (PowerShell, действует в текущем окне).

Версии transformers: в новых версиях get_text_features / get_image_features возвращают не тензор, а объект
BaseModelOutputWithPooling (так упало у Арай). Поэтому здесь используется обычный forward модели:
model(input_ids=..., pixel_values=...).logits_per_image - он одинаков во всех версиях.

НЕПРОВЕРЕНО: в моей песочнице нет ни torch, ни доступа к huggingface.co, код проверен только
синтаксисом и логикой. Если упадёт - пришли трейсбек.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # убирает шумное предупреждение на Windows

import cv2  # noqa: E402

from vlm.describers.base import PREDICT_COLORS, PREDICT_TYPES, Describer, make_answer  # noqa: E402


def _pretty(x: str) -> str:
    return x.replace("_", " ").replace("-", " ")


class SiglipDescriber(Describer):
    method = "cv:siglip"

    def __init__(self, model_id="google/siglip-base-patch16-224", device=None,
                 colors=None, types=None, second_min_prob=0.20):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.colors = list(colors or PREDICT_COLORS)
        self.types = list(types or PREDICT_TYPES)
        self.second_min_prob = second_min_prob
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

        prompts = ([f"a photo of a person wearing {_pretty(c)} clothes." for c in self.colors]
                   + [f"a photo of a person wearing a {_pretty(t)}." for t in self.types])
        self.n_colors = len(self.colors)
        # SigLIP обучен с padding="max_length", без него качество падает. Тексты токенизируем один раз.
        self.text_inputs = self.processor(text=prompts, padding="max_length", max_length=64,
                                          truncation=True, return_tensors="pt").to(self.device)

    def params(self):
        return {"model_id": self.model_id, "device": self.device}

    def weights(self):
        return self.model_id

    def _logits(self, imgs_bgr):
        from PIL import Image
        pil = [Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB)) for i in imgs_bgr]
        pix = self.processor(images=pil, return_tensors="pt")["pixel_values"].to(self.device)
        with self.torch.no_grad():
            out = self.model(**self.text_inputs, pixel_values=pix)
        return out.logits_per_image  # [число_картинок, цвета + типы]

    def describe(self, img_bgr):
        return self.describe_many([img_bgr])[0]

    def describe_many(self, imgs):
        logits = self._logits(imgs)
        pc = logits[:, :self.n_colors].softmax(dim=-1)
        pt = logits[:, self.n_colors:].softmax(dim=-1)
        out = []
        for i in range(len(imgs)):
            cv_, ci = pc[i].topk(2)
            tv, ti = pt[i].topk(1)
            color = self.colors[int(ci[0])]
            color_2 = self.colors[int(ci[1])] if float(cv_[1]) >= self.second_min_prob else None
            typ = self.types[int(ti[0])]
            out.append(make_answer(
                color=color, color_2=color_2, type=typ, confidence=float(cv_[0]),
                description=f"{color} {typ}", method=self.method,
                type_confidence=round(float(tv[0]), 4)))
        return out
