"""T24b. M3: PAR (pedestrian attribute recognition) - ResNet50 с несколькими "головами",
обученная на атрибутах Market-1501. Даёт: цвет верха, цвет низа, рюкзак, сумка, шляпа.

ОГРАНИЧЕНИЯ (писать в отчёт):
  * В Market-1501 только 8 цветов верха: black, white, red, purple, yellow, gray, blue, green.
    Из нашего словаря у модели НЕТ orange, pink, brown, beige (и multicolor), так что на
    кропах с этими цветами она ошибётся всегда. Это потолок метода, а не баг кода.
    Особенно важно для beige: на этапе 2 именно бежевая куртка получала разные цвета.
  * Типа одежды нет (есть только "длинный/короткий рукав") -> type="unknown".
  * Пол и возраст из Market НЕ используем (privacy by design: профили людей не строим).
  * Market-1501 - research only. Источник указывать.
  * Обучен на людях 64x128 с другой камеры -> на ваших кропах точность будет ниже.
  * PETA / RAP здесь не реализованы: у каждого свой формат разметки, нужен свой загрузчик.

НЕПРОВЕРЕНО (в песочнице нет ни torch, ни датасета): структура market_attribute.mat
(поля upblack, upwhite, ... со значениями 1=нет / 2=да; атрибуты заданы на человека,
а не на картинку). Если загрузчик упадёт на другом формате - пришли ошибку.

Обучение (на сервере, ~15-25 мин на одной A10, ЗАЙМЁТ GPU - согласуй окно с Пандой):
    python -m vlm.describers.par_model train \
        --market-root /path/to/Market-1501-v15.09.15 \
        --attr-mat /path/to/market_attribute.mat --out models/par_market.pt --epochs 15 --device cuda:1

Использование: HistColorDescriber-подобно, через eval/run_method.py --method par
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from vlm.describers.base import UNKNOWN, Describer, make_answer

UP_COLORS = ["black", "white", "red", "purple", "yellow", "gray", "blue", "green"]
DOWN_COLORS = ["black", "white", "pink", "purple", "yellow", "gray", "blue", "green", "brown"]
BINARY = ["backpack", "bag", "handbag", "hat"]
HEADS = {"up_color": len(UP_COLORS), "down_color": len(DOWN_COLORS),
         **{b: 2 for b in BINARY}}
IMG_H, IMG_W = 256, 128
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_model(pretrained=True):
    import torch.nn as nn
    import torchvision

    class Par(nn.Module):
        def __init__(self):
            super().__init__()
            w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = torchvision.models.resnet50(weights=w)
            self.backbone.fc = nn.Identity()
            self.heads = nn.ModuleDict({k: nn.Linear(2048, n) for k, n in HEADS.items()})

        def forward(self, x):
            f = self.backbone(x)
            return {k: h(f) for k, h in self.heads.items()}

    return Par()


# ---------------- инференс ----------------


class ParDescriber(Describer):
    method = "par:market"
    needs_full_body = True  # PAR учили на людях целиком

    def __init__(self, ckpt="models/par_market.pt", device=None, second_min_prob=0.20):
        import torch
        self.torch = torch
        self.ckpt = ckpt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.second_min_prob = second_min_prob
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.to(self.device).eval()

    def params(self):
        return {"ckpt": self.ckpt, "device": self.device}

    def weights(self):
        return self.ckpt

    def _prep(self, img_bgr):
        import cv2
        import numpy as np
        torch = self.torch
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
        return torch.from_numpy(x.transpose(2, 0, 1))

    def describe(self, img_bgr):
        return self.describe_many([img_bgr])[0]

    def describe_many(self, imgs):
        torch = self.torch
        x = torch.stack([self._prep(i) for i in imgs]).to(self.device)
        with torch.no_grad():
            out = {k: v.softmax(-1).cpu() for k, v in self.model(x).items()}
        res = []
        for i in range(len(imgs)):
            up = out["up_color"][i]
            vals, idx = up.topk(2)
            color = UP_COLORS[int(idx[0])]
            color_2 = UP_COLORS[int(idx[1])] if float(vals[1]) >= self.second_min_prob else None
            extras = {"down_color": DOWN_COLORS[int(out["down_color"][i].argmax())]}
            for b in BINARY:
                extras[b] = bool(out[b][i][1] > 0.5)
            res.append(make_answer(color=color, color_2=color_2, type=UNKNOWN,
                                   confidence=float(vals[0]), description=f"{color} top",
                                   method=self.method, **extras))
        return res


# ---------------- обучение на Market-1501 ----------------


def _load_market_labels(mat_path):
    """-> {person_id(int): {head: class_index или -100}}. Читаем train-часть .mat."""
    import numpy as np
    from scipy.io import loadmat

    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    root = m["market_attribute"]
    labels = {}
    for part in ("train", "test"):
        s = getattr(root, part)
        ids = [int(x) for x in np.atleast_1d(s.image_index)]
        for pos, pid in enumerate(ids):
            lab = {}
            for head, names in (("up_color", UP_COLORS), ("down_color", DOWN_COLORS)):
                prefix = "up" if head == "up_color" else "down"
                hit = [i for i, n in enumerate(names)
                       if int(np.atleast_1d(getattr(s, prefix + n))[pos]) == 2]
                lab[head] = hit[0] if len(hit) == 1 else -100  # нет или неоднозначно
            for b in BINARY:
                lab[b] = int(np.atleast_1d(getattr(s, b))[pos]) - 1  # 1/2 -> 0/1
            labels[pid] = lab
    return labels


def train(market_root, attr_mat, out, epochs=15, batch=64, lr=3e-4, device="cuda", workers=4):
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    labels = _load_market_labels(attr_mat)
    files = sorted(Path(market_root, "bounding_box_train").glob("*.jpg"))
    items = []
    for f in files:
        pid = int(re.match(r"(-?\d+)_", f.name).group(1))
        if pid > 0 and pid in labels:
            items.append((f, pid))
    pids = sorted({p for _, p in items})
    val_pids = set(pids[:: 10])  # каждый 10-й человек в проверку, без утечки по людям
    tr = [x for x in items if x[1] not in val_pids]
    va = [x for x in items if x[1] in val_pids]
    print(f"train {len(tr)} картинок, val {len(va)} картинок, людей {len(pids)}")

    norm = T.Normalize(MEAN, STD)
    tf_tr = T.Compose([T.Resize((IMG_H, IMG_W)), T.RandomHorizontalFlip(), T.Pad(8),
                       T.RandomCrop((IMG_H, IMG_W)), T.ToTensor(), norm])  # без ColorJitter!
    tf_va = T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor(), norm])

    class DS(Dataset):
        def __init__(self, items, tf):
            self.items, self.tf = items, tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            f, pid = self.items[i]
            x = self.tf(Image.open(f).convert("RGB"))
            return x, {k: torch.tensor(v) for k, v in labels[pid].items()}

    dl = DataLoader(DS(tr, tf_tr), batch_size=batch, shuffle=True, num_workers=workers, drop_last=True)
    dv = DataLoader(DS(va, tf_va), batch_size=batch, num_workers=workers)
    model = build_model(True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            x = x.to(device)
            out_ = model(x)
            loss = sum(F.cross_entropy(out_[k], y[k].to(device), ignore_index=-100) for k in HEADS)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        sched.step()
        model.eval()
        hit = {k: [0, 0] for k in HEADS}
        with torch.no_grad():
            for x, y in dv:
                o = model(x.to(device))
                for k in HEADS:
                    yy = y[k].to(device)
                    ok = yy != -100
                    hit[k][0] += int((o[k].argmax(-1)[ok] == yy[ok]).sum())
                    hit[k][1] += int(ok.sum())
        acc = {k: round(h[0] / max(1, h[1]), 3) for k, h in hit.items()}
        print(f"epoch {ep + 1}/{epochs} loss {tot / len(dl):.3f} val_acc {acc}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print("сохранено:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--market-root", required=True)
    t.add_argument("--attr-mat", required=True)
    t.add_argument("--out", default="models/par_market.pt")
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--device", default="cuda")
    a = ap.parse_args()
    train(a.market_root, a.attr_mat, a.out, a.epochs, a.batch, device=a.device)
