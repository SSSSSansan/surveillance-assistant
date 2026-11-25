"""Прогнать один метод по кропам из cv.crop_people. Ничего не пишет в базу и не трогает
snapshots_vlm.json: результат лежит рядом, отдельным файлом на каждый метод.

    python -m eval.run_method --method hist   --index data/out_snapshots/crops/crops_index.json data/video2/crops/crops_index.json
    python -m eval.run_method --method hist   --index ... --wb-alpha 0 --variant B     # без баланса белого
    python -m eval.run_method --method siglip --index ... --device cuda:1
    python -m eval.run_method --method par    --index ... --par-ckpt models/par_market.pt

Выход:
    results/<method>__crop__<variant>.jsonl      строка на человека: контракт + video_id, snapshot_id, person_idx, сек
    results/<method>__crop__<variant>.runs.json  по записи на видео (метод, веса, параметры, секунды): для таблицы runs

Ключ строки совпадает с ключом базы и snapshots_vlm.json: (video_id, snapshot_id, person_idx).

В Postgres напрямую НЕ пишем: таблицы runs / колонки method в snapshot_people (T15 из STAGE3_PLAN) в
репозитории ещё нет, а схема этапа 2 хранит одного описателя на человека. Победителя в базу кладёт
eval/export_snapshots_json.py + существующий database.snapshots_loader.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2


def build_describer(a):
    if a.method == "hist":
        from cv.color_hist import HistColorDescriber
        return HistColorDescriber(wb_alpha=a.wb_alpha, use_skin_filter=not a.no_skin,
                                  dark_sat=a.dark_sat, neutral_black_v=a.neutral_black_v,
                                  top_skip=a.top_skip)
    if a.method == "siglip":
        from vlm.describers.siglip import SiglipDescriber
        return SiglipDescriber(model_id=a.siglip_model, device=a.device)
    if a.method == "par":
        from vlm.describers.par_model import ParDescriber
        return ParDescriber(ckpt=a.par_ckpt, device=a.device)
    raise SystemExit(f"неизвестный метод {a.method}")


def load_items(index_paths):
    items, seen = [], set()
    for pat in index_paths:
        for p in sorted(glob.glob(pat)):
            for c in json.loads(Path(p).read_text(encoding="utf-8")):
                key = (c["video_id"], c["snapshot_id"], c["person_idx"])
                if c.get("kept") and key not in seen:
                    seen.add(key)
                    items.append(c)
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--method", required=True, choices=["hist", "siglip", "par"])
    ap.add_argument("--index", nargs="+", required=True, help="crops_index.json (можно маски в кавычках)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--variant", default="A", help="prompt_variant в имени файла: A, B... (для ablation)")
    ap.add_argument("--crop-kind", choices=["torso", "full"], default="torso",
                    help="какой кроп давать методу (par всегда full)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default=None)
    ap.add_argument("--wb-alpha", type=float, default=0.5)
    ap.add_argument("--no-skin", action="store_true")
    ap.add_argument("--dark-sat", type=float, default=0.0,
                    help="hist: тёмные пиксели с насыщенностью ниже порога -> black (0 = выключено; пробуй 0.6)")
    ap.add_argument("--top-skip", type=float, default=0.0,
                    help="hist: игнорировать верхнюю долю кропа (голова/волосы). В торсе-кропе это ~20-25%%; пробуй 0.25")
    ap.add_argument("--neutral-black-v", type=float, default=0.30,
                    help="hist: серый темнее этой яркости -> black (по умолчанию 0.30; пробуй 0.22)")
    ap.add_argument("--siglip-model", default="google/siglip-base-patch16-224")
    ap.add_argument("--par-ckpt", default="models/par_market.pt")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    items = load_items(a.index)
    if a.limit:
        items = items[: a.limit]
    if not items:
        raise SystemExit("в индексах нет кропов с kept=true (проверь пути в --index)")
    d = build_describer(a)
    d.prompt_variant = a.variant
    key = "full_path" if (d.needs_full_body or a.crop_kind == "full") else "path"
    print(f"метод {d.method}, вариант {a.variant}, кроп {key}, кропов {len(items)}")

    rows, per_video = [], defaultdict(lambda: {"sec": 0.0, "n": 0})
    for i in range(0, len(items), a.batch):
        chunk = items[i:i + a.batch]
        imgs = [cv2.imread(c[key]) for c in chunk]
        bad = [c["crop_id"] for c, im in zip(chunk, imgs) if im is None]
        if bad:
            raise SystemExit(f"не читаются кропы: {bad[:3]}... Проверь, что запуск из корня проекта")
        t0 = time.perf_counter()
        answers = d.describe_many(imgs)
        dt = time.perf_counter() - t0
        for c, ans in zip(chunk, answers):
            ans["prompt_variant"] = a.variant
            rows.append({"video_id": c["video_id"], "snapshot_id": c["snapshot_id"],
                         "person_idx": c["person_idx"], "crop_id": c["crop_id"],
                         "box_h": c["box_h"], "box": c["box"], "sec": round(dt / len(chunk), 5), **ans})
            per_video[c["video_id"]]["sec"] += dt / len(chunk)
            per_video[c["video_id"]]["n"] += 1
        if (i // a.batch) % 5 == 0:
            print(f"  {min(i + a.batch, len(items))}/{len(items)}")

    started = datetime.now(timezone.utc).isoformat()
    runs = [{"run_id": f"{a.method}-{a.variant}-{uuid.uuid4().hex[:6]}", "video_id": vid,
             "method": d.method, "model": getattr(d, "model_id", d.method), "mode": d.mode,
             "prompt_variant": a.variant, "weights": d.weights(), "started_at": started,
             "duration_sec": round(v["sec"], 3),
             "params": {**d.params(), "crop_kind": key, "n_crops": v["n"]}}
            for vid, v in per_video.items()]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.method}__crop__{a.variant}"
    with open(out / f"{stem}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / f"{stem}.runs.json").write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding="utf-8")
    mean = sum(r["sec"] for r in rows) / len(rows)
    print(f"готово: {(out / (stem + '.jsonl')).as_posix()}  ({len(rows)} строк, {mean * 1000:.1f} мс на кроп)")


if __name__ == "__main__":
    main()