"""T34. Результат метода (jsonl от eval.run_method) -> snapshots_*.json в формате Панды, чтобы его
принял УЖЕ существующий `python -m database.snapshots_loader` без изменений.

    python -m eval.export_snapshots_json --base vlm/snapshots_vlm.json \
        --pred results/hist__crop__A.jsonl --out snapshots_best.json

Как собирается файл: берётся --base (там срезы, рамки, описания Панды), и у людей, для которых есть
строка в --pred, подменяется clothing_color (и clothing_type, только если метод его умеет и сказал не
unknown, либо если указан --type-from-pred). У остальных людей (дубли рамок, мелкие) остаётся то, что было.

ЧЕСТНО ПРО ОГРАНИЧЕНИЯ
  * description берётся из base (его писал VLM) и может ПРОТИВОРЕЧИТЬ новому цвету ("gray jacket" при
    clothing_color=black). Агент ищет по полям, а не по тексту, но на фронте текст виден. Поэтому в каждого
    человека пишется "color_method": <метод>, а при --fix-description описание заменяется на "<color> <type>".
  * Загрузчик заменяет людей среза целиком и ключа method в базе нет: файл ПЕРЕЗАПИШЕТ описания Панды.
    Делай это только когда победитель выбран, и сохрани старый snapshots_vlm.json (он в git).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.score_methods import load_preds


def export(base, preds, type_from_pred=False, fix_description=False):
    changed = 0
    for snap in base["snapshots"]:
        for p in snap.get("people", []):
            r = preds.get((base["video_id"], snap["snapshot_id"], int(p["person_idx"])))
            if not r:
                continue
            p["clothing_color"] = r["color"]
            if r.get("type") and (type_from_pred or r["type"] != "unknown"):
                p["clothing_type"] = r["type"]
            p["color_method"] = r["method"]
            if fix_description:
                p["description"] = f"{p['clothing_color']} {p['clothing_type']}"
            changed += 1
    base["color_method"] = next(iter(preds.values()))["method"] if preds else None
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", required=True, help="snapshots_vlm.json (структура, типы, описания)")
    ap.add_argument("--pred", required=True, help="results/<метод>__crop__<вариант>.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--type-from-pred", action="store_true")
    ap.add_argument("--fix-description", action="store_true")
    a = ap.parse_args()

    base = json.loads(Path(a.base).read_text(encoding="utf-8"))
    preds = load_preds(a.pred)
    n = export(base, preds, a.type_from_pred, a.fix_description)
    Path(a.out).write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(s.get("people", [])) for s in base["snapshots"])
    print(f"подменён цвет у {n} из {total} людей -> {a.out}")
    print("загрузка: python -m database.snapshots_loader", a.out)


if __name__ == "__main__":
    main()
