"""T2. Разбор разметки MEVA (KPF YAML) в простой JSON.

  python cv/meva_gt.py data/meva-data-repo <имя видео без расширения> data/gt.json --fps 30

Ищет файлы <имя>*.activities.yml, переводит кадры в секунды, печатает список
активностей с количеством. Эти названия нужны, чтобы проверить MAPPING в eval_vs_meva.py.
"""
import argparse
import collections
import glob
import json
import yaml


def parse(path, fps):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    acts = []
    for item in data or []:
        act = item.get("act") if isinstance(item, dict) else None
        if not act:
            continue
        names = [k for k, v in act.items() if k.startswith("act") and isinstance(v, dict)]
        if not names:
            continue
        name = next(iter(act[names[0]]))
        spans = []
        for ts in act.get("timespan", []):
            for v in ts.values():
                spans.append(v)
        if not spans:
            continue
        a = min(s[0] for s in spans)
        b = max(s[1] for s in spans)
        actors = [x.get("id1") for x in act.get("actors", []) if isinstance(x, dict)]
        acts.append({"name": name, "t_start": round(a / fps, 2),
                     "t_end": round(b / fps, 2), "actors": actors})
    return acts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("stem")
    p.add_argument("out")
    p.add_argument("--fps", type=float, default=30.0)
    a = p.parse_args()
    files = glob.glob(f"{a.repo}/**/{a.stem}*.activities.yml", recursive=True)
    if not files:
        raise SystemExit("файл разметки не найден: проверь имя видео и что видео входит в размеченную часть")
    acts = []
    for fp in files:
        acts += parse(fp, a.fps)
    acts.sort(key=lambda x: x["t_start"])
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"video_stem": a.stem, "fps": a.fps, "activities": acts}, f,
                  ensure_ascii=False, indent=2)
    print("найдено активностей:", len(acts))
    for k, v in collections.Counter(x["name"] for x in acts).most_common():
        print(f"  {v:3d}  {k}")