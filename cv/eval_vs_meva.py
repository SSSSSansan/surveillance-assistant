"""T8. Сколько размеченных в MEVA событий нашёл наш детектор.

  python cv/eval_vs_meva.py cv/events.json data/gt.json --tol 2

Сопоставление только по времени (допуск --tol секунд): номера треков у нас и в MEVA разные.
Названия активностей смотри в выводе meva_gt.py и при необходимости поправь MAPPING.
"""
import argparse
import json

MAPPING = {
    "person_enters_scene_through_structure": {"entered"},
    "person_exits_scene_through_structure": {"exited"},
    "person_picks_up_object": {"pickup_or_put_candidate"},
    "person_puts_down_object": {"pickup_or_put_candidate"},
}


def overlap(a0, a1, b0, b1, tol):
    return a0 - tol <= b1 and b0 - tol <= a1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("events")
    p.add_argument("gt")
    p.add_argument("--tol", type=float, default=2.0)
    a = p.parse_args()
    ev = json.load(open(a.events, encoding="utf-8"))["events"]
    gt = json.load(open(a.gt, encoding="utf-8"))["activities"]
    print(f"{'активность MEVA':45s} размечено  найдено  recall")
    for name, types in MAPPING.items():
        g = [x for x in gt if x["name"] == name]
        hit = sum(any(e["event_type"] in types and overlap(x["t_start"], x["t_end"], e["t_start"], e["t_end"], a.tol)
                      for e in ev) for x in g)
        print(f"{name:45s} {len(g):9d} {hit:8d}  {hit / len(g):.2f}" if g else f"{name:45s} {0:9d}")
    mapped = set().union(*MAPPING.values())
    ours = [e for e in ev if e["event_type"] in mapped]
    good = sum(any(x["name"] in MAPPING and e["event_type"] in MAPPING[x["name"]]
                   and overlap(x["t_start"], x["t_end"], e["t_start"], e["t_end"], a.tol) for x in gt) for e in ours)
    if ours:
        print(f"\nнаших событий этих типов: {len(ours)}, совпало с разметкой: {good} ({good / len(ours):.2f})")