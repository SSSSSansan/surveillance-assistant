"""T8. Сколько размеченных в MEVA событий нашёл наш детектор.

  python cv/eval_vs_meva.py cv/events.json data/gt.json --tol 2

Сопоставление только по времени (допуск --tol секунд): номера треков у нас и в MEVA разные.
Два подсчёта:
  мягкий   одно наше событие может засчитаться за несколько размеченных (завышает recall)
  строгий  каждое наше событие засчитывается не больше одного раза (честная цифра для отчёта)
"""
import argparse
import json

MAPPING = {
    "person_enters_scene_through_structure": {"enter"},
    "person_exits_scene_through_structure": {"exit"},
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
    used = set()  # индексы наших событий, уже засчитанных в строгом режиме
    print(f"{'активность MEVA':40s} {'размечено':>9s} {'мягко':>6s} {'recall':>7s} {'строго':>7s} {'recall':>7s}")
    strict_total = 0
    for name, types in MAPPING.items():
        g = sorted((x for x in gt if x["name"] == name), key=lambda x: x["t_start"])
        soft = strict = 0
        for x in g:
            cands = [i for i, e in enumerate(ev) if e["event_type"] in types
                     and overlap(x["t_start"], x["t_end"], e["t_start"], e["t_end"], a.tol)]
            soft += bool(cands)
            free = [i for i in cands if i not in used]
            if free:
                used.add(free[0])
                strict += 1
        strict_total += strict
        if g:
            print(f"{name:40s} {len(g):9d} {soft:6d} {soft / len(g):7.2f} {strict:7d} {strict / len(g):7.2f}")
        else:
            print(f"{name:40s} {0:9d}")
    mapped = set().union(*MAPPING.values())
    ours = [e for e in ev if e["event_type"] in mapped]
    if ours:
        print(f"\nнаших событий этих типов: {len(ours)}; "
              f"засчитано строго: {len(used)} (precision {len(used) / len(ours):.2f})")