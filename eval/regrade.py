"""Пересчитать оценки ответов в готовых CSV (eval/results/) заново, БЕЗ повторного запуска моделей.

Нужно, когда поправлен грейдер (eval/grader.py): ответы моделей уже сохранены в CSV, меняется только проверка.
Исходный файл не трогается. Рядом создаётся новый файл с тем же названием модели и свежей меткой времени,
поэтому `py -m eval.compare_models` берёт именно его.
Запуск из корня проекта:
    py -m eval.regrade                      # все файлы событий из eval/results/
    py -m eval.regrade --file eval/results/20261125_004359_qwen2.5_7b.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

from eval import compare_models, grader
from eval.eval_agent import FIELDS

RESULTS = Path(__file__).resolve().parent / "results"


def regrade_rows(rows, time_tol=0.0):
    out = []
    for r in rows:
        r = dict(r)
        for k in ("tool_ok", "args_ok"):          # в CSV это текст «True»/«False»
            r[k] = str(r[k]).strip().lower() == "true"
        r["answer_ok"] = False
        g = grader.grade(r, r.get("answer"), time_tol) if (r.get("answer") or "").strip() else {"correct": False, "detail": r.get("error") or "нет ответа"}
        r["answer_ok"], r["grade_detail"] = g["correct"], g["detail"]
        if r["expected_tool"] != "none" and r["tool_ok"]:
            try:
                first = json.loads(r["first_args"] or "{}")
                r["args_ok"] = grader.args_match(first, json.loads(r["expected_args"]))
            except (ValueError, TypeError):
                pass
        out.append(r)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Пересчёт оценок в готовых CSV")
    p.add_argument("--file", default=None)
    p.add_argument("--dir", default=str(RESULTS))
    p.add_argument("--time-tol", type=float, default=0.0)
    a = p.parse_args(argv)
    if a.file:
        files = [Path(a.file)]
    else:
        files = [v["events"] for v in compare_models.find_files(a.dir).values() if "events" in v]
    if not files:
        print("Нет файлов событий для пересчёта.", file=sys.stderr)
        return 1
    for src in files:
        m = compare_models._NAME.match(src.name)
        model = m.group(3) if m else src.stem
        rows = compare_models.read_rows(src)
        new = regrade_rows(rows, a.time_tol)
        dst = src.parent / f"{time.strftime('%Y%m%d_%H%M%S')}_{model}.csv"
        n = 0
        while dst.exists():                       # две модели за одну секунду: имена не должны совпасть
            n += 1
            dst = src.parent / f"{time.strftime('%Y%m%d_%H%M%S')}{n}_{model}.csv"
        with open(dst, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows({k: r.get(k, "") for k in FIELDS} for r in new)
        was = sum(str(r["answer_ok"]).lower() == "true" for r in rows)
        now = sum(bool(r["answer_ok"]) for r in new)
        wa = sum(str(r["args_ok"]).lower() == "true" for r in rows)
        na = sum(bool(r["args_ok"]) for r in new)
        print(f"{model}: ответы {was} -> {now} из {len(rows)}, аргументы {wa} -> {na}   ({dst.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
