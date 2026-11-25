"""Сравнение моделей на вопросах по срезам (этап 2) на НАСТОЯЩЕЙ базе PostgreSQL.

Правильные ответы посчитаны по данным из SQL (159 человек, видео 2018-03-15...G421.r13). Если данные в базе
изменятся (другое видео, другая разметка VLM), ответы надо пересчитать: см. EXPECTED ниже.

Запуск из корня проекта (DATABASE_URL должен быть задан), по одной модели:
    py -m eval.eval_snapshots --model qwen2.5:3b
    py -m eval.eval_snapshots --model qwen2.5:7b
Для каждого вопроса печатается: верно ли, сколько вызовов функций, секунды, иероглифы в ответе.
Результаты сохраняются в eval/results/snapshots_*.csv.
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

from agent import runner
from agent.llm import OpenAICompatClient

EVAL_DIR = Path(__file__).resolve().parent

# (вопрос, тип проверки, ожидание)
#   yes / no : ответ начинается с «Да/Yes» или «Нет/No»
#   has      : в ответе есть все перечисленные подстроки
EXPECTED = [
    ("Был ли кто-то в красной кофте?", "yes", None),
    ("Был ли кто-то в красной куртке?", "yes", None),
    ("Был ли кто-то в синей кофте?", "yes", None),
    ("Был ли кто-то в зелёной одежде?", "yes", None),
    ("Был ли кто-то в розовой куртке?", "yes", None),
    ("Был ли кто-то в зелёной куртке?", "no", None),
    ("Был ли кто-то в жёлтой одежде?", "no", None),
    ("Был ли кто-то в белой одежде?", "no", None),
    ("Был ли кто-то в красной футболке?", "no", None),
    ("Was there anyone in a red sweater?", "yes", None),
    ("Сколько людей было на 270 секунде?", "has", ["18"]),
    ("Когда было больше всего людей?", "has", ["18", "270"]),
    ("Когда был кто-то в красной одежде?", "has", ["60 с", "300 с"]),
]


def check_answer(answer, kind, want):
    if not answer:
        return False
    if kind == "yes":
        return bool(runner._YES.match(answer))
    if kind == "no":
        return bool(runner._NO.match(answer))
    return all(w in answer for w in want)


def run(client, conn=None, questions=EXPECTED, max_steps=5):
    out = []
    for q, kind, want in questions:
        res = runner.run_agent(q, client, conn=conn, max_steps=max_steps)
        out.append({
            "question": q,
            "expected": kind if kind != "has" else "содержит " + ", ".join(want),
            "ok": check_answer(res.answer, kind, want),
            "tool_calls": len(res.steps),
            "tools": " > ".join(s.name for s in res.steps),
            "arguments": " | ".join(json.dumps(s.arguments, ensure_ascii=False) for s in res.steps),
            "seconds": res.seconds,
            "foreign_script": runner.has_foreign_script(res.answer or "", q),
            "answer": res.answer or "",
            "error": res.error or "",
        })
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Вопросы по срезам: сравнение моделей")
    p.add_argument("--base-url", default=None, help="по умолчанию LLM_BASE_URL")
    p.add_argument("--model", default=None, help="по умолчанию LLM_MODEL")
    p.add_argument("--repeat", type=int, default=1, help="сколько раз повторить весь набор (проверка стабильности)")
    p.add_argument("--only", default=None, help="прогнать только вопросы, в тексте которых есть эта подстрока")
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--out", default=str(EVAL_DIR / "results"))
    a = p.parse_args(argv)

    client = OpenAICompatClient(base_url=a.base_url, model=a.model)
    if not client.model:
        print("Ошибка: не задана модель (--model или LLM_MODEL)", file=sys.stderr)
        return 2

    rows = []
    for i in range(a.repeat):
        qs = [x for x in EXPECTED if not a.only or a.only.lower() in x[0].lower()]
        for r in run(client, None, qs, a.max_steps):
            r["run"] = i + 1
            rows.append(r)
            mark = "OK  " if r["ok"] else "FAIL"
            print(f"{mark} {r['tool_calls']} выз. {r['seconds']:>6} с  {r['question']}"
                  + (f"  | {r['error']}" if r["error"] else "")
                  + (f"  | ответ: {r['answer'][:70]}" if not r["ok"] and r["answer"] else "")
                  + (f"\n       аргументы модели: {r['arguments']}" if not r["ok"] else ""))

    n = len(rows)
    ok = sum(r["ok"] for r in rows)
    print(f"\nМодель: {client.model}")
    print(f"Верных ответов: {ok}/{n}   с иероглифами: {sum(r['foreign_script'] for r in rows)}   "
          f"ошибок модели: {sum(bool(r['error']) for r in rows)}")
    print(f"Среднее: {sum(r['tool_calls'] for r in rows) / n:.1f} вызова функций, "
          f"{sum(r['seconds'] for r in rows) / n:.1f} с на ответ")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.-]+", "_", client.model)
    path = out / f"snapshots_{time.strftime('%Y%m%d_%H%M%S')}_{safe}.csv"
    fields = ["run", "question", "expected", "ok", "tool_calls", "tools", "arguments", "seconds", "foreign_script", "answer", "error"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Результаты: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())