"""Оценка агента на 30 вопросах из eval/questions.csv (T11).

Считает по вопросам:
  * функция  : первая вызванная функция совпала с expected_tool (для вопросов вне базы: функций не вызвано);
  * аргументы: функция верна и аргументы совпали строго (после отбрасывания значений по умолчанию);
  * ответ    : грейдер (eval/grader.py) принял ответ.
Эталоны посчитаны на тестовых данных (seed), поэтому по умолчанию используется демо-база в SQLite.

Запуск из корня проекта, например:
    export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=<имя модели>
    python -m eval.eval_agent                      # демо-база, допуск по времени 0.0 с
    python -m eval.eval_agent --db postgres        # DATABASE_URL, в базе загружены database/seed*
Результаты по каждому вопросу сохраняются в eval/results/*.csv: их стоит просмотреть вручную.
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

from agent import config
from agent.demo_db import make_demo_db
from agent.llm import OpenAICompatClient
from agent.runner import run_agent
from eval import grader

EVAL_DIR = Path(__file__).resolve().parent
FIELDS = ["id", "type", "lang", "question", "expected_answer", "expected_tool", "expected_args",
          "first_tool", "first_args", "n_tool_calls", "tool_ok", "args_ok", "answer", "answer_ok",
          "grade_detail", "error", "seconds"]


def load_questions(path=None):
    with open(path or EVAL_DIR / "questions.csv", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def evaluate_question(row, client, conn=None, embedder=None, time_tol=0.0, max_steps=5):
    res = run_agent(row["question"], client, conn=conn, embedder=embedder, max_steps=max_steps)
    expected_tool = row["expected_tool"]
    first = res.steps[0] if res.steps else None
    if expected_tool == "none":
        tool_ok = first is None and res.error is None
        args_ok = tool_ok
    else:
        tool_ok = first is not None and first.name == expected_tool
        args_ok = bool(tool_ok and isinstance(first.arguments, dict)
                       and grader.args_match(first.arguments, json.loads(row["expected_args"])))
    g = grader.grade(row, res.answer, time_tol) if res.answer else {"correct": False, "detail": res.error or "нет ответа"}
    return {"id": row["id"], "type": row["type"], "lang": row["lang"], "question": row["question"],
            "expected_answer": row["expected_answer"], "expected_tool": expected_tool,
            "expected_args": row["expected_args"],
            "first_tool": first.name if first else "", "first_args": json.dumps(first.arguments, ensure_ascii=False) if first else "",
            "n_tool_calls": len(res.steps), "tool_ok": tool_ok, "args_ok": args_ok,
            "answer": res.answer or "", "answer_ok": g["correct"], "grade_detail": g["detail"],
            "error": res.error or "", "seconds": res.seconds}


def evaluate(client, rows=None, conn=None, embedder=None, time_tol=0.0, max_steps=5):
    rows = rows if rows is not None else load_questions()
    return [evaluate_question(r, client, conn, embedder, time_tol, max_steps) for r in rows]


def summarize(results):
    n = len(results)
    pct = lambda k: round(100.0 * k / n, 1) if n else 0.0
    count = lambda key, subset=None: sum(1 for r in (subset if subset is not None else results) if r[key])
    summary = {"questions": n,
               "tool_ok": count("tool_ok"), "tool_ok_pct": pct(count("tool_ok")),
               "args_ok": count("args_ok"), "args_ok_pct": pct(count("args_ok")),
               "answer_ok": count("answer_ok"), "answer_ok_pct": pct(count("answer_ok")),
               "model_errors": sum(1 for r in results if r["error"]),
               "mean_seconds": round(sum(r["seconds"] for r in results) / n, 3) if n else 0.0,
               "by_type": {}}
    for t in sorted({r["type"] for r in results}):
        sub = [r for r in results if r["type"] == t]
        summary["by_type"][t] = {"n": len(sub), "tool_ok": count("tool_ok", sub),
                                 "args_ok": count("args_ok", sub), "answer_ok": count("answer_ok", sub)}
    return summary


def format_summary(s, time_tol):
    lines = [f"Вопросов: {s['questions']}   допуск по времени: {time_tol} с (ещё не подтверждён командой)",
             f"Правильная функция:  {s['tool_ok']}/{s['questions']} ({s['tool_ok_pct']}%)",
             f"Верные аргументы:    {s['args_ok']}/{s['questions']} ({s['args_ok_pct']}%)",
             f"Верный ответ:        {s['answer_ok']}/{s['questions']} ({s['answer_ok_pct']}%)",
             f"Ошибки связи с моделью: {s['model_errors']}; среднее время ответа: {s['mean_seconds']} с", "",
             "По типам вопросов (функция / аргументы / ответ):"]
    for t, v in s["by_type"].items():
        lines.append(f"  {t:<13} n={v['n']:<3} {v['tool_ok']}/{v['n']}  {v['args_ok']}/{v['n']}  {v['answer_ok']}/{v['n']}")
    return "\n".join(lines)


def save_csv(results, out_dir, label):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.-]+", "_", label) or "model"
    path = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe}.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(results)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description="Оценка агента на 30 вопросах")
    p.add_argument("--base-url", default=None, help="по умолчанию LLM_BASE_URL")
    p.add_argument("--model", default=None, help="по умолчанию LLM_MODEL")
    p.add_argument("--questions", default=None, help="CSV с вопросами (по умолчанию eval/questions.csv)")
    p.add_argument("--db", choices=["demo", "postgres"], default="demo")
    p.add_argument("--time-tol", type=float, default=0.0,
                   help="допуск по времени, с (по умолчанию 0.0: допуск ещё не подтверждён)")
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--out", default=str(EVAL_DIR / "results"))
    a = p.parse_args(argv)

    client = OpenAICompatClient(base_url=a.base_url, model=a.model)
    if not client.model:
        print("Ошибка: не задана модель (--model или LLM_MODEL)", file=sys.stderr)
        return 2
    embedder = config.embedder_from_env()
    conn = make_demo_db(embedder=embedder) if a.db == "demo" else None
    results = evaluate(client, load_questions(a.questions), conn, embedder, a.time_tol, a.max_steps)
    summary = summarize(results)
    path = save_csv(results, a.out, client.model)
    print(format_summary(summary, a.time_tol))
    print(f"\nРезультаты по вопросам: {path}")
    if summary["model_errors"] == summary["questions"]:
        print("Ошибка: ни на один вопрос не получен ответ модели. Проверьте адрес и модель.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
