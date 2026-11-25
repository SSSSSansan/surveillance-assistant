"""Сводная таблица моделей для T30: собирает последние CSV каждой модели из eval/results/.

Что берётся:
  события   eval/results/<дата>_<время>_<модель>.csv            (py -m eval.eval_agent --model ...)
  срезы     eval/results/snapshots_<дата>_<время>_<модель>.csv  (py -m eval.eval_snapshots --model ...)
Для каждой модели берётся самый свежий файл. Считается: верная функция, верные аргументы, верный ответ,
доля ответов с иероглифами, ошибки связи, среднее время; по срезам: верных ответов и среднее время.
Запуск из корня проекта:  py -m eval.compare_models
Результат печатается и сохраняется в eval/results/comparison.md и comparison.csv.
Модели, для которых нет какого-то файла, помечаются «-»: ничего не выдумывается."""
import argparse
import csv
import re
import sys
from pathlib import Path

from agent import runner

RESULTS = Path(__file__).resolve().parent / "results"
_NAME = re.compile(r"^(snapshots_)?(\d{8}_\d{6})_(.+)\.csv$")


def _flag(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def find_files(folder):
    """{модель: {"events": путь, "snapshots": путь}}: самые свежие файлы."""
    found = {}
    for p in sorted(Path(folder).glob("*.csv")):
        m = _NAME.match(p.name)
        if not m:
            continue
        kind = "snapshots" if m.group(1) else "events"
        found.setdefault(m.group(3), {})[kind] = p          # sorted: позднее имя перезаписывает раннее
    return found


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def stats_events(rows):
    n = len(rows)
    if not n:
        return None
    pct = lambda k: sum(k) / n * 100
    return {"n": n,
            "function": pct([_flag(r["tool_ok"]) for r in rows]),
            "args": pct([_flag(r["args_ok"]) for r in rows]),
            "answer": pct([_flag(r["answer_ok"]) for r in rows]),
            "cjk": pct([runner.has_foreign_script(r.get("answer") or "", r.get("question") or "") for r in rows]),
            "errors": sum(1 for r in rows if (r.get("error") or "").strip()),
            "seconds": sum(float(r["seconds"] or 0) for r in rows) / n}


def stats_snapshots(rows):
    n = len(rows)
    if not n:
        return None
    return {"n": n, "ok": sum(_flag(r["ok"]) for r in rows),
            "seconds": sum(float(r["seconds"] or 0) for r in rows) / n,
            "cjk": sum(_flag(r["foreign_script"]) for r in rows)}


def build(folder=RESULTS):
    out = []
    for model, files in sorted(find_files(folder).items()):
        ev = stats_events(read_rows(files["events"])) if "events" in files else None
        sn = stats_snapshots(read_rows(files["snapshots"])) if "snapshots" in files else None
        out.append({"model": model, "events": ev, "snapshots": sn,
                    "files": {k: v.name for k, v in files.items()}})
    return out


HEAD = ["model", "events: function %", "events: args %", "events: answer %", "events: CJK %", "events: link errors",
        "events: sec", "snapshots: correct", "snapshots: sec"]


def table_rows(results):
    f1 = lambda x: f"{x:.1f}"
    rows = []
    for r in results:
        ev, sn = r["events"], r["snapshots"]
        rows.append([r["model"],
                     f1(ev["function"]) if ev else "-", f1(ev["args"]) if ev else "-", f1(ev["answer"]) if ev else "-",
                     f1(ev["cjk"]) if ev else "-", ev["errors"] if ev else "-", f1(ev["seconds"]) if ev else "-",
                     f"{sn['ok']}/{sn['n']}" if sn else "-", f1(sn["seconds"]) if sn else "-"])
    return rows


def markdown(results):
    rows = table_rows(results)
    out = ["| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    out += ["", "Files used:"] + [f"- {r['model']}: " + ", ".join(f"{k} = {v}" for k, v in r["files"].items()) for r in results]
    return "\n".join(out) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="T30: сводная таблица моделей")
    p.add_argument("--dir", default=str(RESULTS))
    a = p.parse_args(argv)
    results = build(a.dir)
    if not results:
        print("В папке нет подходящих CSV. Сначала запустите eval_agent / eval_snapshots для моделей.", file=sys.stderr)
        return 1
    md = markdown(results)
    print(md)
    out = Path(a.dir)
    (out / "comparison.md").write_text(md, encoding="utf-8")
    with open(out / "comparison.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEAD)
        w.writerows(table_rows(results))
    print(f"Сохранено: {out / 'comparison.md'} и comparison.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
