
"""Пересобирает тестовые данные и запускает все тесты T5-T10.
Код возврата 0 только если все прошли.
Тесты идут на SQLite, на PostgreSQL не запускались.

Запуск из корня проекта:
    py -m eval.run_tests
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    "eval.build",
    "eval.test_tools",
    "eval.test_loader",
    "eval.test_t10_tools",
]

failed = []

for module in STEPS:
    print(f"\n######## py -m {module} ########")

    res = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
    )

    out = (res.stdout or "") + (res.stderr or "")
    lines = out.splitlines()

    oks = sum(1 for line in lines if line.startswith("OK"))
    fails = [line for line in lines if line.startswith("FAIL")]
    tail = [
        line for line in lines
        if line.startswith("ИТОГО") or line.startswith("events:")
    ]

    for line in fails:
        print(line)

    if res.returncode != 0 and not fails:
        print(out[-1500:])

    print(f"проверок OK: {oks}; " + "; ".join(tail))

    if res.returncode != 0:
        failed.append(module)

print(
    "\n" + (
        "ВСЕ ТЕСТЫ ПРОШЛИ (SQLite)"
        if not failed
        else "ПРОВАЛЫ: " + ", ".join(failed)
    )
)

sys.exit(1 if failed else 0)
