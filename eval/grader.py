"""Автоматическая проверка ответов агента по eval/questions.csv.

Это ГРУБЫЙ автоматический грейдер для свободного текста модели. Он нужен для быстрого сравнения
моделей и промптов; спорные случаи надо просматривать вручную по CSV с результатами.
Правила (по метрикам стек-документа):
  * да/нет        : первое слово ответа (да/yes или нет/no); «не нашёл» тоже считается «нет»;
  * число         : первое число в ответе равно эталону (точное совпадение);
  * время         : в ответе есть число, отличающееся от эталона не более чем на time_tol секунд;
  * список        : сравниваются времена (число + «с»/«s»): precision и recall; верно только при обоих 1.0;
  * «не нашёл»    : в ответе есть формулировка «не нашёл / не найдено / not found»;
  * вне базы      : в ответе есть «нельзя / нет данных / cannot ...».
Допуск по времени time_tol задаётся параметром. Команда его ещё не подтвердила, поэтому по умолчанию 0.0.
"""
import re

NUM = re.compile(r"\d+(?:\.\d+)?")
TIME_WITH_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:с\b|сек|s\b|sec)", re.I)
YES = {"да", "yes", "yep"}
NO = {"нет", "no", "nope"}
NOT_FOUND = re.compile(r"не\s+нашёл|не\s+нашел|не\s+найден|ничего\s+не\s+найден|not\s+found|"
                       r"no\s+(?:such|matching|confirmed)|(?:could\s*n[o']t|couldn't|did\s+not)\s+find", re.I)
CANNOT = re.compile(r"нельзя|нет\s+(?:этих\s+)?(?:данных|информации)|не\s+могу|невозможно|"
                    r"cannot|can't|no\s+(?:such\s+)?data|does\s+not\s+contain|doesn't\s+contain|"
                    r"don't\s+have|do\s+not\s+have", re.I)


def _first_word(text):
    m = re.match(r"\W*([^\W\d_]+)", (text or "").lower())
    return m.group(1) if m else ""


def _times(text):
    """Времена в тексте: числа с единицей («с», «s»); если единиц нет, то числа с десятичной точкой."""
    with_unit = [float(x) for x in TIME_WITH_UNIT.findall(text or "")]
    if with_unit:
        return with_unit
    return [float(x) for x in NUM.findall(text or "") if "." in x]


def _match_count(expected, got, tol):
    used, matched = set(), 0
    for e in expected:
        for i, g in enumerate(got):
            if i not in used and abs(e - g) <= tol + 1e-9:
                used.add(i)
                matched += 1
                break
    return matched


def grade(row, answer, time_tol=0.0):
    """row: строка questions.csv. Возвращает {"correct": bool, "detail": str, ...}."""
    answer = (answer or "").strip()
    expected = row["expected_answer"].strip()
    qtype = row["type"]
    if not answer:
        return {"correct": False, "detail": "пустой ответ"}

    if qtype == "out_of_scope":
        ok = bool(CANNOT.search(answer))
        return {"correct": ok, "detail": "ответ «нельзя» найден" if ok else "нет формулировки «ответить нельзя»"}

    if expected.lower().startswith("не нашёл") or expected.lower().startswith("не нашел"):
        ok = bool(NOT_FOUND.search(answer))
        return {"correct": ok, "detail": "«не нашёл» найдено" if ok else "ожидалось «не нашёл»"}

    if qtype == "exists":
        fw = _first_word(answer)
        if fw in YES:
            said = "да"
        elif fw in NO or NOT_FOUND.search(answer):
            said = "нет"
        else:
            return {"correct": False, "detail": "не удалось определить да/нет"}
        return {"correct": said == expected.lower(), "detail": f"ответ: {said}, ожидалось: {expected}"}

    if qtype == "count":
        nums = NUM.findall(answer)
        if not nums:
            return {"correct": False, "detail": "в ответе нет числа"}
        ok = float(nums[0]) == float(expected)
        return {"correct": ok, "detail": f"первое число {nums[0]}, ожидалось {expected}"}

    if qtype == "time":
        exp = float(NUM.findall(expected)[0])
        got = _times(answer) or [float(x) for x in NUM.findall(answer)]
        ok = any(abs(exp - g) <= time_tol + 1e-9 for g in got)
        return {"correct": ok, "detail": f"времена в ответе {got}, ожидалось {exp} (допуск {time_tol})"}

    if qtype in ("list", "sequence"):
        exp = _times(expected)
        got = _times(answer)
        m = _match_count(exp, got, time_tol)
        precision = m / len(got) if got else 0.0
        recall = m / len(exp) if exp else 0.0
        ok = bool(exp) and precision == 1.0 and recall == 1.0
        return {"correct": ok, "precision": round(precision, 3), "recall": round(recall, 3),
                "detail": f"времена ответа {got}, ожидалось {exp}"}

    return {"correct": False, "detail": f"неизвестный тип вопроса {qtype!r}"}


# ---------- сравнение аргументов функции ----------

_DEFAULT_LIMIT = 50


def normalize_args(args):
    """Приводит аргументы к сравнимому виду: без None и значений по умолчанию, числа как float,
    строки без пробелов (идентификаторы типа и зоны в нижнем регистре)."""
    out = {}
    for k, v in (args or {}).items():
        if v is None:
            continue
        if isinstance(v, bool):
            out[k] = v
            continue
        if isinstance(v, (int, float)):
            v = float(v)
        elif isinstance(v, str):
            v = v.strip()
            if k in ("event_type", "zone_id", "order", "mode"):
                v = v.lower()
        if k == "order" and v == "asc":
            continue
        if k == "t_from" and v == 0.0:
            continue
        if k == "limit" and v == float(_DEFAULT_LIMIT):
            continue
        out[k] = v
    return out


def args_match(actual, expected):
    return normalize_args(actual) == normalize_args(expected)
