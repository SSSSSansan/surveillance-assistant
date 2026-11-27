"""Agreement between labelers: % of equal labels and Cohen's kappa, for pairs of labelers.

    python -m eval.agreement --people eval/gold/gold_people.csv --frames eval/gold/gold_frames.csv [--out report.md]

Only objects labelled by BOTH raters of a pair are used (the plan: 50 crops are labelled by all three).
Low agreement is not a failure: it is the ceiling of accuracy for any model, and it goes to the report.
"""
import argparse
import itertools
from collections import defaultdict

from eval import metrics
from eval.bench_common import SYNTH_BANNER, is_synthetic, md_table, people_key, read_csv

FIELDS = ("color", "type", "visible")


def pairwise(rows: list, fields=FIELDS) -> list:
    """rows = gold_people rows (all labelers). -> list of dicts, one per (pair, field)."""
    by_labeler = defaultdict(dict)
    for r in rows:
        by_labeler[r["labeler"]][people_key(r)] = r
    out = []
    for a, b in itertools.combinations(sorted(by_labeler), 2):
        common = sorted(set(by_labeler[a]) & set(by_labeler[b]))
        for f in fields:
            xs = [by_labeler[a][k][f] for k in common]
            ys = [by_labeler[b][k][f] for k in common]
            n = len(common)
            out.append({"pair": f"{a} vs {b}", "field": f, "n": n,
                        "agree": (sum(1 for x, y in zip(xs, ys) if x == y) / n) if n else float("nan"),
                        "kappa": metrics.cohen_kappa(xs, ys)})
    return out


def all_raters(rows: list, fields=FIELDS) -> list:
    """Share of objects where ALL raters who labelled the object agree (objects with >= 3 raters only)."""
    by_key = defaultdict(list)
    for r in rows:
        by_key[people_key(r)].append(r)
    out = []
    for f in fields:
        objs = [rs for rs in by_key.values() if len(rs) >= 3]
        full = sum(1 for rs in objs if len({r[f] for r in rs}) == 1)
        out.append({"field": f, "n": len(objs), "all_agree": full / len(objs) if objs else float("nan")})
    return out


def frames_pairwise(rows: list) -> list:
    by_labeler = defaultdict(dict)
    for r in rows:
        by_labeler[r["labeler"]][(r["video_id"], r["snapshot_id"])] = int(float(r["true_people_count"]))
    out = []
    for a, b in itertools.combinations(sorted(by_labeler), 2):
        common = sorted(set(by_labeler[a]) & set(by_labeler[b]))
        d = [abs(by_labeler[a][k] - by_labeler[b][k]) for k in common]
        n = len(common)
        out.append({"pair": f"{a} vs {b}", "n": n,
                    "exact": sum(1 for x in d if x == 0) / n if n else float("nan"),
                    "mean_abs_diff": sum(d) / n if n else float("nan")})
    return out


def build_report(people_rows: list, frame_rows: list) -> str:
    vids = {r["video_id"] for r in people_rows} | {r["video_id"] for r in frame_rows}
    parts = ["# Согласие разметчиков\n"]
    if is_synthetic(*vids):
        parts.append(SYNTH_BANNER)
    if not people_rows and not frame_rows:
        parts.append("Разметки пока нет (в файлах только заголовки).")
        return "\n".join(parts)
    pw = pairwise(people_rows)
    if pw:
        parts += ["## Кропы: пары разметчиков", md_table(
            ["пара", "поле", "общих кропов", "% совпадений", "каппа Коэна"],
            [[r["pair"], r["field"], r["n"], r["agree"], r["kappa"]] for r in pw]), ""]
    al = all_raters(people_rows)
    if any(r["n"] for r in al):
        parts += ["## Кропы: все трое согласны", md_table(
            ["поле", "кропов с 3+ разметчиками", "доля полного согласия"],
            [[r["field"], r["n"], r["all_agree"]] for r in al]), ""]
    fp = frames_pairwise(frame_rows)
    if fp:
        parts += ["## Кадры: число людей", md_table(
            ["пара", "общих кадров", "точное совпадение", "средняя разница по модулю"],
            [[r["pair"], r["n"], r["exact"], r["mean_abs_diff"]] for r in fp]), ""]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--people", default="eval/gold/gold_people.csv")
    ap.add_argument("--frames", default="eval/gold/gold_frames.csv")
    ap.add_argument("--out", default=None, help="write the markdown report here too")
    a = ap.parse_args()
    report = build_report(read_csv(a.people), read_csv(a.frames))
    print(report)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(report)


if __name__ == "__main__":
    main()
