"""Accuracy of the clothing COLOR against the crop height bucket (box_h: <80 / 80-150 / >150 px), one line per method.
Same gold, same matching and same buckets as `eval.score_methods people` (its functions are imported, not copied).

    python3 -m eval.plot_height_accuracy --gold "eval/gold_real/gold_people_*.csv" \
        --pred "server_runs2/flat/*.jsonl" --exclude-file eval/gold_real/fewshot_examples.json \
        --out server_runs2/height_accuracy.png

*_repeat<N> files are skipped (a second pass of the same method); --keep-repeats keeps them.
A bucket with fewer than --min-n crops is drawn as a gap: one or two crops say nothing about accuracy.
Needs matplotlib only (no seaborn / pandas).
"""
import argparse
import glob
import re
import sys
from pathlib import Path

from eval.score_methods import BUCKETS, bucket, load_exclude, load_gold_people, load_preds


def height_accuracy(gold, pred_paths, keep_repeats=False):
    """-> {method: [(bucket name, accuracy or None, n)]} in BUCKETS order."""
    out = {}
    for pat in pred_paths:
        for pth in sorted(glob.glob(str(pat))):
            name = Path(pth).stem
            if not keep_repeats and re.search(r"_repeat\d+$", name):
                continue
            preds = load_preds(pth)
            pairs = [(g, preds[k]) for k, g in gold.items() if k in preds]
            if not pairs:
                print(f"{name}: нет пересечения с эталоном, пропущен", file=sys.stderr)
                continue
            row = []
            for bn, _, _ in BUCKETS:
                sub = [(g, p) for g, p in pairs if bucket(p.get("box_h")) == bn]
                acc = sum(g["color"] == p["color"] for g, p in sub) / len(sub) if sub else None
                row.append((bn, acc, len(sub)))
            out[name] = row
    return out


def plot(data, out, min_n=5, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [b[0] for b in BUCKETS]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # colour = model, line style = prompt variant (_R1_A / _B / _C / _D), so 12+ lines stay readable
    def split(m):
        hit = re.match(r"^(?:pred_clothing_)?(.+?)_R1_([A-D])(.*)$", m)
        return (hit.group(1), hit.group(2) + hit.group(3)) if hit else (m, "")
    models = sorted({split(m)[0] for m in data})
    colors = plt.get_cmap("tab10" if len(models) <= 10 else "tab20")
    styles = {"A": ("-", "o"), "B": ("--", "s"), "C": ("-.", "^"), "D": (":", "D")}
    for method, row in sorted(data.items()):
        model, var = split(method)
        ls, mk = styles.get(var[:1], ("-", "x"))
        ys = [acc if (acc is not None and n >= min_n) else float("nan") for _, acc, n in row]
        label = method.replace("pred_clothing_", "") + "  (n: " + "/".join(str(n) for _, _, n in row) + ")"
        ax.plot(range(len(names)), ys, linestyle=ls, marker=mk, color=colors(models.index(model)), linewidth=1.6,
                label=label)
    hidden = sorted({bn for row in data.values() for bn, acc, n in row if acc is not None and n < min_n})
    if hidden:
        ax.text(0.01, 0.02, f"не показано (меньше {min_n} кропов): " + ", ".join(f"{b} px" for b in hidden),
                transform=ax.transAxes, fontsize=8, color="0.35")
    ax.set_xticks(range(len(names)), [f"{n} px" for n in names])
    ax.set_xlabel("высота кропа box_h")
    ax.set_ylabel("accuracy цвета")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title(title or "Accuracy цвета по высоте кропа")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, nargs="+")
    ap.add_argument("--pred", required=True, nargs="+")
    ap.add_argument("--exclude-file", nargs="*", default=[])
    ap.add_argument("--out", default="server_runs2/height_accuracy.png")
    ap.add_argument("--min-n", type=int, default=5, help="a bucket with fewer crops is not drawn")
    ap.add_argument("--keep-repeats", action="store_true")
    a = ap.parse_args(argv)
    gold, _, _ = load_gold_people(a.gold)
    ex = load_exclude(a.exclude_file)
    gold = {k: v for k, v in gold.items() if k not in ex}
    data = height_accuracy(gold, a.pred, a.keep_repeats)
    if not data:
        print("нет ни одного метода с пересечением с эталоном", file=sys.stderr)
        return 1
    for m, row in sorted(data.items()):
        print(m, " ".join(f"{bn}: {'-' if acc is None else f'{acc:.3f}'} (n={n})" for bn, acc, n in row))
    plot(data, a.out, a.min_n)
    print(f"PNG: {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
