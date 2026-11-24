"""Clothing benchmark: one method (model + mode R1 + prompt variant, or ready predictions) -> metrics.

Run a VLM on the gold crops:
    python -m eval.bench_clothing --gold eval/gold/gold_people.csv --snapshots snapshots.json --frames-dir . \
        --model qwen3-vl-4b --api-url http://127.0.0.1:8765/v1/chat/completions --variant A [--repeat 2]
Score predictions made elsewhere (e.g. CV baselines of Arai), one JSON per line
{"video_id","snapshot_id","person_idx","record":{contract},"seconds":0.1}:
    python -m eval.bench_clothing --gold ... --predictions preds.jsonl --method-name cv:hist

Crops come from --crops-index (crops_index.json of Arai; format ASSUMED, see bench_common) when given,
otherwise they are cut on the fly from the boxes in snapshots.json with vlm/imgutil.py.
Outputs (eval/results/): clothing_<run_id>.md / _metrics.csv / _confusion_color.csv / _confusion_type.csv /
_pred.jsonl, one row appended to clothing_summary.csv, one line appended to runs.jsonl.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from eval import metrics as M
from eval.bench_common import (RESULTS_DIR, SYNTH_BANNER, append_csv_row, append_run, find_frame, is_synthetic,
                               load_crops_index, load_gold_people, load_snapshots, md_table, new_run_id,
                               now_iso, write_csv)

SUMMARY_HEADER = ["run_id", "video_id", "method", "mode", "prompt_variant", "n_scored", "color_acc", "type_acc",
                  "color_macro_f1", "type_macro_f1", "color_top2", "black_share_pred", "black_share_gold",
                  "unknown_share_pred", "sec_per_crop", "repeat_color_disagree", "repeat_type_disagree", "synthetic"]


# ------------------------------------------------------------------ crops ---
def build_items(gold: dict, snaps: list, frames_dir, json_dir, crops_index=None, pad=0.12, min_side=336) -> tuple:
    """-> (items, missing). item = {"key", "data_url", "box_h"} for every gold key that can be cropped."""
    import base64
    import imgutil
    by_id = {s["snapshot_id"]: s for s in snaps}
    cache, items, missing = {}, [], []
    for key in sorted(gold):
        _, sid, idx = key
        url, box_h = None, gold[key].get("box_h")
        if crops_index and key in crops_index:
            ci = crops_index[key]
            try:
                with open(ci["crop_path"], "rb") as f:
                    url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")
            except OSError:
                url = None
            box_h = box_h if box_h is not None else ci.get("box_h")
        elif sid in by_id and idx < len(by_id[sid].get("boxes", [])):
            snap = by_id[sid]
            if sid not in cache:
                fp = find_frame(snap.get("frame_path", ""), frames_dir, json_dir)
                cache[sid] = imgutil.load_image(str(fp)) if fp else None
            if cache[sid] is not None:
                box = snap["boxes"][idx]
                url = imgutil.crop_data_url(cache[sid], box, pad, min_side)
                if box_h is None:
                    box_h = float(box[3]) - float(box[1])
        if url is None:
            missing.append(key)
        else:
            items.append({"key": key, "data_url": url, "box_h": box_h})
    return items, missing


def run_describer(describer, items: list, workers: int = 1) -> dict:
    """-> {key: {"record", "seconds", "status", "error"}}"""
    def one(it):
        r = describer.describe_crop(it["data_url"], it["key"][2])
        return it["key"], {"record": r["record"], "seconds": r["seconds"], "status": r["status"],
                           "error": r["error"]}
    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            return dict(ex.map(one, items))
    return dict(map(one, items))


# ---------------------------------------------------------------- scoring ---
def score(gold: dict, preds: dict, repeat_preds: dict = None) -> dict:
    """gold: consensus dict; preds: {key: {"record","seconds"}}. A gold object without a valid prediction is
    WRONG for accuracy/F1 (counted in n_errors). Objects whose gold value is a tie (None) are skipped."""
    def rec(k):
        p = preds.get(k)
        return p["record"] if p and p.get("record") else None

    out = {}
    ck = [k for k in sorted(gold) if gold[k]["color"] is not None]
    tk = [k for k in sorted(gold) if gold[k]["type"] is not None]
    out["n_gold"], out["n_scored"], out["n_scored_type"] = len(gold), len(ck), len(tk)
    out["n_errors"] = sum(1 for k in gold if rec(k) is None)
    gc = [gold[k]["color"] for k in ck]
    pc = [rec(k)["color"] if rec(k) else None for k in ck]
    p2 = [rec(k).get("color_2") if rec(k) else None for k in ck]
    gt = [gold[k]["type"] for k in tk]
    pt = [rec(k)["type"] if rec(k) else None for k in tk]
    out["color_acc"], out["type_acc"] = M.accuracy(gc, pc), M.accuracy(gt, pt)
    out["color_macro_f1"], out["type_macro_f1"] = M.macro_f1(gc, pc), M.macro_f1(gt, pt)
    out["color_top2"] = M.top2_accuracy(gc, pc, p2)
    out["top2_has_color_2"] = any(x is not None for x in p2)  # False for prompts A/B/C: top-2 == top-1
    out["black_share_pred"] = M.share([x for x in pc if x is not None], "black")
    out["black_share_gold"] = M.share(gc, "black")
    out["unknown_share_pred"] = M.share([x for x in pc if x is not None], "unknown")
    out["color_labels"], out["color_cm"] = M.confusion(gc, pc)
    out["type_labels"], out["type_cm"] = M.confusion(gt, pt)
    buckets = {}
    for name in ("<80", "80-150", ">150", "unknown_h"):
        ks = [k for k in ck if M.bucket_of(gold[k]["box_h"]) == name]
        g = [gold[k]["color"] for k in ks]
        p = [rec(k)["color"] if rec(k) else None for k in ks]
        kt = [k for k in ks if gold[k]["type"] is not None]
        buckets[name] = {"n": len(ks), "color_acc": M.accuracy(g, p) if ks else float("nan"),
                         "type_acc": M.accuracy([gold[k]["type"] for k in kt],
                                                [rec(k)["type"] if rec(k) else None for k in kt]) if kt else float("nan")}
    out["buckets"] = buckets
    secs = [preds[k]["seconds"] for k in gold if k in preds and preds[k].get("seconds") is not None]
    out["sec_per_crop"] = statistics.mean(secs) if secs else float("nan")
    out["sec_per_crop_median"] = statistics.median(secs) if secs else float("nan")
    if repeat_preds is not None:
        both = [k for k in sorted(gold) if rec(k) and repeat_preds.get(k) and repeat_preds[k].get("record")]
        out["repeat_n"] = len(both)
        for f, name in (("color", "repeat_color_disagree"), ("type", "repeat_type_disagree")):
            out[name] = (sum(1 for k in both if rec(k)[f] != repeat_preds[k]["record"][f]) / len(both)
                         if both else float("nan"))
    return out


# ---------------------------------------------------------------- outputs ---
def report_md(meta: dict, m: dict) -> str:
    parts = [f"# Одежда: {meta['method']} · {meta['mode']} · промпт {meta['prompt_variant']} · {meta['run_id']}\n"]
    if meta["synthetic"]:
        parts.append(SYNTH_BANNER)
    parts.append(f"Объектов в эталоне: {m['n_gold']}; оценено по цвету: {m['n_scored']}, по типу: {m['n_scored_type']}; "
                 f"без валидного ответа метода (считаются ошибкой): {m['n_errors']}.\n")
    top2 = "" if m["top2_has_color_2"] else " (метод не даёт color_2 -> top-2 равен top-1)"
    parts.append(md_table(["метрика", "цвет", "тип"], [
        ["accuracy", m["color_acc"], m["type_acc"]],
        ["macro-F1", m["color_macro_f1"], m["type_macro_f1"]],
        ["top-2 accuracy" + top2, m["color_top2"], "—"]]))
    parts.append("\n" + md_table(["показатель", "значение"], [
        ["доля black у метода", m["black_share_pred"]], ["доля black в эталоне", m["black_share_gold"]],
        ["доля unknown (цвет) у метода", m["unknown_share_pred"]],
        ["секунд на кроп (среднее)", m["sec_per_crop"]], ["секунд на кроп (медиана)", m["sec_per_crop_median"]]]))
    parts.append("\n## По высоте кропа (box_h, px)\n" + md_table(
        ["бакет", "кропов", "accuracy цвета", "accuracy типа"],
        [[b, v["n"], v["color_acc"], v["type_acc"]] for b, v in m["buckets"].items()]))
    if "repeat_n" in m:
        parts.append("\n## Повторяемость (два прогона, те же кропы)\n" + md_table(
            ["кропов в обоих прогонах", "расхождение цвета", "расхождение типа"],
            [[m["repeat_n"], m["repeat_color_disagree"], m["repeat_type_disagree"]]]))
    for name, labels, cm in (("цвет", m["color_labels"], m["color_cm"]), ("тип", m["type_labels"], m["type_cm"])):
        parts.append(f"\n## Матрица ошибок: {name} (строки = эталон, столбцы = метод)\n" +
                     md_table(["эталон \\ метод"] + labels, [[g] + [cm[g][p] for p in labels] for g in labels]))
    return "\n".join(parts) + "\n"


def save_outputs(meta: dict, m: dict, preds: dict, out_dir=None) -> dict:
    d = out_dir or RESULTS_DIR
    rid = meta["run_id"]
    paths = {"md": d / f"clothing_{rid}.md", "metrics": d / f"clothing_{rid}_metrics.csv",
             "cm_color": d / f"clothing_{rid}_confusion_color.csv", "cm_type": d / f"clothing_{rid}_confusion_type.csv",
             "pred": d / f"clothing_{rid}_pred.jsonl"}
    d.mkdir(parents=True, exist_ok=True)
    paths["md"].write_text(report_md(meta, m), encoding="utf-8")
    flat = [[k, v] for k, v in m.items() if isinstance(v, (int, float, bool))]
    flat += [[f"{name}_{k}", v[k]] for name, v in m["buckets"].items() for k in ("n", "color_acc", "type_acc")]
    write_csv(paths["metrics"], ["metric", "value"], flat)
    for key, labels, cm in (("cm_color", m["color_labels"], m["color_cm"]), ("cm_type", m["type_labels"], m["type_cm"])):
        write_csv(paths[key], ["gold \\ pred"] + labels, [[g] + [cm[g][p] for p in labels] for g in labels])
    with open(paths["pred"], "w", encoding="utf-8") as f:
        for (v, s, i), p in sorted(preds.items()):
            f.write(json.dumps({"video_id": v, "snapshot_id": s, "person_idx": i, "record": p.get("record"),
                                "seconds": p.get("seconds")}, ensure_ascii=False) + "\n")
    append_csv_row(d / "clothing_summary.csv", SUMMARY_HEADER, [
        rid, meta["video_id"], meta["method"], meta["mode"], meta["prompt_variant"], m["n_scored"], m["color_acc"],
        m["type_acc"], m["color_macro_f1"], m["type_macro_f1"], m["color_top2"], m["black_share_pred"],
        m["black_share_gold"], m["unknown_share_pred"], m["sec_per_crop"], m.get("repeat_color_disagree", ""),
        m.get("repeat_type_disagree", ""), meta["synthetic"]])
    return paths


def exclude_fewshot(gold: dict, fewshot_path) -> dict:
    """Remove the crops used as few-shot examples (their `source`) from the gold: they must not be scored."""
    ex = json.loads(Path(fewshot_path).read_text(encoding="utf-8")).get("examples", [])
    drop = {(e["source"]["video_id"], e["source"]["snapshot_id"], int(e["source"]["person_idx"]))
            for e in ex if isinstance(e.get("source"), dict) and {"video_id", "snapshot_id", "person_idx"} <= set(e["source"])}
    return {k: v for k, v in gold.items() if k not in drop}


def load_predictions(path) -> dict:
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                preds[(r["video_id"], r["snapshot_id"], int(r["person_idx"]))] = {
                    "record": r.get("record"), "seconds": r.get("seconds"), "status": "ok" if r.get("record") else "error"}
    return preds


def run_benchmark(gold: dict, gold_info: dict, video_id: str, describer=None, items=None, predictions=None,
                  repeat_predictions=None, method=None, mode="crop", prompt_variant="A", repeat=1, workers=1, out_dir=None, params=None) -> dict:
    """Core entry (also used by tests). Give a describer + items, OR ready predictions."""
    run_id, started, t0 = new_run_id("clo"), now_iso(), time.perf_counter()
    repeat_preds = None
    if predictions is None:
        preds = run_describer(describer, items, workers)
        if repeat >= 2:
            repeat_preds = run_describer(describer, items, workers)
        for it in items:  # box_h of items without gold box_h (taken from the box) must reach the scoring
            if gold[it["key"]]["box_h"] is None:
                gold[it["key"]]["box_h"] = it["box_h"]
        method, mode, prompt_variant = describer.method, describer.mode, describer.prompt_variant
    else:
        preds, repeat_preds = predictions, repeat_predictions
    m = score(gold, preds, repeat_preds)
    m["gold_info"] = gold_info
    meta = {"run_id": run_id, "video_id": video_id, "method": method, "mode": mode,
            "prompt_variant": prompt_variant, "synthetic": is_synthetic(video_id)}
    paths = save_outputs(meta, m, preds, out_dir)
    if predictions is None:  # ready predictions: runs.jsonl comes from whoever produced them (the server)
      append_run({**meta, "model": getattr(describer, "model", None), "weights": None, "started_at": started,
                "duration_sec": round(time.perf_counter() - t0, 2),
                "params": {"bench": "clothing", "repeat": repeat, "workers": workers, **(params or {})}}, out_dir)
    return {"meta": meta, "metrics": m, "paths": paths}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--snapshots", help="snapshots.json (video_id, frame_path, boxes)")
    ap.add_argument("--frames-dir", default=".")
    ap.add_argument("--crops-index", default=None)
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--predictions-repeat", default=None, help="second pass (…_repeat2.jsonl) for the repeatability metric")
    ap.add_argument("--exclude-fewshot", default=None, help="vlm/fewshot.json: drop its source crops from the gold (fair scoring)")
    ap.add_argument("--method-name", default=None, help="with --predictions: name of the method")
    ap.add_argument("--model", default="qwen3-vl-4b")
    ap.add_argument("--api-url", default="http://127.0.0.1:8765/v1/chat/completions")
    ap.add_argument("--variant", default="A", choices=["A", "B", "C", "D"])
    ap.add_argument("--disable-thinking", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="2 = run twice and measure self-disagreement")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--exclude-visible", default="", help="comma list, e.g. 'не видно'")
    ap.add_argument("--limit", type=int, default=0, help="only the first N gold crops (smoke test)")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    vid = a.video_id
    snaps, json_dir = [], Path(".")
    if a.snapshots:
        sv, snaps, _ = load_snapshots(a.snapshots)
        vid, json_dir = vid or sv, Path(a.snapshots).parent
    excl = tuple(x for x in a.exclude_visible.split(",") if x)
    gold, info = load_gold_people(a.gold, vid, excl)
    if a.exclude_fewshot:
        gold = exclude_fewshot(gold, a.exclude_fewshot)
    if a.limit:
        gold = {k: gold[k] for k in sorted(gold)[:a.limit]}
    if not gold:
        raise SystemExit("gold is empty (no labelled crops for this video_id)")
    vid = vid or sorted(gold)[0][0]
    out_dir = Path(a.out_dir) if a.out_dir else None
    if a.predictions:
        res = run_benchmark(gold, info, vid, predictions=load_predictions(a.predictions),
                            repeat_predictions=load_predictions(a.predictions_repeat) if a.predictions_repeat else None,
                            method=a.method_name or "external", out_dir=out_dir)
    else:
        if not a.snapshots and not a.crops_index:
            raise SystemExit("need --snapshots (to cut crops) or --crops-index")
        from vlm.describers.vlm_api import VlmApiDescriber
        idx = load_crops_index(a.crops_index, vid) if a.crops_index else None
        items, missing = build_items(gold, snaps, a.frames_dir, json_dir, idx)
        if missing:
            print(f"WARN: {len(missing)} gold crops cannot be cut (frame/box/crop missing); they count as errors")
        d = VlmApiDescriber(a.model, a.api_url, "R1", a.variant, a.disable_thinking)
        res = run_benchmark(gold, info, vid, describer=d, items=items, repeat=a.repeat, workers=a.workers,
                            out_dir=out_dir, params={"model": a.model, "api_url": a.api_url,
                                                     "disable_thinking": a.disable_thinking})
    print(res["paths"]["md"].read_text(encoding="utf-8"))
    print("files:", ", ".join(str(p) for p in res["paths"].values()))


if __name__ == "__main__":
    main()
