"""T14 (+T20). snapshots.json (рамки от cv/sample_snapshots.py) -> кропы людей + индексы.

Детекцию тут НЕ повторяем: рамки уже лежат в snapshots.json (или snapshots_vlm.json), их же
получал Панда. Поэтому person_idx здесь = индекс рамки в `boxes` (как у Панды и в базе), и
результаты любых методов сопоставляются с описаниями VLM один в один.

    python -m cv.crop_people --snapshots data/out_snapshots/snapshots.json \
        --frames-root data/out_snapshots --out data/out_snapshots/crops

Выход в --out:
    torso/<snapshot_id>_p<idx>.jpg   кроп ТОРСА (верхние 55% рамки + отступ 12%) - для M1 и M2
    full/<snapshot_id>_p<idx>.jpg    кроп человека ЦЕЛИКОМ (рамка + 12%, как у Панды) - для M3
    crops_index.json                 ВСЕ рамки, у каждой kept=true/false и reason
    frames_index.json                по срезу: people_count (сырой детектор), дубли, мелкие, оставлено

Дубли. Рамка считается дублем, если IoU с БОЛЕЕ КРУПНОЙ рамкой > --iou-nms (0.6). Дубль отсеивается,
а person_idx остальных НЕ сдвигается. Проверено на 159 рамках этапа 2: так находятся ровно 3 дубля
(sn_0000 #4, sn_0007 #5, sn_0009 #17 - везде одинаковая одежда у обеих рамок).
Правило "рамка целиком внутри другой" (--contain-thr) по умолчанию ВЫКЛЮЧЕНО: оно удаляло живых
людей (sn_0003: серая футболка и бежевая рубашка - разные люди внутри одной большой рамки).

Мелкие. Пороги как у Панды (batch_describe_snapshots.py): высота < 60 px или ширина < 30 px.
На видео этапа 2 мелких нет (минимум 74 px высоты, медиана 263 px).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

# ---------- геометрия ----------


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(a, b) -> float:
    """Какая доля МЕНЬШЕЙ рамки лежит внутри большей."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    small = min(_area(a), _area(b))
    return inter / small if small > 0 else 0.0


def find_duplicates(boxes, iou_thr=0.6, contain_thr=1.01):
    """-> {индекс_дубля: индекс_оригинала}. Оригинал = более крупная рамка.
    contain_thr > 1 выключает правило "рамка внутри рамки"."""
    order = sorted(range(len(boxes)), key=lambda i: -_area(boxes[i]))
    kept, dups = [], {}
    for i in order:
        for k in kept:
            if iou(boxes[i], boxes[k]) > iou_thr or containment(boxes[i], boxes[k]) > contain_thr:
                dups[i] = k
                break
        else:
            kept.append(i)
    return dups


def _clip(b, W, H):
    x1, y1 = max(0, int(round(b[0]))), max(0, int(round(b[1])))
    x2, y2 = min(W, int(round(b[2]))), min(H, int(round(b[3])))
    return (x1, y1, x2, y2) if (x2 - x1 >= 2 and y2 - y1 >= 2) else None


def full_box(box, W, H, pad=0.12):
    """Та же геометрия, что у Панды (vlm/imgutil.clamp_box): рамка + pad с каждой стороны."""
    w, h = box[2] - box[0], box[3] - box[1]
    return _clip((box[0] - pad * w, box[1] - pad * h, box[2] + pad * w, box[3] + pad * h), W, H)


def torso_box(box, W, H, pad=0.12, torso_frac=0.55):
    """Верхние torso_frac рамки; отступ pad по бокам и снизу, сверху без отступа (там фон)."""
    w, h = box[2] - box[0], box[3] - box[1]
    return _clip((box[0] - pad * w, box[1], box[2] + pad * w, box[1] + torso_frac * h + pad * h), W, H)


def height_bucket(h):
    return "<80" if h < 80 else "80-150" if h < 150 else ">150"


# ---------- основная работа ----------


def build_index(snap_json, frames_root, out, iou_thr=0.6, contain_thr=1.01, pad=0.12,
                torso_frac=0.55, min_h=60.0, min_w=30.0, write_images=True):
    out = Path(out)
    if write_images:
        (out / "torso").mkdir(parents=True, exist_ok=True)
        (out / "full").mkdir(parents=True, exist_ok=True)
    video_id = snap_json["video_id"]
    crops, frames = [], []
    for s in snap_json["snapshots"]:
        sid, boxes = s["snapshot_id"], s.get("boxes") or []
        img = None
        if write_images:
            fp = Path(frames_root) / s["frame_path"]
            img = cv2.imread(str(fp))
            if img is None:
                print(f"  ! {sid}: не читается кадр {fp}, пропуск среза")
                continue
        dups = find_duplicates(boxes, iou_thr, contain_thr)
        n_small = n_kept = 0
        for idx, box in enumerate(boxes):
            bw, bh = box[2] - box[0], box[3] - box[1]
            reason = None
            if idx in dups:
                reason = f"dup_of_p{dups[idx]:02d}"
            elif bh < min_h or bw < min_w:
                reason, n_small = "too_small", n_small + 1
            item = {"crop_id": f"{sid}_p{idx:02d}", "video_id": video_id, "snapshot_id": sid,
                    "person_idx": idx, "t": s.get("t"), "box": box, "box_h": round(bh, 1),
                    "box_w": round(bw, 1), "h_bucket": height_bucket(bh),
                    "frame_path": s["frame_path"], "kept": reason is None, "reason": reason}
            if reason is None:
                n_kept += 1
                if write_images:
                    H, W = img.shape[:2]
                    tb, fb = torso_box(box, W, H, pad, torso_frac), full_box(box, W, H, pad)
                    if tb is None or fb is None:
                        item["kept"], item["reason"] = False, "bad_box"
                        n_kept -= 1
                    else:
                        tp, fp2 = out / "torso" / f"{item['crop_id']}.jpg", out / "full" / f"{item['crop_id']}.jpg"
                        cv2.imwrite(str(tp), img[tb[1]:tb[3], tb[0]:tb[2]])
                        cv2.imwrite(str(fp2), img[fb[1]:fb[3], fb[0]:fb[2]])
                        item["path"], item["full_path"] = tp.as_posix(), fp2.as_posix()
            crops.append(item)
        frames.append({"video_id": video_id, "snapshot_id": sid, "t": s.get("t"),
                       "frame_path": s["frame_path"],
                       "people_count": s.get("people_count", len(boxes)),  # сырой счёт детектора (T33)
                       "n_dups": len(dups), "n_too_small": n_small,
                       "people_after_dedup": len(boxes) - len(dups), "n_kept": n_kept})
    return crops, frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--snapshots", required=True, help="snapshots.json или snapshots_vlm.json (нужны boxes)")
    ap.add_argument("--frames-root", help="папка, от которой считается frame_path (по умолчанию - папка json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou-nms", type=float, default=0.6)
    ap.add_argument("--contain-thr", type=float, default=1.01, help=">1 = выключено (так и надо)")
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--torso-frac", type=float, default=0.55)
    ap.add_argument("--min-h", type=float, default=60.0)
    ap.add_argument("--min-w", type=float, default=30.0)
    ap.add_argument("--no-images", action="store_true", help="только индексы (без кадров, для проверки)")
    a = ap.parse_args()

    sj = Path(a.snapshots)
    data = json.loads(sj.read_text(encoding="utf-8"))
    root = Path(a.frames_root) if a.frames_root else sj.parent
    crops, frames = build_index(data, root, a.out, a.iou_nms, a.contain_thr, a.pad,
                                a.torso_frac, a.min_h, a.min_w, write_images=not a.no_images)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "crops_index.json").write_text(json.dumps(crops, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "frames_index.json").write_text(json.dumps(frames, ensure_ascii=False, indent=1), encoding="utf-8")

    n_kept = sum(c["kept"] for c in crops)
    reasons = {}
    for c in crops:
        if c["reason"]:
            key = "dup" if c["reason"].startswith("dup_of") else c["reason"]
            reasons[key] = reasons.get(key, 0) + 1
    buckets = {}
    for c in crops:
        if c["kept"]:
            buckets[c["h_bucket"]] = buckets.get(c["h_bucket"], 0) + 1
    print(f"видео {data['video_id']}: срезов {len(frames)}, рамок {len(crops)}, кропов оставлено {n_kept}, "
          f"отсеяно {len(crops) - n_kept} {reasons}")
    print(f"высота рамок по бакетам (кропы): {buckets}")
    print(f"индексы: {out / 'crops_index.json'}, {out / 'frames_index.json'}")


if __name__ == "__main__":
    main()
