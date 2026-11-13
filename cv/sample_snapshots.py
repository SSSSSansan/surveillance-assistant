"""Этап 2 (Арай). Срезы каждые N секунд + подсчёт людей -> snapshots.json и кадры.

  python cv/sample_snapshots.py data/meva/video.avi --out-dir data/out_snapshots

Что делает:
  1. Берёт из видео по одному кадру каждые --interval секунд (по умолчанию 30:
     для 5-минутного видео это t = 0, 30, ..., 270 -> 10 срезов).
  2. Сохраняет кадр в <out-dir>/frames/sn_0000.jpg (чистый, его потом смотрит VLM и фронт).
  3. Считает людей детектором YOLO (тот же, что в run_tracking.py, но БЕЗ трекера:
     срезы между собой не связаны). Рисует рамки в <out-dir>/frames_annotated/ для проверки глазами.
  4. Пишет <out-dir>/snapshots.json в формате из STAGE2_PLAN.md, раздел 4.
     Поля clothing_* и description НЕ заполняются: это сделает Панда через VLM.
     people пока пустой список. Дополнительно кладём "boxes" (рамки людей),
     чтобы Панда могла при желании сопоставить person_idx с рамкой. Лишнее поле можно игнорировать.
"""
import argparse
import json
import os
import cv2
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("--out-dir", default="data/out_snapshots")
p.add_argument("--weights", default="yolo11s.pt")
p.add_argument("--interval", type=float, default=30.0, help="шаг между срезами, секунды")
p.add_argument("--conf", type=float, default=0.3)
p.add_argument("--min-height", type=int, default=0,
                help="игнорировать рамки ниже N пикселей (отсечь мелкий шум), 0 = не отсекать")
p.add_argument("--camera-id", default=None, help="по умолчанию берётся из имени файла (G421)")
p.add_argument("--source", default="meva")
a = p.parse_args()

stem = os.path.splitext(os.path.basename(a.video))[0]
# имя MEVA: 2018-03-15.15-00-00.15-05-00.school.G421.r13 -> камера G421
camera_id = a.camera_id
if camera_id is None:
    parts = stem.split(".")
    camera_id = next((x for x in parts if x.startswith("G") and x[1:].isdigit()), "unknown")

frames_dir = os.path.join(a.out_dir, "frames")
ann_dir = os.path.join(a.out_dir, "frames_annotated")
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(ann_dir, exist_ok=True)

cap = cv2.VideoCapture(a.video)
if not cap.isOpened():
    raise SystemExit(f"Не открывается видео: {a.video}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = n_frames / fps
print(f"видео: {stem}, fps={fps:.2f}, кадров={n_frames}, длительность={duration:.1f} c")

# Номера кадров, которые берём: 0, 30*fps, 60*fps, ... строго меньше длительности.
targets, k = {}, 0
while k * a.interval < duration:
    targets[int(round(k * a.interval * fps))] = k
    k += 1
print(f"срезов будет: {len(targets)} (шаг {a.interval} c)")

model = YOLO(a.weights)
snapshots = []
frame_no = -1
last_wanted = max(targets)
# Читаем подряд (не прыгаем set(POS_FRAMES)): на .avi прыжки бывают неточными.
while frame_no < last_wanted:
    ok = cap.grab()
    if not ok:
        break
    frame_no += 1
    if frame_no not in targets:
        continue
    ok, frame = cap.retrieve()
    if not ok:
        print(f"  не смог прочитать кадр {frame_no}, пропускаю")
        continue
    idx = targets[frame_no]
    sid = f"sn_{idx:04d}"
    cv2.imwrite(os.path.join(frames_dir, sid + ".jpg"), frame)

    r = model.predict(frame, classes=[0], conf=a.conf, verbose=False)[0]
    boxes = []
    for box in r.boxes.xyxy.tolist():
        if (box[3] - box[1]) < a.min_height:
            continue
        boxes.append([round(v, 1) for v in box])
    boxes.sort(key=lambda b: b[0])  # слева направо: порядок стабилен, person_idx можно привязать

    vis = frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(vis, f"#{i}", (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis, f"{sid}  t={frame_no / fps:.1f}s  people={len(boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(ann_dir, sid + ".jpg"), vis)

    snapshots.append({
        "snapshot_id": sid,
        "t": round(frame_no / fps, 3),
        "frame_path": f"frames/{sid}.jpg",  # всегда прямые слэши (баг с Windows-путями был на этапе 1)
        "people_count": len(boxes),
        "boxes": boxes,
        "people": [],
    })
    print(f"  {sid}  t={frame_no / fps:6.1f} c  людей: {len(boxes)}")
cap.release()

out = {"video_id": stem, "camera_id": camera_id, "source": a.source,
       "interval_sec": a.interval, "snapshots": snapshots}
path = os.path.join(a.out_dir, "snapshots.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

counts = [s["people_count"] for s in snapshots]
print(f"\nготово: {len(snapshots)} срезов -> {path}")
if counts:
    print(f"людей: макс {max(counts)} (t={snapshots[counts.index(max(counts))]['t']} c), "
          f"среднее {sum(counts) / len(counts):.1f}")