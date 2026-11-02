"""T3. Детектор + ByteTrack на одном видео -> tracks.json и видео с рамками.

  python cv/run_tracking.py data/meva/video.avi --out-dir data/out --weights yolo11s.pt

Веса меняются параметром --weights. В плане стоит RT-DETRv4-S: если он у тебя уже
запускается локально, оставь свой код, а результат сохрани через tracks_io.save_tracks.
Ultralytics понимает rtdetr-l.pt и yolo11*.pt без доработок.
"""
import argparse
import os
import cv2
from ultralytics import YOLO
from tracks_io import save_tracks

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("--out-dir", default="data/out")
p.add_argument("--weights", default="yolo11s.pt")
p.add_argument("--conf", type=float, default=0.3)
p.add_argument("--stride", type=int, default=3, help="брать каждый N-й кадр (30 fps / 3 = 10 fps)")
a = p.parse_args()

os.makedirs(a.out_dir, exist_ok=True)
stem = os.path.splitext(os.path.basename(a.video))[0]
cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w, h = int(cap.get(3)), int(cap.get(4))
writer = cv2.VideoWriter(os.path.join(a.out_dir, stem + "_tracked.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps / a.stride, (w, h))
model = YOLO(a.weights)
tracks, frame_no = {}, -1
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1
    if frame_no % a.stride:
        continue
    r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                    classes=[0], conf=a.conf, verbose=False)[0]
    if r.boxes is not None and r.boxes.id is not None:
        for box, tid, c in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(), r.boxes.conf.tolist()):
            tracks.setdefault(tid, []).append({"t": round(frame_no / fps, 3), "frame": frame_no,
                                               "bbox": [round(v, 1) for v in box], "conf": round(c, 3)})
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"id {tid}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    writer.write(frame)
cap.release()
writer.release()
save_tracks(os.path.join(a.out_dir, "tracks.json"), stem, fps, w, h, tracks)
print("треков:", len(tracks), "->", a.out_dir)