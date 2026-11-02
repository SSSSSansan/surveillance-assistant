"""T3. Нарисовать зоны мышью на первом кадре и сохранить в zones.json.

  python cv/draw_zones.py data/meva/video.avi cv/zones.json

Управление: клик левой кнопкой добавляет точку, Enter закрывает зону и спрашивает
в консоли её имя и вид (entrance или object), Esc сохраняет и выходит.
entrance = дверь/вход, object = стол или полка, где что-то берут и кладут.
"""
import json
import sys
import cv2

video, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(video)
ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("не открылось видео")

zones, pts = [], []


def on_click(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])


cv2.namedWindow("zones")
cv2.setMouseCallback("zones", on_click)
while True:
    img = frame.copy()
    for z in zones:
        cv2.polylines(img, [__import__("numpy").array(z["polygon"])], True, (0, 255, 0), 2)
    for p in pts:
        cv2.circle(img, tuple(p), 4, (0, 0, 255), -1)
    cv2.imshow("zones", img)
    k = cv2.waitKey(30) & 0xFF
    if k == 13 and len(pts) >= 3:
        zid = input("zone_id (например door, table): ").strip()
        kind = input("вид (entrance/object): ").strip()
        zones.append({"zone_id": zid, "kind": kind, "polygon": pts[:]})
        pts.clear()
    elif k == 27:
        break
cv2.destroyAllWindows()
with open(out, "w", encoding="utf-8") as f:
    json.dump({"width": frame.shape[1], "height": frame.shape[0], "zones": zones}, f, indent=2)
print("сохранено:", out, "зон:", len(zones))