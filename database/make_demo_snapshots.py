"""Генератор ТЕСТОВЫХ срезов для демо и тестов: database/seed_snapshots.json + картинки-заглушки.
Это выдуманные данные (не результат детектора и не результат VLM). Запуск из корня проекта:
    python -m database.make_demo_snapshots
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "demo_frames" / "test_video_01"

# (t, [(цвет, тип, описание), ...], people_count). Один человек с unknown нужен для проверки честных «нет».
PLAN = [
    (0,   [], 0),
    (30,  [("black", "jacket", "black jacket, walking toward the door")], 1),
    (60,  [("red", "sweater", "red sweater, standing near the desk"),
           ("black", "jacket", "black jacket, walking")], 2),
    (90,  [("red", "sweater", "red sweater, sitting at the desk"),
           ("blue", "t-shirt", "blue t-shirt, standing"),
           ("unknown", "unknown", "back turned, partially visible")], 3),
    (120, [("blue", "t-shirt", "blue t-shirt, near the door"),
           ("white", "shirt", "white shirt, holding a bag")], 2),
    (150, [("green", "jacket", "green jacket, standing near the desk"),
           ("white", "shirt", "white shirt, walking")], 2),
    (180, [("green", "jacket", "green jacket, picking up an object"),
           ("black", "hoodie", "black hoodie, sitting"),
           ("gray", "sweater", "gray sweater, standing"),
           ("blue", "t-shirt", "blue t-shirt, walking")], 4),
    (210, [("black", "hoodie", "black hoodie, leaving through the door")], 1),
    (240, [("gray", "sweater", "gray sweater, standing near the desk")], 1),
    (270, [], 0),
]
RGB = {"red": (200, 50, 50), "black": (35, 35, 35), "blue": (50, 90, 200), "white": (235, 235, 235),
       "green": (50, 150, 70), "gray": (140, 140, 140), "unknown": (190, 160, 200)}


def main():
    from PIL import Image, ImageDraw
    FRAMES.mkdir(parents=True, exist_ok=True)
    snaps = []
    for i, (t, people, count) in enumerate(PLAN):
        sid = f"sn_{i:04d}"
        img = Image.new("RGB", (480, 270), (225, 228, 232))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 200, 480, 270], fill=(200, 196, 188))
        d.text((10, 8), f"DEMO FRAME {sid}  t={t}s  people={count}", fill=(60, 60, 60))
        for k, (color, _type, _desc) in enumerate(people):
            x = 40 + k * 100
            d.ellipse([x + 15, 60, x + 55, 100], fill=(224, 190, 160))
            d.rectangle([x, 100, x + 70, 200], fill=RGB[color])
        img.save(FRAMES / f"{sid}.jpg", quality=70)
        snaps.append({"snapshot_id": sid, "t": float(t), "frame_path": f"demo_frames/test_video_01/{sid}.jpg",
                      "people_count": count,
                      "people": [{"person_idx": j, "clothing_color": c, "clothing_type": ty, "description": ds}
                                 for j, (c, ty, ds) in enumerate(people)]})
    out = {"video_id": "test_video_01", "camera_id": "test_cam01", "source": "meva",
           "interval_sec": 30, "snapshots": snaps}
    (ROOT / "database" / "seed_snapshots.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"срезов: {len(snaps)}; кадры: {FRAMES}")


if __name__ == "__main__":
    main()
