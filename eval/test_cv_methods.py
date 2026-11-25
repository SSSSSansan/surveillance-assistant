"""Тесты задач Арай этапа 3 (T14, T17, T20, T24, T32, T33, T34). Без GPU, видео и интернета.
Запуск из корня проекта:  python -m eval.test_cv_methods   (или py -m ...)
"""
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cv.color_hist import HistColorDescriber  # noqa: E402
from cv.crop_people import build_index, find_duplicates, torso_box  # noqa: E402
from cv.track_vote import assign_tracks, load_tracks_file, vote  # noqa: E402
from eval.export_snapshots_json import export  # noqa: E402
from eval.score_methods import load_preds  # noqa: E402
from vlm.describers import base  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"OK   {name}")
    else:
        print(f"FAIL {name} {detail}")
        FAILS.append(name)


# ------------------------------------------------------------------ словарь

def _tuple_from(path, name):
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return tuple(ast.literal_eval(node.value))
    return None


vlm_file = ROOT / "vlm" / "describe_snapshot.py"
check("словарь цветов совпадает с vlm/describe_snapshot.py", _tuple_from(vlm_file, "COLORS") == base.COLORS)
check("словарь типов совпадает с vlm/describe_snapshot.py", _tuple_from(vlm_file, "TYPES") == base.TYPES)

# ------------------------------------------------------------------ цвета (M1)

RGB = {"red": (200, 30, 30), "blue": (30, 60, 200), "green": (30, 150, 60), "black": (20, 20, 20),
       "white": (240, 240, 240), "gray": (128, 128, 128), "beige": (220, 200, 170), "brown": (110, 70, 40),
       "pink": (240, 150, 180), "yellow": (240, 220, 40), "orange": (240, 130, 30), "purple": (120, 50, 160)}
# как в правилах Панды: светло-синий и тёмно-синий -> blue, бордовый -> red
RGB_MAPPED = {"light blue": ((140, 200, 240), "blue"), "navy": ((20, 30, 80), "blue"),
              "burgundy": ((120, 20, 40), "red")}


def fake_crop(rgb, skin=True):
    img = np.full((120, 80, 3), (90, 140, 90), np.uint8)
    img[10:120, 12:68] = rgb[::-1]
    if skin:
        img[0:14, 30:50] = (140, 170, 225)
    return img


d = HistColorDescriber(wb_alpha=0)
bad = [(n, d.describe(fake_crop(c))["color"]) for n, c in RGB.items() if d.describe(fake_crop(c))["color"] != n]
check("M1: 12 цветовых образцов", not bad, str(bad))
bad = [(n, d.describe(fake_crop(c))["color"], w) for n, (c, w) in RGB_MAPPED.items()
       if d.describe(fake_crop(c))["color"] != w]
check("M1: светло-синий/тёмно-синий -> blue, бордовый -> red", not bad, str(bad))
a = HistColorDescriber().describe(fake_crop(RGB["red"]))
check("M1: контракт ответа", all(k in a for k in ("color", "color_2", "type", "confidence", "description",
                                                    "method", "mode", "prompt_variant")) and a["type"] == "unknown")
check("M1: ответ всегда из словаря", all(HistColorDescriber().describe(fake_crop(c))["color"] in base.COLORS
                                         for c in RGB.values()))
check("M1: пустой кроп -> unknown", HistColorDescriber().describe(np.zeros((0, 0, 3), np.uint8))["color"] == "unknown")
img2 = fake_crop(RGB["black"], skin=False)
img2[10:60, 12:68] = RGB["white"][::-1]
a2 = HistColorDescriber(wb_alpha=0).describe(img2)
check("M1: два цвета (белый верх, чёрный низ)", {a2["color"], a2["color_2"]} == {"white", "black"}, str(a2))

# ------------------------------------------------------- РЕАЛЬНЫЕ данные этапа 2 (T14)

real = ROOT / "vlm" / "snapshots_vlm.json"
if real.exists():
    rd = json.loads(real.read_text(encoding="utf-8"))
    total = sum(len(s["boxes"]) for s in rd["snapshots"])
    crops, frames = build_index(rd, ROOT, "unused", write_images=False)
    dup = {(c["snapshot_id"], c["person_idx"]) for c in crops if c["reason"] and c["reason"].startswith("dup")}
    check("реальные данные: 159 рамок", total == 159 and len(crops) == 159)
    check("реальные данные: найдено ровно 3 дубля (sn_0000 #4, sn_0007 #5, sn_0009 #17)",
          dup == {("sn_0000", 4), ("sn_0007", 5), ("sn_0009", 17)}, str(sorted(dup)))
    check("реальные данные: живые люди sn_0003 (серая футболка, бежевая рубашка) НЕ удалены",
          all(c["kept"] for c in crops if c["snapshot_id"] == "sn_0003"))
    check("реальные данные: person_idx не сдвигается (индексы 0..N-1 подряд)",
          all([c["person_idx"] for c in crops if c["snapshot_id"] == s["snapshot_id"]] == list(range(len(s["boxes"])))
              for s in rd["snapshots"]))
    check("реальные данные: people_count сохранён сырым, без дублей на 3 меньше",
          sum(f["people_count"] for f in frames) == 159 and sum(f["people_after_dedup"] for f in frames) == 156)
    check("реальные данные: мелких нет (мин. высота 74 px > 60)", not any(c["reason"] == "too_small" for c in crops))
    cont = find_duplicates([b for b in rd["snapshots"][3]["boxes"]], 0.6, 0.85)
    check("контроль: правило 'рамка внутри рамки' УДАЛИЛО БЫ живых людей (поэтому выключено)", len(cont) > 0)
else:
    print("SKIP реальные данные: нет vlm/snapshots_vlm.json")

# ------------------------------------------------- кропы на синтетическом кадре

tmp = Path(tempfile.mkdtemp())
(tmp / "frames").mkdir()
frame = np.full((400, 600, 3), 70, np.uint8)
cv2.rectangle(frame, (50, 40), (130, 300), (0, 0, 200), -1)
cv2.imwrite(str(tmp / "frames" / "sn_0000.jpg"), frame)
snap = {"video_id": "v", "camera_id": "G421", "source": "meva", "interval_sec": 30.0,
        "snapshots": [{"snapshot_id": "sn_0000", "t": 0.0, "frame_path": "frames/sn_0000.jpg", "people_count": 4,
                       "boxes": [[50, 40, 130, 300], [52, 42, 128, 298], [300, 100, 320, 130], [400, 50, 520, 340]],
                       "people": []}]}
cr, fr = build_index(snap, tmp, tmp / "out")
reasons = [c["reason"] for c in cr]
check("кроппер: дубль, мелкая рамка, 2 оставлены",
      reasons == [None, "dup_of_p00" if False else reasons[1], "too_small", None] and reasons[1].startswith("dup_of")
      and sum(c["kept"] for c in cr) == 2, str(reasons))
check("кроппер: файлы торса и целиком созданы", all(Path(c["path"]).exists() and Path(c["full_path"]).exists()
                                                    for c in cr if c["kept"]))
check("кроппер: дубль оставляет БОЛЬШУЮ рамку", cr[0]["kept"] and not cr[1]["kept"])
tb = torso_box([100, 100, 200, 300], 640, 480, 0.12, 0.55)
check("торс: верхние 55% + отступ", tb == (88, 100, 212, 100 + int(0.55 * 200 + 0.12 * 200)), str(tb))
check("frames_index: сырой счёт и счёт без дублей", fr[0]["people_count"] == 4 and fr[0]["people_after_dedup"] == 3)

# ---------------------------------- run_method -> score -> count -> export (T17, T33, T34)

out = tmp / "out"
(out / "crops_index.json").write_text(json.dumps(cr), encoding="utf-8")
# синтетические кропы разных цветов вместо однотонного кадра
for c, col in zip([x for x in cr if x["kept"]], ["red", "black"]):
    cv2.imwrite(c["path"], fake_crop(RGB[col]))
(out / "frames_index.json").write_text(json.dumps(fr), encoding="utf-8")


def run(*args):
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


r = run("eval.run_method", "--method", "hist", "--index", str(out / "crops_index.json"), "--out", str(tmp / "res"))
check("run_method hist запускается", r.returncode == 0, r.stderr[-300:] + r.stdout[-300:])
rows = [json.loads(line) for line in open(tmp / "res" / "hist__crop__A.jsonl", encoding="utf-8")]
check("run_method: 2 строки, ключ как в базе (video_id, sn_XXXX, person_idx)",
      len(rows) == 2 and rows[0]["snapshot_id"] == "sn_0000" and rows[0]["person_idx"] == 0
      and {x["color"] for x in rows} == {"red", "black"}, str(rows)[:200])

gold = tmp / "gold.csv"
gold.write_text("video_id,snapshot_id,person_idx,clothing_color,clothing_type,visible,box_h,labeler\n"
                "v,sn_0000,0,red,sweater,да,260,a\nv,sn_0000,0,red,sweater,да,260,b\n"
                "v,sn_0000,3,black,jacket,да,290,a\nv,sn_0000,3,gray,jacket,да,290,b\n", encoding="utf-8")
r = run("eval.score_methods", "people", "--gold", str(gold), "--pred", str(tmp / "res" / "hist__crop__A.jsonl"))
check("score people: строки sn_0000, алиас clothing_color, спорный объект = конфликт",
      r.returncode == 0 and "конфликтов разметчиков 1" in r.stdout and "| 1 | 1.000 |" in r.stdout, r.stdout[-400:] + r.stderr[-300:])

# VLM из snapshots_vlm.json как pred
if real.exists():
    vp = load_preds(real)
    check("load_preds понимает snapshots_vlm.json (159 людей)", len(vp) == 159
          and next(iter(vp.values()))["method"].startswith("vlm:"))

fg = tmp / "fg.csv"
fg.write_text("video_id,snapshot_id,true_people_count,labeler\nv,sn_0000,3,a\n", encoding="utf-8")
r = run("eval.score_methods", "count", "--gold", str(fg), "--frames-index", str(out / "frames_index.json"))
check("T33: два счёта (сырой +1, без дублей 0)", r.returncode == 0 and "4 (+1)" in r.stdout and "3 (+0)" in r.stdout,
      r.stdout[-400:] + r.stderr[-300:])

if real.exists():
    base_json = json.loads(real.read_text(encoding="utf-8"))
    fake_preds = {("2018-03-15.15-00-00.15-05-00.school.G421.r13", "sn_0000", 0):
                  {"color": "red", "type": "unknown", "method": "cv:hist"}}
    n = export(base_json, fake_preds)
    p0 = base_json["snapshots"][0]["people"][0]
    check("экспорт: цвет подменён, тип от VLM сохранён, остальные люди целы",
          n == 1 and p0["clothing_color"] == "red" and p0["clothing_type"] == "jacket"
          and base_json["snapshots"][0]["people"][1]["clothing_color"] == "gray", str(p0))
    try:
        from database.snapshots_loader import _validate
        _validate(base_json)
        check("экспорт: файл проходит валидацию вашего snapshots_loader", True)
    except Exception as e:  # noqa: BLE001
        check("экспорт: файл проходит валидацию вашего snapshots_loader", False, repr(e)[:200])

# ------------------------------------------------------------------ T32

tj = tmp / "tracks.json"
tj.write_text(json.dumps({"video_id": "v", "fps": 30, "width": 600, "height": 400, "tracks": [
    {"track_id": 7, "points": [{"t": 0.0, "frame": 0, "bbox": [50, 40, 130, 300], "conf": 0.9},
                               {"t": 30.0, "frame": 900, "bbox": [52, 44, 131, 302], "conf": 0.9}]},
    {"track_id": 8, "points": [{"t": 30.0, "frame": 900, "bbox": [400, 50, 520, 340], "conf": 0.9}]}]}), encoding="utf-8")
tr = load_tracks_file(tj)
cs = [{"snapshot_id": "sn_0000", "person_idx": 0, "t": 0.0, "box": [50, 40, 130, 300]},
      {"snapshot_id": "sn_0001", "person_idx": 0, "t": 30.0, "box": [51, 43, 129, 300]},
      {"snapshot_id": "sn_0001", "person_idx": 1, "t": 30.0, "box": [400, 50, 520, 340]}]
kt = assign_tracks(cs, tr)
check("T32: привязка кропов к трекам из tracks.json (формат tracks_io)",
      kt == {("sn_0000", 0): 7, ("sn_0001", 0): 7, ("sn_0001", 1): 8}, str(kt))
rows_k = {("sn_0000", 0): {"color": "beige", "type": "jacket", "confidence": 0.9},
          ("sn_0001", 0): {"color": "brown", "type": "jacket", "confidence": 0.4},
          ("sn_0001", 1): {"color": "blue", "type": "unknown", "confidence": 0.5}}
win, _ = vote(rows_k, kt)
check("T32: голосование по весу confidence", win == {7: ("beige", "jacket"), 8: ("blue", "unknown")}, str(win))

# ------------------------------------------------- пороги тёмных цветов (M1) и контактный лист

from cv.color_hist import classify_hsv  # noqa: E402

# (30,40,70) RGB: тёмный синеватый: по умолчанию blue, с --dark-sat 0.6 -> black
h_, s_, v_ = __import__("cv.color_hist", fromlist=["x"]).bgr_to_hsv_deg((70, 40, 30))
check("пороги: по умолчанию поведение прежнее (тёмно-синеватый -> blue)", classify_hsv(h_, s_, v_) == "blue")
check("пороги: --dark-sat 0.6 переводит тёмно-синеватый в black", classify_hsv(h_, s_, v_, dark_sat=0.6) == "black")
hn, sn, vn = __import__("cv.color_hist", fromlist=["x"]).bgr_to_hsv_deg((120, 40, 20))  # настоящий тёмно-синий navy: RGB (20,40,120) = BGR (120,40,20)
check("пороги: настоящий navy остаётся blue даже при --dark-sat 0.6", classify_hsv(hn, sn, vn, dark_sat=0.6) == "blue")
hg, sg, vg = __import__("cv.color_hist", fromlist=["x"]).bgr_to_hsv_deg((60, 60, 60))
check("пороги: тёмно-серый по умолчанию black, с --neutral-black-v 0.2 gray",
      classify_hsv(hg, sg, vg) == "black" and classify_hsv(hg, sg, vg, neutral_black_v=0.2) == "gray")

from eval.quick_agreement import contact_sheet  # noqa: E402

sheet_dir = Path(tempfile.mkdtemp())
paths_, items_ = {}, []
for i in range(7):
    pth = sheet_dir / f"c{i}.jpg"
    cv2.imwrite(str(pth), fake_crop(RGB["black"]))
    paths_[("v", "sn_0000", i)] = str(pth)
    items_.append((("v", "sn_0000", i), "black", "blue"))
ok_sheet = contact_sheet(items_, paths_, sheet_dir / "sheet.jpg", per_row=3)
im_ = cv2.imread(str(sheet_dir / "sheet.jpg"))
check("контактный лист: 7 кропов по 3 в ряд собираются в одну картинку", ok_sheet and im_ is not None and im_.shape[0] > 3 * 200)

# ------------------------------------------------- top_skip и разметка нескольких людей

hd = np.full((120, 80, 3), (90, 140, 90), np.uint8)
hd[0:30, 20:60] = (25, 25, 30)
hd[30:120, 12:68] = (170, 200, 220)
c0 = HistColorDescriber(wb_alpha=0, top_skip=0.0).describe(hd)
c1 = HistColorDescriber(wb_alpha=0, top_skip=0.25).describe(hd)
check("M1 top_skip: голова не мешает, уверенность растёт", c0["color"] == c1["color"] == "beige"
      and c1["confidence"] > c0["confidence"], f"{c0['confidence']} -> {c1['confidence']}")

g_dir = Path(tempfile.mkdtemp())
(g_dir / "gold_frames_aray.csv").write_text(
    "video_id,snapshot_id,true_people_count,labeler\nv,sn_0000,3,Aray\nv,sn_0001,,Aray\n", encoding="utf-8")
(g_dir / "gold_frames_saniya.csv").write_text(
    "video_id,snapshot_id,true_people_count,labeler\nv,sn_0000,4,Saniya\nv,sn_0001,,Saniya\n", encoding="utf-8")
fx = g_dir / "fi.json"
fx.write_text(json.dumps([{"video_id": "v", "snapshot_id": "sn_0000", "people_count": 4, "people_after_dedup": 4},
                          {"video_id": "v", "snapshot_id": "sn_0001", "people_count": 9, "people_after_dedup": 9}]))
r = run("eval.score_methods", "count", "--gold", str(g_dir / "gold_frames_*.csv"), "--frames-index", str(fx))
check("T33: маска gold_frames_*.csv, пустые строки пропущены, расхождение разметчиков показано",
      r.returncode == 0 and "Aray - 1 кадров" in r.stdout and "Saniya - 1 кадров" in r.stdout
      and "разошлись на 1 из 1" in r.stdout and "кадров: 1" in r.stdout, r.stdout[-500:] + r.stderr[-300:])

print(f"\nИТОГО: провалено {len(FAILS)}")
sys.exit(1 if FAILS else 0)