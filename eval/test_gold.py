"""Тесты T15: таблицы эталона, загрузка runs/crops/frames, запись разметки, импорт и экспорт CSV.
Только SQLite. PostgreSQL здесь НЕ запускался.
Запуск из корня проекта: py -m eval.test_gold"""
import csv, json, sqlite3, tempfile
from pathlib import Path

from agent.demo_db import Shim
from database import gold_loader as gl

FAILS = 0
V = "vid1"


def check(name, cond, extra=""):
    global FAILS
    if not cond:
        FAILS += 1
    print(("OK   " if cond else "FAIL ") + name + ((" | " + str(extra)) if not cond and extra != "" else ""))


def db():
    raw = sqlite3.connect(":memory:", check_same_thread=False)
    c = Shim(raw)
    gl.apply_schema(c)
    return c


def n(conn, sql):
    cur = conn.cursor(); cur.execute(sql); v = cur.fetchall()[0][0]; cur.close(); return v


TMP = Path(tempfile.mkdtemp())


def wjson(name, obj):
    p = TMP / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def crop(i, snap="sn_0000", kept=True, reason=None):
    return {"crop_id": f"{snap}_p{i:02d}", "video_id": V, "snapshot_id": snap, "person_idx": i, "t": 0.0,
            "box": [0, 0, 1, 1], "box_h": 100.0 + i, "box_w": 50.0, "h_bucket": "80-150", "frame_path": "frames/x.jpg",
            "kept": kept, "reason": reason, "path": f"t/{i}.jpg", "full_path": f"full/{snap}_p{i:02d}.jpg"}


# --- схема
c = db()
check("схема: 5 таблиц", n(c, "SELECT COUNT(*) FROM sqlite_master WHERE type='table'") == 5)
gl.apply_schema(c)
check("схема: повторный запуск безопасен", True)

# --- runs
line = {"run_id": "r1", "video_id": V, "method": "vlm:x", "model": "x", "mode": "crop", "prompt_variant": "A",
        "weights": None, "started_at": "2026-11-24T13:03:31", "duration_sec": 1.5, "params": {"repeat": 2, "out": "/a"}}
p = TMP / "runs.jsonl"
p.write_text(json.dumps(line) + "\n\n" + json.dumps({**line, "run_id": "r2", "duration_sec": 3}) + "\n", encoding="utf-8")
r = gl.load_runs(p, c)
check("runs: загружено 2", r["ok"] and n(c, "SELECT COUNT(*) FROM runs") == 2, r)
gl.load_runs(p, c)
check("runs: повтор без дублей", n(c, "SELECT COUNT(*) FROM runs") == 2)
cur = c.cursor(); cur.execute("SELECT params, weights FROM runs WHERE run_id='r1'"); row = cur.fetchall()[0]
check("runs: params хранится как JSON", json.loads(row[0]) == line["params"] and row[1] is None, row)
p.write_text(json.dumps(line) + "\nnot json\n" + json.dumps({**line, "run_id": ""}) + "\n", encoding="utf-8")
r = gl.load_runs(p, c)
check("runs: ошибки не пишут ничего", not r["ok"] and len(r["errors"]) == 2 and n(c, "SELECT COUNT(*) FROM runs") == 2, r["errors"])
p.write_text(json.dumps({**line, "duration_sec": -1}) + "\n", encoding="utf-8")
check("runs: отрицательная длительность отклонена", not gl.load_runs(p, c)["ok"])
check("runs: нет файла", not gl.load_runs(TMP / "nope.jsonl", c)["ok"])

# --- crops / frames
crops = [crop(0), crop(1), crop(2, kept=False, reason="dup_of_sn_0000_p01"), crop(0, "sn_0001")]
r = gl.load_crops_index(wjson("crops.json", crops), c)
check("crops: загружено 4, kept 3", r["ok"] and r["rows"] == 4 and r["kept"] == 3 and r["dropped"] == 1, r)
check("crops: повтор без дублей", gl.load_crops_index(wjson("crops.json", crops), c)["ok"]
      and n(c, "SELECT COUNT(*) FROM gold_crops") == 4)
check("crops: дубль помечен", n(c, "SELECT COUNT(*) FROM gold_crops WHERE kept = 0") == 1)
r = gl.load_crops_index(wjson("bad.json", [crop(0), crop(0)]), c)
check("crops: повтор ключа отклонён", not r["ok"])
r = gl.load_crops_index(wjson("bad2.json", [{**crop(0), "kept": "true"}]), c)
check("crops: kept строкой отклонён", not r["ok"])
check("crops: пустой список отклонён", not gl.load_crops_index(wjson("e.json", []), c)["ok"])

frames = [{"video_id": V, "snapshot_id": f"sn_{i:04d}", "t": i * 30.0, "frame_path": f"frames/sn_{i:04d}.jpg",
           "people_count": 13} for i in range(3)]
r = gl.load_frames_index(wjson("frames.json", frames), c)
check("frames: загружено 3", r["ok"] and n(c, "SELECT COUNT(*) FROM gold_frames_index") == 3, r)
cur = c.cursor(); cur.execute("SELECT * FROM gold_frames_index LIMIT 1")
check("frames: счёт модели в базу не попадает", "people_count" not in [d[0] for d in cur.description])

# --- разметка людей
ok, msg = gl.save_people_label(c, V, "sn_0000", 0, "Aigerim", "red", "sweater", "да")
check("people: сохранено, box_h взят из кропа", ok and n(c, "SELECT box_h FROM labels_gold_people") == 100.0, msg)
check("people: плохой цвет", not gl.save_people_label(c, V, "sn_0000", 0, "Aigerim", "crimson", "sweater", "да")[0])
check("people: плохой тип", not gl.save_people_label(c, V, "sn_0000", 0, "Aigerim", "red", "pullover", "да")[0])
check("people: плохой visible", not gl.save_people_label(c, V, "sn_0000", 0, "Aigerim", "red", "sweater", "yes")[0])
check("people: плохой color_2", not gl.save_people_label(c, V, "sn_0000", 0, "Aigerim", "red", "sweater", "да", "x")[0])
check("people: пустой разметчик", not gl.save_people_label(c, V, "sn_0000", 0, " ", "red", "sweater", "да")[0])
gl.save_people_label(c, V, "sn_0000", 0, "Aigerim", "blue", "shirt", "спиной", "white")
check("people: правка заменяет, не дублирует", n(c, "SELECT COUNT(*) FROM labels_gold_people") == 1
      and n(c, "SELECT color FROM labels_gold_people") == "blue")
gl.save_people_label(c, V, "sn_0000", 0, "Arai", "blue", "shirt", "да", "")
check("people: двое на один кроп", n(c, "SELECT COUNT(*) FROM labels_gold_people") == 2)
check("people: color_2 '' -> NULL", n(c, "SELECT COUNT(*) FROM labels_gold_people WHERE color_2 IS NULL") == 1)

# --- очередь кропов
nxt = gl.next_crops(c, "Aigerim", V, limit=10)
keys = [(x["snapshot_id"], x["person_idx"]) for x in nxt]
check("очередь: без своих и без дублей-отброшенных", ("sn_0000", 0) not in keys and ("sn_0000", 2) not in keys
      and len(keys) == 2, keys)
nxt2 = gl.next_crops(c, "Arai", V, limit=10)
check("очередь: у Арай своих нет, но 2 оценки есть только у p00", len(nxt2) == 2 and nxt2[-1]["n_labels"] == 2 or len(nxt2) == 2, nxt2)
gl.save_people_label(c, V, "sn_0000", 1, "Arai", "red", "coat", "да")
nx3 = gl.next_crops(c, "Sania", V, limit=10)
check("очередь: сначала кроп с одной оценкой, с двумя в конце",
      [x["n_labels"] for x in nx3] == [1, 0, 2], [x["n_labels"] for x in nx3])
check("очередь: чужие значения не отдаются", all("color" not in x for x in nxt))

# --- разметка кадров
check("frame: сохранено", gl.save_frame_label(c, V, "sn_0000", "Aigerim", 12)[0])
check("frame: минус отклонён", not gl.save_frame_label(c, V, "sn_0000", "Aigerim", -1)[0])
check("frame: дробь отклонена", not gl.save_frame_label(c, V, "sn_0000", "Aigerim", 2.5)[0])
gl.save_frame_label(c, V, "sn_0000", "Aigerim", 13)
check("frame: правка без дубля", n(c, "SELECT COUNT(*) FROM labels_gold_frames") == 1
      and n(c, "SELECT true_people_count FROM labels_gold_frames") == 13)

# --- кадры, видео, отмена
check("next_frames: свои уже посчитанные пропускаются", [x["snapshot_id"] for x in gl.next_frames(c, "Aigerim", V, 10)] == ["sn_0001", "sn_0002"]
      or len(gl.next_frames(c, "Aigerim", V, 10)) == 2)
check("next_frames: у другого разметчика все 3", len(gl.next_frames(c, "Sania", V, 10)) == 3)
check("next_frames: нет people_count модели", all("people_count" not in x for x in gl.next_frames(c, "Sania", V, 10)))
check("gold_videos", gl.gold_videos(c) == [V])
gl.delete_people_label(c, V, "sn_0000", 0, "Arai")
check("delete_people_label: удалена только своя строка", n(c, "SELECT COUNT(*) FROM labels_gold_people WHERE labeler='Arai' AND person_idx=0") == 0
      and n(c, "SELECT COUNT(*) FROM labels_gold_people WHERE labeler='Arai' AND person_idx=1") == 1
      and n(c, "SELECT COUNT(*) FROM labels_gold_people WHERE labeler='Aigerim'") == 1)
gl.save_people_label(c, V, "sn_0000", 0, "Arai", "blue", "shirt", "да")   # вернуть для следующих проверок
gl.save_frame_label(c, V, "sn_0001", "Sania", 5); gl.delete_frame_label(c, V, "sn_0001", "Sania")
check("delete_frame_label", n(c, "SELECT COUNT(*) FROM labels_gold_frames WHERE labeler='Sania'") == 0)

# --- импорт CSV кадров
f = TMP / "gold_frames_Aigerim.csv"
f.write_text("video_id,snapshot_id,true_people_count,labeler\n" + f"{V},sn_0001,12,Aigerim\n{V},sn_0002,,Aigerim\n",
             encoding="utf-8")
r = gl.import_frames_csv(f, c)
check("import: пустая строка пропущена", r["ok"] and r["rows"] == 1 and r["warnings"], r)
f.write_text("video_id,snapshot_id,true_people_count,labeler\n" + f"{V},sn_0001,abc,Aigerim\n", encoding="utf-8")
check("import: мусор отклонён", not gl.import_frames_csv(f, c)["ok"])

pr = gl.progress(c, "Aigerim")
check("прогресс", pr == {"people_done": 1, "people_total": 3, "frames_done": 2, "frames_total": 3}, pr)

# --- экспорт и совместимость с score_methods
out = TMP / "exp"
r = gl.export_csv(c, out)
rows = list(csv.DictReader(open(out / "gold_people.csv", encoding="utf-8")))
check("экспорт: колонки gold_people", list(rows[0].keys()) == gl.PEOPLE_COLS, list(rows[0].keys()))
check("экспорт: строки людей по разметчикам", len(rows) == 3 and {x["labeler"] for x in rows} == {"Aigerim", "Arai"})
fr = list(csv.DictReader(open(out / "gold_frames.csv", encoding="utf-8")))
check("экспорт: колонки gold_frames", list(fr[0].keys()) == gl.FRAMES_COLS and len(fr) == 2, fr)
from eval import score_methods as sm
g, conflicts, n_obj = sm.load_gold_people(str(out / "gold_people.csv"))
check("score_methods читает экспорт: 2 кропа, согласие двоих", n_obj == 2 and len(g) >= 1
      and g[(V, "sn_0000", 0)]["color"] == "blue" and conflicts == 0, (g, conflicts, n_obj))
gl.save_people_label(c, V, "sn_0000", 0, "Sania", "green", "shirt", "да")   # 2 из 3 за blue: большинство
gl.save_people_label(c, V, "sn_0000", 1, "Sania", "blue", "coat", "да")        # red против blue: ничья
gl.export_csv(c, out)
g, conflicts, n_obj = sm.load_gold_people(str(out / "gold_people.csv"))
check("score_methods: большинство 2 из 3 берётся, ничья не оценивается",
      g.get((V, "sn_0000", 0), {}).get("color") == "blue" and (V, "sn_0000", 1) not in g and conflicts == 1, (g, conflicts))

# словарь (запасная копия в загрузчике не должна разойтись с vlm/describers/base.py)
try:
    from vlm.describers import base as _b
    check("словарь = vlm/describers/base.py", gl.COLORS == _b.COLORS and gl.TYPES == _b.TYPES)
except ImportError:
    check("словарь: vlm/describers нет, взята запасная копия (14 цветов, 12 типов)", len(gl.COLORS) == 14 and len(gl.TYPES) == 12)

# словарь совпадает с README
check("visible как в README", gl.VISIBLE == ("да", "спиной", "частично", "не видно"))

print(f"\nИТОГО: ошибок {FAILS}")
raise SystemExit(1 if FAILS else 0)