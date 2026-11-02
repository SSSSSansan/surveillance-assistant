"""Генерирует database/seed_events.json, seed_vlm_A.json, seed_vlm_B.json, seed.sql и eval/questions.csv.
Запуск из корня проекта: python -m eval.build
Эталонные ответы считаются из тех же придуманных событий по правилу видимости
(вариант подсказки A), независимо от tools.py. Данные придуманные, не MEVA."""
import csv, json
from eval.testutil import DB_DIR, EVAL_DIR

VIDEO, CAM, SRC = "test_video_01", "test_cam01", "meva"
VLM_TYPES = ("pick_up", "put_down", "abandoned_object")  # события 6-8 идут в VLM

# (type, track, zone, t_start, t_end, trigger_score)
RAW = [
 ("enter", 7, "door", 12.4, 14.0, .87),
 ("approach_zone", 7, "desk", 15.0, 18.5, .81),
 ("dwell_in_zone", 7, "desk", 18.5, 35.0, .90),
 ("pick_up", 7, "desk", 24.0, 26.0, .64),
 ("leave_zone", 7, "desk", 35.0, 36.2, .78),
 ("exit", 7, "door", 41.2, 43.0, .85),
 ("enter", 8, "door", 60.1, 61.5, .91),
 ("approach_zone", 8, "desk", 63.0, 66.0, .79),
 ("dwell_in_zone", 8, "desk", 66.0, 74.0, .88),
 ("put_down", 8, "desk", 70.5, 72.0, .58),
 ("leave_zone", 8, "desk", 74.0, 75.2, .76),
 ("exit", 8, "door", 78.0, 80.0, .83),
 ("abandoned_object", 8, "desk", 95.0, 120.0, .71),
 ("enter", 9, "door", 150.2, 151.8, .89),
 ("exit", 9, "door", 155.0, 156.4, .84),
 ("enter", 12, "door", 200.0, 201.5, .86),
 ("approach_zone", 12, "desk", 205.0, 208.0, .80),
 ("pick_up", 12, "desk", 206.0, 207.5, .55),
 ("put_down", 12, "desk", 209.0, 210.5, .52),
]

# Результаты VLM: variant -> event_id -> (status, description)
VLM = {
 "A": {
  "e0004": ("confirmed", "A person standing at the desk picks up a small bag and holds it."),
  "e0010": ("confirmed", "The person leans over the desk and places a laptop on it."),
  "e0013": ("confirmed", "A backpack stays on the desk with nobody nearby."),
  "e0018": ("rejected", "The person reaches toward the desk but no object is picked up."),
 },
 "B": {
  "e0004": ("confirmed", "Person 7 in the desk zone picks up a small bag at about 24 s."),
  "e0010": ("rejected", "Person 8 is in the desk zone, but no object is visibly put down."),
  "e0018": ("rejected", "Person 12 is in the desk zone, no pick-up is visible."),
 },
}
# e0019 (put_down, track 12): нет результатов VLM ни в A, ни в B (not_checked).

E = []
for i, (typ, trk, zone, a, b, sc) in enumerate(RAW, 1):
    eid = f"e{i:04d}"
    nv = typ in VLM_TYPES
    E.append({"event_id": eid, "event_type": typ, "track_id": trk, "zone_id": zone,
              "t_start": a, "t_end": b, "trigger_score": sc, "needs_vlm": nv,
              "clip_path": f"clips/{VIDEO}_{eid}.mp4" if nv else None})


def ev_obj(e, variant=None):
    st, desc = ("not_checked", None)
    if variant:
        st, desc = VLM[variant][e["event_id"]]
    return {"event_id": e["event_id"], "event_type": e["event_type"], "track_id": e["track_id"],
            "zone_id": e["zone_id"], "t_start": e["t_start"], "t_end": e["t_end"],
            "trigger_score": e["trigger_score"], "needs_vlm": e["needs_vlm"],
            "clip_path": e["clip_path"], "vlm_status": st, "vlm_description": desc,
            "prompt_variant": variant}


def dump(name, events):
    json.dump({"video_id": VIDEO, "camera_id": CAM, "source": SRC, "events": events},
              open(name, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

dump(DB_DIR / "seed_events.json", [ev_obj(e) for e in E])                      # файл Арай
for v in ("A", "B"):                                                   # файлы Панды
    dump(DB_DIR / f"seed_vlm_{v}.json", [ev_obj(e, v) for e in E if e["event_id"] in VLM[v]])


def sql_val(v):
    if v is None: return "NULL"
    if isinstance(v, bool): return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)): return repr(v)
    return "'" + v.replace("'", "''") + "'"


def ins(table, cols, row, suffix=""):
    return f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(sql_val(row[c]) for c in cols)}){suffix};\n"

with open(DB_DIR / "seed.sql", "w", encoding="utf-8") as f:
    f.write("-- Придуманные события для T5 (19 штук). Не настоящие данные MEVA.\n")
    f.write(f"DELETE FROM vlm_results WHERE video_id = '{VIDEO}';\n")
    f.write(f"DELETE FROM events WHERE video_id = '{VIDEO}';\n")
    f.write(f"DELETE FROM videos WHERE video_id = '{VIDEO}';\n")
    f.write("-- Зоны не удаляются. Добавляются, если их нет; у уже существующей зоны\n"
            "-- описание заполняется только когда оно пустое, чужие данные не затираются.\n")
    f.write(ins("videos", ["video_id", "camera_id", "source"],
                {"video_id": VIDEO, "camera_id": CAM, "source": SRC}))
    for z, d in (("door", "Entrance door"), ("desk", "Work desk")):
        f.write(ins("zones", ["camera_id", "zone_id", "description"],
                    {"camera_id": CAM, "zone_id": z, "description": d},
                    suffix=" ON CONFLICT (camera_id, zone_id) DO UPDATE "
                           "SET description = COALESCE(zones.description, excluded.description)"))
    ecols = ["video_id", "event_id", "event_type", "track_id", "zone_id", "t_start", "t_end",
             "trigger_score", "needs_vlm", "clip_path"]
    for e in E:
        f.write(ins("events", ecols, dict(e, video_id=VIDEO)))
    for v in ("A", "B"):
        for eid, (st, desc) in VLM[v].items():
            f.write(ins("vlm_results",
                        ["video_id", "event_id", "prompt_variant", "vlm_status", "vlm_description"],
                        {"video_id": VIDEO, "event_id": eid, "prompt_variant": v,
                         "vlm_status": st, "vlm_description": desc}))


# ---------- эталонные ответы ----------
def visible(e):
    """Правило ассистента для варианта A."""
    return (not e["needs_vlm"]) or VLM["A"].get(e["event_id"], ("not_checked",))[0] == "confirmed"


def flt(event_type=None, zone_id=None, track_id=None, t_from=None, t_to=None):
    out = []
    for e in E:
        if not visible(e): continue
        if event_type and e["event_type"] != event_type: continue
        if zone_id and e["zone_id"] != zone_id: continue
        if track_id is not None and e["track_id"] != track_id: continue
        if t_from is not None and e["t_end"] < t_from: continue
        if t_to is not None and e["t_start"] > t_to: continue
        out.append(e)
    return sorted(out, key=lambda e: (e["t_start"], e["event_id"]))

def yn(**kw): return "да" if flt(**kw) else "нет"
def n(**kw): return str(len(flt(**kw)))
def first(**kw): return f"{flt(**kw)[0]['t_start']} с"
def last(**kw): return f"{flt(**kw)[-1]['t_start']} с"
def after(track, etype, how_many):
    tl = flt(track_id=track)
    i = next(k for k, e in enumerate(tl) if e["event_type"] == etype)
    return ", затем ".join(f"{e['event_type']} ({e['t_start']} с)" for e in tl[i + 1:][:how_many])

Q = []  # (type, lang, negative, answerable, question, answer, tool, args)
def add(t, lang, q, ans, tool, args, neg=False, ok=True):
    Q.append((t, lang, neg, ok, q, ans, tool, json.dumps(args, ensure_ascii=False)))

# --- 18 обычных вопросов ---
add("list","ru","Перечисли все входы.", "; ".join(f"человек {e['track_id']} ({e['t_start']} с)" for e in flt(event_type="enter")), "query_events", {"event_type":"enter"})
add("exists","en","Did anyone pick up an object?", yn(event_type="pick_up"), "query_events", {"event_type":"pick_up"})
add("exists","ru","Был ли оставленный предмет?", yn(event_type="abandoned_object"), "query_events", {"event_type":"abandoned_object"})
add("count","ru","Сколько раз кто-то входил?", n(event_type="enter"), "count_events", {"event_type":"enter"})
add("count","ru","Сколько было выходов?", n(event_type="exit"), "count_events", {"event_type":"exit"})
add("count","ru","Сколько всего событий произошло у стола?", n(zone_id="desk"), "count_events", {"zone_id":"desk"})
add("count","en","How many events did person 7 have in total?", n(track_id=7), "count_events", {"track_id":7})
add("count","ru","Сколько раз кто-то подходил к столу?", n(event_type="approach_zone", zone_id="desk"), "count_events", {"event_type":"approach_zone","zone_id":"desk"})
add("count","ru","Сколько раз кто-то входил в первые 100 секунд?", n(event_type="enter", t_to=100), "count_events", {"event_type":"enter","t_from":0,"t_to":100})
add("list","en","List all events of person 9.", "; ".join(f"{e['event_type']} ({e['t_start']} с)" for e in flt(track_id=9)), "get_track_timeline", {"track_id":9})
add("time","ru","Когда было первое событие?", first(), "query_events", {"order":"asc","limit":1})
add("time","ru","Когда человек 8 вошёл?", first(event_type="enter", track_id=8), "query_events", {"event_type":"enter","track_id":8})
add("time","ru","Когда был последний выход?", last(event_type="exit"), "query_events", {"event_type":"exit","order":"desc","limit":1})
add("time","en","When was the object picked up?", first(event_type="pick_up"), "query_events", {"event_type":"pick_up"})
add("time","ru","Когда был оставлен предмет?", first(event_type="abandoned_object"), "query_events", {"event_type":"abandoned_object"})
add("sequence","ru","Что было сразу после первого входа?", after(7,"enter",1), "get_track_timeline", {"track_id":7})
add("sequence","ru","Что делал человек 7 после того, как взял предмет?", after(7,"pick_up",2), "get_track_timeline", {"track_id":7})
add("sequence","en","What happened right after person 8 put the object down?", after(8,"put_down",1), "get_track_timeline", {"track_id":8})

# --- 9 отрицательных (правильный ответ «нет» или «не нашёл») ---
add("exists","ru","Выходил ли человек 12?", yn(event_type="exit", track_id=12), "query_events", {"event_type":"exit","track_id":12}, neg=True)
add("exists","ru","Брал ли что-нибудь человек 12?", yn(event_type="pick_up", track_id=12), "query_events", {"event_type":"pick_up","track_id":12}, neg=True)
add("exists","ru","Подходил ли человек 9 к столу?", yn(event_type="approach_zone", track_id=9, zone_id="desk"), "query_events", {"event_type":"approach_zone","track_id":9,"zone_id":"desk"}, neg=True)
add("exists","en","Did anyone enter between 20 and 55 seconds?", yn(event_type="enter", t_from=20, t_to=55), "query_events", {"event_type":"enter","t_from":20,"t_to":55}, neg=True)
add("exists","ru","Был ли оставленный предмет до 90-й секунды?", yn(event_type="abandoned_object", t_to=90), "query_events", {"event_type":"abandoned_object","t_to":90}, neg=True)
add("exists","en","Did anyone put something down before 60 seconds?", yn(event_type="put_down", t_to=60), "query_events", {"event_type":"put_down","t_to":60}, neg=True)
add("exists","ru","Были ли события с человеком 99?", yn(track_id=99), "query_events", {"track_id":99}, neg=True)
add("time","ru","Когда человек 12 вышел?", "не нашёл" if not flt(event_type="exit", track_id=12) else first(event_type="exit", track_id=12), "query_events", {"event_type":"exit","track_id":12}, neg=True)
add("exists","en","Was there an exit after 160 seconds?", yn(event_type="exit", t_from=160), "query_events", {"event_type":"exit","t_from":160}, neg=True)

# --- 3 вопроса, на которые по базе ответить нельзя ---
NO = "ответить нельзя: в базе нет этих данных"
add("out_of_scope","ru","Какого цвета куртка у человека 7?", NO, "none", {}, ok=False)
add("out_of_scope","ru","Сколько входов было на камере cam02?", NO, "none", {}, ok=False)
add("out_of_scope","en","Why did person 8 leave the object on the desk?", NO, "none", {}, ok=False)

assert len(Q) == 30 and sum(q[2] for q in Q) >= 8
for q in Q:
    if q[2]: assert q[5] in ("нет", "не нашёл"), q

with open(EVAL_DIR / "questions.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id","type","lang","is_negative","answerable","question","expected_answer","expected_tool","expected_args"])
    for i, q in enumerate(Q, 1):
        w.writerow([f"q{i:02d}", q[0], q[1], "да" if q[2] else "нет", "да" if q[3] else "нет", q[4], q[5], q[6], q[7]])
print("events:", len(E), "visible(A):", sum(visible(e) for e in E),
      "questions:", len(Q), "negatives:", sum(q[2] for q in Q))
