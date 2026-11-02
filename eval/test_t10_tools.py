"""Тесты функций T10 на SQLite, БЕЗ PostgreSQL:
get_track_timeline, get_event_details, list_zones_cameras, search_descriptions.
Семантический поиск проверяется на HashingEmbedder: это проверка механики (хранение, устаревание,
фильтры видимости), а НЕ качества BGE-M3."""
import csv, sys
from agent import tools
from database import embeddings
from eval import testutil

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if not cond:
        FAILS += 1
    print(("OK   " if cond else "FAIL ") + name + ((" | " + str(extra)) if not cond and extra != "" else ""))


def ids(res, key="events"):
    return [e["event_id"] for e in res[key]]


raw, conn = testutil.fresh()
Q = {r["id"]: r for r in csv.DictReader(open(testutil.eval_path("questions.csv"), encoding="utf-8-sig"))}

# ---------- get_track_timeline ----------
print("== get_track_timeline ==")
tl7 = tools.get_track_timeline(track_id=7, conn=conn)
check("человек 7: 6 событий по порядку", ids(tl7) == [f"e000{i}" for i in range(1, 7)], ids(tl7))
check("время по возрастанию", [e["t_start"] for e in tl7["events"]] == sorted(e["t_start"] for e in tl7["events"]))
check("video_id определён автоматически", tl7["video_id"] == "test_video_01" and tl7["truncated"] is False)
# вопросы из questions.csv, которым нужен get_track_timeline
for qid, (anchor, n) in {"q16": ("enter", 1), "q17": ("pick_up", 2), "q18": ("put_down", 1)}.items():
    q = Q[qid]
    tid = int(__import__("json").loads(q["expected_args"])["track_id"])
    evs = tools.get_track_timeline(track_id=tid, conn=conn)["events"]
    i = next(k for k, e in enumerate(evs) if e["event_type"] == anchor)
    got = ", затем ".join(f"{e['event_type']} ({e['t_start']} с)" for e in evs[i + 1:][:n])
    check(f"{qid} {q['expected_answer']}", q["expected_tool"] == "get_track_timeline" and got == q["expected_answer"], got)
q = Q["q10"]
evs = tools.get_track_timeline(track_id=9, conn=conn)["events"]
check("q10 список событий человека 9",
      q["expected_tool"] == "get_track_timeline"
      and "; ".join(f"{e['event_type']} ({e['t_start']} с)" for e in evs) == q["expected_answer"])
check("человек 12: отклонённое e0018 и непроверенное e0019 скрыты",
      ids(tools.get_track_timeline(track_id=12, conn=conn)) == ["e0016", "e0017"])
check("неизвестный человек -> пустой список, не ошибка",
      tools.get_track_timeline(track_id=99, conn=conn)["count"] == 0)
check("без track_id -> ошибка", tools.get_track_timeline(conn=conn)["ok"] is False)
check("track_id='abc' -> ошибка", tools.get_track_timeline(track_id="abc", conn=conn)["ok"] is False)
check("вариант B: у человека 8 нет put_down (B отклонил) и abandoned_object (нет результата B)",
      "put_down" not in [e["event_type"] for e in tools.get_track_timeline(track_id=8, vlm_variant="B", conn=conn)["events"]]
      and "abandoned_object" not in [e["event_type"] for e in tools.get_track_timeline(track_id=8, vlm_variant="B", conn=conn)["events"]])
raw.execute("INSERT INTO videos VALUES ('test_video_02','test_cam02','own')")
raw.execute("INSERT INTO events (video_id,event_id,event_type,track_id,t_start,t_end) "
            "VALUES ('test_video_02','e0001','enter',7,5,6)")
amb = tools.get_track_timeline(track_id=7, conn=conn)
check("track_id в двух видео без video_id -> ошибка с просьбой указать video_id",
      amb["ok"] is False and "video_id" in amb["error"], amb)
check("с video_id неоднозначности нет",
      tools.get_track_timeline(track_id=7, video_id="test_video_02", conn=conn)["count"] == 1)
raw.execute("DELETE FROM events WHERE video_id='test_video_02'")
raw.execute("DELETE FROM videos WHERE video_id='test_video_02'")

# ---------- get_event_details ----------
print("== get_event_details ==")
d = tools.get_event_details(event_id="e0004", conn=conn)
check("e0004 найдено, есть clip_path и описание",
      d["found"] and d["event"]["clip_path"] == "clips/test_video_01_e0004.mp4"
      and d["event"]["vlm_description"].startswith("A person") and d["event"]["vlm_status"] == "confirmed", d)
check("e0001 (needs_vlm=false) найдено без описания",
      tools.get_event_details(event_id="e0001", conn=conn)["event"]["vlm_description"] is None)
check("e0018 (rejected) -> found=False", tools.get_event_details(event_id="e0018", conn=conn)["found"] is False)
check("e0019 (not_checked) -> found=False", tools.get_event_details(event_id="e0019", conn=conn)["found"] is False)
check("несуществующее -> found=False, ok=True", tools.get_event_details(event_id="e9999", conn=conn)["found"] is False)
check("вариант B: e0010 отклонён -> не найдено",
      tools.get_event_details(event_id="e0010", vlm_variant="B", conn=conn)["found"] is False)
check("вариант B: e0004 описание из B",
      tools.get_event_details(event_id="e0004", vlm_variant="B", conn=conn)["event"]["vlm_description"].startswith("Person 7"))
check("без event_id -> ошибка", tools.get_event_details(conn=conn)["ok"] is False)
dd = tools.debug_get_event_details(event_id="e0018", conn=conn)
check("debug: e0018 найдено, показаны A и B (оба rejected)",
      dd["found"] and [(v["prompt_variant"], v["vlm_status"]) for v in dd["vlm_results"]] == [("A", "rejected"), ("B", "rejected")], dd)
check("debug: e0019 найдено, результатов VLM нет",
      tools.debug_get_event_details(event_id="e0019", conn=conn)["vlm_results"] == [])
check("debug: неизвестный view -> ошибка", tools.debug_get_event_details(event_id="e0004", view="x", conn=conn)["ok"] is False)
raw.execute("INSERT INTO videos VALUES ('test_video_02','test_cam02','own')")
raw.execute("INSERT INTO events (video_id,event_id,event_type,t_start,t_end) VALUES ('test_video_02','e0001','enter',1,2)")
amb = tools.get_event_details(event_id="e0001", conn=conn)
check("event_id в двух видео без video_id -> ошибка", amb["ok"] is False and "video_id" in amb["error"], amb)
check("с video_id находится",
      tools.get_event_details(event_id="e0001", video_id="test_video_02", conn=conn)["event"]["t_start"] == 1.0)

# ---------- list_zones_cameras ----------
print("== list_zones_cameras ==")
lz = tools.list_zones_cameras(conn=conn)
check("две камеры: test_cam01 и test_cam02", [c["camera_id"] for c in lz["cameras"]] == ["test_cam01", "test_cam02"], lz)
c1 = lz["cameras"][0]
check("у test_cam01 зоны desk и door и видео test_video_01",
      [z["zone_id"] for z in c1["zones"]] == ["desk", "door"] and c1["videos"] == [{"video_id": "test_video_01", "source": "meva"}], c1)
check("у test_cam02 нет зон", lz["cameras"][1]["zones"] == [])
check("фильтр по камере", tools.list_zones_cameras(camera_id="test_cam01", conn=conn)["count"] == 1)
check("неизвестная камера -> пусто, не ошибка",
      tools.list_zones_cameras(camera_id="nope", conn=conn) == {"ok": True, "count": 0, "cameras": []})
raw.execute("INSERT INTO zones VALUES ('cam_without_video','z1','Zone')")
check("камера, у которой есть только зоны, тоже в списке",
      "cam_without_video" in [c["camera_id"] for c in tools.list_zones_cameras(conn=conn)["cameras"]])
check("пустая строка camera_id -> ошибка", tools.list_zones_cameras(camera_id="", conn=conn)["ok"] is False)
raw.execute("DELETE FROM zones WHERE camera_id='cam_without_video'")
raw.execute("DELETE FROM events WHERE video_id='test_video_02'")
raw.execute("DELETE FROM videos WHERE video_id='test_video_02'")

# ---------- search_descriptions: по словам ----------
print("== search_descriptions: keyword ==")
kw = lambda q, **k: tools.search_descriptions(query=q, mode="keyword", conn=conn, **k)
r = kw("backpack")
check("'backpack' -> e0013", ids(r, "results") == ["e0013"], ids(r, "results"))
r = kw("desk bag")
check("'desk bag': e0004 первым (оба слова), score 1.0",
      ids(r, "results")[0] == "e0004" and r["results"][0]["score"] == 1.0, r["results"][:1])
check("у остальных score 0.5", all(x["score"] == 0.5 for x in r["results"][1:]))
check("отклонённое e0018 не находится ('reaches')", kw("reaches")["count"] == 0)
check("'person': e0004 и e0010 (e0018 скрыто)", set(ids(kw("person"), "results")) == {"e0004", "e0010"})
check("вариант B: 'Person 7' -> e0004", ids(kw("Person 7", vlm_variant="B"), "results")[:1] == ["e0004"])
check("вариант B: 'backpack' -> пусто (нет результата B)", kw("backpack", vlm_variant="B")["count"] == 0)
check("регистр не важен", ids(kw("BACKPACK"), "results") == ["e0013"])
check("фильтр event_type", ids(kw("desk", event_type="put_down"), "results") == ["e0010"])
check("top_k ограничивает", kw("desk", top_k=1)["count"] == 1)
check("русский запрос по английским описаниям ничего не находит (для этого есть semantic)",
      kw("рюкзак")["count"] == 0)
check("пустой запрос -> ошибка", tools.search_descriptions(query="  ", mode="keyword", conn=conn)["ok"] is False)
check("запрос без слов -> ошибка", tools.search_descriptions(query="?!", mode="keyword", conn=conn)["ok"] is False)
check("неизвестный mode -> ошибка", tools.search_descriptions(query="a", mode="fuzzy", conn=conn)["ok"] is False)
check("top_k=0 -> ошибка", tools.search_descriptions(query="desk", mode="keyword", top_k=0, conn=conn)["ok"] is False)
check("спецсимволы LIKE безопасны", kw("100%_")["count"] == 0)

# ---------- search_descriptions: по смыслу ----------
print("== search_descriptions: semantic (механика, HashingEmbedder) ==")
emb = embeddings.HashingEmbedder()
sem = lambda q, **k: tools.search_descriptions(query=q, mode="semantic", embedder=emb, conn=conn, **k)
r = sem("backpack on the desk")
check("до расчёта эмбеддингов: ничего не найдено, 3 видимых описания пропущено",
      r["ok"] and r["count"] == 0 and r["skipped_no_embedding"] == 3, r)
b = embeddings.build_embeddings(conn, emb)
check("посчитаны эмбеддинги 7 описаний (A: 4, B: 3)", b["ok"] and b["embedded"] == 7 and b["already_current"] == 0, b)
b2 = embeddings.build_embeddings(conn, emb)
check("повторный расчёт ничего не делает", b2["embedded"] == 0 and b2["already_current"] == 7, b2)
check("в таблице 7 строк, дубликатов нет",
      raw.execute("SELECT COUNT(*) FROM description_embeddings").fetchone()[0] == 7)
r = sem("backpack on the desk")
check("поиск: первым e0013", r["ok"] and ids(r, "results")[0] == "e0013" and r["skipped_no_embedding"] == 0, r)
check("скрытое e0018 не попадает в результат, хотя эмбеддинг у него есть",
      "e0018" not in ids(sem("reaches toward the desk no object picked up", top_k=10), "results"))
check("вариант B: видим только e0004",
      ids(sem("person desk bag", vlm_variant="B", top_k=10), "results") == ["e0004"])
check("min_score отсекает посторонний запрос",
      sem("zzz qqq www", min_score=0.3)["count"] == 0)
check("без min_score возвращается top_k даже для постороннего запроса",
      sem("zzz qqq www", top_k=2)["count"] == 2)
check("score в диапазоне [-1, 1]", all(-1.0 <= x["score"] <= 1.0001 for x in sem("desk", top_k=10)["results"]))
# устаревание
raw.execute("UPDATE vlm_results SET vlm_description='A red suitcase is left on the desk.' "
            "WHERE event_id='e0013' AND prompt_variant='A'")
r = sem("suitcase")
check("описание изменилось: устаревший эмбеддинг пропускается и учитывается",
      "e0013" not in ids(r, "results") and r["skipped_no_embedding"] == 1, r)
b3 = embeddings.build_embeddings(conn, emb)
check("пересчитывается только изменённое (1 описание)", b3["embedded"] == 1 and b3["already_current"] == 6, b3)
check("после пересчёта новое описание находится", ids(sem("suitcase"), "results")[:1] == ["e0013"])
# другая модель не смешивается
other = embeddings.HashingEmbedder(dim=32, name="other-model")
check("поиск другой моделью не видит эмбеддинги первой",
      tools.search_descriptions(query="desk", mode="semantic", embedder=other, conn=conn)["skipped_no_embedding"] == 3)
# каскад
raw.execute("DELETE FROM events WHERE video_id='test_video_01' AND event_id='e0004'")
check("удаление события удаляет его эмбеддинги (каскад)",
      raw.execute("SELECT COUNT(*) FROM description_embeddings WHERE event_id='e0004'").fetchone()[0] == 0)


class Broken:
    name = "broken"
    def embed(self, texts): raise RuntimeError("модель не загружена")

r = tools.search_descriptions(query="desk", mode="semantic", embedder=Broken(), conn=conn)
check("сбой модели -> ok=False с понятным текстом, не исключение", r["ok"] is False and "модель не загружена" in r["error"], r)

print(f"\nИТОГО провалов: {FAILS}")
sys.exit(1 if FAILS else 0)
