"""Stage-3 tests (pytest): answer parsing of all modes, metrics on synthetic data, kappa, fake-server runs of the
describers and both benchmarks, agreement, server scripts syntax. Run from the repo root:
    python -m pytest eval/test_stage3_vlm.py -q
Everything here runs against a FAKE server: it proves our code, NOT that a real model answers well."""
import csv
import json
import math
import subprocess
from pathlib import Path

import pytest

import eval.bench_common as bc  # noqa: F401  (puts repo root and vlm/ on sys.path)
from eval import agreement, bench_clothing, bench_count, metrics as M
from eval.fake_vlm_server import FakeVlmServer
from vlm.describers import vlm_api
from vlm.describers import base
from vlm.describers.vlm_api import VlmApiDescriber

import describe_snapshot as ds
import prompts

ROOT = Path(__file__).resolve().parents[1]
IMG = "data:image/jpeg;base64,AA=="
CONTRACT = ("color", "color_2", "type", "confidence", "description", "method", "mode", "prompt_variant")


def validate_record(rec: dict) -> list:
    """Problems of one answer against Arai's contract (base.make_answer keys, base.COLORS / base.TYPES)."""
    problems = [f"missing key {k}" for k in CONTRACT if k not in rec]
    if problems:
        return problems
    if rec["color"] not in base.COLORS:
        problems.append(f"color {rec['color']!r} not in vocabulary")
    if rec["color_2"] is not None and rec["color_2"] not in base.COLORS:
        problems.append(f"color_2 {rec['color_2']!r} not in vocabulary")
    if rec["type"] not in base.TYPES:
        problems.append(f"type {rec['type']!r} not in vocabulary")
    c = rec["confidence"]
    if c is not None and not (isinstance(c, (int, float)) and not isinstance(c, bool) and 0 <= c <= 1):
        problems.append(f"confidence {c!r} not in 0..1")
    return problems


# ------------------------------------------------------------- parsing ---
def test_strip_thinking():
    assert vlm_api.strip_thinking('<think>a {"x":1}</think>\n{"people_count": 2}') == '{"people_count": 2}'
    assert vlm_api.strip_thinking('reasoning...</think>{"a": 1}') == '{"a": 1}'
    assert vlm_api.strip_thinking("plain") == "plain"


@pytest.mark.parametrize("text,expected", [
    ('{"people_count": 5}', 5), ('```json\n{"people_count": "7"}\n```', 7), ('{"people_count": 4.0}', 4),
    ('{"people_count": 0}', 0), ('Sure: {"people_count": 3} done', 3),
    ('{"people_count": -1}', None), ('{"people_count": 2.5}', None), ('{"people_count": "many"}', None),
    ('{"people_count": true}', None), ('{"count": 3}', None), ('no json', None), ('', None),
])
def test_parse_count(text, expected):
    assert vlm_api.parse_count(text)[0] == expected


def test_parse_crop_d():
    p, w = vlm_api.parse_crop_d('{"clothing_color":"Navy","clothing_color_2":"black","clothing_type":"puffer",'
                                '"confidence":0.8,"description":"d"}')
    assert (p["clothing_color"], p["clothing_color_2"], p["clothing_type"], p["confidence"]) == ("blue", "black", "jacket", 0.8)
    p, _ = vlm_api.parse_crop_d('{"clothing_color":"red","clothing_color_2":null,"clothing_type":"coat","confidence":85}')
    assert p["clothing_color_2"] is None and p["confidence"] == 0.85
    p, w = vlm_api.parse_crop_d('{"clothing_color":"red","clothing_color_2":"burgundy","clothing_type":"coat","confidence":"high"}')
    assert p["clothing_color_2"] is None and p["confidence"] is None  # burgundy->red == color; text confidence
    assert vlm_api.parse_crop_d("nothing")[0] is None


def test_parse_boxes():
    txt = json.dumps({"people_count": 4, "people": [
        {"box_idx": 1, "clothing_color": "red", "clothing_type": "coat", "description": "a"},
        {"box_idx": 0, "clothing_color": "gray", "clothing_type": "jacket", "description": "b"},
        {"box_idx": 1, "clothing_color": "blue", "clothing_type": "coat", "description": "dup"},
        {"box_idx": 9, "clothing_color": "blue", "clothing_type": "coat", "description": "out"}]})
    v, w = vlm_api.parse_boxes(txt, 3)
    assert v["people_count"] == 4 and [p["person_idx"] for p in v["people"]] == [0, 1]
    assert len(w) == 2  # duplicate + outside range
    assert vlm_api.parse_boxes('{"people_count": 2}', 2)[0] is None  # no people list
    v, _ = vlm_api.parse_boxes('{"people": []}', 0)
    assert v["people_count"] is None and v["people"] == []


def test_prompt_a_is_stage2_prompt_and_vocab_untouched():
    assert prompts.PROMPT_A == ds.CROP_PROMPT and prompts.PROMPT_R3 == ds.FRAME_PROMPT
    assert len(ds.COLORS) == 14 and len(ds.TYPES) == 12 and "unknown" in ds.COLORS
    assert "[10, 20, 30, 40]" in prompts.prompt_r4([[10, 20, 30, 40]], 1920, 1080) and "1920x1080" in prompts.prompt_r4([[1, 2, 3, 4]], 1920, 1080)


# ------------------------------------------------------------- metrics ---
def test_accuracy_f1():
    g, p = ["a", "a", "b", "b"], ["a", "b", "b", None]
    assert M.accuracy(g, p) == 0.5
    f = M.f1_per_class(g, p)
    assert math.isclose(f["a"], 2 / 3) and math.isclose(f["b"], 0.5)
    assert math.isclose(M.macro_f1(g, p), (2 / 3 + 0.5) / 2)


def test_top2_and_confusion():
    assert M.top2_accuracy(["a", "b", "c"], ["a", "x", "x"], [None, "b", None]) == 2 / 3
    labels, m = M.confusion(["a", "a", "b"], ["a", None, "a"])
    assert m["a"]["a"] == 1 and m["a"]["__none__"] == 1 and m["b"]["a"] == 1 and "__none__" in labels


def test_kappa():
    assert math.isclose(M.cohen_kappa(list("xxyy"), list("xyyy")), 0.5)
    assert M.cohen_kappa(list("xyxy"), list("xyxy")) == 1.0
    assert M.cohen_kappa(list("xx"), list("xx")) == 1.0 and M.cohen_kappa(list("xx"), list("xy")) < 1  # pe==1 edge
    assert M.cohen_kappa(list("xx"), list("yy")) == -0.0 or M.cohen_kappa(list("xx"), list("yy")) <= 0
    assert math.isnan(M.cohen_kappa([], []))


def test_count_metrics_and_buckets():
    m = M.count_metrics([5, 5, 5, 5], [5, 3, 7, None])
    assert m["n_failed"] == 1 and math.isclose(m["mae"], 4 / 3) and math.isclose(m["exact"], 1 / 3)
    assert math.isclose(m["under"], 1 / 3) and math.isclose(m["over"], 1 / 3) and m["bias"] == 0
    assert [M.bucket_of(x) for x in (79, 80, 150, 151, None)] == ["<80", "80-150", "80-150", ">150", "unknown_h"]


def test_majority_ties():
    assert bc.majority(["a", "a", "b"]) == "a" and bc.majority(["a", "b"]) is None and bc.majority(["", None]) is None


# ------------------------------------------------- describers + fake server ---
def _handler(p):
    t = p["messages"][-1]["content"][-1]["text"]
    if "box_idx" in t and "people_count" in t:
        return json.dumps({"people_count": 2, "people": [
            {"box_idx": 0, "clothing_color": "red", "clothing_type": "coat", "description": "a"}]})
    if "How many people" in t:
        return '<think>x</think>{"people_count": 3}'
    if "List EVERY person" in t:
        return '{"people":[{"clothing_color":"red","clothing_type":"coat","description":"d"},' \
               '{"clothing_color":"blue","clothing_type":"hoodie","description":"e"}]}'
    if "Convert it" in t:
        return '{"clothing_color":"gray","clothing_type":"jacket","description":"z"}'
    if "In 1-3 plain" in t:
        return "A gray jacket, back view."
    if "clothing_color_2" in t:
        return '{"clothing_color":"red","clothing_color_2":"brown","clothing_type":"sweater","confidence":0.7,"description":"r"}'
    return '{"clothing_color":"red","clothing_type":"sweater","description":"r"}'


def test_crop_variants_and_contract():
    with FakeVlmServer(_handler) as s:
        for v, expect_tokens in (("A", 100), ("B", 200), ("D", 100)):
            r = VlmApiDescriber("m", s.url, "R1", v).describe_crop(IMG, 4)
            assert r["status"] == "ok" and validate_record(r["record"]) == [] and r["prompt_tokens"] == expect_tokens
            assert r["record"]["method"] == "vlm:m" and r["record"]["prompt_variant"] == v and r["record"]["mode"] == "crop"
        d = VlmApiDescriber("m", s.url, "R1", "D").describe_crop(IMG)["record"]
        assert d["color_2"] == "brown" and d["confidence"] == 0.7
        assert all(req["temperature"] == 0 for req in s.requests)
        # B: step 2 carries no image and contains the step-1 text
        step2 = [r for r in s.requests if "Convert it" in r["messages"][-1]["content"][-1]["text"]][0]
        assert len(step2["messages"][-1]["content"]) == 1 and "A gray jacket, back view." in step2["messages"][-1]["content"][0]["text"]


def test_frame_modes():
    frame = next((ROOT / "demo_frames").rglob("*.jpg"), None)
    assert frame is not None
    with FakeVlmServer(_handler) as s:
        r2 = VlmApiDescriber("m", s.url, "R2").describe_frame(str(frame))
        assert r2["people_count"] == 3 and r2["records"] == []
        r3 = VlmApiDescriber("m", s.url, "frame-list").describe_frame(str(frame))
        assert r3["people_count"] == 2 and len(r3["records"]) == 2 and r3["records"][0]["mode"] == "frame-list"
        r4 = VlmApiDescriber("m", s.url, "R4").describe_frame(str(frame), [[1, 2, 50, 90], [3, 4, 60, 100], [5, 6, 70, 110]])
        assert r4["people_count"] == 2 and len(r4["records"]) == 1  # model's own count, not 3 boxes
        assert "[1, 2, 50, 90]" in s.requests[-1]["messages"][-1]["content"][-1]["text"]
    assert VlmApiDescriber("m", "http://x", "R2").describe_frame("/no/such.jpg")["status"] == "error"


def test_bad_arguments():
    with pytest.raises(ValueError):
        VlmApiDescriber("m", "u", "R9")
    with pytest.raises(ValueError):
        VlmApiDescriber("m", "u", "R2", "B")  # frame modes: variant A only
    with pytest.raises(ValueError):
        VlmApiDescriber("m", "u", "R1", "C", fewshot_path=ROOT / "vlm" / "fewshot.json")  # empty on purpose
    with pytest.raises(ValueError):
        VlmApiDescriber("m", "u", "R1").describe_frame("x.jpg")


def test_fewshot_variant_c(tmp_path):
    fs = tmp_path / "fs.json"
    fs.write_text(json.dumps({"examples": [
        {"image": IMG, "clothing_color": "red", "clothing_type": "coat", "description": "a"},
        {"image": IMG, "clothing_color": "blue", "clothing_type": "hoodie", "description": "b"}]}))
    with FakeVlmServer(_handler) as s:
        r = VlmApiDescriber("m", s.url, "R1", "C", fewshot_path=fs).describe_crop(IMG)
        assert r["status"] == "ok"
        msgs = s.requests[-1]["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
        assert json.loads(msgs[1]["content"])["clothing_color"] == "red"


def test_thinking_flag_and_retry_and_http_error():
    with FakeVlmServer(_handler) as s:
        VlmApiDescriber("m", s.url, "R1", disable_thinking=True).describe_crop(IMG)
        VlmApiDescriber("m", s.url, "R1").describe_crop(IMG)
        assert s.requests[-2]["chat_template_kwargs"] == {"enable_thinking": False}
        assert "chat_template_kwargs" not in s.requests[-1]
    with FakeVlmServer(lambda p: None) as s:  # always HTTP 500
        r = VlmApiDescriber("m", s.url, "R1", retries=1).describe_crop(IMG)
        assert r["status"] == "error" and r["attempts"] == 2 and r["record"] is None
    with FakeVlmServer(lambda p: "not json at all") as s:
        r = VlmApiDescriber("m", s.url, "R1", retries=0).describe_crop(IMG)
        assert r["status"] == "error" and "cannot parse" in r["error"]


# ------------------------------------------------------ benchmarks e2e ---
def _make_project(tmp_path, with_frames=True):
    from PIL import Image
    (tmp_path / "frames").mkdir()
    Image.new("RGB", (400, 300), (120, 40, 40)).save(tmp_path / "frames" / "sn_0000.jpg")
    snaps = {"video_id": "synthetic_e2e", "snapshots": [{
        "snapshot_id": "sn_0000", "frame_path": "frames/sn_0000.jpg", "people_count": 2,
        "boxes": [[10, 10, 110, 210], [200, 100, 250, 160]]}]}
    (tmp_path / "snapshots.json").write_text(json.dumps(snaps))
    with open(tmp_path / "gold_people.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "snapshot_id", "person_idx", "color", "color_2", "type", "visible", "box_h", "labeler"])
        for lab in ("a", "b"):
            w.writerow(["synthetic_e2e", "sn_0000", 0, "red", "", "sweater", "да", 200, lab])
            w.writerow(["synthetic_e2e", "sn_0000", 1, "black", "", "jacket", "да", 60, lab])
    with open(tmp_path / "gold_frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "snapshot_id", "true_people_count", "labeler"])
        w.writerow(["synthetic_e2e", "sn_0000", 5, "a"])
        w.writerow(["synthetic_e2e", "sn_0000", 5, "b"])
    return tmp_path


def test_bench_clothing_end_to_end(tmp_path):
    p = _make_project(tmp_path)
    gold, info = bc.load_gold_people(p / "gold_people.csv", "synthetic_e2e")
    assert len(gold) == 2 and info["n_labelers"] == 2 and info["n_color_ties"] == 0
    vid, snaps, _ = bc.load_snapshots(p / "snapshots.json")
    items, missing = bench_clothing.build_items(gold, snaps, p, p)
    assert len(items) == 2 and not missing
    out = tmp_path / "res"
    with FakeVlmServer(_handler) as s:  # always answers red sweater -> person 0 right, person 1 wrong
        d = VlmApiDescriber("fake", s.url, "R1", "D")
        res = bench_clothing.run_benchmark(gold, info, vid, describer=d, items=items, repeat=2, out_dir=out)
    m = res["metrics"]
    assert m["color_acc"] == 0.5 and m["type_acc"] == 0.5 and m["n_errors"] == 0
    assert m["black_share_pred"] == 0 and m["black_share_gold"] == 0.5
    assert m["buckets"][">150"]["color_acc"] == 1.0 and m["buckets"]["<80"]["color_acc"] == 0.0
    assert m["repeat_color_disagree"] == 0 and m["top2_has_color_2"]
    assert all(pth.exists() for pth in res["paths"].values())
    assert "SYNTHETIC" in res["paths"]["md"].read_text(encoding="utf-8")
    runs = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines()]
    assert len(runs) == 1 and runs[0]["mode"] == "crop" and runs[0]["prompt_variant"] == "D" and runs[0]["model"] == "fake"
    assert set(runs[0]) == {"run_id", "video_id", "method", "model", "mode", "prompt_variant", "weights",
                            "started_at", "duration_sec", "params"}
    assert (out / "clothing_summary.csv").exists()


def test_bench_clothing_failed_requests_count_as_errors(tmp_path):
    p = _make_project(tmp_path)
    gold, info = bc.load_gold_people(p / "gold_people.csv")
    vid, snaps, _ = bc.load_snapshots(p / "snapshots.json")
    items, _ = bench_clothing.build_items(gold, snaps, p, p)
    with FakeVlmServer(lambda q: None) as s:
        res = bench_clothing.run_benchmark(gold, info, vid, describer=VlmApiDescriber("f", s.url, "R1", retries=0),
                                           items=items, out_dir=tmp_path / "r")
    assert res["metrics"]["n_errors"] == 2 and res["metrics"]["color_acc"] == 0.0


def test_bench_clothing_external_predictions(tmp_path):
    p = _make_project(tmp_path)
    gold, info = bc.load_gold_people(p / "gold_people.csv")
    preds = {("synthetic_e2e", "sn_0000", 0): {"record": {"color": "red", "color_2": None, "type": "sweater"}, "seconds": 0.1},
             ("synthetic_e2e", "sn_0000", 1): {"record": {"color": "black", "color_2": None, "type": "jacket"}, "seconds": 0.1}}
    res = bench_clothing.run_benchmark(gold, info, "synthetic_e2e", predictions=preds, method="cv:test", out_dir=tmp_path / "r")
    assert res["metrics"]["color_acc"] == 1.0 and res["meta"]["method"] == "cv:test"
    assert not res["metrics"]["top2_has_color_2"]


def test_bench_count_end_to_end(tmp_path):
    p = _make_project(tmp_path)
    vid, snaps, _ = bc.load_snapshots(p / "snapshots.json")
    gold, _ = bc.load_gold_frames(p / "gold_frames.csv", vid)
    assert gold == {("synthetic_e2e", "sn_0000"): 5}
    with FakeVlmServer(_handler) as s:
        res = bench_count.run_count_benchmark(
            gold, vid, snaps, ["yolo", "R2", "R3", "R4"], lambda mode: VlmApiDescriber("fake", s.url, mode),
            p, p, model="fake", yolo_weights="w.pt", out_dir=tmp_path / "res")
    assert res["yolo"]["mae"] == 3 and res["yolo"]["under"] == 1.0
    assert res["R2"]["mae"] == 2 and res["R3"]["pred"] == [2] and res["R4"]["pred"] == [2]
    assert "SYNTHETIC" in res["_report"].read_text(encoding="utf-8")
    runs = [json.loads(x) for x in (tmp_path / "res" / "runs.jsonl").read_text().splitlines()]
    assert [r["mode"] for r in runs] == ["detector", "frame-count", "frame-list", "boxes"] and runs[0]["weights"] == "w.pt"


# ------------------------------------------------------------ agreement ---
def test_agreement_on_synthetic_fixtures():
    people = bc.read_csv(ROOT / "eval/gold/synthetic/gold_people_synthetic.csv")
    frames = bc.read_csv(ROOT / "eval/gold/synthetic/gold_frames_synthetic.csv")
    pw = agreement.pairwise(people)
    assert len(pw) == 9 and {r["n"] for r in pw if "synth_c" in r["pair"]} == {50} and pw[0]["n"] == 60
    assert all(-1 <= r["kappa"] <= 1 for r in pw)
    assert agreement.all_raters(people)[0]["n"] == 50
    assert agreement.frames_pairwise(frames)[0]["n"] == 22
    assert "SYNTHETIC" in agreement.build_report(people, frames)


def test_agreement_known_values():
    rows = [{"video_id": "v", "snapshot_id": "s", "person_idx": str(i), "labeler": lab, "color": c, "type": "x", "visible": "да"}
            for i, (a, b) in enumerate(zip("xxyy", "xyyy")) for lab, c in (("a", a), ("b", b))]
    r = [x for x in agreement.pairwise(rows) if x["field"] == "color"][0]
    assert r["agree"] == 0.75 and math.isclose(r["kappa"], 0.5)


def test_real_gold_templates_are_header_only():
    for name, cols in (("gold_people.csv", "video_id,snapshot_id,person_idx,color,color_2,type,visible,box_h,labeler"),
                       ("gold_frames.csv", "video_id,snapshot_id,true_people_count,labeler")):
        lines = (ROOT / "eval/gold" / name).read_text(encoding="utf-8").strip().splitlines()
        assert lines == [cols]


# --------------------------------------------------------- server kit ---
@pytest.mark.parametrize("script", sorted(p.name for p in (ROOT / "vlm" / "server").glob("*.sh")))
def test_server_scripts_syntax(script):
    assert subprocess.run(["bash", "-n", str(ROOT / "vlm" / "server" / script)]).returncode == 0


def _fake_home(tmp_path):
    w = tmp_path / "work"
    slug = "models--Qwen--Qwen3-VL-4B-Instruct"
    mine = [w / "hf_cache/hub" / slug, w / "cache/vllm" / slug, w / "cache/triton" / slug, w / "cache/inductor" / slug]
    keep = [w / "hf_cache/hub/models--Other--Keep", w / "cache/vllm/models--Other--Keep", w / "data", w / "runs"]
    for d in mine + keep:
        d.mkdir(parents=True)
        (d / "f.bin").write_text("x")
    return mine, keep


def _cleanup(tmp_path, answer, *args):
    return subprocess.run(["bash", str(ROOT / "vlm/server/cleanup.sh"), *(args or ("Qwen/Qwen3-VL-4B-Instruct",))],
                          input=answer, capture_output=True, text=True,
                          env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "CLEANUP_NO_KILL": "1"})


def test_cleanup_refuses_without_confirmation(tmp_path):
    mine, keep = _fake_home(tmp_path)
    r = _cleanup(tmp_path, "no\n")
    assert r.returncode != 0 and "ничего не удалено" in r.stdout.lower() and all(d.exists() for d in mine + keep)


def test_cleanup_deletes_only_the_named_model_everywhere(tmp_path):
    mine, keep = _fake_home(tmp_path)
    r = _cleanup(tmp_path, "yes\n")
    assert r.returncode == 0 and not any(d.exists() for d in mine) and all(d.exists() for d in keep)


def test_cleanup_also_removes_flashinfer_cache_outside_work(tmp_path):
    _fake_home(tmp_path)
    fi = tmp_path / ".cache/flashinfer"
    fi.mkdir(parents=True)
    (fi / "x.so").write_text("x")
    (tmp_path / ".cache/other").mkdir()
    assert _cleanup(tmp_path, "no\n").returncode != 0 and fi.exists()
    assert _cleanup(tmp_path, "yes\n").returncode == 0
    assert not fi.exists() and (tmp_path / ".cache/other").exists()


def test_cleanup_removes_orphan_blobs_and_locks_only_when_no_model_left(tmp_path):
    mine, keep = _fake_home(tmp_path)
    hub = tmp_path / "work/hf_cache/hub"
    for d in (hub / "blobs", hub / ".locks/models--Qwen--Qwen3-VL-4B-Instruct"):
        d.mkdir(parents=True)
        (d / "f").write_text("x")
    # another model is still in hub -> blobs/.locks stay, no extra question
    r = _cleanup(tmp_path, "yes\nyes\n")
    assert r.returncode == 0 and (hub / "blobs").exists() and (hub / ".locks").exists() and "Осиротевшие" not in r.stdout
    # last model removed, but the answer to the second question is not yes -> kept
    r = _cleanup(tmp_path, "yes\nno\n", "Other/Keep")
    assert r.returncode == 0 and "Осиротевшие" in r.stdout and (hub / "blobs").exists() and (hub / ".locks").exists()
    # a model is downloaded and removed again; hub then has no models--* -> removed after yes; the rest stays
    mine[0].mkdir(parents=True)
    r = _cleanup(tmp_path, "yes\nyes\n")
    assert r.returncode == 0 and not (hub / "blobs").exists() and not (hub / ".locks").exists()
    assert hub.exists() and (tmp_path / "work/data").exists() and (tmp_path / "work/runs").exists()


def test_cleanup_orphans_never_follow_symlinks(tmp_path):
    _fake_home(tmp_path)
    hub = tmp_path / "work/hf_cache/hub"
    for d in hub.glob("models--*"):
        import shutil
        shutil.rmtree(d)
    foreign = tmp_path / "foreign_blobs"
    foreign.mkdir()
    (foreign / "f").write_text("x")
    (hub / "blobs").symlink_to(foreign)
    r = _cleanup(tmp_path, "yes\nyes\n")
    assert r.returncode == 0 and (foreign / "f").exists() and (hub / "blobs").is_symlink() and "Не трогаю" in r.stdout


def test_cleanup_rejects_bad_repo_name(tmp_path):
    mine, _ = _fake_home(tmp_path)
    assert _cleanup(tmp_path, "yes\n", "../../etc").returncode == 2 and all(d.exists() for d in mine)


def test_cleanup_all_needs_exact_word_and_only_removes_work(tmp_path):
    mine, keep = _fake_home(tmp_path)
    (tmp_path / "other.txt").write_text("keep")
    r = _cleanup(tmp_path, "yes\n", "--all")  # wrong word
    assert r.returncode != 0 and (tmp_path / "work").exists()
    r = _cleanup(tmp_path, "delete-work\n", "--all")
    assert r.returncode == 0 and not (tmp_path / "work").exists() and (tmp_path / "other.txt").exists()


def test_cleanup_all_refuses_foreign_work_path(tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    r = subprocess.run(["bash", str(ROOT / "vlm/server/cleanup.sh"), "--all"], input="delete-work\n", capture_output=True,
                       text=True, env={"HOME": str(tmp_path), "WORK": str(other), "PATH": "/usr/bin:/bin", "CLEANUP_NO_KILL": "1"})
    assert r.returncode == 2 and other.exists()


def _serve(*args, env=None):
    e = {"HOME": "/h", "PATH": "/usr/bin:/bin"}
    e.update(env or {})
    return subprocess.run(["bash", str(ROOT / "vlm/server/serve_v1_qwen3vl4b.sh"), *args], capture_output=True, text=True, env=e)


def test_serve_print_defaults_gpu1_caches_and_util():
    r = _serve("--print")
    out = r.stdout
    assert r.returncode == 0 and "--gpu-memory-utilization 0.48" in out and "--port 8765" in out and "GPU=1" in out
    assert "HF_HOME=/h/work/hf_cache" in out and "XDG_CACHE_HOME=/h/work/cache" in out
    assert "TRITON_CACHE_DIR=/h/work/cache/triton/" in out and "VLLM_CACHE_ROOT=/h/work/cache/vllm/" in out
    assert '--limit-mm-per-prompt {"image": 1}' in out  # vLLM 0.34 takes JSON, not image=N (checked on the server)
    assert '{"image": 4}' in _serve("--print", env={"LIMIT_IMAGES": "4"}).stdout
    assert "--enforce-eager" in out and "--max-num-seqs 8" in out and "--max-num-batched-tokens 8192" in out
    assert "VLLM_USE_FLASHINFER_SAMPLER=0" in out and "--max-num-batched-tokens 4096" in _serve("--print", env={"MAX_LEN": "4096"}).stdout
    assert "--enforce-eager" not in _serve("--print", env={"ENFORCE_EAGER": "0"}).stdout
    sw = _serve("--print", env={"VLM_REPO": "me/awq-4b", "VLM_SERVED": "qwen3-vl-4b"}).stdout
    assert "vllm serve me/awq-4b --served-model-name qwen3-vl-4b" in sw and "triton/models--me--awq-4b" in sw
    assert "--gpu-memory-utilization 0.35" in _serve("--print", "--gpu-memory-utilization", "0.35").stdout
    assert _serve("--print", "--gpu-memory-utilization", "1.5").returncode == 2


def test_serve_refuses_busy_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        r = _serve("--print", env={"PORT": str(s.getsockname()[1])})
        assert r.returncode == 1 and "занят" in r.stdout and "vllm serve" not in r.stdout
    finally:
        s.close()


# ------------------------------------- server-side runner: isolated copy ---
def test_server_runner_in_isolated_copy_feeds_local_benchmarks(tmp_path):
    """Copy ONLY the files meant for the server into a clean folder, run run_predictions.py there (no repo on the
    path, python -I), then score the output with the local benchmarks."""
    import shutil
    import sys
    srv = tmp_path / "vlm"
    (srv / "describers").mkdir(parents=True)
    for f in ("prompts.py", "describe_snapshot.py", "imgutil.py", "run_predictions.py", "fewshot.json", "check_fewshot.py"):
        shutil.copy(ROOT / "vlm" / f, srv / f)
    for f in ("__init__.py", "base.py", "vlm_api.py"):
        shutil.copy(ROOT / "vlm/describers" / f, srv / "describers" / f)
    proj = _make_project(tmp_path / "proj") if (tmp_path / "proj").mkdir() is None else None
    runs = tmp_path / "runs"
    with FakeVlmServer(_handler) as s:
        base = [sys.executable, "-I", str(srv / "run_predictions.py")]
        common = ["--snapshots", str(proj / "snapshots.json"), "--frames-dir", str(proj / "frames"), "--model", "fake",
                  "--api-url", s.url, "--workers", "2"]
        r1 = subprocess.run(base + ["crops", *common, "--variant", "A", "--repeat", "2", "--out", str(runs / "pc.jsonl")],
                            capture_output=True, text=True, cwd=tmp_path)
        r2 = subprocess.run(base + ["count", *common, "--mode", "R2", "--out", str(runs / "cnt_R2.jsonl")],
                            capture_output=True, text=True, cwd=tmp_path)
    assert r1.returncode == 0, r1.stderr
    assert r2.returncode == 0, r2.stderr
    assert (runs / "pc.jsonl").exists() and (runs / "pc_repeat2.jsonl").exists()
    run_lines = [json.loads(x) for x in (runs / "runs.jsonl").read_text().splitlines()]
    assert [r["mode"] for r in run_lines] == ["crop", "frame-count"] and run_lines[0]["prompt_variant"] == "A"
    # local scoring of the files
    gold, info = bc.load_gold_people(proj / "gold_people.csv")
    preds = bench_clothing.load_predictions(runs / "pc.jsonl")
    rep = bench_clothing.load_predictions(runs / "pc_repeat2.jsonl")
    res = bench_clothing.run_benchmark(gold, info, "synthetic_e2e", predictions=preds, repeat_predictions=rep,
                                       method="vlm:fake", out_dir=tmp_path / "res")
    assert res["metrics"]["color_acc"] == 0.5 and res["metrics"]["repeat_color_disagree"] == 0
    assert not (tmp_path / "res" / "runs.jsonl").exists()  # scoring ready predictions must not invent a run
    vid, snaps, _ = bc.load_snapshots(proj / "snapshots.json")
    gf, _ = bc.load_gold_frames(proj / "gold_frames.csv", vid)
    pre = {"R2": bench_count.load_counts_file(runs / "cnt_R2.jsonl", vid)}
    out = bench_count.run_count_benchmark(gf, vid, snaps, ["yolo", "R2"], None, precomputed=pre, out_dir=tmp_path / "res2")
    assert out["R2"]["pred"] == [3] and out["R2"]["mae"] == 2 and out["yolo"]["pred"] == [2]
    assert not (tmp_path / "res2" / "runs.jsonl").read_text().count("frame-count")  # only yolo logged locally


def test_server_runner_marks_failures_and_missing_frames(tmp_path):
    import sys
    proj = tmp_path / "p"
    proj.mkdir()
    _make_project(proj)
    snaps = json.loads((proj / "snapshots.json").read_text())
    snaps["snapshots"][0]["frame_path"] = "frames/missing.jpg"
    (proj / "snapshots.json").write_text(json.dumps(snaps))
    with FakeVlmServer(_handler) as s:
        r = subprocess.run([sys.executable, "-I", str(ROOT / "vlm/run_predictions.py"), "crops", "--snapshots", str(proj / "snapshots.json"),
                            "--frames-dir", str(proj / "frames"), "--model", "f", "--api-url", s.url, "--out", str(tmp_path / "o/p.jsonl")],
                           capture_output=True, text=True)
    rows = [json.loads(x) for x in (tmp_path / "o/p.jsonl").read_text().splitlines()]
    assert r.returncode == 0 and len(rows) == 2 and all(x["record"] is None and "frame not found" in x["error"] for x in rows)


# ------------------------------- raw answer, truncated, suspicious, penalties ---
LOOP = '{"people":[' + ',\n'.join(['{"clothing_color":"black","clothing_type":"jacket","description":"same man"}'] * 200)


def test_describer_truncated_answer_is_not_parsed_or_retried():
    frame = next((ROOT / "demo_frames").rglob("*.jpg"))
    with FakeVlmServer(lambda p: {"content": LOOP, "finish_reason": "length"}) as s:
        r = VlmApiDescriber("m", s.url, "R3", retries=1).describe_frame(str(frame))
        assert r["status"] == "truncated" and r["people_count"] is None and r["records"] == []
        assert r["finish_reason"] == "length" and r["raw_response"] == LOOP and r["attempts"] == 1 and len(s.requests) == 1
        c = VlmApiDescriber("m", s.url, "R1").describe_crop(IMG)
        assert c["status"] == "truncated" and c["record"] is None
        n = len(s.requests)
        b = VlmApiDescriber("m", s.url, "R1", "B").describe_crop(IMG)  # step 1 cut -> step 2 not sent
        assert b["status"] == "truncated" and len(s.requests) == n + 1 and b["raw_response"].startswith("STEP1:")
    with FakeVlmServer(_handler) as s:
        r = VlmApiDescriber("m", s.url, "R2").describe_frame(str(frame))
        assert r["status"] == "ok" and r["finish_reason"] == "stop" and r["completion_tokens"] == 10
        assert r["raw_response"].startswith("<think>")  # raw = before stripping <think>


def test_penalties_and_max_tokens_only_sent_when_given():
    with FakeVlmServer(_handler) as s:
        VlmApiDescriber("m", s.url, "R1").describe_crop(IMG)
        VlmApiDescriber("m", s.url, "R1", frequency_penalty=0.5, repetition_penalty=1.1, max_tokens=777).describe_crop(IMG)
        plain, tuned = s.requests
    assert set(plain) == {"model", "temperature", "max_tokens", "messages"} and plain["max_tokens"] == vlm_api.MAX_TOKENS_CROP
    assert tuned["frequency_penalty"] == 0.5 and tuned["repetition_penalty"] == 1.1 and tuned["max_tokens"] == 777
    assert {k: v for k, v in tuned.items() if k not in ("frequency_penalty", "repetition_penalty", "max_tokens")} == \
           {k: v for k, v in plain.items() if k != "max_tokens"}


def _run_count(tmp_path, handler, *extra, boxes=None, min_h="0"):
    import sys
    proj = tmp_path / "p"
    proj.mkdir(exist_ok=True)
    if not (proj / "snapshots.json").exists():
        _make_project(proj)
    if boxes is not None:
        snaps = json.loads((proj / "snapshots.json").read_text())
        snaps["snapshots"][0]["boxes"] = boxes
        (proj / "snapshots.json").write_text(json.dumps(snaps))
    out = tmp_path / "o" / f"c{len(list((tmp_path).glob('o/c*.jsonl')))}.jsonl"
    with FakeVlmServer(handler) as s:
        r = subprocess.run([sys.executable, "-I", str(ROOT / "vlm/run_predictions.py"), "count", "--mode", "R3",
                            "--snapshots", str(proj / "snapshots.json"), "--frames-dir", str(proj / "frames"), "--model", "f",
                            "--api-url", s.url, "--min-box-height", min_h, "--out", str(out), *extra],
                           capture_output=True, text=True)
        reqs = list(s.requests)
    assert r.returncode == 0, r.stderr
    runs = [json.loads(x) for x in (out.parent / "runs.jsonl").read_text().splitlines()]
    return json.loads(out.read_text().splitlines()[0]), reqs, runs[-1]


def test_runner_count_truncated_suspicious_raw_and_params(tmp_path):
    long = LOOP * 20
    row, reqs, run = _run_count(tmp_path, lambda p: {"content": long, "finish_reason": "length"})
    assert row["status"] == "truncated" and row["people_count"] is None and row["suspicious"] is False
    assert row["finish_reason"] == "length" and row["raw_response"] == long[:2000] and row["completion_tokens"] == 10
    assert run["params"]["frequency_penalty"] is None and run["params"]["max_tokens"] is None
    assert "frequency_penalty" not in reqs[0] and "repetition_penalty" not in reqs[0]
    # 0 people but the detector has boxes (both are >= 0 px) -> ok + suspicious
    row, _, _ = _run_count(tmp_path, lambda p: '{"people":[]}')
    assert row["status"] == "ok" and row["people_count"] == 0 and row["suspicious"] is True and row["raw_response"] == '{"people":[]}'
    # boxes are 200 and 60 px high; with --min-box-height 300 none pass the filter -> not suspicious
    row, _, _ = _run_count(tmp_path, lambda p: '{"people":[]}', min_h="300")
    assert row["status"] == "ok" and row["suspicious"] is False
    row, _, _ = _run_count(tmp_path, lambda p: '{"people":[]}', boxes=[])
    assert row["suspicious"] is False
    # penalties: sent and logged
    row, reqs, run = _run_count(tmp_path, _handler, "--frequency-penalty", "0.3", "--repetition-penalty", "1.05",
                                "--max-tokens", "512")
    assert row["status"] == "ok" and row["people_count"] == 2 and row["suspicious"] is False
    assert reqs[0]["frequency_penalty"] == 0.3 and reqs[0]["repetition_penalty"] == 1.05 and reqs[0]["max_tokens"] == 512
    assert run["params"]["frequency_penalty"] == 0.3 and run["params"]["repetition_penalty"] == 1.05 and run["params"]["max_tokens"] == 512


def test_runner_crops_lines_have_raw_answer_and_truncated(tmp_path):
    import sys
    proj = tmp_path / "p"
    proj.mkdir()
    _make_project(proj)
    calls = []

    def handler(p):
        calls.append(1)
        return {"content": '{"clothing_color":"re', "finish_reason": "length"} if len(calls) == 1 else _handler(p)
    with FakeVlmServer(handler) as s:
        r = subprocess.run([sys.executable, "-I", str(ROOT / "vlm/run_predictions.py"), "crops", "--snapshots", str(proj / "snapshots.json"),
                            "--frames-dir", str(proj / "frames"), "--model", "f", "--api-url", s.url, "--workers", "1",
                            "--out", str(tmp_path / "o/p.jsonl")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(x) for x in (tmp_path / "o/p.jsonl").read_text().splitlines()]
    assert [x["status"] for x in rows] == ["truncated", "ok"] and rows[0]["record"] is None
    assert rows[1]["raw_response"].startswith('{"clothing_color":"red"') and rows[1]["finish_reason"] == "stop"
    assert all(x["suspicious"] is False and x["prompt_tokens"] == 100 for x in rows)
    preds = bench_clothing.load_predictions(tmp_path / "o/p.jsonl")  # truncated lines count as failures on the Mac
    assert sum(1 for v in preds.values() if v["status"] == "ok") == 1


def test_exclude_fewshot_sources_from_gold(tmp_path):
    gold = {("v", "s1", 0): {}, ("v", "s1", 1): {}, ("v", "s2", 0): {}}
    fs = tmp_path / "fs.json"
    fs.write_text(json.dumps({"examples": [{"image": "x", "source": {"video_id": "v", "snapshot_id": "s1", "person_idx": 1}},
                                           {"image": "y"}]}))
    assert set(bench_clothing.exclude_fewshot(gold, fs)) == {("v", "s1", 0), ("v", "s2", 0)}


def test_check_fewshot_script(tmp_path):
    import sys
    chk = str(ROOT / "vlm/check_fewshot.py")
    assert subprocess.run([sys.executable, chk], capture_output=True).returncode == 1  # empty template
    (tmp_path / "a.jpg").write_bytes(b"x")
    ex = {"image": "a.jpg", "clothing_color": "red", "clothing_type": "coat", "description": "d",
          "source": {"video_id": "v", "snapshot_id": "s", "person_idx": 0}}
    good = tmp_path / "f.json"
    good.write_text(json.dumps({"examples": [ex, ex, ex]}))
    assert subprocess.run([sys.executable, chk, str(good)], capture_output=True).returncode == 0
    bad = tmp_path / "g.json"
    bad.write_text(json.dumps({"examples": [{**ex, "clothing_color": "navy"}, ex]}))
    assert subprocess.run([sys.executable, chk, str(bad)], capture_output=True).returncode == 1


# ----------------------------------------------------------- summarize_runs ---
def _jl(path, rows, extra_lines=()):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "".join(extra_lines))


def _crop(i, color, typ, sec, status="ok", new=False):
    r = {"video_id": "v", "snapshot_id": "sn_0000", "person_idx": i, "box_h": 100,
         "record": {"color": color, "type": typ, "method": "vlm:m", "mode": "crop", "prompt_variant": "A"}
         if status == "ok" else None, "seconds": sec, "status": status, "error": None}
    if new:
        r.update(raw_response="{}", finish_reason="length" if status == "truncated" else "stop", suspicious=False)
    return r


def test_summarize_runs_old_and_new_formats(tmp_path):
    from eval import summarize_runs as S
    d = tmp_path / "server_runs"
    d.mkdir()
    # old R1 A + its repeat: 4 people, color agrees 3/4, type 4/4 (person 3 is error in pass 1 -> not a pair)
    _jl(d / "pred_clothing_m_R1_A.jsonl", [_crop(0, "black", "coat", 1.0), _crop(1, "unknown", "coat", 2.0),
                                          _crop(2, "red", "jacket", 3.0), _crop(3, None, None, None, status="error")])
    _jl(d / "pred_clothing_m_R1_A_repeat2.jsonl", [_crop(0, "black", "coat", 1.0), _crop(1, "gray", "coat", 1.0),
                                                  _crop(2, "red", "jacket", 1.0), _crop(3, "red", "coat", 1.0)])
    # new R1 D with one truncated line and a broken line
    _jl(d / "pred_clothing_m_R1_D.jsonl", [_crop(0, "black", "coat", 2.0, new=True),
                                          _crop(1, None, None, 4.0, status="truncated", new=True)], ["{broken\n"])
    # old count R3 (no suspicious field) and new count R3 with a tag
    old = [{"snapshot_id": f"s{i}", "mode": "frame-list", "model": "m", "people_count": c, "seconds": 10.0 * (i + 1),
            "status": "ok" if c is not None else "error", "error": None} for i, c in enumerate([4, 6, None])]
    _jl(d / "count_m_R3.jsonl", old)
    new = [{**r, "raw_response": "", "finish_reason": "stop", "suspicious": r["people_count"] == 0} for r in
           [{"snapshot_id": "s0", "mode": "frame-list", "model": "m", "people_count": 0, "seconds": 1.0, "status": "ok"},
            {"snapshot_id": "s1", "mode": "frame-list", "model": "m", "people_count": None, "seconds": 2.0,
             "status": "truncated"}]]
    _jl(d / "count_m_R3_rep1.1.jsonl", new)
    (d / "runs.jsonl").write_text('{"run_id": "x"}\n')
    (d / "vllm.log").write_text("log")

    out = tmp_path / "s.csv"
    assert S.main(["--dir", str(d), "--out", str(out)]) == 0
    rows = {(r["mode"], r["variant"], r["tag"]): r for r in S.collect(d)}
    assert set(rows) == {("R1", "A", ""), ("R1", "D", ""), ("R3", "A", ""), ("R3", "A", "rep1.1")}  # repeat2 is no row
    a = rows[("R1", "A", "")]
    assert a["format"] == "old" and a["rows"] == 4 and a["ok_share"] == 0.75 and a["error_share"] == 0.25
    assert a["truncated_share"] == 0 and a["sec_mean"] == 2.0 and a["sec_p95"] == 3.0
    assert a["black_share"] == round(1 / 3, 4) and a["unknown_share"] == round(1 / 3, 4)
    assert a["repeat_pairs"] == 3 and a["repeat_color_same"] == round(2 / 3, 4) and a["repeat_type_same"] == 1
    dd = rows[("R1", "D", "")]
    assert dd["format"] == "new" and dd["truncated_share"] == 0.5 and dd["bad_lines"] == 1 and dd.get("repeat_pairs") is None
    o = rows[("R3", "A", "")]
    assert o["format"] == "old" and o["count_mean"] == 5 and o["count_std"] == 1 and o["count_min"] == 4
    assert o.get("suspicious") is None  # unknown in the old format, not 0
    n = rows[("R3", "A", "rep1.1")]
    assert n["suspicious"] == 1 and n["truncated_share"] == 0.5 and n["count_mean"] == 0
    csv_rows = list(csv.DictReader(out.open()))
    assert len(csv_rows) == 4 and csv_rows[0]["model"] == "m" and set(S.COLUMNS) == set(csv_rows[0])
    md = S.to_markdown(S.collect(d))
    assert "### Одежда (R1)" in md and "### Счёт людей (R2/R3/R4)" in md and "rep1.1" in md


def test_summarize_runs_empty_or_missing_dir(tmp_path):
    from eval import summarize_runs as S
    assert S.main(["--dir", str(tmp_path / "nope")]) == 2
    (tmp_path / "e").mkdir()
    assert S.main(["--dir", str(tmp_path / "e")]) == 1


# ------------------------------------------------- index mode (Arai's crops) ---
ARAI_FLAT_KEYS = {"video_id", "snapshot_id", "person_idx", "crop_id", "box_h", "box", "sec", "color", "color_2", "type",
                  "confidence", "description", "method", "mode", "prompt_variant"}


def _make_index_project(root):
    """Two videos as cv/crop_people.py writes them: crops_index.json + frames_index.json + full/<crop_id>.jpg.
    p01 is a duplicate (kept=false, no file). In video B the crop file of p02 is missing."""
    from PIL import Image
    for vid, name in (("vidA", "v1"), ("vidB", "v2")):
        d = root / name
        (d / "full").mkdir(parents=True)
        (d / "frames").mkdir()
        Image.new("RGB", (400, 300), (90, 90, 90)).save(d / "frames" / "sn_0000.jpg")
        crops = []
        for idx, (box, reason) in enumerate([([10, 10, 110, 210], None), ([12, 12, 108, 205], "dup_of_p00"),
                                             ([200, 100, 260, 200], None)]):
            cid = f"sn_0000_p{idx:02d}"
            c = {"crop_id": cid, "video_id": vid, "snapshot_id": "sn_0000", "person_idx": idx, "t": 0.0, "box": box,
                 "box_h": float(box[3] - box[1]), "box_w": float(box[2] - box[0]), "h_bucket": ">150",
                 "frame_path": "frames/sn_0000.jpg", "kept": reason is None, "reason": reason}
            if reason is None:
                c["path"], c["full_path"] = f"elsewhere/torso/{cid}.jpg", f"elsewhere/full/{cid}.jpg"
                if not (vid == "vidB" and idx == 2):
                    Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (200, 30, 30)).save(d / "full" / f"{cid}.jpg")
            crops.append(c)
        (d / "crops_index.json").write_text(json.dumps(crops))
        (d / "frames_index.json").write_text(json.dumps([{
            "video_id": vid, "snapshot_id": "sn_0000", "t": 0.0, "frame_path": "frames/sn_0000.jpg", "people_count": 3,
            "n_dups": 1, "n_too_small": 0, "people_after_dedup": 2, "n_kept": 2}]))
    with open(root / "gold_people.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "snapshot_id", "person_idx", "color", "type", "visible", "box_h", "labeler"])
        for vid in ("vidA", "vidB"):
            for idx in (0, 2):
                w.writerow([vid, "sn_0000", idx, "red", "sweater", "да", 200, "a"])


def _runner(args, handler):
    import sys
    with FakeVlmServer(handler) as s:
        r = subprocess.run([sys.executable, "-I", str(ROOT / "vlm/run_predictions.py"), *args, "--model", "f",
                            "--api-url", s.url], capture_output=True, text=True)
        reqs = list(s.requests)
    return r, reqs


def _lines(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def test_runner_crops_index_mode_kept_only_ready_files_and_flat_format(tmp_path):
    import sys
    _make_index_project(tmp_path)
    calls = []

    def handler(p):
        calls.append(1)
        return {"content": '{"clothing_color":"re', "finish_reason": "length"} if len(calls) == 1 else _handler(p)
    out = tmp_path / "o" / "pred_clothing_f_R1_A.jsonl"
    r, reqs = _runner(["crops", "--crops-index", str(tmp_path / "v1/crops_index.json"), str(tmp_path / "v2/crops_index.json"),
                       "--workers", "1", "--out", str(out)], handler)
    assert r.returncode == 0, r.stderr
    assert len(reqs) == 3  # 2 kept in A, 1 kept with a file in B; the duplicate p01 is never sent
    rows = _lines(out)
    assert [(x["video_id"], x["person_idx"], x["status"]) for x in rows] == [
        ("vidA", 0, "truncated"), ("vidA", 2, "ok"), ("vidB", 0, "ok"), ("vidB", 2, "skipped")]
    assert rows[1]["crop_id"] == "sn_0000_p02" and rows[1]["box"] == [200, 100, 260, 200] and rows[1]["box_h"] == 100.0
    assert "crop file not found" in rows[3]["error"] and rows[0]["raw_response"] == '{"clothing_color":"re'
    flat = _lines(tmp_path / "o" / "flat" / "pred_clothing_f_R1_A.jsonl")
    assert len(flat) == 4 and all(ARAI_FLAT_KEYS <= set(x) for x in flat)
    assert flat[1]["color"] == "red" and flat[1]["type"] == "sweater" and flat[1]["method"] == "vlm:f" and flat[1]["mode"] == "crop"
    assert [(x["color"], x["type"], x["status"]) for x in (flat[0], flat[3])] == [
        ("unknown", "unknown", "truncated"), ("unknown", "unknown", "skipped")]
    assert flat[0]["finish_reason"] == "length" and flat[1]["sec"] is not None
    sample = ROOT / "results" / "hist__crop__A.jsonl"  # Arai's real file: our flat line has at least her keys
    if sample.exists():
        assert set(_lines(sample)[0]) <= set(flat[0])
    runs = _lines(tmp_path / "o" / "runs.jsonl")
    assert [x["video_id"] for x in runs] == ["vidA", "vidB"] and runs[0]["params"]["source"] == "crops_index"
    assert not (tmp_path / "o" / "summary_runs.csv").exists()
    # summarize_runs reads only the top folder (the flat copy is no row) and splits the file per video
    from eval import summarize_runs
    summ = summarize_runs.collect(tmp_path / "o")
    assert [(x["file"], x["video_id"], x["rows"], x["ok_share"]) for x in summ] == [
        ("pred_clothing_f_R1_A.jsonl", "vidA", 2, 0.5), ("pred_clothing_f_R1_A.jsonl", "vidB", 2, 0.5)]
    # score_methods reads our flat file; truncated / skipped people are misses (n = 4, 2 right), not dropped
    from eval import score_methods
    assert len(score_methods.load_preds(tmp_path / "o" / "flat" / "pred_clothing_f_R1_A.jsonl")) == 4
    sm = subprocess.run([sys.executable, "-m", "eval.score_methods", "people", "--gold", str(tmp_path / "gold_people.csv"),
                         "--pred", str(tmp_path / "o" / "flat" / "*.jsonl")], capture_output=True, text=True, cwd=ROOT)
    assert sm.returncode == 0, sm.stderr
    assert "| pred_clothing_f_R1_A | 4 | 0.500 |" in sm.stdout


def test_runner_crops_index_root_override_and_errors(tmp_path):
    _make_index_project(tmp_path)
    (tmp_path / "copy").mkdir()
    import shutil
    shutil.copytree(tmp_path / "v1" / "full", tmp_path / "copy" / "full")
    out = tmp_path / "o" / "p.jsonl"
    r, reqs = _runner(["crops", "--crops-index", str(tmp_path / "v1/crops_index.json"), "--crops-root", str(tmp_path / "copy"),
                       "--out", str(out), "--out-flat", str(tmp_path / "flat.jsonl"), "--repeat", "2"], _handler)
    assert r.returncode == 0, r.stderr
    assert len(reqs) == 4
    assert [x["status"] for x in _lines(out)] == ["ok", "ok"] and len(_lines(tmp_path / "flat.jsonl")) == 2
    assert len(_lines(tmp_path / "flat_repeat2.jsonl")) == 2 and len(_lines(tmp_path / "o" / "p_repeat2.jsonl")) == 2
    r, _ = _runner(["crops", "--crops-index", str(tmp_path / "v1/crops_index.json"), "--snapshots", "x", "--out", str(out)], _handler)
    assert r.returncode == 2 and "OR --crops-index" in r.stderr
    r, _ = _runner(["crops", "--crops-index", str(tmp_path / "v1/crops_index.json"), str(tmp_path / "v2/crops_index.json"),
                    "--crops-root", "a", "b", "c", "--out", str(out)], _handler)
    assert r.returncode != 0 and "per index file" in r.stderr
    # two videos have the same crop ids (sn_0000_p00): one shared crops folder is refused
    r, _ = _runner(["crops", "--crops-index", str(tmp_path / "v1/crops_index.json"), str(tmp_path / "v2/crops_index.json"),
                    "--crops-root", str(tmp_path / "copy"), "--out", str(out)], _handler)
    assert r.returncode != 0 and "names repeat in every video" in r.stderr


def test_runner_count_index_mode_r4_gets_only_kept_boxes(tmp_path):
    _make_index_project(tmp_path)
    fi = [str(tmp_path / "v1/frames_index.json"), str(tmp_path / "v2/frames_index.json")]
    ci = [str(tmp_path / "v1/crops_index.json"), str(tmp_path / "v2/crops_index.json")]
    fd = [str(tmp_path / "v1"), str(tmp_path / "v2")]
    out = tmp_path / "o" / "count_f_R4.jsonl"
    r, reqs = _runner(["count", "--mode", "R4", "--frames-index", *fi, "--frames-dir", *fd, "--crops-index", *ci,
                       "--out", str(out)], _handler)
    assert r.returncode == 0, r.stderr
    assert len(reqs) == 2
    for q in reqs:
        t = q["messages"][-1]["content"][-1]["text"]
        assert "0: [10, 10, 110, 210]" in t and "1: [200, 100, 260, 200]" in t and "2: [" not in t and "[12, 12" not in t
    rows = _lines(out)
    assert [(x["video_id"], x["status"], x["people_count"], x["box_person_idx"]) for x in rows] == [
        ("vidA", "ok", 2, [0, 2]), ("vidB", "ok", 2, [0, 2])]
    # R4 without the crops index: refused (the boxes would be unknown)
    r, _ = _runner(["count", "--mode", "R4", "--frames-index", *fi, "--frames-dir", *fd, "--out", str(out)], _handler)
    assert r.returncode == 2 and "needs --crops-index" in r.stderr
    # R2 from frames_index only: suspicious uses n_kept; no box_person_idx outside R4
    out2 = tmp_path / "o" / "count_f_R2.jsonl"
    r, reqs = _runner(["count", "--mode", "R2", "--frames-index", *fi, "--frames-dir", *fd, "--out", str(out2)],
                      lambda p: '{"people_count": 0}')
    assert r.returncode == 0, r.stderr
    rows = _lines(out2)
    assert [(x["people_count"], x["suspicious"]) for x in rows] == [(0, True), (0, True)] and "box_person_idx" not in rows[0]
    assert {x["video_id"] for x in _lines(tmp_path / "o" / "runs.jsonl")} == {"vidA", "vidB"}
    r, _ = _runner(["count", "--frames-index", *fi, "--frames-dir", "a", "b", "c", "--out", str(out2)], _handler)
    assert r.returncode != 0 and "per index file" in r.stderr
    r, _ = _runner(["count", "--frames-index", *fi, "--frames-dir", fd[0], "--out", str(out2)], _handler)
    assert r.returncode != 0 and "names repeat in every video" in r.stderr
    # no frames folder for video B (as now for the second school video): B is skipped with a message, A runs
    out3 = tmp_path / "o" / "count_f_R3.jsonl"
    r, reqs = _runner(["count", "--mode", "R3", "--frames-index", *fi, "--frames-dir", fd[0], str(tmp_path / "nope"),
                       "--out", str(out3)], _handler)
    assert r.returncode == 0, r.stderr
    assert "ПРОПУСК vidB" in r.stderr and str(tmp_path / "nope") in r.stderr and len(reqs) == 1
    assert [x["video_id"] for x in _lines(out3)] == ["vidA"]
    r, reqs = _runner(["count", "--mode", "R2", "--frames-index", fi[1], "--frames-dir", str(tmp_path / "nope"),
                       "--out", str(out3)], _handler)
    assert r.returncode != 0 and "ПРОПУСК vidB" in r.stderr and "запускать нечего" in r.stderr and reqs == []


def test_runner_index_mode_on_real_stage2_frames(tmp_path):
    """Real data, no GPU, fake server: Arai's arai_data/video*/video*_crops_index.json (crops lie right in video1/,
    video2/), the frames of video 1 in arai_data/frames_all; video 2 has no frames yet -> its R4 is skipped."""
    ad = ROOT / "arai_data"
    ci = sorted(str(p) for p in ad.glob("video*/*crops_index.json"))
    if len(ci) < 2:
        pytest.skip("no Arai indices in arai_data/video*/")
    fi = [c.replace("crops_index", "frames_index") for c in ci]
    entries = [c for p in ci for c in json.loads(Path(p).read_text())]
    kept = [c for c in entries if c["kept"]]
    keys = {(c["video_id"], c["snapshot_id"], c["person_idx"]) for c in kept}
    assert len(keys) == len(kept) and len({c["video_id"] for c in kept}) == 2
    assert len({(c["snapshot_id"], c["person_idx"]) for c in kept}) < len(kept)  # sn_ ids repeat across the videos
    out = tmp_path / "o" / "pred_clothing_f_R1_A.jsonl"
    r, reqs = _runner(["crops", "--crops-index", *ci, "--workers", "8", "--out", str(out)], _handler)
    assert r.returncode == 0, r.stderr
    rows = _lines(out)
    assert len(rows) == len(kept) == len(reqs) and all(x["status"] == "ok" for x in rows)
    assert {(x["video_id"], x["snapshot_id"], x["person_idx"]) for x in rows} == keys
    flat = _lines(tmp_path / "o" / "flat" / out.name)
    assert {(x["video_id"], x["snapshot_id"], x["person_idx"]) for x in flat} == keys
    from eval import score_methods, summarize_runs
    assert set(score_methods.load_preds(tmp_path / "o" / "flat" / out.name)) == keys  # nothing glued together
    by_video = {x["video_id"]: x["rows"] for x in summarize_runs.collect(tmp_path / "o")}
    assert by_video == {v: sum(1 for c in kept if c["video_id"] == v) for v in by_video} and len(by_video) == 2
    frames1 = ad / "frames_all"
    if not (frames1 / "frames").is_dir():
        return
    out4 = tmp_path / "o" / "count_f_R4.jsonl"
    r, reqs = _runner(["count", "--mode", "R4", "--frames-index", *fi, "--frames-dir", str(frames1), str(ad / "frames_video2"),
                       "--crops-index", *ci, "--workers", "4", "--out", str(out4)], _handler)
    assert r.returncode == 0, r.stderr
    rows4 = _lines(out4)
    v1 = json.loads(Path(fi[0]).read_text())[0]["video_id"]
    if not (ad / "frames_video2").is_dir():
        assert "ПРОПУСК" in r.stderr and {x["video_id"] for x in rows4} == {v1}
    per_frame = {}
    for c in kept:
        per_frame.setdefault((c["video_id"], c["snapshot_id"]), []).append(c["person_idx"])
    assert all(x["status"] == "ok" and x["box_person_idx"] == sorted(per_frame.get((x["video_id"], x["snapshot_id"]), []))
               for x in rows4)
    import re
    sent = sorted(sum(1 for ln in q["messages"][-1]["content"][-1]["text"].splitlines() if re.match(r"^\d+: \[", ln))
                  for q in reqs)
    assert sent == sorted(len(per_frame.get((x["video_id"], x["snapshot_id"]), [])) for x in rows4)


def test_summarize_runs_two_videos_same_snapshot_ids_are_not_glued(tmp_path):
    from eval import summarize_runs

    def line(v, sid, color, status="ok"):
        return json.dumps({"video_id": v, "snapshot_id": sid, "person_idx": 0, "status": status, "seconds": 1.0,
                           "finish_reason": "stop", "record": {"color": color, "type": "coat"} if status == "ok" else None})
    (tmp_path / "pred_clothing_m_R1_A.jsonl").write_text("\n".join([line("v1", "sn_0000", "red"), line("v2", "sn_0000", "blue"),
                                                                     line("v2", "sn_0001", "x", "truncated")]))
    (tmp_path / "pred_clothing_m_R1_A_repeat2.jsonl").write_text("\n".join([line("v1", "sn_0000", "red"),
                                                                             line("v2", "sn_0000", "black")]))
    (tmp_path / "count_m_R2.jsonl").write_text("\n".join(json.dumps(
        {"video_id": v, "snapshot_id": "sn_0000", "status": "ok", "people_count": n, "suspicious": False}) for v, n in (("v1", 3), ("v2", 9))))
    rows = {(r["mode"], r["video_id"]): r for r in summarize_runs.collect(tmp_path)}
    assert set(rows) == {("R1", "v1"), ("R1", "v2"), ("R2", "v1"), ("R2", "v2")}
    assert rows[("R1", "v1")]["repeat_color_same"] == 1.0 and rows[("R1", "v2")]["repeat_color_same"] == 0.0
    assert rows[("R1", "v2")]["rows"] == 2 and rows[("R1", "v2")]["truncated_share"] == 0.5
    assert rows[("R2", "v1")]["count_mean"] == 3 and rows[("R2", "v2")]["count_mean"] == 9
    assert "| видео |" in summarize_runs.to_markdown(list(rows.values()))
    from eval import bench_count
    with pytest.raises(SystemExit, match="several videos"):
        bench_count.load_counts_file(tmp_path / "count_m_R2.jsonl")
    assert bench_count.load_counts_file(tmp_path / "count_m_R2.jsonl", "v2")[0] == {"sn_0000": {"count": 9, "seconds": None}}


def test_bench_common_crops_index_real_schema_full_crop_kept_and_one_video(tmp_path):
    from eval.bench_common import load_crops_index
    _make_index_project(tmp_path)
    idx = load_crops_index(tmp_path / "v1/crops_index.json", "vidA")
    assert set(idx) == {("vidA", "sn_0000", 0), ("vidA", "sn_0000", 2)}  # the duplicate p01 (kept=false) is out
    assert idx[("vidA", "sn_0000", 0)]["crop_path"] == str(tmp_path / "v1/full/sn_0000_p00.jpg")
    assert load_crops_index(tmp_path / "v1/crops_index.json", "vidB") == {}  # never another video's lines
    real = sorted((ROOT / "arai_data").glob("video*/*crops_index.json"))
    for p in real:  # Arai's layout: the crops lie right next to the index, without full/
        rows = json.loads(p.read_text())
        vid = rows[0]["video_id"]
        got = load_crops_index(p, vid)
        assert len(got) == sum(1 for c in rows if c["kept"]) and all(Path(v["crop_path"]).parent == p.parent for v in got.values())
