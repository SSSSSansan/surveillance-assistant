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
from vlm.describers.base import validate_record
from vlm.describers.vlm_api import VlmApiDescriber

import describe_snapshot as ds
import prompts

ROOT = Path(__file__).resolve().parents[1]
IMG = "data:image/jpeg;base64,AA=="


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
    assert r.returncode == 0 and "--gpu-memory-utilization 0.40" in out and "--port 8765" in out and "GPU=1" in out
    assert "HF_HOME=/h/work/hf_cache" in out and "XDG_CACHE_HOME=/h/work/cache" in out
    assert "TRITON_CACHE_DIR=/h/work/cache/triton/" in out and "VLLM_CACHE_ROOT=/h/work/cache/vllm/" in out
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
