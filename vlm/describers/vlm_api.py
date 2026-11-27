"""VLM over an OpenAI-compatible API (vLLM). One class, four modes, four prompt variants.

    d = VlmApiDescriber(model="qwen3-vl-4b", mode="R1", prompt_variant="A")
    d.describe_crop(data_url, person_idx=3)          # R1
    VlmApiDescriber(model=..., mode="R2").describe_frame("frames/sn_0000.jpg")            # count only
    VlmApiDescriber(model=..., mode="R3").describe_frame("frames/sn_0000.jpg")            # list + clothing
    VlmApiDescriber(model=..., mode="R4").describe_frame("frames/sn_0000.jpg", boxes)     # boxes in prompt

Parsing and vocabulary normalisation are IMPORTED from describe_snapshot (no copies).
temperature is always 0. Only the standard library is needed here (+ Pillow/OpenCV through imgutil
for the frame size in R4, optional).

NOT VERIFIED without the GPU server: `disable_thinking` (chat_template_kwargs.enable_thinking=false)
is accepted by the vLLM build in use; images as base64 data URLs for every model; few-shot (several
images in one request) needs a --limit-mm-per-prompt larger than 1 on the server.

Every result carries raw_response (model text before <think> stripping), prompt_tokens, completion_tokens and
finish_reason. finish_reason == "length" (cut at max_tokens) -> status "truncated", value None: a cut JSON is
never repaired and a cut people list is never counted.
Optional sampling (off by default, so old runs stay reproducible): frequency_penalty, repetition_penalty
(vLLM accepts both in the request body) and max_tokens (replaces the per-mode limit in EVERY request).
"""
import base64
import json
import mimetypes
import re
import sys
import time
import urllib.request

import describe_snapshot as ds
import prompts

from . import base  # noqa: F401  (also makes sure vlm/ is on sys.path via __init__)
from .base import Describer, make_answer

MAX_TOKENS_COUNT = 64
MAX_TOKENS_CROP = ds.MAX_TOKENS_CROP
MAX_TOKENS_FRAME = ds.MAX_TOKENS
MAX_TOKENS_BOXES = 2048
MAX_TOKENS_STEP1 = 200
MIN_CROP_SIDE = 336

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_thinking(text: str) -> str:
    """Remove a <think>...</think> block (Qwen3.x 'thinking' models) before JSON parsing."""
    text = text or ""
    text = _THINK.sub("", text)
    if "</think>" in text:  # the opening tag was cut by the chat template
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


# ---------------------------------------------------------------- parsing ---

def _opt_color(value, warnings, where):
    """color_2: null / '' / 'none' -> None, else normalised with the stage-2 rules."""
    if value is None or str(value).strip().lower() in ("", "none", "null", "n/a"):
        return None
    return ds.normalize_color(value, warnings, where)


def _confidence(value, warnings, where):
    if value is None or isinstance(value, bool):
        return None
    try:
        c = float(value)
    except (TypeError, ValueError):
        warnings.append(f"{where}: confidence {value!r} is not a number -> None")
        return None
    if c != c or c < 0 or c > 1:
        if 1 < c <= 100:  # model answered in percent
            warnings.append(f"{where}: confidence {value!r} read as percent")
            return round(c / 100, 4)
        warnings.append(f"{where}: confidence {value!r} out of 0..1 -> None")
        return None
    return c


def parse_crop_d(text: str, idx: int = 0):
    """Variant D answer -> (person dict, warnings); (None, warnings) if unparseable.
    person has clothing_color, clothing_color_2, clothing_type, confidence, description."""
    person, warnings = ds.parse_person(text, idx)
    if person is None:
        return None, warnings
    data = ds.extract_json(text)
    if isinstance(data, list):
        data = data[0]
    elif isinstance(data, dict) and isinstance(data.get("people"), list):
        data = data["people"][0]
    where = f"person {idx}"
    c2 = _opt_color(data.get("clothing_color_2"), warnings, where)
    if c2 is not None and c2 == person["clothing_color"]:
        warnings.append(f"{where}: color_2 equals color -> None")
        c2 = None
    person["clothing_color_2"] = c2
    person["confidence"] = _confidence(data.get("confidence"), warnings, where)
    return person, warnings


def parse_count(text: str):
    """R2 answer {"people_count": N} -> (int, warnings); (None, warnings) if unusable."""
    data = ds.extract_json(text)
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict) or "people_count" not in data:
        return None, ["no JSON with 'people_count' in the answer"]
    v = data["people_count"]
    if isinstance(v, bool):
        return None, [f"people_count {v!r} is not a number"]
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None, [f"people_count {v!r} is not a number"]
    if f != int(f) or f < 0:
        return None, [f"people_count {v!r} is not a non-negative integer"]
    return int(f), []


def parse_boxes(text: str, n_boxes: int):
    """R4 answer -> ({"people_count": int|None, "people": [person dicts by box]}, warnings).
    A box the model skipped is simply absent from people (the caller decides what that means)."""
    data = ds.extract_json(text)
    if isinstance(data, list):
        data = {"people": data}
    if not isinstance(data, dict) or not isinstance(data.get("people"), list):
        return None, ["no JSON with a 'people' list in the answer"]
    warnings = []
    people, seen = [], set()
    for pos, raw in enumerate(data["people"]):
        if not isinstance(raw, dict):
            warnings.append(f"skipped a non-object entry: {raw!r}")
            continue
        idx = raw.get("box_idx", pos)
        if isinstance(idx, bool) or not isinstance(idx, (int, float)) or int(idx) != idx:
            warnings.append(f"entry {pos}: bad box_idx {idx!r} -> used position {pos}")
            idx = pos
        idx = int(idx)
        if not 0 <= idx < n_boxes:
            warnings.append(f"entry {pos}: box_idx {idx} outside 0..{n_boxes - 1}, skipped")
            continue
        if idx in seen:
            warnings.append(f"entry {pos}: box_idx {idx} repeated, skipped")
            continue
        seen.add(idx)
        people.append(ds._make_person(raw, idx, warnings))
    people.sort(key=lambda p: p["person_idx"])
    count = None
    if "people_count" in data:
        count, w = parse_count(json.dumps({"people_count": data["people_count"]}))
        warnings.extend(w)
    return {"people_count": count, "people": people}, warnings


# ------------------------------------------------------------------ class ---

class VlmApiDescriber(Describer):
    """Arai's Describer contract (vlm/describers/base.py): describe(img_bgr) -> contract answer.
    The server runner uses describe_crop / describe_frame, which also return status, raw_response, tokens."""
    needs_full_body = True  # the VLM gets the whole person (full/<crop_id>.jpg), not the torso

    def __init__(self, model: str = ds.MODEL, api_url: str = ds.API_URL, mode: str = "R1",
                 prompt_variant: str = "A", disable_thinking: bool = False, timeout: int = ds.TIMEOUT_SEC,
                 retries: int = 1, fewshot_path=None, method_name: str = None, frequency_penalty: float = None,
                 repetition_penalty: float = None, max_tokens: int = None):
        mode = prompts.MODE_CODES.get(mode, mode)
        if mode not in prompts.MODES:
            raise ValueError(f"unknown mode {mode!r}; use R1..R4 or one of {prompts.MODES}")
        if prompt_variant not in prompts.VARIANTS:
            raise ValueError(f"unknown prompt variant {prompt_variant!r}; use one of {prompts.VARIANTS}")
        if mode != "crop" and prompt_variant != "A":
            raise ValueError("frame modes (R2/R3/R4) support prompt variant A only")
        self.model, self.api_url, self.mode = model, api_url, mode
        self.prompt_variant, self.disable_thinking = prompt_variant, disable_thinking
        self.timeout, self.retries = timeout, retries
        # None = not sent at all (the request stays exactly as before)
        self.extra = {k: v for k, v in (("frequency_penalty", frequency_penalty),
                                        ("repetition_penalty", repetition_penalty)) if v is not None}
        self.max_tokens = max_tokens
        self.method = method_name or f"vlm:{model}"
        self._fewshot = prompts.load_fewshot(fewshot_path) if (mode == "crop" and prompt_variant == "C") else None

    # ---- transport
    def _chat(self, messages: list, max_tokens: int) -> tuple:
        """-> (text without <think>, meta{prompt_tokens, completion_tokens, finish_reason, raw})"""
        payload = {"model": self.model, "temperature": 0,
                   "max_tokens": self.max_tokens if self.max_tokens is not None else max_tokens, "messages": messages}
        payload.update(self.extra)
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}  # NOT VERIFIED on the server
        req = urllib.request.Request(self.api_url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())
        choice = result["choices"][0]
        text = choice["message"]["content"] or ""
        usage = result.get("usage", {}) or {}
        return strip_thinking(text), {"prompt_tokens": usage.get("prompt_tokens"),
                                      "completion_tokens": usage.get("completion_tokens"),
                                      "finish_reason": choice.get("finish_reason"), "raw": text}

    @staticmethod
    def _user(text: str, image_url: str = None) -> dict:
        content = []
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}

    def _run(self, pipeline, parser, label: str) -> dict:
        """pipeline() -> (text_to_parse, meta) (may do several requests; meta as from _chat).
        parser(text) -> (value|None, warnings). Whole pipeline is retried `retries` times on errors;
        a truncated answer is NOT retried (temperature 0 gives the same answer again)."""
        out = {"status": "error", "value": None, "warnings": [], "raw_response": "", "error": None,
               "attempts": 0, "prompt_tokens": None, "completion_tokens": None, "finish_reason": None,
               "seconds": 0.0}
        t0 = time.perf_counter()
        for attempt in range(1 + self.retries):
            out["attempts"] = attempt + 1
            try:
                text, meta = pipeline()
            except Exception as e:  # network, HTTP error, bad server JSON, timeout
                out["error"] = f"request failed: {type(e).__name__}: {e}"
                print(f"  {label}[attempt {attempt + 1}] {out['error']}", file=sys.stderr)
                continue
            raw = meta["raw"]
            out.update(raw_response=raw, prompt_tokens=meta["prompt_tokens"],
                       completion_tokens=meta["completion_tokens"], finish_reason=meta["finish_reason"])
            if meta["finish_reason"] == "length":
                out.update(status="truncated", value=None, warnings=[],
                           error="answer cut at max_tokens (finish_reason=length); not parsed")
                print(f"  {label}[attempt {attempt + 1}] {out['error']}", file=sys.stderr)
                break
            value, warnings = parser(text)
            if value is None:
                out["error"] = f"cannot parse model answer: {'; '.join(warnings)}"
                print(f"  {label}[attempt {attempt + 1}] {out['error']}\n  raw: {raw!r}", file=sys.stderr)
                continue
            out.update(status="ok", value=value, warnings=warnings, error=None)
            break
        out["seconds"] = round(time.perf_counter() - t0, 3)
        return out

    @staticmethod
    def _join(m1: dict, m2: dict) -> dict:
        """Two requests (variant B) -> one meta: tokens summed, "length" if either step was cut."""
        out = {k: (m1.get(k) or 0) + (m2.get(k) or 0) for k in ("prompt_tokens", "completion_tokens")}
        out["finish_reason"] = "length" if "length" in (m1["finish_reason"], m2["finish_reason"]) else m2["finish_reason"]
        out["raw"] = f"STEP1: {m1['raw']}\nSTEP2: {m2['raw']}"
        return out

    # ---- R1
    def describe_crop(self, crop_data_url: str, person_idx: int = 0, label: str = "") -> dict:
        if self.mode != "crop":
            raise ValueError(f"describe_crop needs mode R1/crop, this describer is {self.mode}")
        v = self.prompt_variant

        if v == "B":
            def pipeline():
                t1, m1 = self._chat([self._user(prompts.PROMPT_B_STEP1, crop_data_url)], MAX_TOKENS_STEP1)
                if m1["finish_reason"] == "length":  # cut description: do not feed it to step 2
                    return t1, {**m1, "raw": f"STEP1: {m1['raw']}"}
                t2, m2 = self._chat([self._user(prompts.PROMPT_B_STEP2.replace("{description}", t1.strip()))],
                                    MAX_TOKENS_CROP)
                return t2, self._join(m1, m2)
            parser = lambda t: ds.parse_person(t, person_idx)
        else:
            if v == "C":
                msgs = []
                for url, answer in self._fewshot:
                    msgs.append(self._user(prompts.PROMPT_C, url))
                    msgs.append({"role": "assistant", "content": answer})
                msgs.append(self._user(prompts.PROMPT_C, crop_data_url))
            else:
                msgs = [self._user(prompts.PROMPT_D if v == "D" else prompts.PROMPT_A, crop_data_url)]

            def pipeline():
                return self._chat(msgs, MAX_TOKENS_CROP)
            parser = (lambda t: parse_crop_d(t, person_idx)) if v == "D" else (lambda t: ds.parse_person(t, person_idx))

        r = self._run(pipeline, parser, label)
        rec = self._record(r["value"]) if r["status"] == "ok" else None
        return self._result(r, record=rec)

    # ---- R2 / R3 / R4
    def describe_frame(self, frame_path: str, boxes: list = None) -> dict:
        if self.mode == "crop":
            raise ValueError("describe_frame needs mode R2/R3/R4, this describer is R1/crop")
        try:
            with open(frame_path, "rb") as f:
                mime = mimetypes.guess_type(frame_path)[0] or "image/jpeg"
                url = f"data:{mime};base64,{base64.b64encode(f.read()).decode('ascii')}"
        except OSError as e:
            return {"status": "error", "mode": self.mode, "people_count": None, "records": [], "warnings": [],
                    "raw_response": "", "error": f"cannot read frame: {e}", "attempts": 0,
                    "prompt_tokens": None, "completion_tokens": None, "finish_reason": None, "seconds": 0.0}

        if self.mode == "boxes":
            boxes = list(boxes or [])
            w = h = None
            try:
                import imgutil
                w, h = imgutil.image_size(imgutil.load_image(frame_path))
            except Exception:  # no OpenCV/Pillow: the size is only a hint in the prompt
                pass
            prompt, max_tokens = prompts.prompt_r4(boxes, w, h), MAX_TOKENS_BOXES
            parser = lambda t: parse_boxes(t, len(boxes))
        elif self.mode == "frame-list":
            prompt, max_tokens, parser = prompts.PROMPT_R3, MAX_TOKENS_FRAME, ds.parse_people
        else:
            prompt, max_tokens, parser = prompts.PROMPT_R2, MAX_TOKENS_COUNT, parse_count

        def pipeline():
            return self._chat([self._user(prompt, url)], max_tokens)

        r = self._run(pipeline, parser, f"{frame_path} ")
        res = self._result(r)
        res.update(mode=self.mode, people_count=None, records=[])
        if r["status"] == "ok":
            v = r["value"]
            if self.mode == "frame-count":
                res["people_count"] = v
            elif self.mode == "frame-list":
                res["people_count"] = len(v)
                res["records"] = [self._record(p) for p in v]
            else:
                res["people_count"] = v["people_count"]
                if v["people_count"] is None:
                    res["warnings"].append("model gave no people_count; not replaced by the number of boxes")
                res["records"] = [self._record(p) for p in v["people"]]
                for rec, p in zip(res["records"], v["people"]):
                    rec["person_idx"] = p["person_idx"]
        return res

    # ---- helpers
    def _record(self, p: dict) -> dict:
        return make_answer(color=p["clothing_color"], color_2=p.get("clothing_color_2"), type=p["clothing_type"],
                           confidence=p.get("confidence"), description=p.get("description", "") or "",
                           method=self.method, mode=self.mode, prompt_variant=self.prompt_variant)

    def describe(self, img_bgr) -> dict:
        """Describer.describe: BGR crop (numpy, as cv2.imread gives) -> one contract answer (mode R1 only).
        A not-ok answer becomes color/type "unknown" with status, finish_reason and error in extras."""
        import cv2
        h, w = img_bgr.shape[:2]
        if max(h, w) < MIN_CROP_SIDE:  # the same upscaling as imgutil.crop_data_url in the server runner
            k = MIN_CROP_SIDE / max(h, w)
            img_bgr = cv2.resize(img_bgr, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise ValueError("cannot encode the crop as JPEG")
        r = self.describe_crop("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode())
        if r["status"] == "ok":
            return r["record"]
        return make_answer(confidence=None, method=self.method, mode=self.mode, prompt_variant=self.prompt_variant,
                           status=r["status"], finish_reason=r["finish_reason"], error=r["error"])

    @staticmethod
    def _result(r: dict, record=None) -> dict:
        return {"status": r["status"], "record": record, "warnings": r["warnings"],
                "raw_response": r["raw_response"], "error": r["error"], "attempts": r["attempts"],
                "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                "finish_reason": r["finish_reason"], "seconds": r["seconds"]}
