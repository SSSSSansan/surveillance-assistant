"""Describe the clothing of people in a snapshot frame with the local vLLM server.

Two modes (see batch_describe_snapshots.py for the full-pipeline use):
  * frame : the whole frame in one request -> list of people.  Poor on crowded frames
            (miscounts people, says "black" for almost everyone). Kept for small scenes.
  * crops : one request per person box (boxes come from Arai's YOLO) -> one person each.
            person_idx = index of the box. This is the main mode.

Colors and types come from FIXED vocabularies (search by clothing in the agent works on
exact values, so "crimson pullover" must not reach the database). Borderline colors are
mapped INTO the current vocabulary (navy -> blue, burgundy -> red, khaki -> beige, ...);
the rule is in the prompt AND applied again in code as a safety net (logged as "mapped").

Usage (debug, one frame or one box):
    python vlm/describe_snapshot.py FRAME_PATH                      # whole frame
    python vlm/describe_snapshot.py FRAME_PATH --box 318,377,518,694  # one person crop

Result dict: status "ok"|"error", people (frame) or person (crop), warnings,
raw_response (always kept for the log), error, attempts, prompt_tokens, completion_tokens.

NOT VERIFIED without the GPU server: base64 image_url for this vLLM; whether upscaling
tiny crops helps.
"""
import argparse
import base64
import json
import mimetypes
import sys
import urllib.request

import imgutil

API_URL = "http://127.0.0.1:8765/v1/chat/completions"
MODEL = "qwen3-vl-4b"
TIMEOUT_SEC = 180
MAX_TOKENS = 1024
MAX_TOKENS_CROP = 256

COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
          "black", "white", "gray", "beige", "multicolor", "unknown")
TYPES = ("sweater", "hoodie", "t-shirt", "shirt", "jacket", "coat", "dress",
         "pants_only", "vest", "uniform", "other", "unknown")

# Spelling variants (silent) ...
COLOR_ALIASES = {"grey": "gray", "violet": "purple", "multi-color": "multicolor",
                 "multi color": "multicolor", "multicoloured": "multicolor",
                 "multicolored": "multicolor"}
TYPE_ALIASES = {"tshirt": "t-shirt", "t shirt": "t-shirt", "tee": "t-shirt",
                "t_shirt": "t-shirt", "pants only": "pants_only",
                "pants-only": "pants_only", "hooded sweatshirt": "hoodie"}
# ... and borderline values mapped INTO the vocabulary (logged as "mapped").
COLOR_MAP = {
    "navy": "blue", "navy blue": "blue", "dark blue": "blue", "dark navy": "blue",
    "light blue": "blue", "sky blue": "blue", "teal": "blue", "turquoise": "blue",
    "burgundy": "red", "maroon": "red", "dark red": "red", "wine": "red", "crimson": "red",
    "khaki": "beige", "tan": "beige", "sand": "beige", "cream": "beige", "camel": "beige",
    "olive": "green", "olive green": "green", "army green": "green", "dark green": "green",
    "light green": "green", "silver": "gray", "dark gray": "gray", "light gray": "gray",
    "charcoal": "gray", "golden": "yellow", "lavender": "purple",
}
TYPE_MAP = {
    "sweatshirt": "sweater", "pullover": "sweater", "cardigan": "sweater", "jumper": "sweater",
    "knit": "sweater", "puffer": "jacket", "puffer jacket": "jacket", "down jacket": "jacket",
    "windbreaker": "jacket", "parka": "jacket", "bomber": "jacket", "bomber jacket": "jacket",
    "blazer": "jacket", "overcoat": "coat", "raincoat": "coat", "trench coat": "coat",
    "polo": "shirt", "blouse": "shirt", "button-up": "shirt", "tank top": "t-shirt",
    "top": "other", "backpack": "other",
}

COLOR_RULES = (
    "Color rules (answer only with the allowed color values): "
    "navy, dark blue, light blue, sky blue -> blue; "
    "burgundy, maroon, dark red -> red; "
    "khaki, tan, sand, cream -> beige; "
    "olive, army green, dark green -> green; "
    "silver, dark gray, charcoal -> gray. "
    "If a color is so dark that you cannot tell navy from black, answer black. "
    "Write the exact shade in the description (for example 'dark navy puffer jacket')."
)
TYPE_RULES = (
    "Type rules: sweatshirt, pullover, cardigan, jumper -> sweater; "
    "puffer, down jacket, windbreaker, parka, bomber, blazer -> jacket; "
    "overcoat, raincoat -> coat; polo, blouse -> shirt; tank top -> t-shirt."
)
# Person seen from behind: the clothing on the back is still visible, so describe it.
# Set to True to force "unknown" for everyone turned away (the older, stricter plan wording).
BACK_TURNED_IS_UNKNOWN = False
BACK_RULE = ("If the person is turned away, still describe the clothing you can see on the back; "
             "use unknown only if you really cannot tell."
             if not BACK_TURNED_IS_UNKNOWN else
             "If the person is turned away so the front of the clothing is not visible, use unknown.")

_VOCAB = (f"Allowed clothing_color values ONLY: {', '.join(COLORS)}.\n"
          f"Allowed clothing_type values ONLY: {', '.join(TYPES)}.")

FRAME_PROMPT = f"""You are given one frame from a surveillance camera.
List EVERY person visible in the frame (also partially visible ones), ordered from LEFT to RIGHT in the image.
For each person give:
- "clothing_color": the main color of the UPPER-BODY clothing (the outermost layer on the torso, not trousers, not a cap).
- "clothing_type": the type of that upper-body clothing.
- "description": one short English sentence (at most 15 words): clothing and what the person is doing / where they stand.
{_VOCAB}
Rules:
- If you are not sure (far away, dark, blurred, hidden by an object) use "unknown". Do not guess.
- {BACK_RULE}
- If only the legs are visible use "pants_only" for the type and "unknown" for the color.
- Use "multicolor" only if no single color dominates.
- Do not invent details that are not visible.
- If there are no people, return an empty list.
{COLOR_RULES}
{TYPE_RULES}
Answer with strict JSON and nothing else, in exactly this form:
{{"people": [{{"clothing_color": "...", "clothing_type": "...", "description": "..."}}]}}"""

CROP_PROMPT = f"""This image is a crop of ONE person from a surveillance camera frame.
Describe ONLY the person in the CENTER of the crop. The person may be partly cut off, and parts of other people may be visible at the edges: ignore them.
Give:
- "clothing_color": the main color of the UPPER-BODY clothing, i.e. the OUTERMOST layer on the torso (jacket, coat, hoodie, sweater, shirt, t-shirt). NOT the trousers, NOT a cap or hat, NOT a bag or backpack.
- "clothing_type": the type of that upper-body clothing.
- "description": one short English sentence (at most 15 words) about the clothing and the pose or action.
{_VOCAB}
Rules:
- If the torso is not visible, or the image is too small, dark or blurred to judge, use "unknown". Do not guess.
- {BACK_RULE}
- If only legs are visible use "pants_only" for the type and "unknown" for the color.
- Use "multicolor" only if no single color dominates.
- Do not invent details that are not visible.
{COLOR_RULES}
{TYPE_RULES}
Answer with strict JSON and nothing else, in exactly this form:
{{"clothing_color": "...", "clothing_type": "...", "description": "..."}}"""


# ---------------------------------------------------------------- request ---

def call_vlm(data_url: str, prompt: str, api_url: str = API_URL, model: str = MODEL,
             timeout: int = TIMEOUT_SEC, max_tokens: int = MAX_TOKENS) -> tuple:
    """One request to vLLM. Returns (text, usage_dict). Raises on HTTP/network errors."""
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    text = result["choices"][0]["message"]["content"] or ""
    return text, result.get("usage", {}) or {}


# ---------------------------------------------------------------- parsing ---

def extract_json(text: str):
    """First valid JSON object (or list of objects) in text: plain, in ```json fences,
    or surrounded by other words. Returns the parsed value or None."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(value, dict) or (isinstance(value, list) and
                                           all(isinstance(x, dict) for x in value)):
                return value
    return None


def _norm(value) -> str:
    return " ".join(str(value).strip().lower().split()) if value is not None else ""


def _normalize(value, aliases, mapping, vocab, missing_fallback, outside_fallback,
               field, warnings, where):
    v = _norm(value)
    if not v:
        warnings.append(f"{where}: {field} missing -> {missing_fallback}")
        return missing_fallback
    v = aliases.get(v, v)
    if v in vocab:
        return v
    if v in mapping:
        warnings.append(f"{where}: {field} {value!r} mapped -> {mapping[v]}")
        return mapping[v]
    warnings.append(f"{where}: {field} {value!r} not in vocabulary -> {outside_fallback}")
    return outside_fallback


def normalize_color(value, warnings: list, where: str) -> str:
    return _normalize(value, COLOR_ALIASES, COLOR_MAP, COLORS, "unknown", "unknown",
                      "clothing_color", warnings, where)


def normalize_type(value, warnings: list, where: str) -> str:
    return _normalize(value, TYPE_ALIASES, TYPE_MAP, TYPES, "unknown", "other",
                      "clothing_type", warnings, where)


def _make_person(raw: dict, idx: int, warnings: list) -> dict:
    where = f"person {idx}"
    desc = raw.get("description")
    desc = " ".join(str(desc).split()) if desc not in (None, "") else ""
    if not desc:
        warnings.append(f"{where}: description missing")
    return {
        "person_idx": idx,
        "clothing_color": normalize_color(raw.get("clothing_color"), warnings, where),
        "clothing_type": normalize_type(raw.get("clothing_type"), warnings, where),
        "description": desc,
    }


def parse_people(text: str):
    """Whole-frame answer -> (people, warnings). (None, warnings) if unparseable."""
    warnings = []
    data = extract_json(text)
    if isinstance(data, dict):
        raw_people = data.get("people")
    elif isinstance(data, list):
        raw_people = data
        warnings.append("model returned a bare list instead of {'people': [...]}")
    else:
        return None, ["no JSON found in the answer"]
    if not isinstance(raw_people, list):
        return None, ["JSON has no 'people' list"]
    people = []
    for raw in raw_people:
        if not isinstance(raw, dict):
            warnings.append(f"skipped a non-object entry in people: {raw!r}")
            continue
        people.append(_make_person(raw, len(people), warnings))
    return people, warnings


def parse_person(text: str, idx: int = 0):
    """Crop answer -> (person, warnings). Also accepts {'people':[{...}]} / [{...}] (takes the first).
    (None, warnings) if there is no object with clothing fields."""
    warnings = []
    data = extract_json(text)
    if isinstance(data, list):
        data = data[0] if data else None
        warnings.append("model returned a list instead of one object (took the first)")
    elif isinstance(data, dict) and isinstance(data.get("people"), list):
        data = data["people"][0] if data["people"] else None
        warnings.append("model returned {'people': [...]} for a crop (took the first)")
    if not isinstance(data, dict):
        return None, ["no JSON object found in the answer"]
    if not any(k in data for k in ("clothing_color", "clothing_type", "description")):
        return None, ["JSON object has none of clothing_color / clothing_type / description"]
    return _make_person(data, idx, warnings), warnings


# ------------------------------------------------------------------- main ---

def _ask(data_url: str, prompt: str, parser, api_url, model, timeout, retries, max_tokens, label):
    """Request + parse with `retries` extra attempts. parser(text) -> (value|None, warnings)."""
    out = {"status": "error", "value": None, "warnings": [], "raw_response": "",
           "error": None, "attempts": 0, "prompt_tokens": None, "completion_tokens": None}
    for attempt in range(1 + retries):
        out["attempts"] = attempt + 1
        try:
            text, usage = call_vlm(data_url, prompt, api_url, model, timeout, max_tokens)
        except Exception as e:  # network, HTTP 4xx/5xx, bad JSON from the server, timeout
            out["error"] = f"request failed: {type(e).__name__}: {e}"
            print(f"  {label}[attempt {attempt + 1}] {out['error']}", file=sys.stderr)
            continue
        out["raw_response"] = text
        out["prompt_tokens"] = usage.get("prompt_tokens")
        out["completion_tokens"] = usage.get("completion_tokens")
        value, warnings = parser(text)
        if value is None:
            out["error"] = f"cannot parse model answer: {'; '.join(warnings)}"
            print(f"  {label}[attempt {attempt + 1}] {out['error']}\n  raw: {text!r}", file=sys.stderr)
            continue
        out.update(status="ok", value=value, warnings=warnings, error=None)
        return out
    return out


def describe_snapshot(frame_path: str, api_url: str = API_URL, model: str = MODEL,
                      timeout: int = TIMEOUT_SEC, retries: int = 1) -> dict:
    """Mode 'frame': whole frame -> {"status","people",...}. One retry on error/unparseable answer."""
    res = {"status": "error", "people": [], "warnings": [], "raw_response": "",
           "error": None, "attempts": 0, "prompt_tokens": None, "completion_tokens": None}
    try:
        with open(frame_path, "rb") as f:  # whole frame as-is, original bytes
            mime = mimetypes.guess_type(frame_path)[0] or "image/jpeg"
            data_url = f"data:{mime};base64,{base64.b64encode(f.read()).decode('ascii')}"
    except OSError as e:
        res["error"] = f"cannot read frame: {e}"
        return res
    r = _ask(data_url, FRAME_PROMPT, parse_people, api_url, model, timeout, retries, MAX_TOKENS, "")
    res.update({k: r[k] for k in ("status", "warnings", "raw_response", "error", "attempts",
                                  "prompt_tokens", "completion_tokens")})
    if r["status"] == "ok":
        res["people"] = r["value"]
    return res


def describe_person_crop(crop_data_url: str, person_idx: int = 0, api_url: str = API_URL,
                         model: str = MODEL, timeout: int = TIMEOUT_SEC, retries: int = 1,
                         label: str = "") -> dict:
    """Mode 'crops': one person crop (data URL) -> {"status","person",...}.
    person["person_idx"] is set to person_idx."""
    r = _ask(crop_data_url, CROP_PROMPT, lambda t: parse_person(t, person_idx), api_url, model,
             timeout, retries, MAX_TOKENS_CROP, label)
    return {"status": r["status"], "person": r["value"], "warnings": r["warnings"],
            "raw_response": r["raw_response"], "error": r["error"], "attempts": r["attempts"],
            "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_path")
    parser.add_argument("--box", default=None, help="x1,y1,x2,y2: describe ONE person crop instead of the whole frame")
    parser.add_argument("--crop-pad", type=float, default=0.12)
    parser.add_argument("--min-crop-side", type=int, default=336)
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SEC)
    args = parser.parse_args()

    if args.box:
        img = imgutil.load_image(args.frame_path)
        url = imgutil.crop_data_url(img, [float(v) for v in args.box.split(",")],
                                    args.crop_pad, args.min_crop_side)
        if url is None:
            sys.exit("invalid box")
        res = describe_person_crop(url, 0, args.api_url, args.model, args.timeout)
    else:
        res = describe_snapshot(args.frame_path, args.api_url, args.model, args.timeout)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    sys.exit(0 if res["status"] == "ok" else 1)
