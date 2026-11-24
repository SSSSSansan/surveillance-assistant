"""All prompts for the stage-3 benchmark in one place.

Variants for ONE-PERSON CROPS (mode R1 'crop'):
  A  current stage-2 prompt, text NOT changed (imported from describe_snapshot.CROP_PROMPT)
  B  two steps: (1) free-text description of the clothing, (2) choose values from the vocabulary
     using ONLY that description (the second request carries no image - NOT VERIFIED that this is the best
     variant of "two steps"; the alternative is to resend the image in step 2)
  C  few-shot: 3 labelled crops from vlm/fewshot.json as previous chat turns, then the same text as A
  D  as A, plus "clothing_color_2" (second most likely color) and "confidence" 0..1
Frame modes (R2 frame-count, R3 frame-list, R4 boxes) use variant A only.

The vocabulary is NOT changed: 14 colors, 12 types, unknown (imported from describe_snapshot).
"""
import base64
import json
import mimetypes
from pathlib import Path

import describe_snapshot as ds

VARIANTS = ("A", "B", "C", "D")
MODES = ("crop", "frame-count", "frame-list", "boxes")
MODE_CODES = {"R1": "crop", "R2": "frame-count", "R3": "frame-list", "R4": "boxes"}

FEWSHOT_PATH = Path(__file__).with_name("fewshot.json")

# ---------------------------------------------------------------- crops ---
PROMPT_A = ds.CROP_PROMPT  # unchanged on purpose: baseline of stage 2

PROMPT_B_STEP1 = """This image is a crop of ONE person from a surveillance camera frame.
Look ONLY at the person in the CENTER of the crop (ignore parts of other people at the edges).
In 1-3 plain English sentences describe the clothing on the UPPER BODY: the outermost layer on the torso,
its color(s) and what kind of garment it is, and whether you see the person from the front or from the back.
If you cannot see the torso or the image is too small, dark or blurred, say so. Do not guess. Do not use JSON."""

PROMPT_B_STEP2 = f"""Here is a description of the upper-body clothing of one person:
\"\"\"{{description}}\"\"\"
Convert it to fixed values. Use ONLY what the description says.
{ds._VOCAB}
Rules:
- If the description says the clothing cannot be judged, use "unknown". Do not guess.
- {ds.BACK_RULE}
- If only legs are visible use "pants_only" for the type and "unknown" for the color.
- Use "multicolor" only if no single color dominates.
{ds.COLOR_RULES}
{ds.TYPE_RULES}
Answer with strict JSON and nothing else, in exactly this form:
{{"clothing_color": "...", "clothing_type": "...", "description": "..."}}"""

# C: same instruction text as A; the examples are added as earlier chat turns
PROMPT_C = PROMPT_A

PROMPT_D = f"""This image is a crop of ONE person from a surveillance camera frame.
Describe ONLY the person in the CENTER of the crop. The person may be partly cut off, and parts of other people may be visible at the edges: ignore them.
Give:
- "clothing_color": the main color of the UPPER-BODY clothing, i.e. the OUTERMOST layer on the torso (jacket, coat, hoodie, sweater, shirt, t-shirt). NOT the trousers, NOT a cap or hat, NOT a bag or backpack.
- "clothing_color_2": your SECOND most likely color for that clothing (a different allowed value), or null if you are sure.
- "clothing_type": the type of that upper-body clothing.
- "confidence": a number from 0 to 1: how sure you are about clothing_color.
- "description": one short English sentence (at most 15 words) about the clothing and the pose or action.
{ds._VOCAB}
Rules:
- If the torso is not visible, or the image is too small, dark or blurred to judge, use "unknown". Do not guess.
- {ds.BACK_RULE}
- If only legs are visible use "pants_only" for the type and "unknown" for the color.
- Use "multicolor" only if no single color dominates.
- Do not invent details that are not visible.
{ds.COLOR_RULES}
{ds.TYPE_RULES}
Answer with strict JSON and nothing else, in exactly this form:
{{"clothing_color": "...", "clothing_color_2": "...", "clothing_type": "...", "confidence": 0.0, "description": "..."}}"""

# --------------------------------------------------------------- frames ---
# Counting rule = the labelling rule of the gold standard (STAGE3_PLAN section 2).
COUNT_RULE = ("Count a person if the head and part of the torso are visible, even if the person is partly hidden; "
              "count people standing in doorways and at the edge of the frame; "
              "do NOT count reflections, and do NOT count people in photos or posters on the walls.")

PROMPT_R2 = f"""You are given one frame from a surveillance camera.
How many people are visible in the frame?
{COUNT_RULE}
Answer with strict JSON and nothing else, in exactly this form:
{{"people_count": 0}}"""

PROMPT_R3 = ds.FRAME_PROMPT  # unchanged: stage-2 whole-frame prompt (list of people + clothing)


def prompt_r4(boxes: list, width: int = None, height: int = None) -> str:
    """Whole frame + the detector's boxes as TEXT. The model must still give its OWN people_count
    (the detector can miss people or draw two boxes for one person)."""
    size = f" The frame is {width}x{height} pixels." if width and height else ""
    lines = "\n".join(f"{i}: [{', '.join(str(int(round(float(v)))) for v in b)}]" for i, b in enumerate(boxes))
    return f"""You are given one frame from a surveillance camera.{size}
A person detector found these boxes (index: [x1, y1, x2, y2] in pixels, origin = top-left corner). It can miss people or draw two boxes around one person, so do not trust it blindly.
{lines or '(no boxes)'}
Do two things:
1. Count the people in the frame yourself ("people_count"). {COUNT_RULE}
2. For EVERY box above, in the same order, give the clothing of the person inside it:
- "box_idx": the index of the box.
- "clothing_color": the main color of the UPPER-BODY clothing (outermost layer on the torso, not trousers, not a cap).
- "clothing_type": the type of that clothing.
- "description": one short English sentence (at most 15 words).
{ds._VOCAB}
Rules:
- If you are not sure (far away, dark, blurred, hidden) use "unknown". Do not guess.
- {ds.BACK_RULE}
- If only the legs are visible use "pants_only" for the type and "unknown" for the color.
{ds.COLOR_RULES}
{ds.TYPE_RULES}
Answer with strict JSON and nothing else, in exactly this form:
{{"people_count": 0, "people": [{{"box_idx": 0, "clothing_color": "...", "clothing_type": "...", "description": "..."}}]}}"""


# -------------------------------------------------------------- few-shot ---
def _data_url_from_file(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def load_fewshot(path=None) -> list:
    """Read vlm/fewshot.json -> list of (image_data_url, answer_json_text).
    Format: {"examples": [{"image": "<file path relative to the json OR a data: URL>",
                           "clothing_color": "...", "clothing_type": "...", "description": "..."}]}
    Raises ValueError if there are no examples: variant C must never silently run as variant A."""
    p = Path(path) if path else FEWSHOT_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read few-shot file {p}: {e}")
    examples = data.get("examples") if isinstance(data, dict) else None
    if not examples:
        raise ValueError(f"{p} has no examples: fill it with labelled crops before using prompt variant C")
    out = []
    for i, ex in enumerate(examples):
        img = str(ex.get("image", ""))
        if not img:
            raise ValueError(f"few-shot example {i}: no 'image'")
        url = img if img.startswith("data:") else _data_url_from_file(p.parent / img)
        answer = {k: ex.get(k) for k in ("clothing_color", "clothing_type", "description")}
        out.append((url, json.dumps(answer, ensure_ascii=False)))
    return out
