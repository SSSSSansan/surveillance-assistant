"""Common answer contract for every clothing method (VLM now; CV baselines of Arai later).

    {"color": "red", "color_2": "brown", "type": "sweater", "confidence": 0.72,
     "description": "red sweater, standing", "method": "vlm:qwen3-vl-4b", "mode": "crop",
     "prompt_variant": "A"}

color_2 and confidence are None when the prompt variant does not ask for them (A, B, C).
Vocabulary is the stage-2 one (14 colors, 12 types, unknown); it is imported, not copied.
"""
import describe_snapshot as ds

COLORS = ds.COLORS
TYPES = ds.TYPES
CONTRACT_KEYS = ("color", "color_2", "type", "confidence", "description", "method", "mode",
                 "prompt_variant")


def make_record(color, type_, method, mode, prompt_variant, color_2=None, confidence=None,
                description="") -> dict:
    return {"color": color, "color_2": color_2, "type": type_, "confidence": confidence,
            "description": description or "", "method": method, "mode": mode,
            "prompt_variant": prompt_variant}


def validate_record(rec: dict) -> list:
    """List of problems (empty = ok). Used by tests and by the benchmarks as a guard."""
    problems = []
    for k in CONTRACT_KEYS:
        if k not in rec:
            problems.append(f"missing key {k}")
    if problems:
        return problems
    if rec["color"] not in COLORS:
        problems.append(f"color {rec['color']!r} not in vocabulary")
    if rec["color_2"] is not None and rec["color_2"] not in COLORS:
        problems.append(f"color_2 {rec['color_2']!r} not in vocabulary")
    if rec["type"] not in TYPES:
        problems.append(f"type {rec['type']!r} not in vocabulary")
    c = rec["confidence"]
    if c is not None and not (isinstance(c, (int, float)) and not isinstance(c, bool) and 0 <= c <= 1):
        problems.append(f"confidence {c!r} not in 0..1")
    return problems


class BaseDescriber:
    """A method that turns an image into contract records. Subclasses implement the two calls.

    describe_crop(crop_data_url, person_idx) -> {"status": "ok"|"error", "record": dict|None, ...}
    describe_frame(frame_path, boxes)        -> {"status", "people_count": int|None, "records": [...], ...}
    Both results also carry: warnings, raw_response, error, attempts, prompt_tokens,
    completion_tokens, seconds.
    """
    method = "base"
    mode = None
    prompt_variant = "A"

    def describe_crop(self, crop_data_url: str, person_idx: int = 0) -> dict:
        raise NotImplementedError

    def describe_frame(self, frame_path: str, boxes: list = None) -> dict:
        raise NotImplementedError
