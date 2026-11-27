"""Checks vlm/fewshot.json for prompt variant C: exactly 3 examples, images exist, values are in the vocabulary,
each example names its source crop (so the benchmark can EXCLUDE it from scoring).

    python3 vlm/check_fewshot.py [path/to/fewshot.json]

Example format:
 {"image": "fewshot/sn_0003_5.jpg", "clothing_color": "red", "clothing_type": "sweater", "description": "...",
  "source": {"video_id": "...", "snapshot_id": "sn_0003", "person_idx": 5}}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import describe_snapshot as ds  # noqa: E402

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("fewshot.json")
problems = []
try:
    ex = json.loads(path.read_text(encoding="utf-8")).get("examples", [])
except (OSError, ValueError) as e:
    sys.exit(f"cannot read {path}: {e}")
print(f"{path}: {len(ex)} examples (need exactly 3)")
if len(ex) != 3:
    problems.append(f"{len(ex)} examples instead of 3")
for i, e in enumerate(ex):
    img = str(e.get("image", ""))
    if not img.startswith("data:") and not (path.parent / img).is_file():
        problems.append(f"example {i}: image file not found: {img!r}")
    if e.get("clothing_color") not in ds.COLORS or e.get("clothing_type") not in ds.TYPES:
        problems.append(f"example {i}: color/type not in the vocabulary")
    s = e.get("source") or {}
    if not all(k in s for k in ("video_id", "snapshot_id", "person_idx")):
        problems.append(f"example {i}: no 'source' (video_id, snapshot_id, person_idx): it could not be excluded from scoring")
for p in problems:
    print("PROBLEM:", p)
print("OK" if not problems else f"{len(problems)} problem(s)")
sys.exit(1 if problems else 0)
