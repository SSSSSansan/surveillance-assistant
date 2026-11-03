"""Run describe_clip.py (A and B) over all clips in events.json that need VLM.

Usage:
    python batch_describe.py EVENTS_JSON CLIPS_DIR OUTPUT_JSON
"""
import argparse
import json
import sys
from pathlib import Path

from describe_clip import describe_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("events_json")
    parser.add_argument("clips_dir")
    parser.add_argument("output_json")
    args = parser.parse_args()

    data = json.load(open(args.events_json, encoding="utf-8"))
    clips_dir = Path(args.clips_dir)

    todo = [e for e in data["events"] if e.get("needs_vlm")]
    print(f"Events needing VLM: {len(todo)}", file=sys.stderr)

    for i, event in enumerate(todo, 1):
        clip_name = Path(event["clip_path"].replace("\\", "/")).name
        clip_path = clips_dir / clip_name

        if not clip_path.exists():
            print(f"[{i}/{len(todo)}] MISSING: {clip_path}", file=sys.stderr)
            event["vlm_status"] = "error"
            continue

        print(f"[{i}/{len(todo)}] {clip_name}", file=sys.stderr)

        # Способ A
        res_a = describe_clip(str(clip_path))
        # Способ B
        res_b = describe_clip(
            str(clip_path),
            track_id=event.get("track_id"),
            zone=event.get("zone_id"),
            time_hint=event.get("t_start"),
        )

        # Записываем B как основной результат (он с метаданными),
        # A сохраняем отдельно для сравнения.
        event["vlm_description"] = res_b["vlm_description"]
        event["vlm_description_a"] = res_a["vlm_description"]
        event["prompt_variant"] = "B"
        event["vlm_status"] = "confirmed"

    json.dump(data, open(args.output_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved: {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()