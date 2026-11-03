"""Send one clip to the local vLLM server and get a description.

Usage:
    python describe_clip.py CLIP_PATH [--track-id N] [--zone ZONE] [--time T]

--track-id/--zone/--time together turn on prompt variant B (clip + metadata).
Without them, it's variant A (clip only).
"""
import argparse
import json
import urllib.request

API_URL = "http://127.0.0.1:8765/v1/chat/completions"
MODEL = "qwen3-vl-4b"


def describe_clip(clip_path: str, track_id=None, zone=None, time_hint=None) -> dict:
    prompt_variant = "A"
    text = "Describe what happens in this video."

    if track_id is not None or zone is not None or time_hint is not None:
        prompt_variant = "B"
        parts = []
        if track_id is not None:
            parts.append(f"person track ID {track_id}")
        if zone is not None:
            parts.append(f"zone '{zone}'")
        if time_hint is not None:
            parts.append(f"around {time_hint} seconds into the video")
        meta = ", ".join(parts)
        text = (
            f"This clip shows {meta}. Describe what happens, "
            "focusing on this person and whether they picked up or put down an object."
        )

    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": f"file://{clip_path}"}},
                {"type": "text", "text": text},
            ],
        }],
    }

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())

    description = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})

    return {
        "prompt_variant": prompt_variant,
        "vlm_description": description,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_path")
    parser.add_argument("--track-id", type=int, default=None)
    parser.add_argument("--zone", default=None)
    parser.add_argument("--time", type=float, default=None)
    args = parser.parse_args()

    out = describe_clip(args.clip_path, args.track_id, args.zone, args.time)
    print(json.dumps(out, indent=2, ensure_ascii=False))
