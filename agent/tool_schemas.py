
"""Схемы инструментов для модели и безопасный вызов функций из agent/tools.py.

Модели отдаются только шесть обычных функций. Отладочные debug_*,
параметры vlm_variant, view, conn, embedder, model и min_score модели
НЕ показываются и вызвать их через call_tool нельзя.
"""

import json

from agent import tools

_EVENT_TYPES = list(tools.EVENT_TYPES)


def _p(type_, description, **extra):
    d = {"type": type_, "description": description}
    d.update(extra)
    return d


_FILTERS = {
    "video_id": _p(
        "string",
        "Video identifier, e.g. 'test_video_01'. "
        "Use this to filter events by video."
    ),
    "camera_id": _p(
        "string",
        "Camera identifier, e.g. 'test_cam01'. "
        "Use ONLY for camera filtering, never for video IDs."
    ),
    "event_type": _p(
        "string",
        "Event type. enter/exit: a person enters/leaves the scene; "
        "approach_zone/leave_zone/dwell_in_zone: movement relative to a zone; "
        "pick_up/put_down: a person takes/puts an object; "
        "abandoned_object: an object is left unattended; "
        "pickup_or_put_candidate: CV candidate for a possible pickup or put-down. "
        "Use this exact spelling; do not invent variants such as pick_up_or_put_candidate.",
        enum=_EVENT_TYPES
    ),
    "zone_id": _p(
        "string",
        "Zone id, e.g. 'door' or 'desk'. Call list_zones_cameras if unsure."
    ),
    "track_id": _p(
        "integer",
        "Person (track) id, unique only inside one video."
    ),
    "t_from": _p(
        "number",
        "Window start, seconds from the start of the video. "
        "Events that overlap the window match."
    ),
    "t_to": _p(
        "number",
        "Window end, seconds from the start of the video."
    ),
}


def _spec(name, description, properties, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required)
            }
        }
    }


SPECS = [
    _spec(
        "query_events",
        "Find confirmed events in the video event journal, sorted by time. "
        "Use it for 'was there...', 'when...' and for listing events. "
        "Never use it to count: use count_events. "
        "Returns an empty list when nothing is found. "
        "For a request containing multiple event types joined by OR (for example pick_up OR put_down), "
        "make one query_events call per event_type; never invent a combined event_type. "
        "If the user provides video_id but no camera_id, pass only video_id.",
        {
            **_FILTERS,
            "order": _p(
                "string",
                "Sort by time: 'asc' (default) or 'desc'.",
                enum=["asc", "desc"]
            ),
            "limit": _p(
                "integer",
                "Maximum number of events (default 50, maximum 200)."
            )
        }
    ),

    _spec(
        "count_events",
        "Count confirmed events that match the filters. "
        "Always use it for 'how many' questions; never count events yourself. "
        "For multiple event types joined by OR, make one count_events call per event_type "
        "and report the separate counts (do not invent a combined type). "
        "If the user provides video_id but no camera_id, pass only video_id.",
        dict(_FILTERS)
    ),

    _spec(
        "get_track_timeline",
        "All confirmed events of one person (track) in time order. "
        "Use it for 'what did person N do' and 'what happened after...'. "
        "If the same track id exists in several videos, pass video_id.",
        {
            "track_id": _p("integer", "Person (track) id."),
            "video_id": _p("string", "Video identifier, e.g. 'test_video_01'."),
            "camera_id": _p("string", "Camera identifier, e.g. 'test_cam01'.")
        },
        required=["track_id"]
    ),

    _spec(
        "get_event_details",
        "Details of one event by id: type, person, zone, times, "
        "clip path and the text description.",
        {
            "event_id": _p("string", "Event id, e.g. 'e0004'."),
            "video_id": _p(
                "string",
                "Video id; required if the event id exists in several videos."
            )
        },
        required=["event_id"]
    ),

    _spec(
        "list_zones_cameras",
        "List cameras, their videos and zones. "
        "Use it to check which cameras and zones exist.",
        {
            "camera_id": _p(
                "string",
                "Camera identifier. Show only this camera."
            )
        }
    ),

    _spec(
        "search_descriptions",
        "Search text descriptions of events written by the vision-language model. "
        "Use this function to find events by their visual descriptions or actions. "
        "IMPORTANT: video_id and camera_id are different identifiers. "
        "video_id identifies a video, for example 'test_video_01'. "
        "camera_id identifies a physical camera, for example 'test_cam01'. "
        "If the user mentions a video ID, ALWAYS pass it as video_id, "
        "NEVER as camera_id. If the user mentions a camera ID, pass it as camera_id. "
        "Do not guess identifiers. "
        "Use mode='semantic' for meaning-based search and mode='keyword' "
        "for exact word matching. Event descriptions are stored in English. "
        "Returns matching events with similarity scores. "
        "If the user provides video_id but no camera_id, pass only video_id. "
        "Never guess a camera_id from a video_id.",
        {
            "query": _p(
                "string",
                "Description of what to find, e.g. "
                "'a person picks up a small bag'."
            ),
            "mode": _p(
                "string",
                "Search mode: semantic for meaning-based search, "
                "keyword for exact words.",
                enum=["semantic", "keyword"]
            ),
            "top_k": _p(
                "integer",
                "Maximum number of results. Default is 5."
            ),
            "video_id": _p(
                "string",
                "VIDEO identifier, e.g. 'test_video_01'. "
                "Use this when filtering by video."
            ),
            "camera_id": _p(
                "string",
                "CAMERA identifier, e.g. 'test_cam01'. "
                "Use ONLY when filtering by camera."
            ),
            "event_type": _FILTERS["event_type"]
        },
        required=["query"]
    ),
]


TOOL_FUNCS = {
    "query_events": tools.query_events,
    "count_events": tools.count_events,
    "get_track_timeline": tools.get_track_timeline,
    "get_event_details": tools.get_event_details,
    "list_zones_cameras": tools.list_zones_cameras,
    "search_descriptions": tools.search_descriptions,
}

ALLOWED_ARGS = {
    s["function"]["name"]: set(s["function"]["parameters"]["properties"])
    for s in SPECS
}


def to_openai_tools():
    """Список инструментов в формате OpenAI-совместимого API."""
    return SPECS


def call_tool(name, arguments, conn=None, embedder=None):
    """Безопасно вызывает функцию и возвращает результат или описание ошибки."""

    if name not in TOOL_FUNCS:
        return {
            "ok": False,
            "error": (
                f"неизвестная функция {name!r}; "
                f"доступны: {', '.join(sorted(TOOL_FUNCS))}"
            )
        }

    if arguments is None:
        arguments = {}

    if not isinstance(arguments, dict):
        return {
            "ok": False,
            "error": "arguments должен быть JSON-объектом"
        }

    unknown = sorted(set(arguments) - ALLOWED_ARGS[name])

    if unknown:
        return {
            "ok": False,
            "error": (
                f"функция {name} не принимает аргументы {unknown}; "
                f"допустимы: {sorted(ALLOWED_ARGS[name])}"
            )
        }

    kwargs = dict(arguments)

    # Repair a known Qwen spelling variant; keep the canonical database value.
    if kwargs.get("event_type") == "pick_up_or_put_candidate":
        kwargs["event_type"] = "pickup_or_put_candidate"

    if name == "search_descriptions" and embedder is not None:
        kwargs["embedder"] = embedder

    try:
        result = TOOL_FUNCS[name](conn=conn, **kwargs)
        json.dumps(result)
        return result

    except Exception as e:
        return {
            "ok": False,
            "error": (
                f"внутренняя ошибка функции "
                f"{name} ({type(e).__name__})"
            )
        }
