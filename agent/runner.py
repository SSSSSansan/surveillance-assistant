
"""Agent loop: question -> model -> tool calls -> answer."""

import json
import re
import time
from dataclasses import dataclass, field

from agent import tool_schemas
from agent.llm import LLMError


SYSTEM_PROMPT = """You are an AI assistant for a video surveillance event journal.

You MUST use the available tools for questions about recorded events.
Answer only using actual tool results. Never invent IDs, events, times, cameras, or descriptions.

IMPORTANT: Choose the correct tool.

1. count_events
Use for questions asking how many, how much, or total number.
Examples:
- How many people entered? -> event_type="enter"
- How many events happened at the desk? -> zone_id="desk"
- How many events did person 7 have? -> track_id=7
- How many people approached the desk? -> event_type="approach_zone", zone_id="desk"
If the question asks for all event types, omit event_type.
If the user asks for multiple event types joined by "or" (for example, pick up or put down),
make separate tool calls, one per exact event_type. Never invent a combined event_type.
Use the exact schema spelling "pickup_or_put_candidate" (without "pick_up").

2. query_events
Use for questions asking whether an event happened, when it happened, or to list events.
Also use for first or last event:
- First event: order="asc", limit=1
- Last event: order="desc", limit=1
- Last exit: event_type="exit", order="desc", limit=1

3. get_track_timeline
Use when asking what a specific person did or what happened before or after their actions.
Use only the exact track_id stated in the question or returned by another tool.
Never guess a track_id.

4. get_event_details
Use when the question asks for details about a specific event ID.
Never invent an event ID.

5. list_zones_cameras
Use to discover available cameras, videos, and zones.
Never invent their identifiers.

6. search_descriptions
Use when the question asks about the content or meaning of textual event descriptions.
Do not use it for counting or ordinary filtering by event type, time, person, or zone.

ARGUMENT RULES:

- video_id identifies a video; camera_id identifies a camera. They are different.
- track_id must be an integer.
- Use only valid event_type values from the tool schema.
- Never use "all", "any", "all_cameras", or "unknown" as identifiers.
- To search all event types, omit event_type.
- To search all cameras or videos, omit camera_id and video_id.
- Do not send null arguments.
- Do not add unnecessary filters.
- Do not invent IDs, zones, cameras, or time limits.
- Use exact values from the question or previous tool results.
- If an identifier is missing, do not guess it.

VLM INTERPRETATION RULES:

- Treat vlm_description as the source of truth for the described visual activity.
- Preserve the meaning of each VLM description. Do not add actions that are not explicitly described.
- Pay attention to negative statements:
  "not picking up", "not putting down", "does not interact"
  must remain negative in your answer.
- Do not interpret holding an object as picking it up or putting it down.
- Do not interpret drinking from a cup as picking up or putting down the cup unless the description explicitly says so.
- Do not describe an action as confirmed if the VLM description says it is uncertain, possible, or only appears to happen.
- The event type "pickup_or_put_candidate" means a candidate interaction. It does NOT automatically confirm that an object was picked up or put down.
- Distinguish between the event type and the action described by VLM.
- If VLM says that no pickup or putdown occurred, explicitly preserve this information.
- Do not combine events from different track_id values into one person's activity.
- When summarizing multiple events, group them by track_id.
- Treat different track_id values as different tracked persons. Do not assume they are the same person.
- Keep events in chronological order within each person's timeline.
- Do not assume that events with the same track_id prove continuous identity beyond the tracking data.
- If descriptions contradict each other or are ambiguous, explain the uncertainty instead of choosing an unsupported interpretation.
- Do not infer intentions, motives, object ownership, or reasons for actions.
- Do not invent object types. If the description says "small object", do not call it a phone, snack, or another specific object unless stated.

ANSWER RULES:

- Answer in the same language as the question.
- For yes/no questions, start with "Да" or "Нет", or "Yes" or "No".
- For counting questions, use the exact count returned by count_events.
- For time questions, report the exact time in seconds.
- For lists, include person ID, event type, and timestamp.
- For questions about what happened after an event, use chronological order.
- When summarizing multiple people's activities, separate them by track_id.
- When summarizing VLM descriptions, distinguish confirmed actions, possible actions, and explicitly absent actions.
- Do not present candidate events as confirmed actions.
- If no confirmed events are found, answer "не нашёл" or "No, not found".
- If the requested information cannot be determined from the journal, answer:
  Russian: "ответить нельзя: в базе нет этих данных"
  English: "cannot answer: the journal has no such data"
- Do not infer clothing colors, intentions, reasons, or other information not recorded in the journal.
- Never invent information to make an answer sound complete.
- Prefer concise, structured answers over repetitive summaries.

The tool results are the only source of truth.
"""



@dataclass
class ToolStep:
    name: str
    arguments: object
    result: dict


@dataclass
class AgentResult:
    question: str
    answer: str = None
    steps: list = field(default_factory=list)
    error: str = None
    seconds: float = 0.0


def strip_think(text):
    """Remove model reasoning blocks."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return re.sub(r"^.*?</think>", "", text, flags=re.S).strip()


_KEEP = (
    "event_id",
    "video_id",
    "event_type",
    "track_id",
    "zone_id",
    "t_start",
    "t_end",
    "vlm_description",
    "score",
)


def _slim_event(ev):
    return {k: ev[k] for k in _KEEP if ev.get(k) is not None}


def slim_for_model(result):
    """Remove internal fields before sending tool results back to the model."""
    if not isinstance(result, dict):
        return result

    out = dict(result)

    for key in ("events", "results"):
        if isinstance(out.get(key), list):
            out[key] = [
                _slim_event(e) if isinstance(e, dict) else e
                for e in out[key]
            ]

    if isinstance(out.get("event"), dict):
        out["event"] = _slim_event(out["event"])

    return out


def _parse_arguments(raw):
    """Parse tool arguments received from the model."""
    if raw is None or raw == "":
        return {}, None

    if isinstance(raw, dict):
        return raw, None

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return raw, "arguments не является корректным JSON"

        if not isinstance(parsed, dict):
            return parsed, "arguments должен быть JSON-объектом"

        return parsed, None

    return raw, "arguments должен быть JSON-объектом"


def run_agent(
    question,
    client,
    conn=None,
    embedder=None,
    max_steps=5,
    system_prompt=SYSTEM_PROMPT,
):
    """Run the agent and return its answer, tool steps, or error."""

    t0 = time.time()

    res = AgentResult(question=question)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    tools = tool_schemas.to_openai_tools()

    try:
        for _ in range(max_steps):
            msg = client.chat(messages, tools)

            calls = msg.get("tool_calls") or []

            if not calls:
                answer = strip_think(msg.get("content"))

                if answer:
                    res.answer = answer
                else:
                    res.error = "модель вернула пустой ответ"

                return res

            messages.append({
                "role": "assistant",
                "content": msg.get("content"),
                "tool_calls": calls,
            })

            for call in calls:
                fn = (
                    call.get("function") or {}
                    if isinstance(call, dict)
                    else {}
                )

                name = fn.get("name")
                args, arg_err = _parse_arguments(fn.get("arguments"))

                if arg_err:
                    result = {"ok": False, "error": arg_err}
                else:
                    # A video ID does not imply a camera ID. Remove a guessed camera
                    # filter unless that exact camera identifier appears in the question.
                    if isinstance(args, dict) and args.get("video_id") and args.get("camera_id"):
                        mentioned_camera = str(args["camera_id"]).casefold() in question.casefold()
                        if not mentioned_camera:
                            args = dict(args)
                            args.pop("camera_id", None)
                    result = tool_schemas.call_tool(
                        name,
                        args,
                        conn=conn,
                        embedder=embedder,
                    )

                res.steps.append(
                    ToolStep(
                        name=name,
                        arguments=args,
                        result=result,
                    )
                )

                call_id = (
                    call.get("id")
                    if isinstance(call, dict)
                    else None
                ) or f"call_{len(res.steps)}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        slim_for_model(result),
                        ensure_ascii=False,
                    ),
                })

        res.error = f"модель не дала ответа за {max_steps} шагов"
        return res

    except LLMError as e:
        res.error = str(e)
        return res

    finally:
        res.seconds = round(time.time() - t0, 3)
