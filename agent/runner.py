"""Agent loop: question -> model -> tool calls -> answer."""

import json
import re
import time
from dataclasses import dataclass, field

from agent import snapshot_tools as _snap
from agent import tool_schemas
from agent.llm import LLMError


SYSTEM_PROMPT = """LANGUAGE RULE: answer in the language of the user's question (Russian question -> Russian answer,
English question -> English answer). Never use Chinese characters.

You are an AI assistant for a video surveillance event journal.

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

SNAPSHOT ARCHIVE (one frame every ~30 seconds: number of people and their clothing).
Snapshots are NOT linked to track_id. They answer "was anyone in ...", never "what did person N wear".

7. search_by_clothing
Use when asking whether anyone wore a given clothing, or when someone was in it.
Examples:
- Was anyone in a red sweater? -> color="red", clothing_type="sweater"
- When was someone in a green jacket? -> color="green", clothing_type="jacket"
Pass colour and type in English (colours: red, green, blue, black, white, gray; types: sweater, t-shirt, shirt, jacket, coat).
Russian -> English types: свитер, джемпер -> sweater; толстовка, худи -> hoodie; куртка, ветровка -> jacket;
пальто -> coat; футболка -> t-shirt; рубашка -> shirt; платье -> dress.
For "кофта" pass the Russian word itself: clothing_type="кофта" (the tool searches both sweaters and hoodies).
Never turn "кофта" into jacket.
Russian -> English colours: красный, бордовый -> red; зелёный -> green; синий, голубой -> blue; чёрный -> black;
белый -> white; серый -> gray; жёлтый -> yellow; оранжевый -> orange; коричневый -> brown; розовый -> pink;
фиолетовый -> purple; бежевый -> beige.
If the question says just "clothes" or "clothing" (for example "green clothes"), pass only color and omit clothing_type.
If it names only a type, pass only clothing_type. Never pass "any", "all" or "clothes" as a value.
Omit the argument you do not know; do not send empty strings.
If found is true, start with "Да" and give the times of the matches in seconds.
If found is false, answer "Нет, не нашёл".
Never use it for events or for the clothing of a specific track_id.

8. count_people_at
Use for "how many people were in the frame at time X". Pass t in seconds (1:30 = 90).
Do NOT use count_events for this question.
Snapshots are about 30 seconds apart: if the returned snapshot time t differs from the asked time,
say which time the answer is for.

9. get_snapshot_summary
Use for the maximum or average number of people, the most crowded moment, or what clothes people wore.
Answer from max_people, max_people_t, avg_people and the people lists. Do not count yourself.

ARGUMENT RULES:

- video_id identifies a video; camera_id identifies a camera. They are different.
- track_id must be an integer.
- Use only valid event_type values from the tool schema.
- Never use "all", "any", "all_cameras", or "unknown" as identifiers.
- To search all event types, omit event_type.
- To search all cameras or videos, omit camera_id and video_id.
- Never invent video_id or camera_id. If the user did not name a video, omit video_id.
- If a tool returns ok:false, never answer "not found": call it again without the failing identifier, or report the error.
  Do not copy identifiers from tool descriptions or earlier answers unless the user named them.
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
- Clothing is known ONLY from the snapshot archive (search_by_clothing, count_people_at, get_snapshot_summary).
  Never infer clothing from events or from VLM event descriptions.
- Snapshots are not linked to track_id: for the clothing of a specific person (track_id) answer that it cannot be determined.
- Do not infer intentions, reasons, or other information not recorded in the journal.
- Never invent information to make an answer sound complete.
- Prefer concise, structured answers over repetitive summaries.

The tool results are the only source of truth.
Answer only in the language of the question. Never use Chinese characters unless the question is in Chinese.
"""



# Вопросы про срезы (одежда, число людей на кадре) идут по короткому пути: короткий промпт и только 3 функции.
# Длинный промпт целиком (события этапа 1 + срезы, ~19 тыс. знаков) локальная модель на CPU разбирает десятки секунд;
# короткий в 4-5 раз меньше. Вопросы про события и всё неясное идут полным промптом, как раньше.
SNAPSHOT_TOOLS = ("search_by_clothing", "count_people_at", "get_snapshot_summary")

SNAPSHOT_PROMPT = """LANGUAGE RULE: answer in the language of the user's question (Russian question -> Russian answer,
English question -> English answer). Never use Chinese characters.

You answer questions about a snapshot archive of a video: one frame every ~30 seconds, with the number of people
and the clothing of each person. You MUST call a tool; never answer from memory. Never invent identifiers.

TOOLS
1. search_by_clothing(color, clothing_type, t_from, t_to): was anyone wearing such clothing, and when.
   "Was anyone in a red sweater?" -> color="red", clothing_type="sweater".
   Pass colour and type in English. Colours: red, orange, yellow, green, blue, purple, pink, brown, black, white,
   gray, beige. Russian: красный, бордовый -> red; зелёный -> green; синий, голубой -> blue; чёрный -> black;
   белый -> white; серый -> gray; жёлтый -> yellow; оранжевый -> orange; коричневый -> brown; розовый -> pink;
   фиолетовый -> purple; бежевый -> beige.
   Types: sweater, hoodie, t-shirt, shirt, jacket, coat, dress, vest. Russian: свитер, джемпер -> sweater;
   толстовка, худи -> hoodie; куртка, ветровка -> jacket; пальто -> coat; футболка -> t-shirt; рубашка -> shirt.
   For "кофта" pass clothing_type="кофта" itself (the tool searches sweaters and hoodies). Never turn "кофта" into jacket.
   If the question says just "clothes"/"одежда", pass only color and omit clothing_type.
2. count_people_at(t): how many people at time t, in seconds (1:30 = 90).
3. get_snapshot_summary(): maximum or average number of people, the most crowded moment.

RULES
- Omit video_id and every argument you do not know. Never invent a video_id. Do not send null or empty values.
- Snapshots are not linked to persons: the clothing of one particular person (person 7, track 3) cannot be determined.
- If a tool returns ok:false, call it again without the failing argument, or report the error.
- Use only tool results. For yes/no start with "Да"/"Нет" ("Yes"/"No"). Times are in seconds.
"""

_RE_EVENT_WORDS = re.compile(
    r"событи|вход|выход|вошё?л|вышел|вышла|подошё?л|подход|взял|положил|оставл|предмет|зон[аеыу]|стол|track|трек|"
    r"человек\w*\s*\d|person\s*\d|track\s*\d|event|enter|exit|pick|approach|zone|desk|камер|camera|описани|description", re.IGNORECASE)
_RE_FRAME_Q = re.compile(
    r"на\s+\d+\s*(с\b|сек|мин)|at\s+\d+\s*(s\b|sec|min)|больше всего люд|максимум люд|средн\w*\s+(число|количество)\s+люд|"
    r"в кадре|на кадре|на срез|срез|snapshot|most people|most crowded|crowded|maximum number of people|"
    r"average number of people", re.IGNORECASE)


def is_snapshot_question(question):
    """Question about the snapshot archive (clothing / people on a frame) and not about events."""
    q = question or ""
    if _RE_EVENT_WORDS.search(q):
        return False
    colors, types = _snap.find_clothing_terms(q)
    return bool(colors or types or _snap.has_generic_word(q) or _RE_FRAME_Q.search(q))


# Вопросы про события (этап 1) идут с промптом этапа 1 и только 6 функциями событий. Полный промпт (события + срезы,
# ~8 тыс. знаков, 9 функций) локальная 3B-модель разбирала очень долго и путала функции: на 30 вопросах по событиям
# верных функций было 5 из 30 против 16-21 из 30 со старым промптом. Текстовый ответ без функций на этом пути
# законен («ответить нельзя»), поэтому его не повторяют.
EVENTS_PROMPT = """LANGUAGE RULE: answer in the language of the user's question (Russian question -> Russian answer,
English question -> English answer). Never use Chinese characters.

You are an AI assistant for a video surveillance event journal.

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
EVENT_ROUTE_TOOLS = ("query_events", "count_events", "get_track_timeline", "get_event_details",
                     "list_zones_cameras", "search_descriptions")


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
    prompt_mode: str = "full"      # "snapshots": короткий промпт по срезам; "events": промпт этапа 1; "full": полный; "code": ответил код


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

    # этап 2: срезы. Пути к кадрам, id срезов и длинные описания людей модели не нужны: короткий результат
    # слабой модели проще превратить в ответ (и он не переполняет контекст). Полный результат остаётся в интерфейсе.
    for key in ("matches", "snapshots"):
        if isinstance(out.get(key), list):
            out[key] = [
                {k: (_slim_people(v) if k == "people" else v) for k, v in x.items()
                 if k not in ("frame_path", "described", "snapshot_id", "video_id")}
                if isinstance(x, dict) else x
                for x in out[key]
            ]
    if isinstance(out.get("people"), list):
        out["people"] = _slim_people(out["people"])
    out.pop("frame_path", None)

    return out


def _slim_people(people):
    return [{k: p[k] for k in ("clothing_color", "clothing_type") if k in p} if isinstance(p, dict) else p
            for p in people]


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


_YES = re.compile(r"^\W*(да|yes)\b", re.IGNORECASE)
_NO = re.compile(r"^\W*(нет|no)\b", re.IGNORECASE)
_CYR = re.compile(r"[а-яё]", re.IGNORECASE)


def _clothing_answer(result, question):
    """Yes/no answer for a single search_by_clothing call, built from the function result (not by the model)."""
    ru = bool(_CYR.search(question))
    if result.get("found"):
        times = ", ".join(f"{m['t']:g} " + ("с" if ru else "s") for m in result.get("matches", []))
        return f"Да. Найдено на срезах: {times}." if ru else f"Yes. Found in snapshots at: {times}."
    unc, und = result.get("uncertain_people", 0), result.get("snapshots_not_described", 0)
    if ru:
        ans = "Нет, не нашёл."
        if unc or und:
            ans += f" Но у {unc} чел. одежда не определена, {und} срезов не описано, поэтому «нет» не точно."
    else:
        ans = "No, not found."
        if unc or und:
            ans += f" But {unc} people had undetermined clothing and {und} snapshots were not described, so this is not certain."
    return ans


def fix_clothing_answer(answer, steps, question):
    """If the only tool call was search_by_clothing and the model's Yes/No contradicts `found`, use the code answer."""
    if len(steps) != 1 or steps[0].name != "search_by_clothing":
        return answer
    result = steps[0].result
    if not isinstance(result, dict) or not result.get("ok") or not isinstance(result.get("found"), bool):
        return answer
    contradicts = (result["found"] and _NO.match(answer)) or (not result["found"] and _YES.match(answer))
    return _clothing_answer(result, question) if contradicts else answer


_YESNO = re.compile(r"^\W*(был[аои]?\s+ли|были\s+ли|есть\s+ли|бывал[аои]?\s+ли|встречал\w*\s+ли|"
                    r"was\s+there|were\s+there|is\s+there|are\s+there|did\b|does\b|has\b|have\b)", re.IGNORECASE)


def is_yes_no_question(question):
    return bool(_YESNO.match(question or ""))


def _first_ok_search(steps):
    for s in steps:
        r = s.result
        if s.name == "search_by_clothing" and isinstance(r, dict) and r.get("ok") and isinstance(r.get("found"), bool):
            return r
    return None


def shortcut_answer(steps, question):
    """After the first function call: if the question is simple (one tool, answer fully in the result),
    the answer text is built by code. A question with «или/or» goes on to the model (two searches needed)."""
    if len(steps) != 1:
        return None
    step = steps[0]
    if len(_RE_QWORDS.findall(question)) >= 2:
        return None   # составной вопрос («когда … и сколько …»): одним шаблоном не ответить, продолжает модель
    if step.name == "search_by_clothing" and _RE_OR.search(question):
        return None
    return template_answer(step, question)


def fallback_answer(steps, question):
    """Answer built from the function results when the model does not finish (loops or returns nothing).

    Yes/no question: from the first successful search_by_clothing. Otherwise only if every step is the same
    successful clothing search or people-count call. Only for questions about the snapshot archive: for an
    event question ("was there an abandoned object?") a clothing search result says nothing."""
    if not is_snapshot_question(question):
        return None
    if is_yes_no_question(question):
        r = _first_ok_search(steps)
        if r is not None:
            return _clothing_answer(r, question)
    if not steps or len({s.name for s in steps}) != 1:
        return None
    last = steps[-1]
    r = last.result
    if not isinstance(r, dict) or not r.get("ok"):
        return None
    if last.name == "search_by_clothing" and isinstance(r.get("found"), bool):
        return _clothing_answer(r, question)
    if last.name == "count_people_at" and r.get("found") and "people_count" in r:
        ru = bool(_CYR.search(question))
        t, req = r.get("t", 0), r.get("requested_t", r.get("t", 0))
        if ru:
            return f"На срезе {t:g} с (запрошено {req:g} с) было {r['people_count']} чел."
        return f"At the snapshot at {t:g} s (requested {req:g} s) there were {r['people_count']} people."
    return None


_RE_COUNT_Q = re.compile(r"сколько|how many|количеств|число|count|number of", re.IGNORECASE)
_RE_AVG_Q = re.compile(r"средн|average|mean", re.IGNORECASE)
_RE_CLOTHES_Q = re.compile(r"одежд|во что|носил|одет|одеты|wear|wore|clothes|clothing|dressed", re.IGNORECASE)
_RE_TIME_Q = re.compile(r"\d+\s*(с\b|сек|мин|s\b|sec|min)|\d+:\d\d|секунд|минут|second|minute", re.IGNORECASE)
_RE_OR = re.compile(r"\b(или|or)\b", re.IGNORECASE)
_RE_AND = re.compile(r"\b(и|and)\b|,", re.IGNORECASE)
_RE_QWORDS = re.compile(r"сколько|когда|был\w*\s+ли|были\s+ли|есть\s+ли|какой|какая|какие|кто(?![\w-])|что(?![\w-])|как(?![\w-])|"
                        r"how many|when\b|was there|were there|what\b|who\b|which\b", re.IGNORECASE)
_HAN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def _people_word(n):
    n = abs(int(n))
    if 11 <= n % 100 <= 14:
        return "человек"
    return {1: "человек", 2: "человека", 3: "человека", 4: "человека"}.get(n % 10, "человек")


def _filters_text(args):
    if not isinstance(args, dict):
        return ""
    parts = [f"{k}={args[k]}" for k in ("event_type", "zone_id", "track_id", "camera_id", "video_id")
             if args.get(k) not in (None, "")]
    if args.get("t_from") is not None or args.get("t_to") is not None:
        parts.append(f"t={args.get('t_from', 0)}–{args.get('t_to', '…')}")
    return ", ".join(parts)


def _clothing_count_answer(result, ru):
    """«Сколько людей было в красном»: считает код по результату поиска."""
    n, snaps = result.get("people_matched", 0), result.get("count", 0)
    if ru:
        return f"Найдено {n} {_people_word(n)} на срезах: {snaps}."
    return f"Found {n} people across {snaps} snapshots."


_RE_LI = re.compile(r"\bли\b", re.IGNORECASE)
_RE_EN_YESNO = re.compile(r"^\W*(did|was|were|is|are|does|do|has|have|could|can)\b", re.IGNORECASE)
_RE_WH = re.compile(r"когда|сколько|кто(?![\w-])|что(?![\w-])|какой|какая|какие|где|куда|откуда|"
                    r"\b(when|how|who|what|which|where)\b", re.IGNORECASE)


def is_event_exists_question(question):
    """«Выходил ли человек 12?», «Did anyone pick up an object?»: ответ да/нет по числу найденных событий.
    Не сюда: вопросы про срезы, счёт, время, составные вопросы («или», «и», запятая)."""
    q = question or ""
    no_range = re.sub(r"\d+(?:[.,]\d+)?\s*(?:and|и)\s*\d", " ", q, flags=re.IGNORECASE)   # «между 20 и 55»: это не составной вопрос
    if is_snapshot_question(q) or _RE_WH.search(q) or _RE_COUNT_Q.search(q) or _RE_OR.search(q) or _RE_AND.search(no_range):
        return False
    return bool(_RE_LI.search(q) or _RE_EN_YESNO.match(q))


def template_answer(step, question):
    """Answer text built by code from ONE successful tool result (the model only chose the function).

    Returns None when the question needs the model (e.g. 'what clothes', several tools, errors)."""
    r = step.result
    if not isinstance(r, dict) or not r.get("ok"):
        return None
    ru = bool(_CYR.search(question))
    if step.name in SNAPSHOT_TOOLS and not is_snapshot_question(question):
        return None      # модель позвала функцию срезов на вопрос про события: её результат не отвечает на вопрос

    if step.name == "search_by_clothing" and isinstance(r.get("found"), bool):
        if r["found"] and _RE_COUNT_Q.search(question) and not is_yes_no_question(question):
            return _clothing_count_answer(r, ru)
        return _clothing_answer(r, question)

    if step.name == "count_people_at":
        if _RE_CLOTHES_Q.search(question):
            return None
        if not r.get("found") or "people_count" not in r:
            return "На этом видео нет срезов." if ru else "There are no snapshots for this video."
        t, req = r.get("t", 0), r.get("requested_t", r.get("t", 0))
        n = r["people_count"]
        if ru:
            ask = "" if t == req else f" (запрошено {req:g} с)"
            return f"На срезе {t:g} с{ask} было {n} {_people_word(n)}."
        ask = "" if t == req else f" (requested {req:g} s)"
        return f"At the snapshot at {t:g} s{ask} there were {n} people."

    if step.name == "get_snapshot_summary":
        if _RE_CLOTHES_Q.search(question) or not r.get("snapshots_count"):
            return None
        top, ts = r.get("max_people"), r.get("max_people_t") or []
        avg, cnt = r.get("avg_people"), r.get("snapshots_count")
        when = ", ".join(f"{t:g} с" if ru else f"{t:g} s" for t in ts)
        if _RE_AVG_Q.search(question) and not re.search(r"макс|max|больше всего|most", question, re.IGNORECASE):
            return (f"В среднем {avg:g} чел. на срез (срезов: {cnt})." if ru
                    else f"On average {avg:g} people per snapshot ({cnt} snapshots).")
        if ru:
            return f"Максимум: {top} {_people_word(top)}, на срезе {when}. В среднем {avg:g} чел. на срез (срезов: {cnt})."
        return f"Maximum: {top} people, at {when}. Average {avg:g} per snapshot ({cnt} snapshots)."

    if step.name in ("count_events", "query_events") and "count" in r and is_event_exists_question(question):
        n, flt = int(r["count"]), _filters_text(step.arguments)
        tail = f" ({flt})" if flt else ""
        if ru:
            return f"Да. Найдено событий: {n}{tail}." if n else f"Нет. Событий не найдено{tail}."
        return f"Yes. Events found: {n}{tail}." if n else f"No. No events found{tail}."

    if step.name == "count_events" and "count" in r:
        if not _RE_COUNT_Q.search(question) or _RE_OR.search(question) or _RE_AND.search(question):
            return None
        flt = _filters_text(step.arguments)
        if ru:
            return f"Найдено событий: {r['count']}" + (f" ({flt})." if flt else ".")
        return f"Events found: {r['count']}" + (f" ({flt})." if flt else ".")
    return None


def has_foreign_script(answer, question):
    """Chinese/Japanese/Korean characters in the answer when the question has none."""
    return bool(_HAN.search(answer or "")) and not _HAN.search(question or "")


def has_wrong_language(answer, question):
    """Вопрос без кириллицы (английский), а ответ русский: 3+ кириллические буквы."""
    return bool(re.search(r"[a-z]", question or "", re.IGNORECASE)) and not _CYR.search(question or "") \
        and len(_CYR.findall(answer or "")) >= 3


def _bad_language(answer, question):
    return has_foreign_script(answer, question) or has_wrong_language(answer, question)


def guard_language(answer, client, messages, steps, question):
    """Answer with Chinese characters or in another language than the question: one retry with a strict
    instruction; if it fails again, the code answer."""
    if not _bad_language(answer, question):
        return answer
    ru = bool(_CYR.search(question))
    strict = {"role": "user", "content": (
        "Ответь ещё раз ТОЛЬКО по-русски, без китайских иероглифов, по результатам функций выше." if ru else
        "Answer again ONLY in English, with no Chinese characters, using the tool results above.")}
    try:
        msg = client.chat(messages + [strict], None)
        again = strip_think(msg.get("content")) if isinstance(msg, dict) else ""
    except LLMError:
        again = ""
    if again and not _bad_language(again, question):
        return again
    for s in reversed(steps):
        t = template_answer(s, question)
        if t:
            return t
    fb = fallback_answer(steps, question)
    if fb:
        return fb
    return ("Модель ответила не на языке вопроса; результаты функций смотрите в блоке «Function calls»." if ru
            else "The model answered in the wrong language; see the tool results in 'Function calls'.")


def repair_clothing_args(args, question):
    """Check search_by_clothing arguments against the question; return (args, note or None).

    A weak model loses the clothing type ("red t-shirt" -> only red), turns "кофта" into jacket or invents
    a filter. The colour and type named in the question are known by code, so a mismatch is corrected here.
    Questions with several colours or types ("red or blue") are left to the model."""
    if not isinstance(args, dict):
        return args, None
    colors, types = _snap.find_clothing_terms(question)
    if len(colors) > 1 or len(types) > 1:
        return args, None
    new = dict(args)
    if colors:
        have = args.get("color")
        if not have or not isinstance(have, str) or _snap.normalize_color(have) != colors[0]:
            new["color"] = colors[0]
    if types:
        have = args.get("clothing_type")
        if not have or not isinstance(have, str) or _snap.type_set(have) != _snap.type_set(types[0]):
            new["clothing_type"] = types[0]
    elif args.get("clothing_type") and _snap.has_generic_word(question):
        new.pop("clothing_type")
    # Окно времени: если время в вопросе не названо, модель его выдумывает (t_to=0, t_to=1200: окно [0; 0] отбрасывает
    # все срезы, кроме нулевого). Тогда окно убираем. Назвали время ("на 90 с", "после минуты") - решает модель.
    has_time = bool(_RE_TIME_Q.search(question))
    zero_window = args.get("t_from") in (0, None) and args.get("t_to") == 0
    if zero_window or not has_time:
        for k in ("t_from", "t_to"):
            new.pop(k, None)
    if new == args:
        return args, None
    keys = ["color", "clothing_type"] + [k for k in ("t_from", "t_to") if args.get(k) != new.get(k)]
    return new, {"from": {k: args.get(k) for k in keys}, "to": {k: new.get(k) for k in keys}}


EVENT_TOOLS = ("count_events", "query_events", "get_track_timeline", "get_event_details", "search_descriptions")


def known_video_ids(conn):
    try:
        return {r[0] for r in _snap._rows(conn, "SELECT video_id FROM videos")}
    except Exception:  # noqa: BLE001
        return None


_RE_BEFORE_N = re.compile(r"(?:\bдо|\bранее|\bраньше|\bbefore|\bby|\bearlier than)\s+(?:the\s+)?(\d+(?:[.,]\d+)?)", re.IGNORECASE)
_RE_AFTER_N = re.compile(r"(?:\bпосле|\bпозже|\bafter|\blater than)\s+(?:the\s+)?(\d+(?:[.,]\d+)?)", re.IGNORECASE)


def _num(v):
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _swap_time_direction(new, question):
    """«до 90 секунды»: слабая модель ставит t_from=90 вместо t_to=90 (и наоборот для «после»). Единственная граница,
    совпадающая с числом из вопроса, переносится в нужное поле."""
    before, after = _RE_BEFORE_N.search(question or ""), _RE_AFTER_N.search(question or "")
    if before and not after and _num(new.get("t_from")) == _num(before.group(1)) and new.get("t_to") in (None, "") \
            and _num(new.get("t_from")) is not None:
        new["t_to"] = new.pop("t_from")
    elif after and not before and _num(new.get("t_to")) == _num(after.group(1)) and new.get("t_from") in (None, "") \
            and _num(new.get("t_to")) is not None:
        new["t_from"] = new.pop("t_to")


def repair_event_args(args, question, conn=None):
    """Check arguments of the event tools. A weak model invents video_id ("video_001"), a time window the
    question never mentions (t_to=1200) and a [0; 0] window; the filter then returns nothing. Such arguments
    are dropped (an unknown video_id only when it is not in the archive). Returns (args, note or None)."""
    if not isinstance(args, dict):
        return args, None
    new = dict(args)
    vid = new.get("video_id")
    if "video_id" in new:
        known = known_video_ids(conn) if vid else None
        if not vid or (known is not None and vid not in known):
            new.pop("video_id")
    zero_window = new.get("t_from") in (0, None) and new.get("t_to") == 0
    if zero_window or not _RE_TIME_Q.search(question):
        for k in ("t_from", "t_to"):
            new.pop(k, None)
    _swap_time_direction(new, question)
    if new == args:
        return args, None
    keys = [k for k in ("video_id", "t_from", "t_to") if args.get(k) != new.get(k)]
    return new, {"from": {k: args.get(k) for k in keys}, "to": {k: new.get(k) for k in keys}}


_RE_PERSON_REF = re.compile(r"(человек\w*|person|track|трек\w*)\s*#?\s*\d+", re.IGNORECASE)
_RE_CLOTHING_Q = re.compile(r"цвет|одет|одежд|куртк|кофт|футболк|свитер|толстовк|худи|пальто|рубашк|"
                            r"wear|wore|cloth|colou?r|jacket|sweater|shirt|hoodie|coat", re.IGNORECASE)
_RE_WHY = re.compile(r"\b(почему|зачем|why)\b", re.IGNORECASE)
_RE_CAMERA = re.compile(r"\bcam\w*?\d+\b", re.IGNORECASE)


def out_of_scope_answer(question, conn=None):
    """Questions the archive cannot answer, recognised by code (a weak model invents facts for them):
    clothing of one particular person (snapshots are not linked to tracks), reasons ("why"), an unknown camera.
    Returns the answer text or None."""
    ru = bool(_CYR.search(question))
    reason = None
    if _RE_PERSON_REF.search(question) and _RE_CLOTHING_Q.search(question):
        reason = ("одежда относится к срезам и не привязана к конкретному человеку (track_id)" if ru else
                  "clothing is stored per snapshot and is not linked to a particular person (track_id)")
    elif _RE_WHY.search(question):
        reason = "в базе только события и срезы, причин и намерений нет" if ru else "the archive holds events and snapshots, not reasons or intentions"
    else:
        m = _RE_CAMERA.search(question)
        if m:
            try:
                cams = {str(r[0]).casefold() for r in _snap._rows(conn, "SELECT DISTINCT camera_id FROM videos")}
            except Exception:  # noqa: BLE001
                cams = None
            word = m.group(0).casefold()
            if cams is not None and not any(c == word or c.endswith(word) for c in cams):
                reason = f"камеры {m.group(0)} нет в базе" if ru else f"camera {m.group(0)} is not in the archive"
    if not reason:
        return None
    return (f"Ответить нельзя: в базе нет этих данных ({reason})." if ru
            else f"Cannot answer: the archive has no such data ({reason}).")


def _empty_hint(msg):
    """What the model actually returned when the text is empty: helps to see why (thinking, token limit...)."""
    if not isinstance(msg, dict):
        return ""
    keys = sorted(k for k, v in msg.items() if v)
    hint = f" (поля ответа модели: {', '.join(keys) or 'пусто'})"
    if any(k in keys for k in ("reasoning", "reasoning_content", "thinking")):
        hint += ("; модель рассуждала, но ответ не успел появиться: увеличьте LLM_MAX_TOKENS "
                 "или отключите режим размышления у модели")
    return hint


def _final_pass(client, messages, steps, question):
    """One more model call WITHOUT tools: it cannot call functions again and has to write the answer."""
    nudge = {"role": "user", "content": "Use the tool results above and give the final answer now. Do not call any tools."}
    try:
        msg = client.chat(messages + [nudge], None)
    except LLMError:
        return None
    answer = strip_think(msg.get("content")) if isinstance(msg, dict) else ""
    if not answer:
        return None
    answer = guard_language(answer, client, messages, steps, question)
    return fix_clothing_answer(answer, steps, question)


def run_agent(
    question,
    client,
    conn=None,
    embedder=None,
    max_steps=5,
    system_prompt=SYSTEM_PROMPT,
    _compact=True,
):
    """Run the agent and return its answer, tool steps, or error."""

    t0 = time.time()

    res = AgentResult(question=question)
    if _compact and system_prompt is SYSTEM_PROMPT:
        oos = out_of_scope_answer(question, conn)
        if oos:                      # вопрос, на который архив ответить не может: отвечает код, модель не нужна
            res.answer, res.prompt_mode = oos, "code"
            res.seconds = round(time.time() - t0, 3)
            return res
    compact = _compact and system_prompt is SYSTEM_PROMPT
    route_tools = None
    if compact and is_snapshot_question(question):
        system_prompt, route_tools, res.prompt_mode = SNAPSHOT_PROMPT, SNAPSHOT_TOOLS, "snapshots"
    elif compact:
        system_prompt, route_tools, res.prompt_mode = EVENTS_PROMPT, EVENT_ROUTE_TOOLS, "events"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    tools = tool_schemas.to_openai_tools()
    if compact and route_tools:
        tools = [t for t in tools if t["function"]["name"] in route_tools]
    seen_calls = set()

    try:
        for _ in range(max_steps):
            msg = client.chat(messages, tools)

            calls = msg.get("tool_calls") or []

            if not calls:
                answer = strip_think(msg.get("content"))

                if res.prompt_mode == "snapshots" and not res.steps:
                    # вопрос похож на срезы, но модель не вызвала функций: повторяем полным промптом. На пути событий текстовый ответ
                    # без функций законен («ответить нельзя», приветствие), его не повторяем
                    again = run_agent(question, client, conn=conn, embedder=embedder, max_steps=max_steps,
                                      _compact=False)
                    again.seconds = round(time.time() - t0, 3)
                    return again

                if answer:
                    answer = guard_language(answer, client, messages, res.steps, question)
                    res.answer = fix_clothing_answer(answer, res.steps, question)
                else:
                    res.answer = fallback_answer(res.steps, question)
                    if not res.answer and res.steps:
                        res.answer = _final_pass(client, messages, res.steps, question)
                    if not res.answer:
                        res.error = "модель вернула пустой ответ" + _empty_hint(msg)

                return res

            messages.append({
                "role": "assistant",
                "content": msg.get("content"),
                "tool_calls": calls,
            })

            repeated = False
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
                    fixed = None
                    if name == "search_by_clothing":
                        args, fixed = repair_clothing_args(args, question)
                    elif name in EVENT_TOOLS:
                        args, fixed = repair_event_args(args, question, conn)
                    result = tool_schemas.call_tool(
                        name,
                        args,
                        conn=conn,
                        embedder=embedder,
                    )
                    if fixed and isinstance(result, dict):
                        result = dict(result, arguments_corrected=fixed)

                key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False, default=str))
                if key in seen_calls:
                    repeated = True
                    if isinstance(result, dict):
                        result = dict(result, repeated_call="you already called this with the same arguments: "
                                                           "use the result above and write the final answer now")
                seen_calls.add(key)

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

            short = shortcut_answer(res.steps, question)
            if short:
                res.answer = short   # yes/no about clothing: the function result already contains the answer
                return res

            if repeated:
                # the model repeats the same call instead of answering: finish from the function result
                res.answer = fallback_answer(res.steps, question) or _final_pass(client, messages, res.steps, question)
                if res.answer:
                    return res

        res.answer = fallback_answer(res.steps, question) or _final_pass(client, messages, res.steps, question)
        if not res.answer:
            res.error = f"модель не дала ответа за {max_steps} шагов"
        return res

    except LLMError as e:
        res.error = str(e)
        return res

    finally:
        res.seconds = round(time.time() - t0, 3)