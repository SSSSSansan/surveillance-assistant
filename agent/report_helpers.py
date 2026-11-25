"""Расчёты для страницы «Отчёт» (T29): без Streamlit, чтобы тестировать отдельно.

Что считается по срезам видео (таблицы snapshots и snapshot_people):
  шапка, плитки показателей, число людей по времени, распределения цветов и типов, таблица по срезам,
  экспорт в Markdown и CSV.
Таблицы runs, labels_gold_* появятся в T15: пока их нет, отчёт честно пишет «нет данных о прогоне»,
а не выдумывает модель и время обработки. Всё, что пришло из базы (id, описания), экранируется при выводе в HTML.
Проверялось на SQLite и заглушках; PostgreSQL и браузер не проверялись."""
import csv
import html
import io
import statistics
from collections import Counter

from agent import snapshot_tools
from agent.archive_helpers import color_hex, fmt_clock

NO_RUN = "no run data (the runs table is added in T15)"


# ---------- данные из базы ----------

def video_info(conn, video_id):
    """camera_id и source из таблицы videos; пустой словарь, если записи нет."""
    try:
        rows = snapshot_tools._rows(conn, "SELECT camera_id, source FROM videos WHERE video_id = %s", [video_id])
    except Exception:  # noqa: BLE001
        _rollback(conn)
        return {}
    return {"camera_id": rows[0][0], "source": rows[0][1]} if rows else {}


def load_runs(conn, video_id):
    """Прогоны видео из таблицы runs (T15). Таблицы пока нет: пустой список, без ошибки.
    После неудачного запроса PostgreSQL помечает транзакцию как сломанную: делаем rollback."""
    cols = ["run_id", "method", "model", "mode", "prompt_variant", "weights", "started_at", "duration_sec"]
    try:
        rows = snapshot_tools._rows(conn, f"SELECT {', '.join(cols)} FROM runs WHERE video_id = %s "
                                          "ORDER BY started_at DESC", [video_id])
    except Exception:  # noqa: BLE001
        _rollback(conn)
        return []
    return [dict(zip(cols, r)) for r in rows]


def _rollback(conn):
    try:
        if conn is not None:
            conn.rollback()
    except Exception:  # noqa: BLE001
        pass


# ---------- расчёт отчёта ----------

def snapshot_step(snaps):
    """Шаг между срезами, секунды: медиана разностей (None, если срез один)."""
    ts = sorted(s["t"] for s in snaps)
    diffs = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    return statistics.median(diffs) if diffs else None


def distribution(snaps, key):
    """Сколько людей с каждым значением clothing_color / clothing_type: [(значение, число)], по убыванию."""
    c = Counter((p.get(key) or "unknown") for s in snaps for p in s.get("people", []))
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))


def _status(s):
    n, d = s.get("people_count", 0), len(s.get("people", []))
    if n == 0:
        return "empty"
    if d == 0:
        return "not described"
    if d < n:
        return f"partial ({d}/{n})"
    return "described"


def snapshot_rows(snaps):
    """Строки таблицы по срезам."""
    rows = []
    for s in snaps:
        cnt = Counter(p.get("clothing_color") or "unknown" for p in s.get("people", []))
        known = [(c, k) for c, k in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])) if c != "unknown"][:3]
        rows.append({
            "snapshot_id": s.get("snapshot_id"), "t": s["t"], "time": fmt_clock(s["t"]),
            "people": int(s.get("people_count", 0)),
            "gold_people": s.get("gold_people"),          # эталон по кадру (T15/T21): пока None
            "top_colors": ", ".join(f"{c} {k}" for c, k in known) or "-",
            "unknown": cnt.get("unknown", 0),
            "status": _status(s), "frame_path": s.get("frame_path"),
        })
    return rows


def build_report(video_id, snaps, info=None, runs=None, t_range=None):
    """Весь отчёт одним словарём. t_range=(от, до) оставляет срезы в этом окне времени."""
    snaps = sorted(snaps, key=lambda s: s["t"])
    if t_range:
        lo, hi = t_range
        snaps = [s for s in snaps if lo <= s["t"] <= hi]
    counts = [int(s.get("people_count", 0)) for s in snaps]
    top = max(counts) if counts else 0
    people = [p for s in snaps for p in s.get("people", [])]
    colors, types = distribution(snaps, "clothing_color"), distribution(snaps, "clothing_type")
    unknown_color = sum(k for c, k in colors if c == "unknown")
    unknown_type = sum(k for c, k in types if c == "unknown")
    detections = sum(counts)
    run = (runs or [None])[0]
    return {
        "video_id": video_id, "camera_id": (info or {}).get("camera_id"), "source": (info or {}).get("source"),
        "first_t": snaps[0]["t"] if snaps else None, "last_t": snaps[-1]["t"] if snaps else None,
        "step": snapshot_step(snaps), "run": run, "runs_count": len(runs or []),
        "snapshots": len(snaps), "detections": detections,
        "max_people": top, "max_people_t": [s["t"] for s in snaps if s.get("people_count") == top] if snaps else [],
        "avg_people": round(detections / len(snaps), 2) if snaps else 0.0,
        "described_people": len(people),
        "coverage": round(len(people) / detections, 3) if detections else None,
        "unknown_color": unknown_color, "unknown_type": unknown_type,
        "unknown_color_share": round(unknown_color / len(people), 3) if people else None,
        "unknown_type_share": round(unknown_type / len(people), 3) if people else None,
        "colors": colors, "types": types,
        "series": [(s["t"], int(s.get("people_count", 0))) for s in snaps],
        "rows": snapshot_rows(snaps),
    }


def pct(x):
    return "-" if x is None else f"{x * 100:.0f}%"


def header_fields(r):
    """Поля шапки: (название, значение). Данные о прогоне берутся только из runs."""
    run = r["run"]
    fields = [("Video", r["video_id"]), ("Camera", r["camera_id"] or "-"), ("Source", r["source"] or "-"),
              ("Covered", f"{fmt_clock(r['last_t'])} ({r['last_t']:g} s)" if r["last_t"] is not None else "-"),
              ("Snapshot step", f"{r['step']:g} s" if r["step"] else "-")]
    if run:
        fields += [("Method", run.get("method") or "-"), ("Model", run.get("model") or "-"),
                   ("Mode", run.get("mode") or "-"), ("Prompt", run.get("prompt_variant") or "-"),
                   ("Weights", run.get("weights") or "-"), ("Run started", str(run.get("started_at") or "-")),
                   ("Run time", f"{run['duration_sec']:g} s" if run.get("duration_sec") is not None else "-")]
    return fields, (None if run else NO_RUN)


def tiles(r):
    """Плитки показателей: (название, значение, пояснение)."""
    when = ", ".join(fmt_clock(t) for t in r["max_people_t"]) or "-"
    run = r["run"]
    return [("Snapshots", r["snapshots"], ""),
            ("People detections", r["detections"], "sum over snapshots"),
            ("Max people", r["max_people"], f"at {when}"),
            ("Avg people", r["avg_people"], "per snapshot"),
            ("Unknown color", pct(r["unknown_color_share"]), f"{r['unknown_color']} of {r['described_people']} people"),
            ("Processing time", f"{run['duration_sec']:g} s" if run and run.get("duration_sec") is not None else "-",
             "from runs" if run else "no run data")]


# ---------- HTML-фрагменты ----------

def bars_html(items, color_of=None, limit=12):
    """Горизонтальные полосы распределения. Подписи экранируются, цвет только из белого списка archive_helpers."""
    items = items[:limit]
    if not items:
        return '<div class="muted">no data</div>'
    total = sum(k for _, k in items) or 1
    peak = max(k for _, k in items) or 1
    out = ['<div class="rbars">']
    for name, k in items:
        color = color_of(name) if color_of else "var(--primary)"
        out.append(f'<div class="rbar"><span class="rbar-name">{html.escape(str(name))}</span>'
                   f'<span class="rbar-track"><span class="rbar-fill" style="width:{k / peak * 100:.1f}%;background:{color}">'
                   f'</span></span><span class="rbar-val">{k} · {k / total * 100:.0f}%</span></div>')
    out.append("</div>")
    return "".join(out)


def colors_bars(r):
    return bars_html(r["colors"], color_of=color_hex)


def types_bars(r):
    return bars_html(r["types"])


def header_html(r):
    fields, note = header_fields(r)
    cells = "".join(f'<div class="rh-cell"><span class="rh-k">{html.escape(k)}</span>'
                    f'<span class="rh-v">{html.escape(str(v))}</span></div>' for k, v in fields)
    tail = f'<div class="muted rh-note">{html.escape(note)}</div>' if note else ""
    return f'<div class="rhead">{cells}</div>{tail}'


def chart_svg(series, gold=None, width=760, height=230):
    """Число людей по времени: линия, область, пик отмечен точкой. gold: [(t, n)] эталон (пунктир), если есть."""
    if not series:
        return '<div class="muted">no snapshots</div>'
    l, r_, t_, b = 40, 14, 22, 28
    w, h = width - l - r_, height - t_ - b
    ts = [t for t, _ in series]
    t0, t1 = min(ts), max(ts)
    ymax = max([n for _, n in series] + [n for _, n in (gold or [])] + [1])
    ymax = ymax + (ymax % 2)                         # ровная верхняя линия сетки

    def X(t):
        return l + (w / 2 if t1 == t0 else (t - t0) / (t1 - t0) * w)

    def Y(n):
        return t_ + h - n / ymax * h

    def path(points):
        return " ".join(f"{X(t):.1f},{Y(n):.1f}" for t, n in points)

    grid = "".join(f'<line x1="{l}" x2="{l + w}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="g"/>'
                   f'<text x="{l - 6}" y="{Y(v) + 4:.1f}" class="ax" text-anchor="end">{v:g}</text>'
                   for v in (0, ymax / 2, ymax))
    ticks = sorted({ts[0], ts[len(ts) // 2], ts[-1]})
    xl = "".join(f'<text x="{X(t):.1f}" y="{height - 8}" class="ax" text-anchor="middle">{fmt_clock(t)}</text>' for t in ticks)
    area = (f'<polygon points="{X(ts[0]):.1f},{Y(0):.1f} {path(series)} {X(ts[-1]):.1f},{Y(0):.1f}" class="area"/>'
            if len(series) > 1 else "")
    line = f'<polyline points="{path(series)}" class="line"/>' if len(series) > 1 else ""
    top = max(n for _, n in series)
    peaks = "".join(f'<circle cx="{X(t):.1f}" cy="{Y(n):.1f}" r="5" class="peak"/>'
                    f'<text x="{X(t):.1f}" y="{Y(n) - 10:.1f}" class="pk" text-anchor="middle">{n} · {fmt_clock(t)}</text>'
                    for t, n in series if n == top)
    dots = "".join(f'<circle cx="{X(t):.1f}" cy="{Y(n):.1f}" r="2.5" class="dot"/>' for t, n in series)
    gl = f'<polyline points="{path(gold)}" class="gold"/>' if gold else ""
    return (f'<svg viewBox="0 0 {width} {height}" class="rchart" role="img" '
            f'aria-label="People per snapshot over time, peak {top}">{grid}{area}{line}{gl}{dots}{peaks}{xl}</svg>')


# ---------- события этапа 1 (T29, пункт 6) ----------

def load_events(conn, video_id):
    """События видео из таблицы events (этап 1). Нет таблицы или записей: пустой список, без ошибки."""
    cols = ["event_id", "event_type", "track_id", "zone_id", "t_start", "t_end", "trigger_score", "needs_vlm", "clip_path"]
    try:
        rows = snapshot_tools._rows(conn, f"SELECT {', '.join(cols)} FROM events WHERE video_id = %s "
                                          "ORDER BY t_start, event_id", [video_id])
    except Exception:  # noqa: BLE001
        _rollback(conn)
        return []
    return [dict(zip(cols, r)) for r in rows]


def build_events(events, t_range=None):
    """События в окне времени (по t_start): число по типам (по убыванию), строки для таблицы."""
    evs = [e for e in events if not t_range or t_range[0] <= float(e["t_start"]) <= t_range[1]]
    counts = sorted(Counter(e["event_type"] for e in evs).items(), key=lambda kv: (-kv[1], kv[0]))
    rows = []
    for e in evs:
        rows.append({"event_id": e["event_id"], "type": e["event_type"], "time": fmt_clock(e["t_start"]),
                     "t_start": float(e["t_start"]), "t_end": float(e["t_end"]),
                     "track": "-" if e.get("track_id") is None else e["track_id"], "zone": e.get("zone_id") or "-",
                     "score": "-" if e.get("trigger_score") is None else round(float(e["trigger_score"]), 2),
                     "needs_vlm": bool(e.get("needs_vlm")), "clip_path": e.get("clip_path")})
    return {"total": len(evs), "counts": counts, "rows": rows, "types": [t for t, _ in counts]}


def events_timeline_svg(ev, width=760):
    """Таймлайн: строка на тип события, отрезок от t_start до t_end и точка в начале."""
    rows = ev["rows"]
    if not rows:
        return '<div class="muted">no events</div>'
    types = ev["types"]
    l, r_, t_, b, step = 120, 14, 12, 28, 26
    height = t_ + step * len(types) + b
    w = width - l - r_
    t0 = min(x["t_start"] for x in rows)
    t1 = max(x["t_end"] for x in rows)

    def X(t):
        return l + (w / 2 if t1 == t0 else (t - t0) / (t1 - t0) * w)

    def Y(i):
        return t_ + step * i + step / 2

    lab = "".join(f'<line x1="{l}" x2="{l + w}" y1="{Y(i):.1f}" y2="{Y(i):.1f}" class="g"/>'
                  f'<text x="{l - 8}" y="{Y(i) + 4:.1f}" class="ax" text-anchor="end">{html.escape(str(tp))}</text>'
                  for i, tp in enumerate(types))
    marks = "".join(
        f'<line x1="{X(x["t_start"]):.1f}" x2="{max(X(x["t_end"]), X(x["t_start"]) + 2):.1f}" y1="{Y(types.index(x["type"])):.1f}" '
        f'y2="{Y(types.index(x["type"])):.1f}" class="evline"/>'
        f'<circle cx="{X(x["t_start"]):.1f}" cy="{Y(types.index(x["type"])):.1f}" r="4" class="evdot">'
        f'<title>{html.escape(str(x["type"]))} {x["time"]}</title></circle>' for x in rows)
    ticks = sorted({t0, (t0 + t1) / 2, t1})
    xl = "".join(f'<text x="{X(t):.1f}" y="{height - 8}" class="ax" text-anchor="middle">{fmt_clock(t)}</text>' for t in ticks)
    return (f'<svg viewBox="0 0 {width} {height}" class="rchart" role="img" '
            f'aria-label="Events timeline, {ev["total"]} events">{lab}{marks}{xl}</svg>')


REPORT_CSS = """
<style>
.rhead{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px 18px;background:var(--card);
 border:1px solid var(--border);border-radius:16px;padding:16px 20px;box-shadow:0 8px 24px var(--shadow)}
.rh-cell{display:flex;flex-direction:column;min-width:0}.rh-k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.rh-v{font-size:15px;font-weight:600;color:var(--text);overflow-wrap:anywhere}.rh-note{margin-top:8px}
.rpanel{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:14px 18px;box-shadow:0 8px 24px var(--shadow)}
.rtitle{font-weight:700;color:var(--text);margin:18px 0 8px}
.rchart{width:100%;height:auto}.rchart .g{stroke:var(--border);stroke-width:1}.rchart .ax{fill:var(--muted);font-size:11px}
.rchart .line{fill:none;stroke:var(--primary);stroke-width:2.5;stroke-linejoin:round}.rchart .area{fill:var(--soft)}
.rchart .gold{fill:none;stroke:var(--accent);stroke-width:2;stroke-dasharray:6 4}.rchart .dot{fill:var(--primary)}
.rchart .peak{fill:#fff;stroke:var(--primary);stroke-width:3}.rchart .pk{fill:var(--text);font-size:12px;font-weight:700}
.rbars{display:flex;flex-direction:column;gap:7px}.rbar{display:grid;grid-template-columns:96px 1fr 84px;align-items:center;gap:10px;font-size:13px}
.rbar-name{color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rbar-track{background:var(--soft);border-radius:8px;height:12px;overflow:hidden}.rbar-fill{display:block;height:100%;border-radius:8px;box-shadow:inset 0 0 0 1px rgba(0,0,0,.14)}
.rbar-val{color:var(--muted);text-align:right}
.rchart .evline{stroke:var(--accent);stroke-width:3;stroke-linecap:round;opacity:.55}.rchart .evdot{fill:var(--primary)}
</style>
"""


# ---------- экспорт ----------

def report_markdown(r, ev=None):
    """Весь отчёт одним Markdown-файлом. ev: результат build_events (раздел «События»), если есть."""
    fields, note = header_fields(r)
    out = [f"# Snapshot report: {r['video_id']}", "", "## Video and run", ""]
    out += [f"- **{k}:** {v}" for k, v in fields]
    if note:
        out.append(f"- _{note}_")
    out += ["", "## Key figures", ""]
    out += [f"- **{name}:** {val}" + (f" ({sub})" if sub else "") for name, val, sub in tiles(r)]
    out += ["", "## People per snapshot", "", "| time | people |", "|---|---|"]
    out += [f"| {fmt_clock(t)} ({t:g} s) | {n} |" for t, n in r["series"]]
    for title, key in (("Clothing colors", "colors"), ("Clothing types", "types")):
        out += ["", f"## {title}", "", "| value | people | share |", "|---|---|---|"]
        total = sum(k for _, k in r[key]) or 1
        out += [f"| {v} | {k} | {k / total * 100:.0f}% |" for v, k in r[key]]
    out += ["", "## Snapshots", "", "| time | snapshot | people | gold | top colors | unknown | VLM status |",
            "|---|---|---|---|---|---|---|"]
    out += [f"| {x['time']} | {x['snapshot_id']} | {x['people']} | {x['gold_people'] if x['gold_people'] is not None else '-'} "
            f"| {x['top_colors']} | {x['unknown']} | {x['status']} |" for x in r["rows"]]
    if ev is not None:
        out += ["", "## Events", ""]
        if not ev["total"]:
            out.append("No events for this video in this time range.")
        else:
            out += ["| type | events |", "|---|---|"] + [f"| {t} | {n} |" for t, n in ev["counts"]]
            out += ["", "| time | event | type | track | zone | score | clip |", "|---|---|---|---|---|---|---|"]
            out += [f"| {x['time']} | {x['event_id']} | {x['type']} | {x['track']} | {x['zone']} | {x['score']} | {x['clip_path'] or '-'} |"
                    for x in ev["rows"]]
    return "\n".join(out) + "\n"


def _csv_safe(v):
    """Значение из базы не должно превращаться в формулу при открытии CSV в Excel."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


CSV_FIELDS = ["snapshot_id", "t", "time", "people", "gold_people", "top_colors", "unknown", "status", "frame_path"]


def snapshots_csv(r):
    """Таблица по срезам в CSV (с BOM, чтобы Excel открыл кириллицу и UTF-8 правильно)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore")
    w.writeheader()
    for x in r["rows"]:
        row = {k: ("" if x.get(k) is None else x[k]) for k in CSV_FIELDS}
        if row["top_colors"] == "-":
            row["top_colors"] = ""
        w.writerow({k: _csv_safe(v) for k, v in row.items()})
    return "﻿" + buf.getvalue()