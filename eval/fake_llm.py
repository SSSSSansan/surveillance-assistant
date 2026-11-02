"""Заглушки модели для проверки ПРОВОДКИ (цикл агента, HTTP-клиент, оценка, страница).
Это не модели и их результаты НЕ говорят о качестве настоящей LLM: «оракул» знает правильные
вызовы из questions.csv и поэтому всегда выбирает функцию верно."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _last_user(messages):
    return next(m["content"] for m in reversed(messages) if m["role"] == "user")


def _tool_call(name, arguments, call_id="call_1"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}]}


class OracleClient:
    """Вызывает функцию и аргументы из questions.csv, а ответ строит из РЕАЛЬНОГО результата функции
    (кроме трёх вопросов на последовательность и вопросов вне базы: их ответ берётся из эталона).
    Так проверяется весь путь: функция -> результат -> ответ -> грейдер."""
    def __init__(self, rows):
        self.by_q = {r["question"]: r for r in rows}
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        row = self.by_q[_last_user(messages)]
        if messages[-1]["role"] != "tool":
            if row["expected_tool"] == "none":
                return {"role": "assistant", "content": row["expected_answer"]}
            return _tool_call(row["expected_tool"], row["expected_args"])
        result = json.loads(messages[-1]["content"])
        return {"role": "assistant", "content": self._answer(row, result)}

    @staticmethod
    def _answer(row, result):
        t = row["type"]
        if t == "exists":
            return "Да" if result.get("count") else "Нет, не нашёл"
        if t == "count":
            return str(result["count"])
        if t == "time":
            evs = result.get("events") or []
            return f"{evs[0]['t_start']} с" if evs else "Не нашёл"
        if t == "list":
            evs = result.get("events") or []
            if row["expected_tool"] == "get_track_timeline":
                return "; ".join(f"{e['event_type']} ({e['t_start']} с)" for e in evs)
            return "; ".join(f"человек {e['track_id']} ({e['t_start']} с)" for e in evs)
        return row["expected_answer"]  # sequence, out_of_scope


class WrongToolClient:
    """Всегда вызывает count_events без аргументов и отвечает «0»."""
    def chat(self, messages, tools=None):
        if messages[-1]["role"] != "tool":
            return _tool_call("count_events", "{}")
        return {"role": "assistant", "content": "0"}


class NoToolClient:
    """Никогда не вызывает функции и отвечает «Да»."""
    def chat(self, messages, tools=None):
        return {"role": "assistant", "content": "Да"}


class FakeOpenAIServer:
    """Локальный HTTP-сервер с /v1/chat/completions. responder(client_obj) отвечает на запросы."""
    def __init__(self, client):
        self.client = client
        self.requests = []
        self.headers = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
                outer.requests.append(body)
                outer.headers.append(dict(self.headers))
                if self.path != "/v1/chat/completions":
                    self.send_response(404); self.end_headers(); return
                try:
                    msg = outer.client.chat(body["messages"], body.get("tools"))
                except Exception:  # сбой «модели» -> HTTP 500, как у настоящего сервера
                    self.send_response(500); self.send_header("Content-Length", "0"); self.end_headers(); return
                data = json.dumps({"choices": [{"index": 0, "message": msg}]}, ensure_ascii=False).encode("utf-8")
                if not isinstance(msg, dict):  # заглушка неверного формата: отдаём как есть
                    data = json.dumps({"unexpected": msg}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):  # тишина в выводе тестов
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def __enter__(self):
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
