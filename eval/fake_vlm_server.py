"""Fake OpenAI-compatible /v1/chat/completions server for tests (stdlib only, localhost, random port).

    with FakeVlmServer(lambda payload: '{"people_count": 3}') as srv:
        url = srv.url            # http://127.0.0.1:PORT/v1/chat/completions
        srv.requests             # list of received JSON payloads

The handler gets the request payload and returns the answer text (finish_reason "stop"), or a dict
{"content": text, "finish_reason": "length"} to imitate an answer cut at max_tokens, or raises / returns None
for HTTP 500. A zero count is just the text '{"people_count": 0}'.
It is a stand-in for the real vLLM: it proves our code builds requests and parses answers,
NOT that a real model answers well.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class FakeVlmServer:
    def __init__(self, handler):
        self.handler, self.requests = handler, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                outer.requests.append(payload)
                try:
                    text = outer.handler(payload)
                except Exception:
                    text = None
                if text is None:
                    self.send_response(500)
                    self.end_headers()
                    return
                finish = "stop"
                if isinstance(text, dict):
                    text, finish = text["content"], text.get("finish_reason", "stop")
                body = json.dumps({"choices": [{"message": {"content": text}, "finish_reason": finish}],
                                   "usage": {"prompt_tokens": 100, "completion_tokens": 10}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self._srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self._srv.server_port}/v1/chat/completions"

    def __enter__(self):
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._srv.shutdown()
        self._srv.server_close()
