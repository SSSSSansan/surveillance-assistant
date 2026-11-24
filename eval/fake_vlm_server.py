"""Fake OpenAI-compatible /v1/chat/completions server for tests (stdlib only, localhost, random port).

    with FakeVlmServer(lambda payload: '{"people_count": 3}') as srv:
        url = srv.url            # http://127.0.0.1:PORT/v1/chat/completions
        srv.requests             # list of received JSON payloads

The handler gets the request payload and returns the answer text (or raises / returns None for HTTP 500).
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
                body = json.dumps({"choices": [{"message": {"content": text}}],
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
