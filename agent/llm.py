"""Клиент LLM для OpenAI-совместимого API (vLLM, Ollama и т. п.) без сторонних пакетов.

Сервер должен поддерживать вызов функций (tools) для выбранной модели. Для vLLM обычно нужны
флаги --enable-auto-tool-choice и --tool-call-parser <парсер для модели>: уточните в документации
vLLM для вашей модели. Какая именно модель агента будет основной, пока не решено (выбор замером).
"""
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request

from agent import config


class LLMError(RuntimeError):
    """Модель недоступна или вернула неожиданный ответ."""


_LOOPBACK = ("localhost", "127.0.0.1", "::1")


class OpenAICompatClient:
    def __init__(self, base_url=None, model=None, api_key=None, timeout=120,
                 temperature=0.0, max_tokens=512):
        self.base_url = (base_url or config.llm_base_url()).rstrip("/")
        self.model = config.llm_model() if model is None else model
        self.api_key = api_key or config.llm_api_key()
        self.timeout, self.temperature, self.max_tokens = timeout, temperature, max_tokens

    def chat(self, messages, tools=None):
        """Один запрос к модели. Возвращает сообщение ассистента (dict с content и tool_calls)."""
        if not self.model:
            raise LLMError("не задана модель (LLM_MODEL или поле «модель»)")
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        url = self.base_url + "/chat/completions"
        req = urllib.request.Request(
            url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        host = urllib.parse.urlparse(url).hostname
        opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
                  if host in _LOOPBACK else urllib.request.build_opener())  # локальную модель — мимо прокси
        try:
            with opener.open(req, timeout=self.timeout) as r:
                body = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace")
            raise LLMError(f"HTTP {e.code} от {url}: {detail}") from None
        except urllib.error.URLError as e:
            raise LLMError(f"не удалось подключиться к {url}: {e.reason}") from None
        except (socket.timeout, TimeoutError):
            raise LLMError(f"таймаут {self.timeout} с при обращении к {url}") from None
        except ValueError:
            raise LLMError(f"ответ {url} не является JSON") from None
        try:
            msg = body["choices"][0]["message"]
            if not isinstance(msg, dict):
                raise TypeError
        except (KeyError, IndexError, TypeError):
            raise LLMError("неожиданный формат ответа модели (нет choices[0].message)") from None
        return msg
