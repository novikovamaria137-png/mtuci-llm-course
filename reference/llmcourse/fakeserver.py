"""Минимальный OpenAI-совместимый сервер. Урок 2.5.

Нужен, чтобы доказать переносимость кода, не имея ни ключа, ни GPU.
Сервер отвечает на POST /v1/chat/completions в том же формате, что и
настоящий поставщик, — а значит, клиент из урока 2.1 не должен заметить
разницы. Если не заметит, переносимость не декларация, а факт.

Это НЕ языковая модель. Он ничего не генерирует, а возвращает заготовку.
Проверяется совместимость интерфейса, а не качество ответов.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class _Handler(BaseHTTPRequestHandler):
    delay = 0.0                      # искусственная задержка, секунды

    def log_message(self, *args):    # тишина в выводе ноутбука
        pass

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            self._json({"object": "list", "data": [
                {"id": "fake-local-model", "object": "model", "owned_by": "local"}]})
        else:
            self._json({"error": {"message": "not found"}}, code=404)

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json({"error": {"message": "not found"}}, code=404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages", [])
        user = next((m["content"] for m in reversed(messages)
                     if m.get("role") == "user"), "")

        if self.delay:
            time.sleep(self.delay)

        text = (f"[локальный сервер] Получено сообщений: {len(messages)}. "
                f"Последний вопрос: {str(user)[:60]}")

        prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 3
        completion_tokens = len(text) // 3

        self._json({
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "fake-local-model"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })

    def _json(self, payload, code=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start(port=0, delay=0.0):
    """Запускает сервер в отдельном потоке. Возвращает (base_url, stop)."""
    _Handler.delay = delay
    srv = HTTPServer(("127.0.0.1", port), _Handler)
    real_port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    def stop():
        srv.shutdown()
        srv.server_close()

    return f"http://127.0.0.1:{real_port}/v1", stop
