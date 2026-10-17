"""A stand-in for `scripts/invokeai-web.py`: the HTTP routes and socket.io events the family E2E runner uses.

The flow test copies this file to `<code_dir>/scripts/invokeai-web.py`, so the runner launches it exactly as it would
launch InvokeAI. It reads the scratch root's `invokeai.yaml` for its port and log file, logs in InvokeAI's plain
format, and executes a queued graph by walking its nodes: the model loader logs a full load once per cache flush, the
`fake_denoise` node sends preview frames, and the non-intermediate node returns a PNG of its width and height.
When `<code_dir>/crash-at-enqueue` holds N, the server deletes the file and dies while running its N-th queue item.
Self-contained on purpose: it runs in a fresh interpreter with only the code directory on the path.
"""

import argparse
import io
import json
import logging
import os
import re
import socketserver
import threading
import time
from datetime import datetime
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

import socketio
import yaml
from PIL import Image

STEP_S = 0.01
CRASH_MARKER = Path(__file__).resolve().parents[1] / "crash-at-enqueue"


class _Server(socketserver.ThreadingMixIn, WSGIServer):
    daemon_threads = True


class _QuietHandler(WSGIRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass


class FakeInvoke:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.sio = socketio.Server(async_mode="threading")
        self.sio.on("subscribe_queue", self.subscribe)
        self.lock = threading.Lock()
        self.items: dict[int, dict] = {}
        self.images: dict[str, bytes] = {}
        self.resident: set[str] = set()
        self.uploads = 0

    def subscribe(self, sid, data):
        self.sio.enter_room(sid, data["queue_id"])

    def emit(self, name: str, data: dict) -> None:
        self.sio.emit(name, {"queue_id": "default", **data}, room="default")

    def execute(self, item_id: int, crash: bool) -> None:
        item = self.items[item_id]
        graph = item["batch"]["graph"]
        item.update(status="in_progress", started_at=str(datetime.now()))
        self.emit("queue_item_status_changed", {"item_id": item_id, "status": "in_progress"})
        if crash:
            time.sleep(STEP_S)
            os._exit(3)
        mapping, results = {}, {}
        for node_id, node in graph["nodes"].items():
            prepared = f"{node_id}-prepared"
            mapping[node_id] = [prepared]
            invocation = {"id": prepared, "type": node["type"]}
            base = {"item_id": item_id, "invocation": invocation, "invocation_source_id": node_id}
            self.emit("invocation_started", base)
            result: dict = {"type": "fake_output"}
            if "model" in node:
                key = node["model"]["key"]
                if key not in self.resident:
                    load = {"config": {"key": key}, "submodel_type": "transformer"}
                    self.sio.emit("model_load_started", load, room="default")
                    time.sleep(STEP_S)
                    self.logger.info(
                        f"[MODEL CACHE] Loaded model '{key}:transformer' (FakeTransformer) onto cuda device #0 in "
                        "0.01s. Total model size: 1024.00MB, VRAM: 1024.00MB (100.0%)"
                    )
                    self.sio.emit("model_load_complete", load, room="default")
                    self.resident.add(key)
            if node["type"] == "fake_denoise":
                for step in range(1, node["steps"] + 1):
                    time.sleep(STEP_S)
                    frame = {"percentage": step / node["steps"], "message": "Denoising", "image": {"dataURL": "x"}}
                    self.emit("invocation_progress", base | frame)
            if not node.get("is_intermediate", False):
                buffer = io.BytesIO()
                Image.new("RGB", (node["width"], node["height"]), (200, 80, 40)).save(buffer, format="PNG")
                name = f"output-{item_id}.png"
                self.images[name] = buffer.getvalue()
                result = {"type": "image_output", "image": {"image_name": name}}
            results[prepared] = result
            time.sleep(STEP_S)
            self.emit("invocation_complete", base | {"result": result})
        item.update(
            status="completed",
            completed_at=str(datetime.now()),
            session={"source_prepared_mapping": mapping, "results": results},
        )
        self.emit("queue_item_status_changed", {"item_id": item_id, "status": "completed"})

    def __call__(self, environ, start_response):
        method, path = environ["REQUEST_METHOD"], environ["PATH_INFO"]
        length = int(environ.get("CONTENT_LENGTH") or 0)
        body = environ["wsgi.input"].read(length) if length else b""

        def reply(payload, status="200 OK", content_type="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            start_response(status, [("Content-Type", content_type), ("Content-Length", str(len(data)))])
            return [data]

        if path == "/api/v1/app/version":
            return reply({"version": "0.0.0-fake"})
        if path == "/api/v1/app/app_deps":
            return reply({"torch": "fake"})
        if path == "/api/v1/app/runtime_config":
            return reply({"set_fields": [], "config": {"multiuser": False, "fp8_compute": False}})
        if path == "/api/v2/models/":
            return reply({"models": [{"key": "fake-model", "base": "fake", "type": "main", "default_settings": {}}]})
        if path == "/api/v2/models/capabilities":
            return reply({"fake": {}})
        if path == "/api/v2/models/empty_model_cache":
            self.logger.info("Emptying model cache.")
            self.resident.clear()
            return reply({"models_cleared": 0, "bytes_freed": 0})
        if path == "/api/v1/images/upload":
            with self.lock:
                self.uploads += 1
                return reply({"image_name": f"upload-{self.uploads}.png"}, status="201 Created")
        if path == "/api/v1/queue/default/enqueue_batch":
            with self.lock:
                item_id = len(self.items) + 1
                self.items[item_id] = {
                    "item_id": item_id,
                    "status": "pending",
                    "batch": json.loads(body)["batch"],
                    "session": {},
                }
                crash = CRASH_MARKER.exists() and CRASH_MARKER.read_text() == str(item_id)
                if crash:
                    CRASH_MARKER.unlink()
            threading.Thread(target=self.execute, args=(item_id, crash), daemon=True).start()
            return reply({"item_ids": [item_id]}, status="201 Created")
        if path == "/api/v1/queue/default/status":
            states = [item["status"] for item in self.items.values()]
            return reply(
                {
                    "queue": {"pending": states.count("pending"), "in_progress": states.count("in_progress")},
                    "processor": {},
                }
            )
        if match := re.fullmatch(r"/api/v1/queue/default/i/(\d+)", path):
            item = self.items[int(match[1])]
            return reply({k: v for k, v in item.items() if k != "batch"})
        if match := re.fullmatch(r"/api/v1/images/i/([^/]+)/full", path):
            return reply(self.images[match[1]], content_type="image/png")
        return reply({"detail": f"no route {method} {path}"}, status="404 Not Found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    root = Path(parser.parse_args().root)
    config = yaml.safe_load((root / "invokeai.yaml").read_text(encoding="utf-8"))
    log_file = next(h.split("=", 1)[1] for h in config["log_handlers"] if h.startswith("file="))
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s]::[%(name)s]::%(levelname)s --> %(message)s"))
    logger = logging.getLogger("InvokeAI")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    fake = FakeInvoke(logger)
    app = socketio.WSGIApp(fake.sio, fake, socketio_path="/ws/socket.io")
    port = config["port"]
    while True:
        try:
            httpd = make_server("127.0.0.1", port, app, server_class=_Server, handler_class=_QuietHandler)
            break
        except OSError:
            port += 1
    logger.info(f"Invoke running on http://127.0.0.1:{port} (Press CTRL+C to quit)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
