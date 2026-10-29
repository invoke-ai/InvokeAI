"""A stand-in for `scripts/invokeai-web.py`: the HTTP routes and socket.io events the family E2E runner uses.

The flow test copies this file to `<code_dir>/scripts/invokeai-web.py`, so the runner launches it exactly as it would
launch InvokeAI. It reads the scratch root's `invokeai.yaml` for its port and log file, logs in InvokeAI's plain
format, and executes a queued graph by walking its nodes: the model loader logs a full load once per cache flush, the
`fake_denoise` node sends preview frames, and the non-intermediate node returns a PNG of its width and height.
When `<code_dir>/crash-at-enqueue` holds N, the server deletes the file and dies while running its N-th queue item.
The working-memory calibrate route enqueues like enqueue_batch and records one point per model node in a store file
under the root's db_dir, as the real server does, and the request's `graph_models` with each run; a request without
them is refused with 422, as the real server refuses it. `<code_dir>/calibration-refused` holds `{"at": N or null, "status":
"<HTTP status line>", "detail": ...}`: the route answers the N-th request (every one for null) with that error.
`<code_dir>/calibration-unavailable` makes the status report calibration unavailable, with the file's text as reason.
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
CODE_DIR = Path(__file__).resolve().parents[1]
CRASH_MARKER = CODE_DIR / "crash-at-enqueue"
CALIBRATION_REFUSED = CODE_DIR / "calibration-refused"
CALIBRATION_UNAVAILABLE = CODE_DIR / "calibration-unavailable"
DEVICE_CLASS_ID = "0123456789abcdef"
MODEL_CONFIG = {
    "key": "fake-model",
    "name": "Fake",
    "base": "fake",
    "type": "main",
    "format": "checkpoint",
    "variant": "normal",
    "hash": "blake3:fakehash",
    "path": "/models/fake.safetensors",
    "source": "https://example.com/fake.safetensors?token=fake-token",
    "source_type": "url",
    "default_settings": {},
}


class _Server(socketserver.ThreadingMixIn, WSGIServer):
    daemon_threads = True


class _QuietHandler(WSGIRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass


class FakeInvoke:
    def __init__(self, logger: logging.Logger, store_dir: Path) -> None:
        self.logger = logger
        self.store_file = store_dir / f"{DEVICE_CLASS_ID}.json"
        self.calibrate_requests = 0
        self.calibration_runs = 0
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
        run_id = item["batch"].get("origin", "").removeprefix("invokeai:wm-calibration:")
        if run_id != item["batch"].get("origin", ""):
            self.record_points(
                run_id, item_id, [node_id for node_id, node in graph["nodes"].items() if "model" in node]
            )
        item.update(
            status="completed",
            completed_at=str(datetime.now()),
            session={"source_prepared_mapping": mapping, "results": results},
        )
        self.emit("queue_item_status_changed", {"item_id": item_id, "status": "completed"})

    def enqueue(self, batch: dict) -> int:
        with self.lock:
            item_id = len(self.items) + 1
            self.items[item_id] = {"item_id": item_id, "status": "pending", "batch": batch, "session": {}}
            crash = CRASH_MARKER.exists() and CRASH_MARKER.read_text() == str(item_id)
            if crash:
                CRASH_MARKER.unlink()
        threading.Thread(target=self.execute, args=(item_id, crash), daemon=True).start()
        return item_id

    def store(self) -> dict:
        if self.store_file.is_file():
            return json.loads(self.store_file.read_text(encoding="utf-8"))
        return {"schema_version": 1, "device_class": {"id": DEVICE_CLASS_ID}, "runs": [], "points": [], "envs": []}

    def record_points(self, run_id: str, item_id: int, node_ids: list[str]) -> None:
        with self.lock:
            store = self.store()
            store["points"] += [{"run_id": run_id, "queue_item_id": item_id, "node_id": n} for n in node_ids]
            self.store_file.parent.mkdir(parents=True, exist_ok=True)
            self.store_file.write_text(json.dumps(store), encoding="utf-8")

    def calibrate(self, model_key: str, body: dict) -> tuple[dict, str]:
        self.calibrate_requests += 1
        if CALIBRATION_REFUSED.exists():
            refusal = json.loads(CALIBRATION_REFUSED.read_text())
            if refusal["at"] in (None, self.calibrate_requests):
                return {"detail": refusal["detail"]}, refusal["status"]
        if not isinstance(body.get("graph_models"), dict):
            return {"detail": "graph_models: Field required"}, "422 Unprocessable Entity"
        self.calibration_runs += 1
        run_id = f"run-{self.calibration_runs}"
        with self.lock:
            store = self.store()
            store["runs"].append({"run_id": run_id, "model_key": model_key, "graph_models": body["graph_models"]})
            self.store_file.parent.mkdir(parents=True, exist_ok=True)
            self.store_file.write_text(json.dumps(store), encoding="utf-8")
        batch = {"graph": body["graph"], "data": body.get("data"), "origin": f"invokeai:wm-calibration:{run_id}"}
        return {"run_id": run_id, "device_class_id": DEVICE_CLASS_ID, "item_ids": [self.enqueue(batch)]}, "201 Created"

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
            return reply({"models": [MODEL_CONFIG]})
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
            return reply({"item_ids": [self.enqueue(json.loads(body)["batch"])]}, status="201 Created")
        if match := re.fullmatch(r"/api/v2/models/i/([^/]+)/working_memory/calibrate", path):
            payload, status = self.calibrate(match[1], json.loads(body))
            return reply(payload, status=status)
        if path == "/api/v2/models/working_memory/status":
            reason = CALIBRATION_UNAVAILABLE.read_text() if CALIBRATION_UNAVAILABLE.exists() else None
            status = {"available": reason is None, "unavailable_reason": reason}
            return reply(status | {"current_device_class_id": DEVICE_CLASS_ID})
        if path == "/api/v2/models/working_memory/export":
            if not self.store_file.is_file() or environ.get("QUERY_STRING") != f"device_class_id={DEVICE_CLASS_ID}":
                return reply({"detail": "no store"}, status="404 Not Found")
            return reply(self.store_file.read_bytes())
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

    fake = FakeInvoke(logger, root / config.get("db_dir", "databases") / "working_memory")
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
