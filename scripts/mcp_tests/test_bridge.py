"""Lightweight bridge tests: run without importing the InvokeAI inference application."""

import gzip
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import jsonschema
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SCRIPT = Path(__file__).resolve().parents[1] / "invokeai-mcp.py"
spec = importlib.util.spec_from_file_location("invokeai_mcp", SCRIPT)
assert spec and spec.loader
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://localhost:9090",
        "http://127.0.0.1.evil.test",
        "http://127.0.0.1/api",
        "http://user:secret@127.0.0.1",
        "http://127.0.0.1?x=y",
        "http://127.0.0.1#x",
        "http://127.0.0.1:0",
        "http://127.0.0.1:99999",
        "file:///etc/passwd",
        "http://[invalid",
    ],
)
def test_rejects_non_loopback_origins_and_embedded_secrets(url):
    with pytest.raises(ValueError, match="literal loopback"):
        bridge.validate_base_url(url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:9090", "https://[::1]:9090/"])
def test_accepts_literal_loopback(url):
    assert bridge.validate_base_url(url) == url.rstrip("/")


@pytest.fixture
def api():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.command, self.path, self.headers.get("Authorization")))
            status, body = responses.get(self.path, (404, b"private-error-details"))
            assert self.headers.get("Accept-Encoding") == "identity"
            self.send_response(status)
            if body.startswith(b"\x1f\x8b"):
                self.send_header("Content-Encoding", "gzip")
            if status == 302:
                self.send_header("Location", "/unexpected")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            calls.append((self.command, self.path, self.headers.get("Authorization")))
            posts.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            status, body = responses.get(self.path, (404, b"private-error-details"))
            if status == 0:  # Simulate response lost after the server read the complete submission.
                self.close_connection = True
                return
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    calls = []
    responses = {}
    posts = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls, responses, posts
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.asyncio
async def test_stdio_tools_use_fixed_read_routes_and_sanitize_failures(api):
    url, calls, responses, _ = api
    responses.update(
        {
            "/api/v1/workflows/?page=0&per_page=20": (200, b'{"items": [], "total": 0}'),
            "/api/v1/workflows/i/abc-123": (200, b'{"workflow": {"name": "example"}}'),
            "/api/v1/queue/default/status": (200, b'{"queue": {"pending": 0}}'),
            "/api/v1/queue/default/i/4": (200, b'{"item_id": 4, "status": "completed"}'),
        }
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SCRIPT), "--base-url", url],
        env={"INVOKEAI_MCP_TOKEN": "test-only-token", "HTTP_PROXY": "http://127.0.0.1:1"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            assert {t.name for t in tools} == {
                "list_workflows",
                "get_workflow",
                "get_queue_status",
                "get_queue_item",
                "get_submission",
                "get_batch_status",
            }
            assert all(t.annotations and t.annotations.readOnlyHint for t in tools)
            for name, args, expected in [
                ("list_workflows", {}, {"items": [], "total": 0}),
                ("get_workflow", {"workflow_id": "abc-123"}, {"workflow": {"name": "example"}}),
                ("get_queue_status", {}, {"queue": {"pending": 0}}),
                ("get_queue_item", {"item_id": 4}, {"item_id": 4, "status": "completed"}),
            ]:
                result = await session.call_tool(name, args)
                assert not result.isError
                assert result.structuredContent == expected
            assert [path for _, path, _ in calls] == list(responses)
            assert all(method == "GET" and auth == "Bearer test-only-token" for method, _, auth in calls)

            # Schema validation must reject route injection and unlimited queries before any request.
            count = len(calls)
            for name, args in [
                ("get_workflow", {"workflow_id": "../../auth"}),
                ("get_queue_status", {"queue_id": "x?foo=1"}),
                ("get_queue_item", {"item_id": 0}),
                ("list_workflows", {"per_page": 0}),
                ("list_workflows", {"per_page": 101}),
                ("list_workflows", {"page": -1}),
            ]:
                assert (await session.call_tool(name, args)).isError
            assert len(calls) == count

            for status, body, message in [
                (401, b"private-error-details", "HTTP 401"),
                (403, b"private-error-details", "HTTP 403"),
                (404, b"private-error-details", "HTTP 404"),
                (500, b"private-error-details", "HTTP 500"),
                (302, b"private-error-details", "HTTP 302"),
                (200, b"private-error-details", "invalid JSON"),
                (200, b"[]", "unexpected JSON shape"),
                (200, gzip.compress(b"x" * (bridge.MAX_RESPONSE_BYTES + 1)), "compressed content"),
                (200, b"x" * (bridge.MAX_RESPONSE_BYTES + 1), "2 MiB"),
            ]:
                responses["/api/v1/workflows/i/abc-123"] = status, body
                before = len(calls)
                result = await session.call_tool("get_workflow", {"workflow_id": "abc-123"})
                assert result.isError
                output = result.model_dump_json()
                assert message in output
                assert "private-error-details" not in output
                assert "test-only-token" not in output
                assert len(calls) == before + 1  # No redirects or retries.
            assert not any(path == "/unexpected" for _, path, _ in calls)


@pytest.mark.asyncio
async def test_empty_token_rejected_and_connection_failure_is_safe():
    with pytest.raises(ValueError, match="bearer token"):
        bridge.create_server("http://127.0.0.1:9090", "bad\ntoken")
    with pytest.raises(ValueError, match="bearer token"):
        bridge.create_server("http://127.0.0.1:9090", "")
    # A released ephemeral socket has no listener; no production service is contacted.
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = server.server_port
    server.server_close()
    mcp = bridge.create_server(f"http://127.0.0.1:{port}")
    with pytest.raises(Exception, match="connection failed or timed out") as error:
        await mcp.call_tool("get_queue_status", {})
    assert str(port) not in str(error.value)


@pytest.mark.asyncio
async def test_execution_is_opt_in_and_validates_before_submission(api):
    url, calls, responses, posts = api
    readonly = bridge.create_server(url)
    assert "run_saved_workflow" not in {tool.name for tool in await readonly.list_tools()}
    server = bridge.create_server(url, allowed_workflows=("abc-123",))
    args = {"workflow_id": "not-allowed", "request_id": str(uuid4())}
    with pytest.raises(Exception, match="allowlist"):
        await server.call_tool("run_saved_workflow", args)
    assert not calls
    args["workflow_id"] = "abc-123"
    responses["/openapi.json"] = (200, b'{"components": {"schemas": {}}}')
    with pytest.raises(Exception, match="contract"):
        await server.call_tool("run_saved_workflow", args)
    assert not posts
    spec = json.loads((SCRIPT.parents[1] / "invokeai/frontend/api/openapi.json").read_text())
    responses["/openapi.json"] = (200, json.dumps(spec).encode())
    responses["/api/v1/workflows/i/abc-123"] = (200, b'{"call_saved_workflow_compatibility":{"is_callable":false}}')
    with pytest.raises(Exception, match="not compatible"):
        await server.call_tool("run_saved_workflow", args)
    assert not posts


@pytest.mark.asyncio
async def test_stdio_submission_preserves_retry_payload_and_reads_receipt(api):
    url, calls, responses, posts = api
    spec = json.loads((SCRIPT.parents[1] / "invokeai/frontend/api/openapi.json").read_text())
    request_id = str(uuid4())
    receipt = {"batch_id": request_id, "enqueued": 1, "requested": 1, "item_ids": [42]}
    receipt_path = f"/api/v1/queue/default/enqueue_batch/receipt?idempotency_key=mcp%3A{request_id}"
    responses.update(
        {
            "/openapi.json": (200, json.dumps(spec).encode()),
            "/api/v1/workflows/i/abc-123": (200, b'{"call_saved_workflow_compatibility":{"is_callable":true}}'),
            "/api/v1/queue/default/enqueue_batch": (0, b""),
            receipt_path: (200, json.dumps(receipt).encode()),
            f"/api/v1/queue/default/b/{request_id}/status": (200, b'{"completed":1,"total":1}'),
            "/api/v1/queue/default/i/42": (
                200,
                b'{"status":"completed","session":{"results":{"prepared-call":{"values":{"image":{"image_name":"test.png"}}}}}}',
            ),
        }
    )
    params = StdioServerParameters(
        command=sys.executable, args=[str(SCRIPT), "--base-url", url, "--allow-workflow", "abc-123"]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            args = {"workflow_id": "abc-123", "request_id": request_id}
            failed = await session.call_tool("run_saved_workflow", args)
            assert failed.isError
            assert request_id in failed.model_dump_json()
            assert "never a new ID" in failed.model_dump_json()
            assert len(posts) == 1  # No automatic replay after an ambiguous outcome.
            result = await session.call_tool("get_submission", {"request_id": request_id})
            assert not result.isError and result.structuredContent == receipt
            # The payload matches the real generated endpoint schema, not a copied fixture schema.
            endpoint = spec["paths"]["/api/v1/queue/{queue_id}/enqueue_batch"]["post"]
            body_schema = endpoint["requestBody"]["content"]["application/json"]["schema"]
            jsonschema.validate(posts[0], {**spec, **body_schema})
            assert posts[0]["batch"]["idempotency_key"] == "mcp:" + request_id
            assert posts[0]["batch"]["runs"] == 1
            assert posts[0]["batch"]["graph"]["nodes"]["call"]["workflow_inputs"] == {}
            responses["/api/v1/queue/default/enqueue_batch"] = (200, json.dumps(receipt).encode())
            retried = await session.call_tool("run_saved_workflow", args)
            assert not retried.isError
            assert len(posts) == 2 and posts[0] == posts[1]
            status = await session.call_tool("get_batch_status", {"batch_id": request_id})
            assert not status.isError and status.structuredContent["completed"] == 1
            result = await session.call_tool("get_queue_item", {"item_id": 42})
            assert not result.isError
            assert (
                result.structuredContent["session"]["results"]["prepared-call"]["values"]["image"]["image_name"]
                == "test.png"
            )
            responses["/api/v1/queue/default/enqueue_batch"] = (409, b"private conflict details")
            conflict = await session.call_tool("run_saved_workflow", args)
            assert conflict.isError and "HTTP 409" in conflict.model_dump_json()
            assert "private conflict" not in conflict.model_dump_json()


@pytest.mark.asyncio
async def test_restart_recovers_receipt_without_execution_or_acknowledgement(api):
    url, calls, responses, posts = api
    request_id = str(uuid4())
    receipt = {"batch_id": request_id, "enqueued": 1, "requested": 1, "item_ids": [42]}
    path = f"/api/v1/queue/default/enqueue_batch/receipt?idempotency_key=mcp%3A{request_id}"
    responses[path] = (200, json.dumps(receipt).encode())
    # Two separate stdio processes, no execution allowlist or workflow-read prerequisite.
    for _ in range(2):
        params = StdioServerParameters(command=sys.executable, args=[str(SCRIPT), "--base-url", url])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("get_submission", {"request_id": request_id})
                assert not result.isError and result.structuredContent == receipt
    assert not posts
    assert [p for _, p, _ in calls] == [path, path]


@pytest.mark.asyncio
async def test_receipt_quota_refusal_does_not_acknowledge_or_replay(api):
    url, calls, responses, posts = api
    schema = json.loads((SCRIPT.parents[1] / "invokeai/frontend/api/openapi.json").read_text())
    responses["/openapi.json"] = (200, json.dumps(schema).encode())
    responses["/api/v1/workflows/i/abc-123"] = (200, b'{"call_saved_workflow_compatibility":{"is_callable":true}}')
    responses["/api/v1/queue/default/enqueue_batch"] = (429, b"private quota details")
    server = bridge.create_server(url, allowed_workflows=("abc-123",))
    request_id = str(uuid4())
    with pytest.raises(Exception, match="HTTP 429") as error:
        await server.call_tool("run_saved_workflow", {"workflow_id": "abc-123", "request_id": request_id})
    message = str(error.value)
    assert request_id in message and "stop new submissions" in message
    assert "private quota details" not in message
    assert len(posts) == 1
    assert not any("acknowledge" in path for _, path, _ in calls)
