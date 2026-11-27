import io
import json
from collections import deque
from types import SimpleNamespace

import pytest
import requests
from pydantic import ValidationError
from urllib3.response import HTTPResponse

from invokeai.app.invocations.minimax_h3.minimax_h3_context_ir import MiniMaxH3ContextIRInvocation
from invokeai.app.invocations.text_encoder.minimax_h3_text_encoder import MiniMaxH3TextEncoderInvocation
from invokeai.app.services.external_generation import minimax_context_ir
from invokeai.app.services.external_generation.errors import (
    ExternalProviderNotConfiguredError,
    ExternalProviderRequestError,
)
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from tests.test_nodes import create_edge


def task(status="succeeded", **changes):
    return {"task": {"id": "h3-job", "task_type": "h3_context_ir", "status": status, **changes}}


def context(base_url=None, api_key="test-key", canceled=lambda: False):
    settings = SimpleNamespace(external_minimax_base_url=base_url, external_minimax_api_key=api_key)
    return SimpleNamespace(config=SimpleNamespace(get=lambda: settings), util=SimpleNamespace(is_canceled=canceled))


@pytest.fixture
def transport(monkeypatch):
    replies = deque()
    requests_sent = []

    def send(session, request, **kwargs):
        requests_sent.append(request)
        reply = replies.popleft()
        if isinstance(reply, Exception):
            raise reply
        status, body, *extra = reply
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        response = requests.Response()
        response.status_code = status
        response.request = request
        if extra:
            response.headers.update(extra[0])
        response.url = request.url
        response.raw = HTTPResponse(body=io.BytesIO(raw), headers=response.headers, preload_content=False)
        if 300 <= status < 400:
            response.headers["Location"] = "https://outside.example/redirection"
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    monkeypatch.setattr(minimax_context_ir, "_POLL_INTERVAL", 0)
    return replies, requests_sent


@pytest.mark.parametrize(
    ("base_url", "host"), [(None, "https://api.minimax.io"), ("https://api.minimax.cn/", "https://api.minimax.cn")]
)
def test_workflow_routes_the_enhanced_text_to_the_local_h3_prompt(base_url, host, transport):
    replies, sent = transport
    replies.extend(
        [
            (200, {"task_id": "h3-job"}),
            (200, task("queued")),
            (503, {}),
            (200, task("running")),
            (200, task(content={"prompt": "[Shot 1] A slow pan over the red kite."})),
        ]
    )
    graph = Graph()
    graph.add_node(MiniMaxH3ContextIRInvocation(id="expand", prompt="A red kite", duration=7, ratio="9:16"))
    graph.add_node(MiniMaxH3TextEncoderInvocation(id="encode"))
    graph.add_edge(create_edge("expand", "value", "encode", "prompt"))
    state = GraphExecutionState(graph=graph)
    producer = state.next()
    assert isinstance(producer, MiniMaxH3ContextIRInvocation)
    result = producer.invoke(context(base_url))
    state.complete(producer.id, result)
    consumer = state.next()
    assert isinstance(consumer, MiniMaxH3TextEncoderInvocation)
    assert consumer.prompt == "[Shot 1] A slow pan over the red kite."
    assert [(r.method, r.url) for r in sent] == [
        ("POST", host + "/v2/h3_context_ir"),
        *[("GET", host + "/v2/query/video_generation/h3-job")] * 4,
    ]
    assert json.loads(sent[0].body) == {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "A red kite"}],
        "duration": 7,
        "ratio": "9:16",
    }
    assert all(r.headers["Authorization"] == "Bearer test-key" for r in sent)


@pytest.mark.parametrize("fields", [{"duration": 3}, {"duration": 16}, {"ratio": "adaptive"}, {"prompt": "p" * 7001}])
def test_node_rejects_unsupported_text_task_inputs(fields):
    with pytest.raises(ValidationError):
        MiniMaxH3ContextIRInvocation(**{"prompt": "A red kite", **fields})


def test_blank_prompt_is_rejected_before_submission(transport):
    with pytest.raises(ValueError, match="non-blank"):
        MiniMaxH3ContextIRInvocation(prompt=" \n ").invoke(context())
    assert not transport[1]


@pytest.mark.parametrize(
    ("key", "origin"),
    [
        (None, None),
        ("test-key", "https://api.minimax.io.example"),
        ("test-key", "http://api.minimax.cn"),
        ("key\nvalue", None),
    ],
)
def test_missing_credential_or_untrusted_destination_does_not_send_any_request(transport, key, origin):
    with pytest.raises(ExternalProviderNotConfiguredError):
        MiniMaxH3ContextIRInvocation(prompt="A red kite").invoke(context(origin, key))
    assert not transport[1]


@pytest.mark.parametrize(
    "reply",
    [
        requests.Timeout("secret-token"),
        (502, {"message": "secret-token"}),
        (302, {}),
        (200, b"not json: secret-token"),
        (200, {"task_id": "../other-account"}),
        (200, b"x" * (minimax_context_ir._MAX_RESPONSE_BYTES + 1)),
    ],
)
def test_submission_with_no_usable_receipt_is_never_retried_or_redirected(transport, reply):
    replies, sent = transport
    replies.append(reply)
    with pytest.raises(ExternalProviderRequestError, match="history") as err:
        MiniMaxH3ContextIRInvocation(prompt="A red kite").invoke(context())
    assert "secret-token" not in str(err.value)
    assert len(sent) == 1


@pytest.mark.parametrize(
    "response",
    [
        (200, task("failed", error={"message": "sensitive detail"})),
        (200, task("cancelled")),
        (200, task(content={"url": "wrong result"})),
        (200, task(content={"prompt": "  "})),
        (200, task(id="foreign", content={"prompt": "foreign prompt"})),
        (200, task(task_type="generation", content={"prompt": "wrong task"})),
        (200, task("not-a-status")),
        (302, {}),
        (200, b'{"sensitive detail":'),
        (200, b"x" * (minimax_context_ir._MAX_RESPONSE_BYTES + 1)),
        (401, {"message": "sensitive detail"}),
    ],
)
def test_invalid_remote_task_never_returns_another_prompt_or_submits_again(transport, response):
    replies, sent = transport
    replies.extend([(200, {"task_id": "h3-job"}), response])
    with pytest.raises(ExternalProviderRequestError, match="h3-job") as err:
        MiniMaxH3ContextIRInvocation(prompt="A red kite").invoke(context())
    assert "sensitive detail" not in str(err.value)
    assert [r.method for r in sent] == ["POST", "GET"]


def test_persistent_poll_failures_leave_a_recoverable_task_id(transport):
    replies, sent = transport
    replies.extend([(200, {"task_id": "h3-job"}), requests.ConnectionError("secret-token"), (429, {}), (503, {})])
    with pytest.raises(ExternalProviderRequestError, match="h3-job"):
        MiniMaxH3ContextIRInvocation(prompt="A red kite").invoke(context())
    assert [r.method for r in sent] == ["POST", "GET", "GET", "GET"]


@pytest.mark.parametrize("before_submit", [True, False])
def test_cancellation_prevents_submission_or_stops_polling(transport, before_submit):
    replies, sent = transport
    replies.extend([(200, {"task_id": "h3-job"}), (200, task("running"))])
    with pytest.raises(CanceledException):
        MiniMaxH3ContextIRInvocation(prompt="A red kite").invoke(
            context(canceled=lambda: before_submit or len(sent) >= 2)
        )
    assert [r.method for r in sent] == ([] if before_submit else ["POST", "GET"])


@pytest.fixture
def clock(monkeypatch):
    class TestClock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = TestClock()
    monkeypatch.setattr(minimax_context_ir, "time", clock)
    return clock


@pytest.mark.parametrize("cancel", [True, False])
def test_long_retry_after_respects_local_cancellation_and_deadline(transport, monkeypatch, clock, cancel):
    replies, sent = transport
    replies.extend([(200, {"task_id": "h3-job"}), (429, {}, {"Retry-After": "9999"})])
    monkeypatch.setattr(minimax_context_ir, "_TASK_TIMEOUT", 2)
    settings = context(canceled=lambda: cancel and clock.now >= 0.5)
    if cancel:
        with pytest.raises(CanceledException):
            MiniMaxH3ContextIRInvocation(prompt="A red kite").invoke(settings)
    else:
        with pytest.raises(ExternalProviderRequestError, match="Timed out.*h3-job"):
            MiniMaxH3ContextIRInvocation(prompt="A red kite").invoke(settings)
    assert clock.now == (0.5 if cancel else 2)
    assert [r.method for r in sent] == ["POST", "GET"]


@pytest.mark.parametrize(
    ("status", "retry_after", "expected_delay"),
    [(429, "12", 12), (503, "Thu, 01 Jan 1970 00:00:12 GMT", 12), (429, "invalid", 3), (503, "1", 3)],
)
def test_rate_limited_queries_honor_retry_after(transport, monkeypatch, clock, status, retry_after, expected_delay):
    replies, sent = transport
    replies.extend(
        [
            (200, {"task_id": "h3-job"}),
            (status, {}, {"Retry-After": retry_after}),
            (200, task(content={"prompt": "Expanded."})),
        ]
    )
    monkeypatch.setattr("urllib3.util.retry.time", SimpleNamespace(time=lambda: 0.0))
    monkeypatch.setattr(minimax_context_ir, "_POLL_INTERVAL", 3)
    assert MiniMaxH3ContextIRInvocation(prompt="A red kite").invoke(context()).value == "Expanded."
    assert clock.now == expected_delay
    assert [r.method for r in sent] == ["POST", "GET", "GET"]


def test_truncated_query_responses_retain_the_known_task_id(transport):
    replies, sent = transport
    replies.extend([(200, {"task_id": "h3-job"}), *[(200, b"{", {"Content-Length": "99"})] * 3])
    with pytest.raises(ExternalProviderRequestError, match="h3-job"):
        MiniMaxH3ContextIRInvocation(prompt="A red kite").invoke(context())
    assert [r.method for r in sent] == ["POST", "GET", "GET", "GET"]
