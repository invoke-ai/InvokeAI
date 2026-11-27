"""GET /api/v2/models/fp8_storage_support — which models FP8 Storage actually reaches.

The rows come from the loader registry at import time: no service, no database, the same response for
every install. A client fetches it once and joins it against its model records, so what matters here is
that the wire shape is the one a client can join on, and that the key really is `(base, type, format)`.

The route is authenticated like every other one in this router, which is the only reason these tests
patch `ApiDependencies` at all.
"""

from typing import Any, Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.invoker import Invoker
from invokeai.backend.model_manager.load.fp8_capability import FP8_STORAGE_MODEL_TYPES

URL = "/api/v2/models/fp8_storage_support"


class _MockApiDependencies(ApiDependencies):
    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker  # type: ignore[misc]


@pytest.fixture
def client(monkeypatch: Any, mock_invoker: Invoker) -> Iterator[TestClient]:
    mock_invoker.services.users = MagicMock()
    mock_deps = _MockApiDependencies(mock_invoker)
    for module in ("invokeai.app.api_app", "invokeai.app.api.auth_dependencies"):
        monkeypatch.setattr(f"{module}.ApiDependencies", mock_deps)
    yield TestClient(app)


def _answers(client: TestClient) -> dict[tuple[str, str, str], bool]:
    response = client.get(URL)
    assert response.status_code == 200
    return {(row["base"], row["type"], row["format"]): row["supported"] for row in response.json()}


def test_the_key_is_the_loader_key_and_all_three_parts_change_the_answer(client: TestClient) -> None:
    """Why the table exists in this shape rather than as a field on the architecture row.

    Same base, different type: FLUX main casts, FLUX ControlNet does not. Same base and type, different
    format: a Wan checkpoint casts, a Wan GGUF must never be re-encoded. Either half alone would have to
    be wrong about one of these.
    """
    answers = _answers(client)

    assert answers[("flux", "main", "checkpoint")] is True
    assert answers[("flux", "controlnet", "checkpoint")] is False
    assert answers[("wan", "main", "checkpoint")] is True
    assert answers[("wan", "main", "gguf_quantized")] is False


def test_it_serves_only_the_types_that_can_carry_the_setting(client: TestClient) -> None:
    """A row for a VAE or a LoRA would be a field nobody reads; a client reads a missing key as no."""
    answers = _answers(client)

    assert {type_ for _base, type_, _format in answers} == {t.value for t in FP8_STORAGE_MODEL_TYPES}


def test_the_row_carries_the_answer_and_not_the_reason(client: TestClient) -> None:
    """The reason a loader gives is deliberately not served -- both markers hide the control the same
    way, so a client has nothing to do with the difference. That is a decision worth pinning: it is the
    kind of field that gets added later "because we have it" and then has to be supported forever."""
    rows = client.get(URL).json()

    assert rows, "an empty table would make every other assertion here vacuous"
    assert {key for row in rows for key in row} == {"base", "type", "format", "supported"}


def test_the_rows_are_unique_and_ordered(client: TestClient) -> None:
    """A client builds a map from this, and a duplicated key would silently pick a winner."""
    rows = client.get(URL).json()
    keys = [(row["type"], row["base"], row["format"]) for row in rows]

    assert len(set(keys)) == len(keys)
    assert keys == sorted(keys)
