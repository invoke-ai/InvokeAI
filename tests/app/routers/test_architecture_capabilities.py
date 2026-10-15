"""GET /api/v2/models/capabilities — the static architecture table.

The rows come from what the architectures declare at import time — no service, no database, the same
response for every install. The route is still authenticated like every other one in this router,
which is the only reason these tests need `ApiDependencies` patched at all.
"""

from typing import Any, Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.invoker import Invoker
from invokeai.backend.architectures import architecture_capabilities, generative_bases

URL = "/api/v2/models/capabilities"


class _MockApiDependencies(ApiDependencies):
    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker  # type: ignore[misc]


@pytest.fixture
def client(monkeypatch: Any, mock_invoker: Invoker) -> Iterator[TestClient]:
    """A client whose auth dependency can resolve a default user.

    The route itself needs nothing from the invoker; `CurrentUserOrDefault` does.
    """
    mock_invoker.services.users = MagicMock()
    mock_deps = _MockApiDependencies(mock_invoker)
    for module in ("invokeai.app.api_app", "invokeai.app.api.auth_dependencies"):
        monkeypatch.setattr(f"{module}.ApiDependencies", mock_deps)
    yield TestClient(app)


def test_it_serves_a_row_for_every_architecture(client: TestClient) -> None:
    response = client.get(URL)
    assert response.status_code == 200

    rows = response.json()
    base_rows = [r for r in rows if r["variant"] is None]
    assert {r["base"] for r in base_rows} == {b.value for b in generative_bases()}
    assert len(base_rows) == len(generative_bases()), "one row per architecture, no duplicates"


def test_variant_rows_override_their_base_row(client: TestClient) -> None:
    """FLUX is the clearest case: three variants, three genuinely different answers."""
    rows = client.get(URL).json()
    flux = {r["variant"]: r for r in rows if r["base"] == "flux"}

    assert set(flux) == {None, "schnell", "dev_fill"}
    assert flux["schnell"]["defaults"]["steps"] == 4
    assert flux["dev_fill"]["defaults"]["guidance"] == 30.0
    assert flux[None]["defaults"]["steps"] == 28, "the base row is dev"


def test_the_variant_is_the_value_a_client_holds(client: TestClient) -> None:
    """Not `str(enum)`, which would serialize as `FluxVariantType.DevFill`."""
    variants = {r["variant"] for r in client.get(URL).json() if r["variant"] is not None}
    assert all("." not in v for v in variants), variants
    assert "dev_fill" in variants


def test_the_rows_come_from_the_registry_not_a_service(client: TestClient, mock_invoker: Invoker) -> None:
    """The response is the same table for every install. Nothing about it is looked up.

    Asserted by the model manager service never being touched: if a later version resolved
    capabilities per model record, this is where it would show -- including a lookup that happened
    to produce the same table, which the rendering test below would not catch.
    """
    mock_invoker.services.model_manager = MagicMock()
    assert client.get(URL).status_code == 200
    # `mock_calls`, not `assert_not_called()`: that only sees the mock invoked as a function, so a
    # `services.model_manager.store.get_model(...)` would pass it. `mock_calls` records the chain.
    assert mock_invoker.services.model_manager.mock_calls == []


def test_the_response_matches_what_the_registry_renders(client: TestClient) -> None:
    """The route is a pass-through; anything it added would be a second source of truth."""
    served = client.get(URL).json()
    rendered = [row.model_dump(mode="json") for row in architecture_capabilities()]
    assert served == rendered


def test_the_rows_are_ordered_stably(client: TestClient) -> None:
    """Sorted by base, so a client diffing two responses sees only real changes."""
    rows = client.get(URL).json()
    bases = [r["base"] for r in rows]
    assert bases == sorted(bases)
