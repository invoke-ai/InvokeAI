"""Tests for the /v1/image_map endpoints: serving, staleness, and user scoping."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_index.image_index_base import ImageIndexServiceBase
from invokeai.app.services.image_index.image_index_common import ImageIndexStatus
from invokeai.app.services.image_index.image_index_records_sqlite import ImageIndexRecordsSqlite
from invokeai.app.services.image_index.projection import scope_hash
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database

MODEL_ID = "test-model-hash"
DIM = 4
SYSTEM_USER_ID = "system"


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


class FakeImageIndexService(ImageIndexServiceBase):
    """Records projection/search requests instead of running a worker."""

    def __init__(self, model_id: str | None = MODEL_ID) -> None:
        self._model_id = model_id
        self.index_records: ImageIndexRecordsSqlite | None = None
        self.projection_requests: list[tuple[str, bool]] = []
        self.spent_failed_scopes: dict[str, str] = {}
        self.search_calls: list[tuple[str | None, int]] = []
        self.search_results: list[tuple[str, float]] = []
        self.text_unavailable = False
        self.embedded_texts: list[str] = []
        self.embedded_images: list = []
        self.vocab_invalidations = 0
        self.vocab_state: tuple[str, str | None] = ("idle", None)

    @property
    def model_id(self) -> str | None:
        return self._model_id

    def get_status(self) -> ImageIndexStatus | None:
        if self._model_id is None:
            return None
        return ImageIndexStatus(total=5, embedded=3)

    def embed_text(self, text: str) -> np.ndarray:
        from invokeai.app.services.image_index.image_index_base import TextSearchUnavailableError

        if self.text_unavailable:
            raise TextSearchUnavailableError("no text encoder installed")
        self.embedded_texts.append(text)
        vector = np.zeros(DIM, dtype=np.float32)
        vector[0] = 1.0
        return vector

    def embed_image(self, image) -> np.ndarray:
        self.embedded_images.append(image)
        vector = np.zeros(DIM, dtype=np.float32)
        vector[0] = 1.0
        return vector

    def get_accessible_embeddings(self, user_id: str | None) -> tuple[list[str], np.ndarray]:
        # Wired to the real records store by the mock_services fixture so
        # endpoints exercising the accessible matrix see seeded embeddings.
        if self.index_records is None:
            return [], np.empty((0, 0), dtype=np.float32)
        names = self.index_records.list_accessible_embedded_images(user_id, MODEL_ID)
        return self.index_records.get_embeddings(names, MODEL_ID)

    def search_similar(self, user_id: str | None, query_embedding: np.ndarray, limit: int) -> list[tuple[str, float]]:
        self.search_calls.append((user_id, limit))
        return self.search_results[:limit]

    def get_vocab_embeddings(self) -> tuple[list[str], np.ndarray]:
        from invokeai.app.services.image_index.image_index_base import TextSearchUnavailableError

        if self.text_unavailable:
            raise TextSearchUnavailableError("no text encoder installed")
        # A tiny vocabulary aligned with the seeded DIM-dimensional space:
        # phrase i points along axis i.
        vocabulary = ["alpha", "beta", "gamma", "delta"]
        return vocabulary, np.eye(DIM, dtype=np.float32)

    def invalidate_vocab(self) -> None:
        self.vocab_invalidations += 1
        self.vocab_state = ("building", None)

    def get_vocab_build_state(self) -> tuple[str, str | None]:
        if self._model_id is None:
            return "unavailable", None
        return self.vocab_state

    def request_projection(
        self,
        user_id: str,
        all_images: bool = False,
        failed_scope: str | None = None,
        user_initiated: bool = False,
    ) -> bool:
        if self._model_id is None:
            return False
        # A stand-in for the refusal, NOT a model of it: the real service decides
        # from its own view of the cached row and spends the budget on the worker
        # thread, long after this call returns. These tests pin what the ROUTER
        # does with an accept and a refusal; that the real service actually
        # refuses (and for the same rows) is pinned in
        # tests/app/services/image_index/test_image_index_service.py.
        if user_initiated:
            # A person asked, so the budget resets — as the real service does.
            self.spent_failed_scopes.pop(user_id, None)
        elif failed_scope is not None:
            if self.spent_failed_scopes.get(user_id) == failed_scope:
                return False
            self.spent_failed_scopes[user_id] = failed_scope
        self.projection_requests.append((user_id, all_images))
        return True


@pytest.fixture
def image_index_service() -> FakeImageIndexService:
    return FakeImageIndexService()


@pytest.fixture
def mock_services(image_index_service: FakeImageIndexService) -> InvocationServices:
    from invokeai.app.services.board_image_records.board_image_records_sqlite import SqliteBoardImageRecordStorage
    from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
    from invokeai.app.services.boards.boards_default import BoardService
    from invokeai.app.services.bulk_download.bulk_download_default import BulkDownloadService
    from invokeai.app.services.client_state_persistence.client_state_persistence_sqlite import (
        ClientStatePersistenceSqlite,
    )
    from invokeai.app.services.images.images_default import ImageService
    from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
    from invokeai.app.services.invocation_stats.invocation_stats_default import InvocationStatsService
    from invokeai.app.services.project_records.project_records_sqlite import ProjectRecordsSqlite
    from invokeai.app.services.users.users_default import UserService
    from tests.test_nodes import TestEventService

    configuration = InvokeAIAppConfig(use_memory_db=True, node_cache_size=0)
    logger = InvokeAILogger.get_logger()
    db = create_mock_sqlite_database(configuration, logger)

    services = InvocationServices(
        board_image_records=SqliteBoardImageRecordStorage(db=db),
        board_images=None,  # type: ignore
        board_records=SqliteBoardRecordStorage(db=db),
        boards=BoardService(),
        bulk_download=BulkDownloadService(),
        configuration=configuration,
        events=TestEventService(),
        image_files=None,  # type: ignore
        image_records=SqliteImageRecordStorage(db=db),
        images=ImageService(),
        invocation_cache=MemoryInvocationCache(max_cache_size=0),
        logger=logging,  # type: ignore
        model_images=None,  # type: ignore
        model_manager=None,  # type: ignore
        download_queue=None,  # type: ignore
        names=None,  # type: ignore
        performance_statistics=InvocationStatsService(),
        session_processor=None,  # type: ignore
        session_queue=None,  # type: ignore
        urls=None,  # type: ignore
        workflow_records=None,  # type: ignore
        tensors=None,  # type: ignore
        conditioning=None,  # type: ignore
        style_preset_records=None,  # type: ignore
        style_preset_image_files=None,  # type: ignore
        workflow_thumbnails=None,  # type: ignore
        model_relationship_records=None,  # type: ignore
        model_relationships=None,  # type: ignore
        client_state_persistence=ClientStatePersistenceSqlite(db=db),
        project_records=ProjectRecordsSqlite(db=db),
        users=UserService(db),
        wildcard_records=None,  # type: ignore
        system_prompt_records=None,  # type: ignore
        videos=None,  # type: ignore
        video_files=None,  # type: ignore
        video_records=None,  # type: ignore
        board_video_records=None,  # type: ignore
        gallery=None,  # type: ignore
        image_index_records=(index_records := ImageIndexRecordsSqlite(db=db)),
        image_index=image_index_service,
        external_generation=None,  # type: ignore
    )
    image_index_service.index_records = index_records

    return services


@pytest.fixture
def mock_invoker(mock_services: InvocationServices) -> Invoker:
    return Invoker(services=mock_services)


@pytest.fixture
def client(monkeypatch, mock_invoker: Invoker) -> TestClient:
    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.image_map.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers._access.ApiDependencies", mock_deps)
    return TestClient(app)


def _records(mock_invoker: Invoker) -> ImageIndexRecordsSqlite:
    return mock_invoker.services.image_index_records


def _seed_embedded_image(mock_invoker: Invoker, image_name: str, user_id: str = SYSTEM_USER_ID) -> None:
    mock_invoker.services.image_records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=16,
        height=16,
        has_workflow=False,
        user_id=user_id,
    )
    rng = np.random.default_rng(abs(hash(image_name)) % (2**32))
    vec = rng.standard_normal(DIM).astype(np.float32)
    _records(mock_invoker).upsert_embedding(image_name, MODEL_ID, vec / np.linalg.norm(vec))


def _save_unembedded_image(mock_invoker: Invoker, image_name: str, user_id: str = SYSTEM_USER_ID) -> None:
    mock_invoker.services.image_records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=16,
        height=16,
        has_workflow=False,
        user_id=user_id,
    )


def _seed_projection(mock_invoker: Invoker, user_id: str, image_names: list[str], coords: np.ndarray) -> None:
    accessible = _records(mock_invoker).list_accessible_embedded_images(
        None if user_id == SYSTEM_USER_ID else user_id, MODEL_ID
    )
    _records(mock_invoker).set_projection(
        user_id, MODEL_ID, scope_hash(MODEL_ID, accessible), "{}", image_names, coords
    )


# --- Single-user mode (system admin) ---


def test_points_disabled_when_indexer_not_running(
    monkeypatch, mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    image_index_service._model_id = None
    response = client.get("/api/v1/image_map/points")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "disabled"
    assert body["points"] == []


def test_points_model_missing_when_enabled_without_model(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    # Indexing enabled in config, but the service found no installed model at
    # start: the client must be able to tell this apart from "disabled".
    image_index_service._model_id = None
    mock_invoker.services.configuration.image_index_enabled = True
    body = client.get("/api/v1/image_map/points").json()
    assert body["state"] == "model_missing"
    assert body["model_name"] == mock_invoker.services.configuration.image_index_model
    assert body["points"] == []


def test_points_empty_when_nothing_embedded(client: TestClient) -> None:
    response = client.get("/api/v1/image_map/points")
    assert response.status_code == 200
    assert response.json()["state"] == "empty"


def test_points_computing_and_enqueues_when_cache_missing(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    _seed_embedded_image(mock_invoker, "a.png")

    response = client.get("/api/v1/image_map/points")

    body = response.json()
    assert body["state"] == "computing"
    assert body["stale"] is True
    # System user is admin in single-user mode -> all_images scope.
    assert image_index_service.projection_requests == [(SYSTEM_USER_ID, True)]


def test_points_served_with_live_eps_clustering(mock_invoker: Invoker, client: TestClient) -> None:
    names = ["a.png", "b.png", "c.png", "d.png"]
    for name in names:
        _seed_embedded_image(mock_invoker, name)
    # Two tight pairs far apart (span 30 -> the server-side eps clamp is ~1.5).
    coords = np.array([[0.0, 0.0], [0.4, 0.0], [30.0, 30.0], [30.4, 30.0]], dtype=np.float32)
    _seed_projection(mock_invoker, SYSTEM_USER_ID, names, coords)

    clustered = client.get("/api/v1/image_map/points", params={"eps": 0.5, "min_samples": 2}).json()
    assert clustered["state"] == "ready"
    assert clustered["stale"] is False
    assert clustered["point_count"] == 4
    labels = {p["image_name"]: p["cluster"] for p in clustered["points"]}
    assert labels["a.png"] == labels["b.png"] != labels["c.png"] == labels["d.png"]
    assert labels["a.png"] != -1
    assert clustered["cluster_eps"] == 0.5

    # A much smaller eps dissolves the pairs into noise — recluster without recompute.
    noisy = client.get("/api/v1/image_map/points", params={"eps": 0.05, "min_samples": 2}).json()
    assert {p["cluster"] for p in noisy["points"]} == {-1}

    # No eps: the adaptive default resolves to a concrete value, reported so a
    # later request can reproduce the exact clustering.
    adaptive = client.get("/api/v1/image_map/points", params={"min_samples": 2}).json()
    assert adaptive["cluster_eps"] is not None
    pinned = client.get("/api/v1/image_map/points", params={"eps": adaptive["cluster_eps"], "min_samples": 2}).json()
    assert [p["cluster"] for p in pinned["points"]] == [p["cluster"] for p in adaptive["points"]]


def test_stale_projection_filters_now_inaccessible_names_and_requests_refresh(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    _seed_embedded_image(mock_invoker, "keep.png")
    _seed_embedded_image(mock_invoker, "gone.png")
    coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    _seed_projection(mock_invoker, SYSTEM_USER_ID, ["keep.png", "gone.png"], coords)
    # The image disappears after the projection was cached.
    mock_invoker.services.image_records.delete("gone.png")

    body = client.get("/api/v1/image_map/points").json()

    assert body["stale"] is True
    assert [p["image_name"] for p in body["points"]] == ["keep.png"]
    assert image_index_service.projection_requests == [(SYSTEM_USER_ID, True)]


def test_refresh_endpoint_enqueues(image_index_service: FakeImageIndexService, client: TestClient) -> None:
    response = client.post("/api/v1/image_map/refresh")
    assert response.status_code == 202
    assert response.json()["enqueued"] is True
    assert image_index_service.projection_requests == [(SYSTEM_USER_ID, True)]


def test_status_endpoint(mock_invoker: Invoker, client: TestClient) -> None:
    body = client.get("/api/v1/image_map/status").json()
    assert body["enabled"] is True
    assert body["index"] == {"total": 5, "embedded": 3, "failed": 0}
    assert body["projection"]["state"] == "empty"

    _seed_embedded_image(mock_invoker, "a.png")
    _seed_projection(mock_invoker, SYSTEM_USER_ID, ["a.png"], np.zeros((1, 2), dtype=np.float32))
    body = client.get("/api/v1/image_map/status").json()
    assert body["projection"]["state"] == "ready"
    assert body["projection"]["stale"] is False
    assert body["projection"]["point_count"] == 1


def test_status_model_missing_when_enabled_without_model(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    image_index_service._model_id = None
    mock_invoker.services.configuration.image_index_enabled = True
    body = client.get("/api/v1/image_map/status").json()
    assert body["enabled"] is False
    assert body["model_name"] == mock_invoker.services.configuration.image_index_model
    assert body["projection"]["state"] == "model_missing"


def test_status_disabled(image_index_service: FakeImageIndexService, client: TestClient) -> None:
    image_index_service._model_id = None
    body = client.get("/api/v1/image_map/status").json()
    assert body["enabled"] is False
    assert body["model_name"] is None
    assert body["projection"]["state"] == "disabled"


def test_eps_validation(client: TestClient) -> None:
    assert client.get("/api/v1/image_map/points", params={"eps": 0}).status_code == 422
    assert client.get("/api/v1/image_map/points", params={"eps": 99}).status_code == 422


# --- Multiuser mode: per-user scoping ---


@pytest.fixture
def multiuser(monkeypatch, mock_invoker: Invoker):
    from invokeai.app.services.auth.token_service import set_jwt_secret

    set_jwt_secret("test-secret-key-for-unit-tests-only-do-not-use-in-production")
    mock_invoker.services.configuration.multiuser = True


def _create_user(mock_invoker: Invoker, email: str, is_admin: bool = False) -> str:
    user = mock_invoker.services.users.create(
        UserCreateRequest(email=email, display_name=email, password="TestPass123", is_admin=is_admin)
    )
    return user.user_id


def _login(client: TestClient, email: str) -> dict[str, str]:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": "TestPass123", "remember_me": False})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['token']}"}


def test_multiuser_projection_and_scope_are_per_user(
    multiuser, mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    _create_user(mock_invoker, "admin@test.com", is_admin=True)
    user1_id = _create_user(mock_invoker, "user1@test.com")
    user2_id = _create_user(mock_invoker, "user2@test.com")
    user1_headers = _login(client, "user1@test.com")
    user2_headers = _login(client, "user2@test.com")

    # user1 has a private (unboarded) embedded image and a cached projection.
    _seed_embedded_image(mock_invoker, "private1.png", user_id=user1_id)
    accessible1 = _records(mock_invoker).list_accessible_embedded_images(user1_id, MODEL_ID)
    _records(mock_invoker).set_projection(
        user1_id, MODEL_ID, scope_hash(MODEL_ID, accessible1), "{}", accessible1, np.zeros((1, 2), dtype=np.float32)
    )

    # user1 sees their own point.
    body1 = client.get("/api/v1/image_map/points", headers=user1_headers).json()
    assert [p["image_name"] for p in body1["points"]] == ["private1.png"]
    assert body1["stale"] is False

    # user2 has no cache and nothing accessible: empty, nothing enqueued, and
    # user1's private image name never appears.
    body2 = client.get("/api/v1/image_map/points", headers=user2_headers).json()
    assert body2["state"] == "empty"
    assert body2["points"] == []
    assert (user2_id, False) not in image_index_service.projection_requests

    # A non-admin refresh is scoped to their own images, not all_images.
    client.post("/api/v1/image_map/refresh", headers=user2_headers)
    assert (user2_id, False) in image_index_service.projection_requests

    # Global index counts are admin-only in the status endpoint.
    assert client.get("/api/v1/image_map/status", headers=user1_headers).json()["index"] is None


def test_multiuser_stale_cache_never_leaks_revoked_names(multiuser, mock_invoker: Invoker, client: TestClient) -> None:
    _create_user(mock_invoker, "admin@test.com", is_admin=True)
    user1_id = _create_user(mock_invoker, "user1@test.com")
    user2_id = _create_user(mock_invoker, "user2@test.com")
    user2_headers = _login(client, "user2@test.com")

    # user2's cached projection contains user1's image (e.g. from a share
    # that has since been revoked). It must be filtered out when served.
    _seed_embedded_image(mock_invoker, "was-shared.png", user_id=user1_id)
    _seed_embedded_image(mock_invoker, "own2.png", user_id=user2_id)
    _records(mock_invoker).set_projection(
        user2_id,
        MODEL_ID,
        "stale-hash",
        "{}",
        ["was-shared.png", "own2.png"],
        np.zeros((2, 2), dtype=np.float32),
    )

    body = client.get("/api/v1/image_map/points", headers=user2_headers).json()

    assert body["stale"] is True
    assert [p["image_name"] for p in body["points"]] == ["own2.png"]

    # The status endpoint's point_count is also filtered to the current scope.
    status_body = client.get("/api/v1/image_map/status", headers=user2_headers).json()
    assert status_body["projection"]["point_count"] == 1


def test_cluster_labels_computed_only_over_accessible_points(
    multiuser, mock_invoker: Invoker, client: TestClient
) -> None:
    # Density-chaining through a hidden (inaccessible) point must not fuse the
    # visible points into a cluster — that both mislabels them and leaks the
    # hidden point's existence between them.
    _create_user(mock_invoker, "admin@test.com", is_admin=True)
    user1_id = _create_user(mock_invoker, "user1@test.com")
    user2_id = _create_user(mock_invoker, "user2@test.com")
    user2_headers = _login(client, "user2@test.com")

    for name in ["p1.png", "p2.png", "far.png"]:
        _seed_embedded_image(mock_invoker, name, user_id=user2_id)
    _seed_embedded_image(mock_invoker, "hidden.png", user_id=user1_id)
    # p1 and p2 are 3.0 apart (beyond eps 2.0); hidden sits between them, 1.5
    # from each — close enough to chain them if it were clustered too. far.png
    # widens the span so the eps clamp does not bind.
    _records(mock_invoker).set_projection(
        user2_id,
        MODEL_ID,
        "stale-hash",
        "{}",
        ["p1.png", "hidden.png", "p2.png", "far.png"],
        np.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0], [0.0, 60.0]], dtype=np.float32),
    )

    body = client.get("/api/v1/image_map/points", params={"eps": 2.0, "min_samples": 2}, headers=user2_headers).json()

    assert [p["image_name"] for p in body["points"]] == ["p1.png", "p2.png", "far.png"]
    assert {p["cluster"] for p in body["points"]} == {-1}


# --- Refresh throttling and clustering reuse ---


@pytest.fixture(autouse=True)
def _reset_image_map_router_state():
    """The throttle and cluster cache are module state, so they outlive a test."""
    from invokeai.app.api.routers import image_map as image_map_router_module

    image_map_router_module._refresh_claims.clear()
    image_map_router_module._cluster_cache.clear()
    yield
    image_map_router_module._refresh_claims.clear()
    image_map_router_module._cluster_cache.clear()


def test_refresh_is_throttled_per_user(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    from invokeai.app.api.routers import image_map as image_map_router_module

    assert client.post("/api/v1/image_map/refresh").json()["enqueued"] is True
    # A recompute takes minutes; a second request inside the window is refused
    # without reaching the single shared index worker at all.
    for _ in range(5):
        assert client.post("/api/v1/image_map/refresh").json()["enqueued"] is False
    assert len(image_index_service.projection_requests) == 1

    # Once the interval has passed the next request is accepted again.
    image_map_router_module._refresh_claims[SYSTEM_USER_ID] -= image_map_router_module.MIN_REFRESH_INTERVAL_SECONDS + 1
    assert client.post("/api/v1/image_map/refresh").json()["enqueued"] is True
    assert len(image_index_service.projection_requests) == 2


def test_refresh_throttle_is_not_consumed_when_nothing_was_enqueued(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    """A refused enqueue (indexer down) must not lock the user out of the next real one."""
    image_index_service._model_id = None

    assert client.post("/api/v1/image_map/refresh").json()["enqueued"] is False

    image_index_service._model_id = MODEL_ID
    assert client.post("/api/v1/image_map/refresh").json()["enqueued"] is True


def test_refresh_throttle_does_not_gate_the_points_recovery_path(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    """/points enqueues the one-shot retry of a failed projection.

    Throttling inside request_projection instead of the route would suppress that
    recovery, so a failed fit would go back to being permanent.
    """
    _seed_embedded_image(mock_invoker, "a.png")
    # An empty projection over a non-empty gallery: the shape a failed fit leaves.
    _seed_projection(mock_invoker, SYSTEM_USER_ID, [], np.empty((0, 2), dtype=np.float32))

    assert client.post("/api/v1/image_map/refresh").json()["enqueued"] is True
    before = len(image_index_service.projection_requests)
    body = client.get("/api/v1/image_map/points").json()

    assert len(image_index_service.projection_requests) == before + 1
    assert body["state"] == "computing"


def test_repeat_points_requests_reuse_the_clustering(monkeypatch, mock_invoker: Invoker, client: TestClient) -> None:
    """/points is polled, and between polls nothing it clusters has changed."""
    from invokeai.app.api.routers import image_map as image_map_router_module

    calls = {"n": 0}
    real_cluster = image_map_router_module.cluster_at_eps

    def counting_cluster(coords, eps, min_samples):
        calls["n"] += 1
        return real_cluster(coords, eps, min_samples)

    monkeypatch.setattr(image_map_router_module, "cluster_at_eps", counting_cluster)

    names = ["a.png", "b.png", "c.png", "d.png"]
    for name in names:
        _seed_embedded_image(mock_invoker, name)
    coords = np.array([[0.0, 0.0], [0.4, 0.0], [30.0, 30.0], [30.4, 30.0]], dtype=np.float32)
    _seed_projection(mock_invoker, SYSTEM_USER_ID, names, coords)

    first = client.get("/api/v1/image_map/points", params={"eps": 0.5, "min_samples": 2}).json()
    for _ in range(4):
        repeat = client.get("/api/v1/image_map/points", params={"eps": 0.5, "min_samples": 2}).json()
        assert repeat["points"] == first["points"]
        assert repeat["cluster_eps"] == first["cluster_eps"]
    assert calls["n"] == 1, "identical repeat polls must not recluster"

    # Every clustering input is part of the key.
    client.get("/api/v1/image_map/points", params={"eps": 0.05, "min_samples": 2}).json()
    assert calls["n"] == 2, "a different eps must recluster"
    client.get("/api/v1/image_map/points", params={"eps": 0.5, "min_samples": 3}).json()
    assert calls["n"] == 3, "a different min_samples must recluster"

    # A recomputed projection (same scope, new coordinates) must not be served
    # from the entry the previous one left behind.
    _seed_projection(mock_invoker, SYSTEM_USER_ID, names, coords[::-1].copy())
    reprojected = client.get("/api/v1/image_map/points", params={"eps": 0.5, "min_samples": 2}).json()
    assert calls["n"] == 4, "a rewritten projection must recluster"
    assert reprojected["points"] != first["points"]


def test_cluster_cache_is_bounded(monkeypatch, mock_invoker: Invoker, client: TestClient) -> None:
    from invokeai.app.api.routers import image_map as image_map_router_module

    names = ["a.png", "b.png", "c.png", "d.png"]
    for name in names:
        _seed_embedded_image(mock_invoker, name)
    _seed_projection(
        mock_invoker,
        SYSTEM_USER_ID,
        names,
        np.array([[0.0, 0.0], [0.4, 0.0], [30.0, 30.0], [30.4, 30.0]], dtype=np.float32),
    )

    # eps is caller-controlled and unthrottled. Varying it must cost the caller
    # their OWN entry and nothing else: a shared pool let one client evict every
    # other user's labels with a handful of requests.
    for i in range(image_map_router_module._CLUSTER_CACHE_USERS * 3):
        client.get("/api/v1/image_map/points", params={"eps": 0.1 + i * 0.01, "min_samples": 2})

    assert len(image_map_router_module._cluster_cache) == 1
    assert set(image_map_router_module._cluster_cache) == {SYSTEM_USER_ID}


def test_cluster_cache_evicts_by_user_and_never_crosses_them() -> None:
    """The cache is keyed by user, so an identical key for two users is two entries.

    Asserted directly on the helpers: seeding two users whose rows collide on
    every other key component is not reachable through the API (each user's
    scope hash and updated_at differ), so a round-trip test of this would pass
    with the user dimension removed entirely.
    """
    from invokeai.app.api.routers import image_map as image_map_router_module

    key: image_map_router_module._ClusterCacheKey = ("scope", "2026-01-01 00:00:00.000", "current", 0.2, 10)
    mine = np.array([0, 1], dtype=np.int64)
    theirs = np.array([1, 0], dtype=np.int64)

    image_map_router_module._cluster_cache_put("user1", key, mine, 0.2)
    image_map_router_module._cluster_cache_put("user2", key, theirs, 0.2)

    assert image_map_router_module._cluster_cache_get("user1", key)[0] is mine
    assert image_map_router_module._cluster_cache_get("user2", key)[0] is theirs
    assert image_map_router_module._cluster_cache_get("user3", key) is None

    # A stale key for a user who has an entry is a miss, not another user's value.
    assert image_map_router_module._cluster_cache_get("user1", ("other", None, "current", 0.2, 10)) is None

    # One caller varying its own key churns only its own slot, however long it
    # goes on — this is the eviction a shared pool got wrong.
    for i in range(image_map_router_module._CLUSTER_CACHE_USERS * 3):
        image_map_router_module._cluster_cache_put("churn", ("scope", None, "current", 0.1 + i, 10), mine, 0.2)
    assert len(image_map_router_module._cluster_cache) == 3
    assert image_map_router_module._cluster_cache_get("user1", key)[0] is mine


def test_a_spent_retry_stops_points_from_asking_again(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    """The client listens for projection_ready and refetches on it.

    So a /points that requests a recompute on every poll closes a cycle with the
    worker's unconditional emit: request -> short-circuit -> emit -> refetch ->
    request, at the worker's poll rate for the life of the process. Passing the
    failed scope lets the service refuse once the retry is spent, which breaks it.
    """
    _seed_embedded_image(mock_invoker, "a.png")
    # An empty projection over a non-empty gallery: the shape a failed fit leaves.
    _seed_projection(mock_invoker, SYSTEM_USER_ID, [], np.empty((0, 2), dtype=np.float32))

    first = client.get("/api/v1/image_map/points").json()
    assert first["state"] == "computing", "the one retry is requested"
    assert len(image_index_service.projection_requests) == 1

    for _ in range(5):
        later = client.get("/api/v1/image_map/points").json()
        assert later["state"] == "empty", "a spent retry must settle into an honest empty"
    assert len(image_index_service.projection_requests) == 1, "no further recomputes may be requested"


def test_an_all_non_finite_projection_is_not_permanently_blank(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    """point_count > 0 while every coordinate is NaN — what a database written before
    the writer's isfinite guard can still hold.

    Deciding the retry on the cached count rather than on what is actually servable
    left this row serving "empty, not stale" with nothing ever asking for a recompute.
    """
    names = ["a.png", "b.png"]
    for name in names:
        _seed_embedded_image(mock_invoker, name)
    _seed_projection(mock_invoker, SYSTEM_USER_ID, names, np.full((2, 2), np.nan, dtype=np.float32))

    body = client.get("/api/v1/image_map/points").json()

    assert body["points"] == []
    assert body["state"] == "computing", "a row with nothing servable must ask for a recompute"
    assert [user for user, _ in image_index_service.projection_requests] == [SYSTEM_USER_ID]


# --- Semantic search ---


def test_search_requires_exactly_one_query_kind(client: TestClient) -> None:
    assert client.get("/api/v1/image_map/search").status_code == 422
    assert client.get("/api/v1/image_map/search", params={"image_name": "a.png", "q": "cats"}).status_code == 422


def test_search_disabled_index_conflicts(image_index_service: FakeImageIndexService, client: TestClient) -> None:
    image_index_service._model_id = None
    assert client.get("/api/v1/image_map/search", params={"q": "cats"}).status_code == 409


def test_text_search_returns_ranked_results(image_index_service: FakeImageIndexService, client: TestClient) -> None:
    image_index_service.search_results = [("a.png", 0.9), ("b.png", 0.5)]

    body = client.get("/api/v1/image_map/search", params={"limit": 10, "q": "a red barn"}).json()

    assert image_index_service.embedded_texts == ["a red barn"]
    # System user is admin in single-user mode -> global (None) scope.
    assert image_index_service.search_calls == [(None, 10)]
    assert body["results"] == [
        {"image_name": "a.png", "score": 0.9},
        {"image_name": "b.png", "score": 0.5},
    ]


def test_text_search_unavailable_encoder_conflicts_with_message(
    image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    image_index_service.text_unavailable = True

    response = client.get("/api/v1/image_map/search", params={"q": "cats"})

    assert response.status_code == 409
    assert "text encoder" in response.json()["detail"]


def test_image_search_uses_stored_embedding(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    _seed_embedded_image(mock_invoker, "ref.png")
    image_index_service.search_results = [("ref.png", 1.0), ("close.png", 0.8)]

    body = client.get("/api/v1/image_map/search", params={"image_name": "ref.png"}).json()

    assert [r["image_name"] for r in body["results"]] == ["ref.png", "close.png"]
    # No text was embedded; the stored image embedding was the query.
    assert image_index_service.embedded_texts == []


def test_image_search_embeds_unindexed_reference_on_demand(
    monkeypatch, mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    from PIL import Image

    _save_unembedded_image(mock_invoker, "not-indexed.png")
    monkeypatch.setattr(mock_invoker.services.images, "get_pil_image", lambda name: Image.new("RGB", (4, 4)))
    image_index_service.search_results = [("a.png", 0.7)]

    body = client.get("/api/v1/image_map/search", params={"image_name": "not-indexed.png"}).json()

    # The reference had no stored embedding, so its file was embedded live.
    assert len(image_index_service.embedded_images) == 1
    assert [r["image_name"] for r in body["results"]] == ["a.png"]


def test_image_search_unembeddable_reference_is_404(monkeypatch, mock_invoker: Invoker, client: TestClient) -> None:
    # No stored embedding AND the file is gone. Raised explicitly rather than
    # relying on the mock store's own AttributeError: this asserts the mapping
    # for the exception a real missing file produces, which is a plain
    # Exception subclass and not an OSError.
    from invokeai.app.services.image_files.image_files_common import ImageFileNotFoundException

    _save_unembedded_image(mock_invoker, "not-indexed.png")

    def _missing(name):
        raise ImageFileNotFoundException()

    monkeypatch.setattr(mock_invoker.services.images, "get_pil_image", _missing)

    assert client.get("/api/v1/image_map/search", params={"image_name": "not-indexed.png"}).status_code == 404


def test_image_search_rejects_an_oversized_stored_reference(
    monkeypatch, mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    # Assets and intermediates are never indexed, so this on-demand branch is the
    # normal path for exactly the images that can be huge. Without a cap the
    # convert("RGB") inside embed_image materializes hundreds of MB on a request
    # thread; the uploaded/downloaded path has always capped, this one had not.
    from PIL import Image

    from invokeai.app.api.routers.image_map import MAX_SEARCH_IMAGE_PIXELS

    _save_unembedded_image(mock_invoker, "huge.png")

    side = int(MAX_SEARCH_IMAGE_PIXELS**0.5) + 64
    oversized = SimpleNamespace(width=side, height=side)
    monkeypatch.setattr(mock_invoker.services.images, "get_pil_image", lambda name: oversized)

    response = client.get("/api/v1/image_map/search", params={"image_name": "huge.png"})

    assert response.status_code == 415
    # Refused before anything tried to decode it.
    assert image_index_service.embedded_images == []

    # A reference inside the cap still embeds.
    monkeypatch.setattr(mock_invoker.services.images, "get_pil_image", lambda name: Image.new("RGB", (4, 4)))
    assert client.get("/api/v1/image_map/search", params={"image_name": "huge.png"}).status_code == 200


def test_image_search_reports_an_encoder_fault_as_a_server_error(
    monkeypatch, mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    # A missing file is a 404, but an encoder fault, a stopped index or an OOM is
    # ours — reporting those as "its file may be missing" sends whoever is
    # debugging to the wrong place entirely.
    from PIL import Image

    _save_unembedded_image(mock_invoker, "not-indexed.png")
    monkeypatch.setattr(mock_invoker.services.images, "get_pil_image", lambda name: Image.new("RGB", (4, 4)))

    def _boom(pil):
        raise RuntimeError("The image index is not running")

    monkeypatch.setattr(image_index_service, "embed_image", _boom)

    assert client.get("/api/v1/image_map/search", params={"image_name": "not-indexed.png"}).status_code == 500


def test_cluster_labels_skips_the_embedding_gather_when_nothing_clustered(
    monkeypatch, mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    # Above MAX_CLUSTERED_POINTS every id is -1 by design, and label_clusters
    # returns {} for that. Gathering the accessible rows first copies
    # len(visible) x D float32 for nothing — gigabytes on the large galleries
    # /points is written for, once per points refresh.
    _seed_embedded_image(mock_invoker, "a.png")
    _seed_projection(mock_invoker, SYSTEM_USER_ID, ["a.png"], np.zeros((1, 2), dtype=np.float32))

    monkeypatch.setattr(
        "invokeai.app.api.routers.image_map.compute_clusters",
        lambda coords, eps=None, min_samples=None: np.full((coords.shape[0],), -1, dtype=np.int64),
    )

    gathered = []
    original = image_index_service.get_accessible_embeddings

    def _spy(scope_user):
        gathered.append(scope_user)

        return original(scope_user)

    monkeypatch.setattr(image_index_service, "get_accessible_embeddings", _spy)

    response = client.get("/api/v1/image_map/cluster_labels")

    assert response.status_code == 200
    assert response.json()["labels"] == {}
    assert gathered == [], "the accessible matrix was gathered for a fully-unclustered map"


def test_search_by_image_upload_returns_ranked_results(
    image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(200, 30, 30)).save(buffer, format="PNG")
    image_index_service.search_results = [("a.png", 0.9), ("b.png", 0.4)]

    response = client.post(
        "/api/v1/image_map/search_by_image",
        params={"limit": 5},
        files={"image": ("ref.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    assert len(image_index_service.embedded_images) == 1
    assert image_index_service.search_calls == [(None, 5)]
    assert [r["image_name"] for r in response.json()["results"]] == ["a.png", "b.png"]


def test_search_by_image_requires_exactly_one_source(client: TestClient) -> None:
    assert client.post("/api/v1/image_map/search_by_image").status_code == 422
    assert (
        client.post(
            "/api/v1/image_map/search_by_image",
            params={"image_url": "https://example.com/a.png"},
            files={"image": ("ref.png", b"\x89PNG", "image/png")},
        ).status_code
        == 422
    )


def test_search_by_image_rejects_bad_inputs(client: TestClient) -> None:
    # Bytes that are not a decodable image.
    response = client.post(
        "/api/v1/image_map/search_by_image", files={"image": ("ref.png", b"not an image", "image/png")}
    )
    assert response.status_code == 415

    # Non-http(s) schemes and private/loopback hosts are refused outright —
    # including hostname and non-canonical IP-literal spellings of loopback,
    # which resolve via getaddrinfo rather than a strict literal parse.
    for bad_url in (
        "ftp://example.com/a.png",
        "http://127.0.0.1/a.png",
        "http://localhost/a.png",
        "http://127.1/a.png",
        "http://2130706433/a.png",
    ):
        assert client.post("/api/v1/image_map/search_by_image", params={"image_url": bad_url}).status_code == 422, (
            bad_url
        )


def test_search_by_image_revalidates_redirect_targets(monkeypatch, client: TestClient) -> None:
    # A public URL redirecting to a private address must be refused: requests'
    # automatic redirect following is disabled and every hop is re-validated.
    import requests

    class FakeRedirect:
        status_code = 302
        is_redirect = True
        headers = {"location": "http://127.0.0.1:9090/steal"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        assert kwargs.get("allow_redirects") is False
        return FakeRedirect()

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, **kw: [
            (None, None, None, None, ("127.0.0.1" if host != "public.example" else "93.184.216.34", port))
        ],
    )

    response = client.post("/api/v1/image_map/search_by_image", params={"image_url": "http://public.example/a.png"})

    assert response.status_code == 422
    # The first (public) hop was fetched; the redirect target failed
    # validation before any second request was issued.
    assert calls == ["http://public.example/a.png"]


def test_multiuser_image_search_enforces_read_access_and_user_scope(
    multiuser, mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    _create_user(mock_invoker, "admin@test.com", is_admin=True)
    user1_id = _create_user(mock_invoker, "user1@test.com")
    user2_id = _create_user(mock_invoker, "user2@test.com")
    user2_headers = _login(client, "user2@test.com")

    # user1's private image: user2 cannot use it as a search reference.
    _seed_embedded_image(mock_invoker, "private1.png", user_id=user1_id)
    response = client.get("/api/v1/image_map/search", params={"image_name": "private1.png"}, headers=user2_headers)
    assert response.status_code == 403

    # A text search from a non-admin is scoped to their own user id.
    client.get("/api/v1/image_map/search", params={"q": "boats"}, headers=user2_headers)
    assert image_index_service.search_calls == [(user2_id, 100)]

    # search_by_image must scope identically to /search.
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    client.post(
        "/api/v1/image_map/search_by_image",
        files={"image": ("ref.png", buffer.getvalue(), "image/png")},
        headers=user2_headers,
    )
    assert image_index_service.search_calls[-1] == (user2_id, 100)


# --- Cluster labels ---


def test_cluster_labels_align_with_served_clusters(mock_invoker: Invoker, client: TestClient) -> None:
    # Two tight pairs; each pair's images embed along a distinct axis, so the
    # expected label is that axis's vocabulary phrase.
    def axis_vec(index: int) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        v[index] = 1.0
        return v

    for name, vec in [
        ("a1.png", axis_vec(0)),
        ("a2.png", axis_vec(0)),
        ("b1.png", axis_vec(1)),
        ("b2.png", axis_vec(1)),
    ]:
        _save_unembedded_image(mock_invoker, name)
        _records(mock_invoker).upsert_embedding(name, MODEL_ID, vec)
    coords = np.array([[0.0, 0.0], [0.4, 0.0], [30.0, 30.0], [30.4, 30.0]], dtype=np.float32)
    _seed_projection(mock_invoker, SYSTEM_USER_ID, ["a1.png", "a2.png", "b1.png", "b2.png"], coords)

    points = client.get("/api/v1/image_map/points", params={"eps": 0.5, "min_samples": 2}).json()
    labels = client.get("/api/v1/image_map/cluster_labels", params={"eps": 0.5, "min_samples": 2, "top_k": 2}).json()[
        "labels"
    ]

    label_by_name = {p["image_name"]: p["cluster"] for p in points["points"]}
    a_cluster = str(label_by_name["a1.png"])
    b_cluster = str(label_by_name["b1.png"])
    assert labels[a_cluster]["label"] == "alpha"
    assert labels[b_cluster]["label"] == "beta"
    assert labels[a_cluster]["score"] > 0.9
    assert len(labels[a_cluster]["alternates"]) == 1

    # Adaptive default: with eps omitted on BOTH endpoints, each resolves the
    # same adaptive value over the same visible set, so labels still align.
    points = client.get("/api/v1/image_map/points", params={"min_samples": 2}).json()
    assert points["cluster_eps"] is not None
    labels_response = client.get("/api/v1/image_map/cluster_labels", params={"min_samples": 2, "top_k": 2}).json()
    # Matching fingerprints are the client's proof the two responses were
    # computed over the same visible set.
    assert labels_response["visible_hash"] == points["visible_hash"]
    labels = labels_response["labels"]
    label_by_name = {p["image_name"]: p["cluster"] for p in points["points"]}
    assert labels[str(label_by_name["a1.png"])]["label"] == "alpha"
    assert labels[str(label_by_name["b1.png"])]["label"] == "beta"

    # Pinned round trip: the reported eps is accepted back verbatim.
    labels = client.get(
        "/api/v1/image_map/cluster_labels",
        params={"eps": points["cluster_eps"], "min_samples": 2, "top_k": 2},
    ).json()["labels"]
    assert labels[str(label_by_name["a1.png"])]["label"] == "alpha"
    assert labels[str(label_by_name["b1.png"])]["label"] == "beta"


def test_cluster_labels_survive_a_non_finite_row_the_way_points_does(mock_invoker: Invoker, client: TestClient) -> None:
    """A projection row predating the writer's isfinite guard.

    /points drops the non-finite rows, so labels computed over the undropped set
    hash a different name list: every response fails the visible_hash comparison
    the client is told to make, and all labels are discarded. Handing the NaN to
    sklearn also raised, 500ing the user until their gallery changed.
    """
    names = ["a1.png", "a2.png", "bad.png"]
    for index, name in enumerate(names):
        vector = np.zeros(DIM, dtype=np.float32)
        vector[index % 2] = 1.0
        _save_unembedded_image(mock_invoker, name)
        _records(mock_invoker).upsert_embedding(name, MODEL_ID, vector)
    coords = np.array([[0.0, 0.0], [0.2, 0.0], [np.nan, np.nan]], dtype=np.float32)
    _seed_projection(mock_invoker, SYSTEM_USER_ID, names, coords)

    points = client.get("/api/v1/image_map/points", params={"min_samples": 2}).json()
    labels_response = client.get("/api/v1/image_map/cluster_labels", params={"min_samples": 2})

    assert labels_response.status_code == 200
    assert [p["image_name"] for p in points["points"]] == ["a1.png", "a2.png"]
    assert labels_response.json()["visible_hash"] == points["visible_hash"], (
        "labels the client cannot match to its points are labels it throws away"
    )


def test_cluster_labels_unavailable_text_encoder_conflicts(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    _seed_embedded_image(mock_invoker, "a.png")
    _seed_projection(mock_invoker, SYSTEM_USER_ID, ["a.png"], np.zeros((1, 2), dtype=np.float32))
    image_index_service.text_unavailable = True

    assert client.get("/api/v1/image_map/cluster_labels").status_code == 409


def test_cluster_labels_empty_without_projection(client: TestClient) -> None:
    assert client.get("/api/v1/image_map/cluster_labels").json() == {
        "labels": {},
        "updated_at": None,
        "visible_hash": None,
    }


# --- Supplementary vocabulary ---


def test_vocab_get_returns_terms_and_state(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    _records(mock_invoker).set_custom_vocab_terms(["zebra", "aardvark"])

    body = client.get("/api/v1/image_map/vocab").json()

    assert body["terms"] == ["aardvark", "zebra"]
    assert body["state"] == "idle"
    assert body["error"] is None
    # The client sizes its input constraints from these.
    assert body["max_terms"] > 0
    assert body["max_term_length"] > 0


def test_vocab_get_reports_unavailable_when_indexer_not_running(
    image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    image_index_service._model_id = None
    body = client.get("/api/v1/image_map/vocab").json()
    # Terms are still served: they persist and apply when indexing next runs.
    assert body["state"] == "unavailable"


def test_vocab_put_normalizes_dedupes_stores_and_invalidates(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    response = client.put(
        "/api/v1/image_map/vocab",
        json={"terms": ["  Golden   Retriever ", "golden retriever", "", "Zebra"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["terms"] == ["golden retriever", "zebra"]
    assert body["state"] == "building"
    # Stored, and the embedding cache was invalidated after the commit.
    assert _records(mock_invoker).get_custom_vocab_terms() == ["golden retriever", "zebra"]
    assert image_index_service.vocab_invalidations == 1


def test_vocab_put_replaces_rather_than_merges(mock_invoker: Invoker, client: TestClient) -> None:
    client.put("/api/v1/image_map/vocab", json={"terms": ["zebra"]})
    client.put("/api/v1/image_map/vocab", json={"terms": ["okapi"]})
    assert _records(mock_invoker).get_custom_vocab_terms() == ["okapi"]

    client.put("/api/v1/image_map/vocab", json={"terms": []})
    assert _records(mock_invoker).get_custom_vocab_terms() == []


def test_vocab_put_rejects_an_overlong_term_and_stores_nothing(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    _records(mock_invoker).set_custom_vocab_terms(["zebra"])

    response = client.put("/api/v1/image_map/vocab", json={"terms": ["ok", "x" * 65]})

    assert response.status_code == 422
    assert "64" in response.json()["detail"]
    # The stored list is untouched and nothing was invalidated.
    assert _records(mock_invoker).get_custom_vocab_terms() == ["zebra"]
    assert image_index_service.vocab_invalidations == 0


def test_vocab_put_rejects_too_many_terms(mock_invoker: Invoker, client: TestClient) -> None:
    response = client.put("/api/v1/image_map/vocab", json={"terms": [f"term {i}" for i in range(501)]})
    assert response.status_code == 422
    assert _records(mock_invoker).get_custom_vocab_terms() == []


def test_vocab_writes_are_admin_only(multiuser, mock_invoker: Invoker, client: TestClient) -> None:
    _create_user(mock_invoker, "admin@test.com", is_admin=True)
    _create_user(mock_invoker, "user1@test.com")
    admin_headers = _login(client, "admin@test.com")
    user_headers = _login(client, "user1@test.com")

    denied = client.put("/api/v1/image_map/vocab", json={"terms": ["zebra"]}, headers=user_headers)
    assert denied.status_code == 403
    assert _records(mock_invoker).get_custom_vocab_terms() == []

    allowed = client.put("/api/v1/image_map/vocab", json={"terms": ["zebra"]}, headers=admin_headers)
    assert allowed.status_code == 200

    # The list itself is readable by any user.
    read = client.get("/api/v1/image_map/vocab", headers=user_headers)
    assert read.status_code == 200
    assert read.json()["terms"] == ["zebra"]
