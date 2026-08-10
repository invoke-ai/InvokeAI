"""Tests for the /v1/image_map endpoints: serving, staleness, and user scoping."""

import logging

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
    """Records projection requests instead of running a worker."""

    def __init__(self, model_id: str | None = MODEL_ID) -> None:
        self._model_id = model_id
        self.projection_requests: list[tuple[str, bool]] = []

    @property
    def model_id(self) -> str | None:
        return self._model_id

    def get_status(self) -> ImageIndexStatus | None:
        if self._model_id is None:
            return None
        return ImageIndexStatus(total=5, embedded=3)

    def request_projection(self, user_id: str, all_images: bool = False) -> bool:
        if self._model_id is None:
            return False
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

    return InvocationServices(
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
        image_index_records=ImageIndexRecordsSqlite(db=db),
        image_index=image_index_service,
        external_generation=None,  # type: ignore
    )


@pytest.fixture
def mock_invoker(mock_services: InvocationServices) -> Invoker:
    return Invoker(services=mock_services)


@pytest.fixture
def client(monkeypatch, mock_invoker: Invoker) -> TestClient:
    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.image_map.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)
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

    # Distinct eps values are distinct keys; the cache must not grow per request.
    for i in range(image_map_router_module._CLUSTER_CACHE_ENTRIES * 3):
        client.get("/api/v1/image_map/points", params={"eps": 0.1 + i * 0.01, "min_samples": 2})

    assert len(image_map_router_module._cluster_cache) == image_map_router_module._CLUSTER_CACHE_ENTRIES


def test_cluster_cache_never_serves_one_users_labels_to_another(
    multiuser, mock_invoker: Invoker, client: TestClient
) -> None:
    """user_id is in the cache key, so a collision on every other component still cannot cross users."""
    user1_id = _create_user(mock_invoker, "cacheuser1@test.com")
    user2_id = _create_user(mock_invoker, "cacheuser2@test.com")
    user1_headers = _login(client, "cacheuser1@test.com")
    user2_headers = _login(client, "cacheuser2@test.com")

    _seed_embedded_image(mock_invoker, "mine.png", user_id=user1_id)
    _seed_embedded_image(mock_invoker, "theirs.png", user_id=user2_id)
    _seed_projection(mock_invoker, user1_id, ["mine.png"], np.array([[0.0, 0.0]], dtype=np.float32))
    _seed_projection(mock_invoker, user2_id, ["theirs.png"], np.array([[0.0, 0.0]], dtype=np.float32))

    first = client.get("/api/v1/image_map/points", headers=user1_headers).json()
    second = client.get("/api/v1/image_map/points", headers=user2_headers).json()

    assert [p["image_name"] for p in first["points"]] == ["mine.png"]
    assert [p["image_name"] for p in second["points"]] == ["theirs.png"]
