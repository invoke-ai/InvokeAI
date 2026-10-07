"""Tests for the /v1/image_map endpoints: serving, staleness, and user scoping."""

import logging
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import AbstractSet

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.board_video_records.board_video_records_sqlite import SqliteBoardVideoRecordStorage
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.gallery.gallery_default import SqliteGalleryService
from invokeai.app.services.image_index.image_index_base import ImageIndexServiceBase
from invokeai.app.services.image_index.image_index_common import (
    ImageIndexStatus,
    IndexedItem,
)
from invokeai.app.services.image_index.image_index_records_sqlite import ImageIndexRecordsSqlite
from invokeai.app.services.image_index.projection import (
    DEFAULT_CLUSTER_MIN_SAMPLES,
    cluster_with_diagnostics,
    scope_hash,
)
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.video_records.video_records_sqlite import SqliteVideoRecordStorage
from invokeai.app.services.videos.videos_default import VideoService
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database

MODEL_ID = "test-model-hash"
DIM = 4
SYSTEM_USER_ID = "system"


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


def _png_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeImageIndexService(ImageIndexServiceBase):
    """Records projection/search requests instead of running a worker."""

    def __init__(self, model_id: str | None = MODEL_ID) -> None:
        self._model_id = model_id
        self.index_records: ImageIndexRecordsSqlite | None = None
        self.projection_requests: list[tuple[str, bool]] = []
        self.spent_failed_scopes: dict[str, str] = {}
        self.search_calls: list[tuple[str | None, int]] = []
        self.search_kinds: list[tuple[str, ...] | None] = []
        self.search_within: list[AbstractSet[str] | None] = []
        self.search_results: list[tuple[str, float]] = []
        self.text_unavailable = False
        self.embedded_texts: list[str] = []
        self.embedded_images: list = []
        self.vocab_invalidations = 0
        self.vocab_state: tuple[str, str | None] = ("idle", None)
        self.replacing = False

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def replacing_model(self) -> bool:
        return self.replacing

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

    def get_accessible_embeddings(self, user_id: str | None) -> tuple[list[IndexedItem], np.ndarray]:
        # Wired to the real records store by the mock_services fixture so
        # endpoints exercising the accessible matrix see seeded embeddings.
        if self.index_records is None:
            return [], np.empty((0, 0), dtype=np.float32)
        items = self.index_records.list_accessible_embedded_items(user_id, MODEL_ID)
        return self.index_records.get_embeddings(items, MODEL_ID)

    def search_similar(
        self,
        user_id: str | None,
        query_embedding: np.ndarray,
        limit: int,
        kinds: tuple[str, ...] | None = None,
        within: AbstractSet[str] | None = None,
    ) -> list[tuple[IndexedItem, float]]:
        self.search_calls.append((user_id, limit))
        self.search_kinds.append(kinds)
        self.search_within.append(within)
        results = [
            (item, score)
            for item, score in self.search_results
            if (kinds is None or item.kind in kinds) and (within is None or item.name in within)
        ]
        return results[:limit]

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


def _video_service(thumbnails: Path, video_records: SqliteVideoRecordStorage) -> VideoService:
    """A video service that resolves thumbnails to real files, which is all these endpoints read.

    It goes through the record store first, like the real one: a name with no video raises
    rather than resolving to a fabricated file, which is what makes the not-found paths real.
    """
    videos = VideoService()

    def get_path(video_name: str, thumbnail: bool = False) -> str:
        assert thumbnail, "these endpoints only ever read a video's thumbnail"
        video_records.get(video_name)
        path = thumbnails / f"{video_name}.webp"
        if not path.exists():
            Image.new("RGB", (16, 16), "teal").save(path, "WEBP")
        return str(path)

    videos.get_path = get_path  # type: ignore[method-assign]
    return videos


@pytest.fixture
def mock_services(image_index_service: FakeImageIndexService, tmp_path: Path) -> InvocationServices:
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
        database=db,
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
        videos=_video_service(tmp_path, video_records := SqliteVideoRecordStorage(db=db)),
        video_files=None,  # type: ignore
        video_records=video_records,
        board_video_records=SqliteBoardVideoRecordStorage(db=db),
        gallery=SqliteGalleryService(db=db),
        image_index_records=(index_records := ImageIndexRecordsSqlite(db=db)),
        image_index=image_index_service,
        intermediates=None,  # type: ignore
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
    _records(mock_invoker).upsert_embedding(IndexedItem("image", image_name), MODEL_ID, vec / np.linalg.norm(vec))


def _seed_embedded_video(mock_invoker: Invoker, video_name: str, user_id: str = SYSTEM_USER_ID) -> None:
    mock_invoker.services.video_records.save(
        video_name=video_name,
        video_origin=ResourceOrigin.INTERNAL,
        video_category=ImageCategory.GENERAL,
        width=16,
        height=16,
        duration=2.0,
        fps=24.0,
        has_workflow=False,
        user_id=user_id,
    )
    rng = np.random.default_rng(abs(hash(video_name)) % (2**32))
    vec = rng.standard_normal(DIM).astype(np.float32)
    _records(mock_invoker).upsert_embedding(IndexedItem("video", video_name), MODEL_ID, vec / np.linalg.norm(vec))


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


def imgs(*names: str) -> list[IndexedItem]:
    """The image-namespace items for these names."""
    return [IndexedItem("image", name) for name in names]


def vids(*names: str) -> list[IndexedItem]:
    """The video-namespace items for these names."""
    return [IndexedItem("video", name) for name in names]


def _seed_projection(mock_invoker: Invoker, user_id: str, items: list[IndexedItem], coords: np.ndarray) -> None:
    accessible = _records(mock_invoker).list_accessible_embedded_items(
        None if user_id == SYSTEM_USER_ID else user_id, MODEL_ID
    )
    _records(mock_invoker).set_projection(user_id, MODEL_ID, scope_hash(MODEL_ID, accessible), "{}", items, coords)


# --- Single-user mode (system admin) ---


def test_points_disabled_when_indexer_not_running(
    monkeypatch, mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    image_index_service._model_id = None
    # Indexing is on by default, so "disabled" has to be asked for explicitly.
    mock_invoker.services.configuration.image_index_enabled = False
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
    assert body["model_id"] == MODEL_ID
    assert body["stale"] is True
    # System user is admin in single-user mode -> all_images scope.
    assert image_index_service.projection_requests == [(SYSTEM_USER_ID, True)]


def test_points_served_with_live_eps_clustering(mock_invoker: Invoker, client: TestClient) -> None:
    names = ["a.png", "b.png", "c.png", "d.png"]
    for name in names:
        _seed_embedded_image(mock_invoker, name)
    # Two tight pairs far apart (span 30 -> the server-side eps clamp is ~1.5).
    coords = np.array([[0.0, 0.0], [0.4, 0.0], [30.0, 30.0], [30.4, 30.0]], dtype=np.float32)
    _seed_projection(mock_invoker, SYSTEM_USER_ID, imgs(*names), coords)

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
    _seed_projection(mock_invoker, SYSTEM_USER_ID, imgs("keep.png", "gone.png"), coords)
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
    _seed_projection(mock_invoker, SYSTEM_USER_ID, imgs("a.png"), np.zeros((1, 2), dtype=np.float32))
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


@pytest.mark.parametrize(
    "call_endpoint",
    [
        pytest.param(lambda client: client.get("/api/v1/image_map/points"), id="points"),
        pytest.param(lambda client: client.get("/api/v1/image_map/status"), id="status"),
        pytest.param(lambda client: client.post("/api/v1/image_map/refresh"), id="refresh"),
        # Gallery search runs without the map open, so it must recover on its own.
        pytest.param(lambda client: client.get("/api/v1/image_map/search", params={"q": "a cat"}), id="search"),
        pytest.param(
            lambda client: client.post(
                "/api/v1/image_map/search_by_image", files={"image": ("ref.png", _png_bytes(), "image/png")}
            ),
            id="search_by_image",
        ),
    ],
)
def test_endpoints_activate_a_model_installed_since_startup(
    call_endpoint, mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    # The map's own message links the encoder install, so the next request has
    # to pick the model up rather than repeating "not installed" until a restart.
    # Asserted per endpoint: each one has to make the attempt on its own, and a
    # request that found the model already active proves nothing.
    image_index_service._model_id = None
    mock_invoker.services.configuration.image_index_enabled = True
    activations = 0

    def activate() -> bool:
        nonlocal activations
        activations += 1
        image_index_service._model_id = MODEL_ID
        return True

    image_index_service.try_activate = activate  # type: ignore[method-assign]

    response = call_endpoint(client)

    assert response.status_code in (200, 202)
    assert activations == 1
    assert image_index_service.model_id == MODEL_ID


def test_points_serve_the_index_once_it_is_activated(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    image_index_service._model_id = None
    mock_invoker.services.configuration.image_index_enabled = True

    def activate() -> bool:
        image_index_service._model_id = MODEL_ID
        return True

    image_index_service.try_activate = activate  # type: ignore[method-assign]

    assert client.get("/api/v1/image_map/points").json()["state"] == "empty"
    assert client.get("/api/v1/image_map/status").json()["enabled"] is True


@pytest.mark.parametrize("endpoint", ["points", "status"])
def test_reads_revalidate_an_encoder_deleted_after_activation(
    endpoint: str, mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    mock_invoker.services.configuration.image_index_enabled = True

    def revalidate() -> bool:
        image_index_service._model_id = None
        return False

    image_index_service.try_activate = revalidate  # type: ignore[method-assign]
    body = client.get(f"/api/v1/image_map/{endpoint}").json()
    projection = body["projection"] if endpoint == "status" else body
    assert projection["state"] == "model_missing"
    assert body["model_id"] is None
    assert body["model_name"] == mock_invoker.services.configuration.image_index_model
    if endpoint == "status":
        assert body["enabled"] is False
        assert body["index"] is None


@pytest.mark.parametrize("endpoint", ["points", "status"])
def test_reads_report_a_draining_replacement_as_computing_rather_than_missing(
    endpoint: str, mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    # The replacement is installed; offering to install it again would invite a
    # duplicate install while the retired encoder's work drains.
    mock_invoker.services.configuration.image_index_enabled = True
    image_index_service._model_id = None
    image_index_service.replacing = True
    body = client.get(f"/api/v1/image_map/{endpoint}").json()
    projection = body["projection"] if endpoint == "status" else body
    assert projection["state"] == "computing"
    assert body["model_id"] is None
    assert body["model_name"] is None

    mock_invoker.services.configuration.image_index_enabled = False
    body = client.get(f"/api/v1/image_map/{endpoint}").json()
    assert (body["projection"] if endpoint == "status" else body)["state"] == "disabled"


def test_search_during_a_replacement_says_the_model_is_switching(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    mock_invoker.services.configuration.image_index_enabled = True
    image_index_service._model_id = None
    image_index_service.replacing = True
    response = client.get("/api/v1/image_map/search", params={"q": "a cat"})
    assert response.status_code == 409
    assert "switching" in response.json()["detail"]


@pytest.mark.parametrize("endpoint", ["points", "status"])
def test_encoder_fingerprint_changes_without_a_missing_model_response(
    endpoint: str, mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    _seed_embedded_image(mock_invoker, "a.png")
    _seed_projection(mock_invoker, SYSTEM_USER_ID, imgs("a.png"), np.zeros((1, 2), dtype=np.float32))
    before = client.get(f"/api/v1/image_map/{endpoint}").json()
    assert before["model_id"] == MODEL_ID
    assert (before["projection"] if endpoint == "status" else before)["point_count"] == 1

    # The browser can miss the whole removal/reinstall while closed or suspended.
    # The new encoder must identify itself before it has any projected points.
    image_index_service._model_id = "replacement-model-hash"
    after = client.get(f"/api/v1/image_map/{endpoint}").json()
    assert after["model_id"] == "replacement-model-hash"
    projection = after["projection"] if endpoint == "status" else after
    assert projection["state"] == "empty"
    assert projection["point_count"] == 0


def test_status_disabled(mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient) -> None:
    image_index_service._model_id = None
    mock_invoker.services.configuration.image_index_enabled = False
    body = client.get("/api/v1/image_map/status").json()
    assert body["enabled"] is False
    assert body["model_name"] is None
    assert body["projection"]["state"] == "disabled"


def test_search_refuses_a_model_replaced_after_query_embedding(
    image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    def embed_then_replace(text: str) -> np.ndarray:
        image_index_service._model_id = "replacement-model"
        return np.ones(DIM, dtype=np.float32)

    image_index_service.embed_text = embed_then_replace  # type: ignore[method-assign]
    response = client.get("/api/v1/image_map/search", params={"q": "a cat"})
    assert response.status_code == 409
    assert image_index_service.search_calls == []


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
    accessible1 = _records(mock_invoker).list_accessible_embedded_items(user1_id, MODEL_ID)
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
        imgs("was-shared.png", "own2.png"),
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
        imgs("p1.png", "hidden.png", "p2.png", "far.png"),
        np.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0], [0.0, 60.0]], dtype=np.float32),
    )

    body = client.get("/api/v1/image_map/points", params={"eps": 2.0, "min_samples": 2}, headers=user2_headers).json()

    assert [p["image_name"] for p in body["points"]] == ["p1.png", "p2.png", "far.png"]
    assert {p["cluster"] for p in body["points"]} == {-1}


# --- Refresh throttling and clustering reuse ---


@pytest.fixture(autouse=True)
def _reset_image_map_router_state():
    """The throttle, cluster cache and diagnostics log are module state, so they outlive a test."""
    from invokeai.app.api.routers import image_map as image_map_router_module

    image_map_router_module._refresh_claims.clear()
    image_map_router_module._cluster_cache.clear()
    image_map_router_module._cluster_diagnostics_logged.clear()
    yield
    image_map_router_module._refresh_claims.clear()
    image_map_router_module._cluster_cache.clear()
    image_map_router_module._cluster_diagnostics_logged.clear()


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
    served: list[float | None] = []
    real_cluster = image_map_router_module.cluster_with_diagnostics

    def counting_cluster(coords, eps=None, min_samples=DEFAULT_CLUSTER_MIN_SAMPLES):
        calls["n"] += 1
        labels, diagnostics = real_cluster(coords, eps, min_samples)
        served.append(diagnostics.resolved_eps)

        return labels, diagnostics

    monkeypatch.setattr(image_map_router_module, "cluster_with_diagnostics", counting_cluster)

    names = ["a.png", "b.png", "c.png", "d.png"]
    for name in names:
        _seed_embedded_image(mock_invoker, name)
    coords = np.array([[0.0, 0.0], [0.4, 0.0], [30.0, 30.0], [30.4, 30.0]], dtype=np.float32)
    _seed_projection(mock_invoker, SYSTEM_USER_ID, imgs(*names), coords)

    first = client.get("/api/v1/image_map/points", params={"eps": 0.5, "min_samples": 2}).json()
    for _ in range(4):
        repeat = client.get("/api/v1/image_map/points", params={"eps": 0.5, "min_samples": 2}).json()
        assert repeat["points"] == first["points"]
        assert repeat["cluster_eps"] == first["cluster_eps"]
    assert calls["n"] == 1, "identical repeat polls must not recluster"
    # The eps the client is told to pass back has to be the one that produced
    # these labels. Resolving it a second time anywhere in the endpoint would
    # re-apply the 0.01 floor to a budget-shrunk value and report a different
    # number than DBSCAN ran at.
    assert first["cluster_eps"] == served[0]

    # Every clustering input is part of the key.
    client.get("/api/v1/image_map/points", params={"eps": 0.05, "min_samples": 2}).json()
    assert calls["n"] == 2, "a different eps must recluster"
    client.get("/api/v1/image_map/points", params={"eps": 0.5, "min_samples": 3}).json()
    assert calls["n"] == 3, "a different min_samples must recluster"

    # A recomputed projection (same scope, new coordinates) must not be served
    # from the entry the previous one left behind.
    _seed_projection(mock_invoker, SYSTEM_USER_ID, imgs(*names), coords[::-1].copy())
    reprojected = client.get("/api/v1/image_map/points", params={"eps": 0.5, "min_samples": 2}).json()
    assert calls["n"] == 4, "a rewritten projection must recluster"
    assert reprojected["points"] != first["points"]


def test_points_logs_why_a_map_came_back_unclustered(
    caplog, monkeypatch, mock_invoker: Invoker, client: TestClient
) -> None:
    """An all-noise map is indistinguishable from a working one in the response.

    The point cap is the gate a large gallery hits, and nothing the client
    receives mentions it: every point simply arrives with cluster -1.
    """
    from invokeai.app.api.routers import image_map as image_map_router_module

    monkeypatch.setattr("invokeai.app.services.image_index.projection.MAX_CLUSTERED_POINTS", 1)
    names = ["a.png", "b.png"]
    for name in names:
        _seed_embedded_image(mock_invoker, name)
    _seed_projection(mock_invoker, SYSTEM_USER_ID, imgs(*names), np.zeros((2, 2), dtype=np.float32))

    with caplog.at_level(logging.DEBUG):
        body = client.get("/api/v1/image_map/points").json()

    assert [point["cluster"] for point in body["points"]] == [-1, -1]
    assert _diagnostics_levels(caplog) == ["INFO"], "a skipped clustering must be reported, and at INFO"
    line = _diagnostics_records(caplog)[0].getMessage()
    assert "MAX_CLUSTERED_POINTS" in line
    assert "points=2" in line and "unclustered=2" in line and "clusters=0" in line

    # A second request re-clusters (a skipped clustering is never cached, so
    # this is not a cache hit) and must stay quiet: the map refreshes on every
    # gallery change, and a gallery stuck above the cap would otherwise emit a
    # line per refresh forever.
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        client.get("/api/v1/image_map/points")
    assert image_map_router_module._cluster_cache == {}, "the repeat must have re-clustered, not read a cache"
    assert _diagnostics_levels(caplog) == []

    # Enough time passing re-arms it, so "open the map again while I watch the
    # log" works without restarting the server.
    caplog.clear()
    stale = time.monotonic() - image_map_router_module._CLUSTER_DIAGNOSTICS_REPEAT_AFTER_SECONDS - 1
    signature = image_map_router_module._cluster_diagnostics_logged[SYSTEM_USER_ID][0]
    image_map_router_module._cluster_diagnostics_logged[SYSTEM_USER_ID] = (signature, stale)
    with caplog.at_level(logging.DEBUG):
        client.get("/api/v1/image_map/points")
    assert _diagnostics_levels(caplog) == ["INFO"]

    # A clustering that found clusters explains itself, so it goes to debug.
    caplog.clear()
    monkeypatch.setattr("invokeai.app.services.image_index.projection.MAX_CLUSTERED_POINTS", 50_000)
    with caplog.at_level(logging.DEBUG):
        client.get("/api/v1/image_map/points", params={"min_samples": 2})
    assert _diagnostics_levels(caplog) == ["DEBUG"]
    healthy = _diagnostics_records(caplog)[0].getMessage()
    assert "MAX_CLUSTERED_POINTS" not in healthy and "resolved_eps=" in healthy


def test_a_map_with_no_visible_points_is_not_reported_as_a_clustering(
    caplog, mock_invoker: Invoker, client: TestClient
) -> None:
    """Nothing clustered because there was nothing to cluster; `state` says so already."""
    _seed_embedded_image(mock_invoker, "a.png")
    # A projection whose only row is not in the accessible set: the visible
    # mask empties, and clustering is handed zero points.
    _seed_projection(mock_invoker, SYSTEM_USER_ID, imgs("gone.png"), np.zeros((1, 2), dtype=np.float32))

    with caplog.at_level(logging.DEBUG):
        client.get("/api/v1/image_map/points")

    assert _diagnostics_levels(caplog) == []


def test_cluster_diagnostics_log_is_bounded_and_kept_per_user() -> None:
    from invokeai.app.api.routers import image_map as image_map_router_module

    services = SimpleNamespace(logger=logging)
    diagnostics = cluster_with_diagnostics(np.zeros((4, 2), dtype=np.float32), eps=0.5, min_samples=2)[1]
    for index in range(image_map_router_module._CLUSTER_DIAGNOSTICS_USERS + 3):
        image_map_router_module._log_cluster_diagnostics(services, f"user{index}", diagnostics)

    logged = image_map_router_module._cluster_diagnostics_logged
    assert len(logged) == image_map_router_module._CLUSTER_DIAGNOSTICS_USERS
    assert "user0" not in logged, "the oldest user must be evicted, not the newest"
    assert f"user{image_map_router_module._CLUSTER_DIAGNOSTICS_USERS + 2}" in logged
    # Per user: one user's line must never silence another's.
    assert len({entry[0] for entry in logged.values()}) == 1


def test_a_suppressed_repeat_does_not_push_back_the_re_arm() -> None:
    """The line re-arms 600s after it was last EMITTED, not last suppressed.

    Refreshing the stored timestamp on every quiet request would hold the line
    off for as long as the user keeps the map open — which is exactly when
    they are trying to reproduce it.
    """
    from invokeai.app.api.routers import image_map as image_map_router_module

    services = SimpleNamespace(logger=logging)
    diagnostics = cluster_with_diagnostics(np.zeros((4, 2), dtype=np.float32), eps=0.5, min_samples=2)[1]
    image_map_router_module._log_cluster_diagnostics(services, "user", diagnostics)
    emitted_at = image_map_router_module._cluster_diagnostics_logged["user"][1]

    for _ in range(3):
        image_map_router_module._log_cluster_diagnostics(services, "user", diagnostics)

    assert image_map_router_module._cluster_diagnostics_logged["user"][1] == emitted_at


def test_cluster_diagnostics_log_keeps_the_users_it_keeps_hearing_from() -> None:
    """Recency has to mean "last seen", not "last logged".

    A user whose clustering never changes is the quiet path, and it is exactly
    the entry worth keeping: evict it and their next request logs again, which
    is the repetition the guard exists to prevent.
    """
    from invokeai.app.api.routers import image_map as image_map_router_module

    services = SimpleNamespace(logger=logging)
    diagnostics = cluster_with_diagnostics(np.zeros((4, 2), dtype=np.float32), eps=0.5, min_samples=2)[1]
    cap = image_map_router_module._CLUSTER_DIAGNOSTICS_USERS
    for index in range(cap):
        image_map_router_module._log_cluster_diagnostics(services, f"user{index}", diagnostics)

    # user0 says the same thing again — suppressed, but still active.
    image_map_router_module._log_cluster_diagnostics(services, "user0", diagnostics)
    image_map_router_module._log_cluster_diagnostics(services, "newcomer", diagnostics)

    logged = image_map_router_module._cluster_diagnostics_logged
    assert len(logged) == cap
    assert "user0" in logged, "a suppressed repeat must refresh the entry's recency"
    assert "user1" not in logged, "the genuinely least recent user is the one to evict"


def _diagnostics_records(caplog) -> list[logging.LogRecord]:
    """The clustering-diagnostics lines captured so far."""
    return [record for record in caplog.records if "Image map: clustered" in record.getMessage()]


def _diagnostics_levels(caplog) -> list[str]:
    return [record.levelname for record in _diagnostics_records(caplog)]


def test_cluster_cache_is_bounded(monkeypatch, mock_invoker: Invoker, client: TestClient) -> None:
    from invokeai.app.api.routers import image_map as image_map_router_module

    names = ["a.png", "b.png", "c.png", "d.png"]
    for name in names:
        _seed_embedded_image(mock_invoker, name)
    _seed_projection(
        mock_invoker,
        SYSTEM_USER_ID,
        imgs(*names),
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
    _seed_projection(mock_invoker, SYSTEM_USER_ID, imgs(*names), np.full((2, 2), np.nan, dtype=np.float32))

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
    image_index_service.search_results = [(IndexedItem("image", "a.png"), 0.9), (IndexedItem("image", "b.png"), 0.5)]

    body = client.get("/api/v1/image_map/search", params={"limit": 10, "q": "a red barn"}).json()

    assert image_index_service.embedded_texts == ["a red barn"]
    # System user is admin in single-user mode -> global (None) scope.
    assert image_index_service.search_calls == [(None, 10)]
    assert body["results"] == [
        {"image_name": "a.png", "kind": "image", "score": 0.9},
        {"image_name": "b.png", "kind": "image", "score": 0.5},
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
    image_index_service.search_results = [
        (IndexedItem("image", "ref.png"), 1.0),
        (IndexedItem("image", "close.png"), 0.8),
    ]

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
    image_index_service.search_results = [(IndexedItem("image", "a.png"), 0.7)]

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
    _seed_projection(mock_invoker, SYSTEM_USER_ID, imgs("a.png"), np.zeros((1, 2), dtype=np.float32))

    monkeypatch.setattr("invokeai.app.services.image_index.projection.MAX_CLUSTERED_POINTS", 0)

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
    image_index_service.search_results = [(IndexedItem("image", "a.png"), 0.9), (IndexedItem("image", "b.png"), 0.4)]

    response = client.post(
        "/api/v1/image_map/search_by_image",
        params={"limit": 5},
        files={"image": ("ref.png", _png_bytes(), "image/png")},
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


def test_search_can_be_scoped_to_a_board_the_uncategorized_items_or_a_date(
    image_index_service: FakeImageIndexService, mock_invoker: Invoker, client: TestClient
) -> None:
    _seed_embedded_image(mock_invoker, "on-board.png")
    _seed_embedded_video(mock_invoker, "on-board.mp4")
    _seed_embedded_image(mock_invoker, "loose.png")
    board = mock_invoker.services.board_records.save("Cats", SYSTEM_USER_ID).board_id
    mock_invoker.services.board_image_records.add_image_to_board(board, "on-board.png")
    mock_invoker.services.board_video_records.add_video_to_board(board, "on-board.mp4")
    image_index_service.search_results = [
        (IndexedItem("image", "loose.png"), 0.9),
        (IndexedItem("video", "on-board.mp4"), 0.8),
        (IndexedItem("image", "on-board.png"), 0.7),
    ]
    today = str(mock_invoker.services.image_records.get("loose.png").created_at)[:10]

    def names(**params: object) -> list[str]:
        response = client.get("/api/v1/image_map/search", params={"q": "cat", "include_videos": True, **params})
        assert response.status_code == 200, response.text
        return [result["image_name"] for result in response.json()["results"]]

    assert names() == ["loose.png", "on-board.mp4", "on-board.png"]
    assert names(board_id=board) == ["on-board.mp4", "on-board.png"]
    assert names(board_id="none") == ["loose.png"]
    assert names(created_date=today) == ["loose.png", "on-board.mp4", "on-board.png"]
    assert names(created_date="2001-01-01") == []
    # An empty scope answers without embedding the query.
    assert image_index_service.embedded_texts == ["cat"] * 4
    # The scope reaches the service, which ranks within it before applying the limit.
    assert image_index_service.search_within[0] is None
    assert image_index_service.search_within[1] == {"on-board.png", "on-board.mp4"}
    assert client.get("/api/v1/image_map/search", params={"q": "cat", "created_date": "not-a-date"}).status_code == 422

    buffer = BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    by_image = client.post(
        "/api/v1/image_map/search_by_image",
        params={"board_id": board},
        files={"image": ("ref.png", buffer.getvalue(), "image/png")},
    )
    assert [result["image_name"] for result in by_image.json()["results"]] == ["on-board.png"]


def test_multiuser_board_scoped_search_requires_read_access(
    multiuser, mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    _create_user(mock_invoker, "admin@test.com", is_admin=True)
    user1_id = _create_user(mock_invoker, "user1@test.com")
    _create_user(mock_invoker, "user2@test.com")
    user2_headers = _login(client, "user2@test.com")
    _seed_embedded_image(mock_invoker, "private1.png", user_id=user1_id)
    board = mock_invoker.services.board_records.save("Private", user1_id).board_id
    mock_invoker.services.board_image_records.add_image_to_board(board, "private1.png")

    response = client.get("/api/v1/image_map/search", params={"q": "boats", "board_id": board}, headers=user2_headers)

    assert response.status_code == 403
    # Refused before the query is embedded or ranked.
    assert image_index_service.embedded_texts == []
    assert image_index_service.search_calls == []


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
        _records(mock_invoker).upsert_embedding(IndexedItem("image", name), MODEL_ID, vec)
    coords = np.array([[0.0, 0.0], [0.4, 0.0], [30.0, 30.0], [30.4, 30.0]], dtype=np.float32)
    _seed_projection(mock_invoker, SYSTEM_USER_ID, imgs("a1.png", "a2.png", "b1.png", "b2.png"), coords)

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
        _records(mock_invoker).upsert_embedding(IndexedItem("image", name), MODEL_ID, vector)
    coords = np.array([[0.0, 0.0], [0.2, 0.0], [np.nan, np.nan]], dtype=np.float32)
    _seed_projection(mock_invoker, SYSTEM_USER_ID, imgs(*names), coords)

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
    _seed_projection(mock_invoker, SYSTEM_USER_ID, imgs("a.png"), np.zeros((1, 2), dtype=np.float32))
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


# --- Per-image labels ---


def test_image_labels_rank_the_vocabulary_for_one_image(mock_invoker: Invoker, client: TestClient) -> None:
    # Mostly axis 1, leaning to axis 0, with the rest strictly ordered below
    # them: "beta" wins and "alpha" is the first alternate. Every component is
    # distinct so the ranking never depends on how argpartition/argsort — both
    # unstable — happen to break a tie.
    vector = np.array([0.6, 0.8, 0.3, 0.1], dtype=np.float32)
    _save_unembedded_image(mock_invoker, "leaning.png")
    _records(mock_invoker).upsert_embedding(
        IndexedItem("image", "leaning.png"), MODEL_ID, vector / np.linalg.norm(vector)
    )

    body = client.get("/api/v1/image_map/image_labels", params={"image_name": "leaning.png"}).json()

    assert body["label"] == "beta"
    assert body["alternates"] == ["alpha", "gamma"]
    assert body["score"] == pytest.approx(0.8 / float(np.linalg.norm(vector)))

    # top_k bounds the total label count (best + alternates).
    body = client.get("/api/v1/image_map/image_labels", params={"image_name": "leaning.png", "top_k": 1}).json()
    assert body["alternates"] == []


def test_image_labels_unindexed_image_is_404(mock_invoker: Invoker, client: TestClient) -> None:
    _save_unembedded_image(mock_invoker, "no-embedding.png")

    response = client.get("/api/v1/image_map/image_labels", params={"image_name": "no-embedding.png"})

    assert response.status_code == 404


def test_image_labels_disabled_index_conflicts(image_index_service: FakeImageIndexService, client: TestClient) -> None:
    image_index_service._model_id = None

    assert client.get("/api/v1/image_map/image_labels", params={"image_name": "a.png"}).status_code == 409


def test_image_labels_unavailable_text_encoder_conflicts(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    _seed_embedded_image(mock_invoker, "a.png")
    image_index_service.text_unavailable = True

    assert client.get("/api/v1/image_map/image_labels", params={"image_name": "a.png"}).status_code == 409


def test_multiuser_image_labels_enforce_read_access(multiuser, mock_invoker: Invoker, client: TestClient) -> None:
    _create_user(mock_invoker, "admin@test.com", is_admin=True)
    user1_id = _create_user(mock_invoker, "user1@test.com")
    _create_user(mock_invoker, "user2@test.com")
    user2_headers = _login(client, "user2@test.com")

    # user1's private image: user2 must not learn its labels (or that it has any).
    _seed_embedded_image(mock_invoker, "private1.png", user_id=user1_id)
    response = client.get(
        "/api/v1/image_map/image_labels", params={"image_name": "private1.png"}, headers=user2_headers
    )

    assert response.status_code == 403


def _seed_degenerate_embedding(mock_invoker: Invoker, image_name: str, vector: np.ndarray) -> None:
    """Write an embedding blob straight to the table, bypassing the writer's guards.

    `embedding_to_blob` refuses non-finite and all-zero vectors, but rows
    predating that guard are still in existing databases, and the read path
    validates only the blob's length.
    """
    _save_unembedded_image(mock_invoker, image_name)
    records = _records(mock_invoker)
    blob = np.ascontiguousarray(vector, dtype=np.float32).tobytes()
    with records._db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO image_embeddings (image_name, model_id, dim, embedding) VALUES (?, ?, ?, ?);",
            (image_name, MODEL_ID, vector.shape[0], blob),
        )


@pytest.mark.parametrize(
    "vector",
    [
        pytest.param(np.full(DIM, np.nan, dtype=np.float32), id="non-finite"),
        pytest.param(np.zeros(DIM, dtype=np.float32), id="all-zero"),
    ],
)
def test_image_labels_refuse_a_degenerate_stored_embedding(
    mock_invoker: Invoker, client: TestClient, vector: np.ndarray
) -> None:
    """A degenerate row must not yield confident nonsense.

    Normalizing by a zero or non-finite norm makes every score NaN, and
    argpartition then returns arbitrary rows — three unrelated vocabulary
    phrases presented as this image's tags, with a `score` that serializes as
    JSON null against a schema declaring it a float.
    """
    _seed_degenerate_embedding(mock_invoker, "degenerate.png", vector)

    response = client.get("/api/v1/image_map/image_labels", params={"image_name": "degenerate.png"})

    assert response.status_code == 404
    assert "label" not in response.json(), "a refusal must not carry an arbitrary vocabulary phrase"
    # The row IS present: this must be the degenerate-vector refusal, not the
    # not-indexed 404, or the test would pass without exercising the guard.
    assert response.json()["detail"] == "This item's stored embedding cannot be labeled"


def test_image_labels_corrupt_blob_is_404_not_500(mock_invoker: Invoker, client: TestClient) -> None:
    """A row whose blob length disagrees with its `dim` column.

    `blob_to_embedding` raises on it. Unhandled that is a 500, and because
    this endpoint is driven by pointer movement it would refire several times
    a second for as long as the point is hovered.
    """
    _save_unembedded_image(mock_invoker, "corrupt.png")
    records = _records(mock_invoker)
    with records._db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO image_embeddings (image_name, model_id, dim, embedding) VALUES (?, ?, ?, ?);",
            ("corrupt.png", MODEL_ID, DIM, np.zeros(DIM - 1, dtype=np.float32).tobytes()),
        )

    response = client.get("/api/v1/image_map/image_labels", params={"image_name": "corrupt.png"})

    assert response.status_code == 404


def test_image_labels_dim_mismatch_with_the_vocabulary_is_404_not_500(
    mock_invoker: Invoker, client: TestClient
) -> None:
    """A stored embedding from a different-width model than the vocab matrix."""
    _save_unembedded_image(mock_invoker, "wide.png")
    records = _records(mock_invoker)
    wide = np.ones(DIM + 3, dtype=np.float32)
    with records._db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO image_embeddings (image_name, model_id, dim, embedding) VALUES (?, ?, ?, ?);",
            ("wide.png", MODEL_ID, wide.shape[0], wide.tobytes()),
        )

    response = client.get("/api/v1/image_map/image_labels", params={"image_name": "wide.png"})

    assert response.status_code == 404


# --- Videos on the map ---


def test_points_carry_the_kind_of_each_item(mock_invoker: Invoker, client: TestClient) -> None:
    # A client resolves a point's thumbnail from a kind-specific endpoint, so a video point
    # that claimed to be an image would render as a broken image.
    _seed_embedded_image(mock_invoker, "a.png")
    _seed_embedded_video(mock_invoker, "clip.mp4")
    _seed_projection(
        mock_invoker,
        SYSTEM_USER_ID,
        [IndexedItem("image", "a.png"), IndexedItem("video", "clip.mp4")],
        np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )

    body = client.get("/api/v1/image_map/points", params={"include_videos": True}).json()

    assert body["state"] == "ready"
    assert {(p["image_name"], p["kind"]) for p in body["points"]} == {("a.png", "image"), ("clip.mp4", "video")}


def test_a_new_video_makes_the_projection_stale(mock_invoker: Invoker, client: TestClient) -> None:
    # Staleness is about the projection, not about what this client renders: the fit has to
    # cover the video before any client can be shown it, and the scope hash is taken over the
    # whole accessible set for exactly that reason. One refit per new item, the same as for a
    # new image — not a poll loop, since the refit clears it.
    _seed_embedded_image(mock_invoker, "a.png")
    _seed_projection(mock_invoker, SYSTEM_USER_ID, imgs("a.png"), np.zeros((1, 2), dtype=np.float32))
    assert client.get("/api/v1/image_map/points").json()["stale"] is False

    _seed_embedded_video(mock_invoker, "clip.mp4")

    assert client.get("/api/v1/image_map/points").json()["stale"] is True


def test_deleted_image_is_filtered_from_cached_projection_without_reembedding(
    mock_invoker: Invoker, image_index_service: FakeImageIndexService, client: TestClient
) -> None:
    _seed_embedded_image(mock_invoker, "kept.png")
    _seed_embedded_image(mock_invoker, "removed.png")
    _seed_projection(
        mock_invoker,
        SYSTEM_USER_ID,
        imgs("kept.png", "removed.png"),
        np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32),
    )

    mock_invoker.services.image_records.delete("removed.png")

    response = client.get("/api/v1/image_map/points")

    assert response.status_code == 200
    body = response.json()
    assert body["stale"] is True
    assert [point["image_name"] for point in body["points"]] == ["kept.png"]
    assert image_index_service.embedded_images == []
    assert image_index_service.projection_requests == [(SYSTEM_USER_ID, True)]


def test_search_by_video_reference_uses_its_stored_embedding(
    image_index_service: FakeImageIndexService, mock_invoker: Invoker, client: TestClient
) -> None:
    _seed_embedded_video(mock_invoker, "ref.mp4")
    image_index_service.search_results = [
        (IndexedItem("video", "ref.mp4"), 1.0),
        (IndexedItem("image", "close.png"), 0.8),
    ]

    response = client.get("/api/v1/image_map/search", params={"video_name": "ref.mp4", "include_videos": True})

    assert response.status_code == 200
    assert response.json()["results"] == [
        {"image_name": "ref.mp4", "kind": "video", "score": 1.0},
        {"image_name": "close.png", "kind": "image", "score": 0.8},
    ]
    # The stored embedding was used; nothing was embedded on demand.
    assert image_index_service.embedded_images == []


def test_search_refuses_both_an_image_and_a_video_reference(client: TestClient) -> None:
    response = client.get("/api/v1/image_map/search", params={"image_name": "a.png", "video_name": "clip.mp4"})

    assert response.status_code == 422


def test_image_labels_accept_a_video(
    image_index_service: FakeImageIndexService, mock_invoker: Invoker, client: TestClient
) -> None:
    _seed_embedded_video(mock_invoker, "clip.mp4")

    response = client.get("/api/v1/image_map/image_labels", params={"image_name": "clip.mp4", "kind": "video"})

    assert response.status_code == 200
    assert response.json()["label"] in {"alpha", "beta", "gamma", "delta"}


def test_image_labels_do_not_read_a_video_through_the_image_namespace(
    mock_invoker: Invoker, client: TestClient
) -> None:
    # Without the kind, a video name would be looked up among images: a 404 at best, and at
    # worst another item's embedding if the namespaces ever shared names.
    _seed_embedded_video(mock_invoker, "clip.mp4")

    response = client.get("/api/v1/image_map/image_labels", params={"image_name": "clip.mp4"})

    assert response.status_code == 404


def test_points_hide_videos_unless_the_client_asks_for_them(mock_invoker: Invoker, client: TestClient) -> None:
    # A client that resolves every point through the images endpoints — which the shipped
    # gallery does — would render a video point as a broken tile, so serving them is opt-in.
    _seed_embedded_image(mock_invoker, "a.png")
    _seed_embedded_video(mock_invoker, "clip.mp4")
    _seed_projection(
        mock_invoker,
        SYSTEM_USER_ID,
        [IndexedItem("image", "a.png"), IndexedItem("video", "clip.mp4")],
        np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )

    body = client.get("/api/v1/image_map/points").json()

    assert [p["image_name"] for p in body["points"]] == ["a.png"]
    # The projection covers both items, so hiding one must not make the cache look stale —
    # that would have the client asking for a recompute on every poll, forever.
    assert body["stale"] is False
    assert body["state"] == "ready"


def test_hidden_videos_are_left_out_of_the_status_count(mock_invoker: Invoker, client: TestClient) -> None:
    _seed_embedded_image(mock_invoker, "a.png")
    _seed_embedded_video(mock_invoker, "clip.mp4")
    _seed_projection(
        mock_invoker,
        SYSTEM_USER_ID,
        [IndexedItem("image", "a.png"), IndexedItem("video", "clip.mp4")],
        np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )

    default_body = client.get("/api/v1/image_map/status").json()
    opted_in_body = client.get("/api/v1/image_map/status", params={"include_videos": True}).json()

    assert default_body["projection"]["point_count"] == 1
    assert opted_in_body["projection"]["point_count"] == 2


def test_cluster_labels_cover_the_same_items_as_points(mock_invoker: Invoker, client: TestClient) -> None:
    # The client compares the two responses' visible_hash and discards labels computed over a
    # different set, so both endpoints must apply the same kind filter.
    _seed_embedded_image(mock_invoker, "a.png")
    _seed_embedded_video(mock_invoker, "clip.mp4")
    _seed_projection(
        mock_invoker,
        SYSTEM_USER_ID,
        [IndexedItem("image", "a.png"), IndexedItem("video", "clip.mp4")],
        np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )

    for params in ({}, {"include_videos": True}):
        points = client.get("/api/v1/image_map/points", params=params).json()
        labels = client.get("/api/v1/image_map/cluster_labels", params=params).json()
        assert points["visible_hash"] == labels["visible_hash"], params


def test_search_hides_videos_unless_the_client_asks_for_them(
    image_index_service: FakeImageIndexService, mock_invoker: Invoker, client: TestClient
) -> None:
    image_index_service.search_results = [(IndexedItem("video", "clip.mp4"), 0.9), (IndexedItem("image", "a.png"), 0.5)]

    default_body = client.get("/api/v1/image_map/search", params={"q": "cat"}).json()
    opted_in_body = client.get("/api/v1/image_map/search", params={"q": "cat", "include_videos": True}).json()

    assert [result["image_name"] for result in default_body["results"]] == ["a.png"]
    assert [result["image_name"] for result in opted_in_body["results"]] == ["clip.mp4", "a.png"]
    # The service is told which kinds to rank, rather than the endpoint trimming the ranking
    # afterwards — which would return fewer hits than the caller's limit.
    assert image_index_service.search_kinds == [("image",), ("image", "video")]


def test_video_search_reference_embeds_the_thumbnail_on_demand(
    image_index_service: FakeImageIndexService, mock_invoker: Invoker, client: TestClient
) -> None:
    # A video that is not in the index — an intermediate, or one generated seconds ago — is
    # still usable as a reference: its thumbnail is embedded for this one query. Asking the
    # images store for it would 404 instead.
    mock_invoker.services.video_records.save(
        video_name="fresh.mp4",
        video_origin=ResourceOrigin.INTERNAL,
        video_category=ImageCategory.GENERAL,
        width=16,
        height=16,
        duration=2.0,
        fps=24.0,
        has_workflow=False,
        user_id=SYSTEM_USER_ID,
    )
    image_index_service.search_results = [(IndexedItem("image", "a.png"), 0.6)]

    response = client.get("/api/v1/image_map/search", params={"video_name": "fresh.mp4"})

    assert response.status_code == 200
    assert len(image_index_service.embedded_images) == 1
    # The thumbnail, not the video file: 16x16 is what the fixture writes.
    assert image_index_service.embedded_images[0].size == (16, 16)


def test_video_search_reference_reports_a_missing_video_as_not_found(mock_invoker: Invoker, client: TestClient) -> None:
    response = client.get("/api/v1/image_map/search", params={"video_name": "no-such.mp4"})

    assert response.status_code == 404


def test_multiuser_video_reference_enforces_read_access(multiuser, mock_invoker: Invoker, client: TestClient) -> None:
    # The video access clause is a different join against different tables than the image one,
    # so it needs its own end-to-end check: a wrong join leaks another user's private video.
    _create_user(mock_invoker, "admin@test.com", is_admin=True)
    user1_id = _create_user(mock_invoker, "user1@test.com")
    _create_user(mock_invoker, "user2@test.com")
    user2_headers = _login(client, "user2@test.com")

    _seed_embedded_video(mock_invoker, "private1.mp4", user_id=user1_id)

    refused = client.get("/api/v1/image_map/search", params={"video_name": "private1.mp4"}, headers=user2_headers)
    labels_refused = client.get(
        "/api/v1/image_map/image_labels",
        params={"image_name": "private1.mp4", "kind": "video"},
        headers=user2_headers,
    )

    assert refused.status_code == 403
    assert labels_refused.status_code == 403


def test_multiuser_video_on_a_shared_board_is_readable(multiuser, mock_invoker: Invoker, client: TestClient) -> None:
    from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardVisibility

    _create_user(mock_invoker, "admin@test.com", is_admin=True)
    user1_id = _create_user(mock_invoker, "user1@test.com")
    _create_user(mock_invoker, "user2@test.com")
    user2_headers = _login(client, "user2@test.com")

    _seed_embedded_video(mock_invoker, "shared1.mp4", user_id=user1_id)
    board = mock_invoker.services.board_records.save("Shared", user1_id).board_id
    mock_invoker.services.board_records.update(board, BoardChanges(board_visibility=BoardVisibility.Shared))
    mock_invoker.services.board_video_records.add_video_to_board(board, "shared1.mp4")

    response = client.get("/api/v1/image_map/search", params={"video_name": "shared1.mp4"}, headers=user2_headers)

    assert response.status_code == 200


def test_a_video_only_gallery_answers_empty_instead_of_asking_forever(
    image_index_service: FakeImageIndexService, mock_invoker: Invoker, client: TestClient
) -> None:
    # Nothing servable here is the kind filter's doing, not a failed fit: the projection covers
    # the video and its coordinates are fine. Diagnosing it as a failure asks the worker for a
    # recompute, which short-circuits on the matching scope hash and emits projection_ready —
    # which the client answers with another /points, for the life of the process.
    _seed_embedded_video(mock_invoker, "clip.mp4")
    _seed_projection(
        mock_invoker, SYSTEM_USER_ID, [IndexedItem("video", "clip.mp4")], np.zeros((1, 2), dtype=np.float32)
    )

    first = client.get("/api/v1/image_map/points").json()
    second = client.get("/api/v1/image_map/points").json()

    assert first["state"] == "empty"
    assert first["stale"] is False
    assert first["points"] == []
    # A settled answer, not one that changes every time the client asks again.
    assert second == first
    # The mechanism, not just the symptom: the endpoint must not ask for a recompute at all.
    # Each request is answered with a projection_ready event, which is what brings the client
    # back for another /points.
    assert image_index_service.projection_requests == []
