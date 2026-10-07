# conftest.py is a special pytest file. Fixtures defined in this file will be accessible to all tests in this directory
# without needing to explicitly import them. (https://docs.pytest.org/en/6.2.x/fixture.html)


# We import the model_installer and torch_device fixtures here so that they can be used by all tests. Flake8 does not
# play well with fixtures (F401 and F811), so this is cleaner than importing in all files that use these fixtures.
import logging
import shutil
import sys
from pathlib import Path

import pytest

from invokeai.app.services.board_image_records.board_image_records_default import BoardImageRecordStorage
from invokeai.app.services.board_records.board_records_default import BoardRecordStorage
from invokeai.app.services.board_video_records.board_video_records_default import BoardVideoRecordStorage
from invokeai.app.services.boards.boards_default import BoardService
from invokeai.app.services.bulk_download.bulk_download_default import BulkDownloadService
from invokeai.app.services.client_state_persistence.client_state_persistence_default import ClientStatePersistence
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.external_generation.external_generation_default import ExternalGenerationService
from invokeai.app.services.gallery.gallery_default import GalleryService
from invokeai.app.services.image_index.image_index_default import ImageIndexService
from invokeai.app.services.image_index.image_index_records_default import ImageIndexRecords
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.intermediates.intermediates_default import IntermediatesService
from invokeai.app.services.intermediates.intermediates_records_default import IntermediatesRecords
from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invocation_stats.invocation_stats_default import InvocationStatsService
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.project_records.project_records_default import ProjectRecordsStorage
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.system_prompt_records.system_prompt_records_default import (
    SystemPromptRecordsStorage,
)
from invokeai.app.services.users.users_default import UserService
from invokeai.app.services.video_records.video_records_default import VideoRecordStorage
from invokeai.app.services.wildcard_records.wildcard_records_default import WildcardRecordsStorage
from invokeai.app.services.workflow_records.workflow_records_default import WorkflowRecordsStorage
from invokeai.backend.util.logging import InvokeAILogger
from tests.backend.model_manager.model_manager_fixtures import *  # noqa: F403
from tests.fixtures.database import (  # noqa: F401
    _external_application_schema,
    _external_test_schema,
    _migrated_sqlite,
    database,
    empty_database,
    external_test_db_url,
)
from tests.fixtures.races import lost_races  # noqa: F401
from tests.fixtures.sqlite_database import create_mock_sqlite_database  # noqa: F401
from tests.test_nodes import TestEventService

# Fixtures that put a test on the backend under test, which `-m uses_database` selects.
_DATABASE_FIXTURES = {"empty_database", "database"}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # tryfirst: the marks must exist before `-m` deselects by them.
    external = external_test_db_url() is not None
    for item in items:
        if _DATABASE_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.uses_database)
        if external and item.get_closest_marker("sqlite_only") is not None:
            item.add_marker(pytest.mark.skip(reason="needs SQLite, but INVOKEAI_TEST_DB_URL names another backend"))


@pytest.fixture(autouse=True)
def _clear_deferred_empty_cache():
    """`TorchDevice._empty_cache_deferred` is process-global: a test that exercises a peer-aware
    skip must not make a later test perform a real (GPU-initializing) empty_cache."""
    from invokeai.backend.util.devices import TorchDevice

    TorchDevice._empty_cache_deferred.clear()
    yield
    TorchDevice._empty_cache_deferred.clear()


@pytest.fixture
def mock_sqlite_database() -> Database:
    """The in-memory database behind `mock_services`, for tests that need SQL of their own or a service that
    `mock_services` leaves out (the session queue)."""
    return create_mock_sqlite_database(InvokeAIAppConfig(use_memory_db=True), InvokeAILogger.get_logger())


@pytest.fixture
def mock_services(mock_sqlite_database: Database) -> InvocationServices:
    # Image indexing is on by default, but `model_manager` below is None: starting
    # the indexer against these stub services would fail while resolving the
    # embedding model. Tests that exercise the index enable it themselves.
    configuration = InvokeAIAppConfig(use_memory_db=True, node_cache_size=0, image_index_enabled=False)
    logger = InvokeAILogger.get_logger()
    db = mock_sqlite_database

    # NOTE: none of these are actually called by the test invocations
    return InvocationServices(
        board_image_records=BoardImageRecordStorage(db),
        board_images=None,  # type: ignore
        board_records=BoardRecordStorage(db),
        boards=BoardService(),
        bulk_download=BulkDownloadService(),
        configuration=configuration,
        events=TestEventService(),
        image_files=None,  # type: ignore
        image_records=ImageRecordStorage(db),
        images=ImageService(),
        invocation_cache=MemoryInvocationCache(max_cache_size=0),
        logger=logging,  # type: ignore
        model_images=None,  # type: ignore
        model_manager=None,  # type: ignore
        download_queue=None,  # type: ignore
        external_generation=ExternalGenerationService({}, logger),
        names=None,  # type: ignore
        performance_statistics=InvocationStatsService(),
        session_processor=None,  # type: ignore
        session_queue=None,  # type: ignore
        urls=None,  # type: ignore
        workflow_records=WorkflowRecordsStorage(db),
        tensors=None,  # type: ignore
        conditioning=None,  # type: ignore
        style_preset_records=None,  # type: ignore
        style_preset_image_files=None,  # type: ignore
        system_prompt_records=SystemPromptRecordsStorage(db),
        workflow_thumbnails=None,  # type: ignore
        model_relationship_records=None,  # type: ignore
        model_relationships=None,  # type: ignore
        client_state_persistence=ClientStatePersistence(db),
        project_records=ProjectRecordsStorage(db),
        users=UserService(db),
        wildcard_records=WildcardRecordsStorage(db),
        videos=None,  # type: ignore
        video_files=None,  # type: ignore
        video_records=VideoRecordStorage(db),
        board_video_records=BoardVideoRecordStorage(db),
        # Real SQLite-backed gallery service: the virtual-boards router reads dates and
        # per-date item names through it, and MagicMock cannot exercise the filter SQL.
        gallery=GalleryService(db),
        image_index_records=ImageIndexRecords(db),
        image_index=ImageIndexService(),
        intermediates=IntermediatesService(records=IntermediatesRecords(db), logger=logger),
    )


@pytest.fixture()
def mock_invoker(mock_services: InvocationServices) -> Invoker:
    return Invoker(services=mock_services)


@pytest.fixture(scope="module")
def invokeai_root_dir(tmp_path_factory) -> Path:
    root_template = Path(__file__).parent.resolve() / "backend/model_manager/data/invokeai_root"
    temp_dir: Path = tmp_path_factory.mktemp("data") / "invokeai_root"
    shutil.copytree(root_template, temp_dir)
    return temp_dir


# --- Peak memory reporting -------------------------------------------------------------------
#
# Running the suite across xdist workers multiplies its memory footprint, and the failure mode is
# silent: the runner is killed mid-run, so there is no summary, no failing test and no clue which
# file was responsible. Reporting each worker's peak turns a future blow-up into a number that
# moves in the CI log before it takes a runner down. `--max-worker-restart=0` in the workflow
# makes the death itself fail the run immediately rather than after sixteen restarts.

_worker_peak_rss: dict[str, int] = {}


def _peak_rss_bytes() -> int:
    if sys.platform == "win32":
        import psutil

        return psutil.Process().memory_info().peak_wset
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    return peak if sys.platform == "darwin" else peak * 1024


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Hands this worker's peak to the controller; `workeroutput` exists only in a worker."""
    output = getattr(session.config, "workeroutput", None)
    if output is not None:
        output["peak_rss_bytes"] = _peak_rss_bytes()


def pytest_testnodedown(node, error) -> None:  # noqa: ANN001  # xdist types are not exported
    peak = getattr(node, "workeroutput", {}).get("peak_rss_bytes")
    if peak is not None:
        _worker_peak_rss[node.gateway.id] = peak


def pytest_terminal_summary(terminalreporter) -> None:  # noqa: ANN001
    if not _worker_peak_rss:
        return
    peaks = _worker_peak_rss.values()
    terminalreporter.write_line(
        f"peak RSS: {max(peaks) / 2**30:.2f}GB worst worker, "
        f"{sum(peaks) / 2**30:.2f}GB summed over {len(_worker_peak_rss)} workers"
    )
