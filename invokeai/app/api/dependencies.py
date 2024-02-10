# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from logging import Logger

from invokeai.app.services.item_storage.item_storage_memory import ItemStorageMemory
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.version.invokeai_version import __version__

from ..services.board_image_records.board_image_records_sqlite import SqliteBoardImageRecordStorage
from ..services.board_images.board_images_default import BoardImagesService
from ..services.board_records.board_records_sqlite import SqliteBoardRecordStorage
from ..services.boards.boards_default import BoardService
from ..services.config import InvokeAIAppConfig
from ..services.download import DownloadQueueService
from ..services.image_files.image_files_disk import DiskImageFileStorage
from ..services.image_records.image_records_sqlite import SqliteImageRecordStorage
from ..services.images.images_default import ImageService
from ..services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from ..services.invocation_processor.invocation_processor_default import DefaultInvocationProcessor
from ..services.invocation_queue.invocation_queue_memory import MemoryInvocationQueue
from ..services.invocation_services import InvocationServices
from ..services.invocation_stats.invocation_stats_default import InvocationStatsService
from ..services.invoker import Invoker
from ..services.latents_storage.latents_storage_disk import DiskLatentsStorage
from ..services.latents_storage.latents_storage_forward_cache import ForwardCacheLatentsStorage
from ..services.model_manager.model_manager_default import ModelManagerService
from ..services.names.names_default import SimpleNameService
from ..services.session_processor.session_processor_default import DefaultSessionProcessor
from ..services.session_queue.session_queue_sqlite import SqliteSessionQueue
from ..services.shared.graph import GraphExecutionState
from ..services.urls.urls_default import LocalUrlService
from ..services.workflow_records.workflow_records_sqlite import SqliteWorkflowRecordsStorage
from .events import FastAPIEventService


# TODO: is there a better way to achieve this?
def check_internet() -> bool:
    """
    Return true if the internet is reachable.
    It does this by pinging huggingface.co.
    """
    import urllib.request

    host = "http://huggingface.co"
    try:
        urllib.request.urlopen(host, timeout=1)
        return True
    except Exception:
        return False


logger = InvokeAILogger.get_logger()


class ApiDependencies:
    """Contains and initializes all dependencies for the API"""

    invoker: Invoker

    @staticmethod
    def initialize(config: InvokeAIAppConfig, event_handler_id: int, logger: Logger = logger) -> None:
        logger.info(f"InvokeAI version {__version__}")
        logger.info(f"Root directory = {str(config.root_path)}")
        logger.debug(f"Internet connectivity is {config.internet_available}")

        output_folder = config.output_path
        image_files = DiskImageFileStorage(f"{output_folder}/images")

        db = init_db(config=config, logger=logger, image_files=image_files)

        configuration = config
        logger = logger

        board_image_records = SqliteBoardImageRecordStorage(db=db)
        board_images = BoardImagesService()
        board_records = SqliteBoardRecordStorage(db=db)
        boards = BoardService()
        events = FastAPIEventService(event_handler_id)
        graph_execution_manager = ItemStorageMemory[GraphExecutionState]()
        image_records = SqliteImageRecordStorage(db=db)
        images = ImageService()
        invocation_cache = MemoryInvocationCache(max_cache_size=config.node_cache_size)
        latents = ForwardCacheLatentsStorage(DiskLatentsStorage(f"{output_folder}/latents"))
        download_queue_service = DownloadQueueService(event_bus=events)
        model_manager = ModelManagerService.build_model_manager(
            app_config=configuration, db=db, download_queue=download_queue_service, events=events
        )
        names = SimpleNameService()
        performance_statistics = InvocationStatsService()
        processor = DefaultInvocationProcessor()
        queue = MemoryInvocationQueue()
        session_processor = DefaultSessionProcessor()
        session_queue = SqliteSessionQueue(db=db)
        urls = LocalUrlService()
        workflow_records = SqliteWorkflowRecordsStorage(db=db)

        services = InvocationServices(
            board_image_records=board_image_records,
            board_images=board_images,
            board_records=board_records,
            boards=boards,
            configuration=configuration,
            events=events,
            graph_execution_manager=graph_execution_manager,
            image_files=image_files,
            image_records=image_records,
            images=images,
            invocation_cache=invocation_cache,
            latents=latents,
            logger=logger,
            model_manager=model_manager,
            download_queue=download_queue_service,
            names=names,
            performance_statistics=performance_statistics,
            processor=processor,
            queue=queue,
            session_processor=session_processor,
            session_queue=session_queue,
            urls=urls,
            workflow_records=workflow_records,
        )

        ApiDependencies.invoker = Invoker(services)
        db.clean()

    @staticmethod
    def shutdown() -> None:
        if ApiDependencies.invoker:
            ApiDependencies.invoker.stop()
