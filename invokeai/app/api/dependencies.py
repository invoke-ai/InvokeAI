# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from logging import Logger

from invokeai.app.services.board_image_record_storage import SqliteBoardImageRecordStorage
from invokeai.app.services.board_images import BoardImagesService
from invokeai.app.services.board_record_storage import SqliteBoardRecordStorage
from invokeai.app.services.boards import BoardService
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.image_record_storage import SqliteImageRecordStorage
from invokeai.app.services.images import ImageService
from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from invokeai.app.services.resource_name import SimpleNameService
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionProcessor
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.db import SqliteDatabase
from invokeai.app.services.urls import LocalUrlService
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.version.invokeai_version import __version__

from ..services.default_graphs import create_system_graphs
from ..services.graph import GraphExecutionState, LibraryGraph
from ..services.image_file_storage import DiskImageFileStorage
from ..services.invocation_queue import MemoryInvocationQueue
from ..services.invocation_services import InvocationServices
from ..services.invocation_stats import InvocationStatsService
from ..services.invoker import Invoker
from ..services.latent_storage import DiskLatentsStorage, ForwardCacheLatentsStorage
from ..services.model_manager_service import ModelManagerService
from ..services.processor import DefaultInvocationProcessor
from ..services.sqlite import SqliteItemStorage
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
    def initialize(config: InvokeAIAppConfig, event_handler_id: int, logger: Logger = logger):
        logger.info(f"InvokeAI version {__version__}")
        logger.info(f"Root directory = {str(config.root_path)}")
        logger.debug(f"Internet connectivity is {config.internet_available}")

        output_folder = config.output_path

        db = SqliteDatabase(config, logger)

        configuration = config
        logger = logger

        board_image_records = SqliteBoardImageRecordStorage(db=db)
        board_images = BoardImagesService()
        board_records = SqliteBoardRecordStorage(db=db)
        boards = BoardService()
        events = FastAPIEventService(event_handler_id)
        graph_execution_manager = SqliteItemStorage[GraphExecutionState](db=db, table_name="graph_executions")
        graph_library = SqliteItemStorage[LibraryGraph](db=db, table_name="graphs")
        image_files = DiskImageFileStorage(f"{output_folder}/images")
        image_records = SqliteImageRecordStorage(db=db)
        images = ImageService()
        invocation_cache = MemoryInvocationCache(max_cache_size=config.node_cache_size)
        latents = ForwardCacheLatentsStorage(DiskLatentsStorage(f"{output_folder}/latents"))
        model_manager = ModelManagerService(config, logger)
        names = SimpleNameService()
        performance_statistics = InvocationStatsService()
        processor = DefaultInvocationProcessor()
        queue = MemoryInvocationQueue()
        session_processor = DefaultSessionProcessor()
        session_queue = SqliteSessionQueue(db=db)
        urls = LocalUrlService()

        services = InvocationServices(
            board_image_records=board_image_records,
            board_images=board_images,
            board_records=board_records,
            boards=boards,
            configuration=configuration,
            events=events,
            graph_execution_manager=graph_execution_manager,
            graph_library=graph_library,
            image_files=image_files,
            image_records=image_records,
            images=images,
            invocation_cache=invocation_cache,
            latents=latents,
            logger=logger,
            model_manager=model_manager,
            names=names,
            performance_statistics=performance_statistics,
            processor=processor,
            queue=queue,
            session_processor=session_processor,
            session_queue=session_queue,
            urls=urls,
        )

        create_system_graphs(services.graph_library)

        ApiDependencies.invoker = Invoker(services)

        db.clean()

    @staticmethod
    def shutdown():
        if ApiDependencies.invoker:
            ApiDependencies.invoker.stop()
