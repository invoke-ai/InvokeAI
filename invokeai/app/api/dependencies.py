# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import sqlite3
from logging import Logger

from invokeai.app.services.board_image_record_storage import SqliteBoardImageRecordStorage
from invokeai.app.services.board_images import BoardImagesService, BoardImagesServiceDependencies
from invokeai.app.services.board_record_storage import SqliteBoardRecordStorage
from invokeai.app.services.boards import BoardService, BoardServiceDependencies
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.image_record_storage import SqliteImageRecordStorage
from invokeai.app.services.images import ImageService, ImageServiceDependencies
from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from invokeai.app.services.resource_name import SimpleNameService
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionProcessor
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.urls import LocalUrlService
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.version.invokeai_version import __version__

from ..services.default_graphs import create_system_graphs
from ..services.download_manager import DownloadQueueService
from ..services.graph import GraphExecutionState, LibraryGraph
from ..services.image_file_storage import DiskImageFileStorage
from ..services.invocation_queue import MemoryInvocationQueue
from ..services.invocation_services import InvocationServices
from ..services.invocation_stats import InvocationStatsService
from ..services.invoker import Invoker
from ..services.latent_storage import DiskLatentsStorage, ForwardCacheLatentsStorage
from ..services.model_install_service import ModelInstallService
from ..services.model_loader_service import ModelLoadService
from ..services.model_record_service import ModelRecordServiceBase
from ..services.processor import DefaultInvocationProcessor
from ..services.sqlite import SqliteItemStorage
from ..services.thread import lock
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

        events = FastAPIEventService(event_handler_id)

        output_folder = config.output_path

        # TODO: build a file/path manager?
        if config.use_memory_db:
            db_location = ":memory:"
        else:
            db_path = config.db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_location = str(db_path)

        logger.info(f"Using database at {db_location}")
        db_conn = sqlite3.connect(db_location, check_same_thread=False)  # TODO: figure out a better threading solution

        if config.log_sql:
            db_conn.set_trace_callback(print)
        db_conn.execute("PRAGMA foreign_keys = ON;")

        graph_execution_manager = SqliteItemStorage[GraphExecutionState](
            conn=db_conn, table_name="graph_executions", lock=lock
        )

        urls = LocalUrlService()
        image_record_storage = SqliteImageRecordStorage(conn=db_conn, lock=lock)
        image_file_storage = DiskImageFileStorage(f"{output_folder}/images")
        names = SimpleNameService()
        latents = ForwardCacheLatentsStorage(DiskLatentsStorage(f"{output_folder}/latents"))

        board_record_storage = SqliteBoardRecordStorage(conn=db_conn, lock=lock)
        board_image_record_storage = SqliteBoardImageRecordStorage(conn=db_conn, lock=lock)

        boards = BoardService(
            services=BoardServiceDependencies(
                board_image_record_storage=board_image_record_storage,
                board_record_storage=board_record_storage,
                image_record_storage=image_record_storage,
                url=urls,
                logger=logger,
            )
        )

        board_images = BoardImagesService(
            services=BoardImagesServiceDependencies(
                board_image_record_storage=board_image_record_storage,
                board_record_storage=board_record_storage,
                image_record_storage=image_record_storage,
                url=urls,
                logger=logger,
            )
        )

        images = ImageService(
            services=ImageServiceDependencies(
                board_image_record_storage=board_image_record_storage,
                image_record_storage=image_record_storage,
                image_file_storage=image_file_storage,
                url=urls,
                logger=logger,
                names=names,
                graph_execution_manager=graph_execution_manager,
            )
        )

        download_queue = DownloadQueueService(event_bus=events, config=config)
        model_record_store = ModelRecordServiceBase.open(config, conn=db_conn, lock=lock)
        model_loader = ModelLoadService(config, model_record_store)
        model_installer = ModelInstallService(config, queue=download_queue, store=model_record_store, event_bus=events)

        services = InvocationServices(
            events=events,
            latents=latents,
            images=images,
            boards=boards,
            board_images=board_images,
            queue=MemoryInvocationQueue(),
            graph_library=SqliteItemStorage[LibraryGraph](conn=db_conn, lock=lock, table_name="graphs"),
            graph_execution_manager=graph_execution_manager,
            processor=DefaultInvocationProcessor(),
            configuration=config,
            performance_statistics=InvocationStatsService(graph_execution_manager),
            logger=logger,
            download_queue=download_queue,
            model_record_store=model_record_store,
            model_loader=model_loader,
            model_installer=model_installer,
            session_queue=SqliteSessionQueue(conn=db_conn, lock=lock),
            session_processor=DefaultSessionProcessor(),
            invocation_cache=MemoryInvocationCache(max_cache_size=config.node_cache_size),
        )

        create_system_graphs(services.graph_library)

        ApiDependencies.invoker = Invoker(services)

        try:
            lock.acquire()
            db_conn.execute("VACUUM;")
            db_conn.commit()
            logger.info("Cleaned database")
        finally:
            lock.release()

    @staticmethod
    def shutdown():
        if ApiDependencies.invoker:
            ApiDependencies.invoker.stop()
