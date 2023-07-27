# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from logging import Logger
import os
from invokeai.app.services.board_image_record_storage import (
    SqliteBoardImageRecordStorage,
)
from invokeai.app.services.board_images import (
    BoardImagesService,
    BoardImagesServiceDependencies,
)
from invokeai.app.services.board_record_storage import SqliteBoardRecordStorage
from invokeai.app.services.boards import BoardService, BoardServiceDependencies
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.image_record_storage import SqliteImageRecordStorage
from invokeai.app.services.images import ImageService, ImageServiceDependencies
from invokeai.app.services.resource_name import SimpleNameService
from invokeai.app.services.urls import LocalUrlService
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.version.invokeai_version import __version__

from ..services.default_graphs import create_system_graphs
from ..services.latent_storage import DiskLatentsStorage, ForwardCacheLatentsStorage
from ..services.graph import GraphExecutionState, LibraryGraph
from ..services.image_file_storage import DiskImageFileStorage
from ..services.invocation_queue import MemoryInvocationQueue
from ..services.invocation_services import InvocationServices
from ..services.invoker import Invoker
from ..services.processor import DefaultInvocationProcessor
from ..services.sqlite import SqliteItemStorage
from ..services.model_manager_service import ModelManagerService
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
    except:
        return False


logger = InvokeAILogger.getLogger()


class ApiDependencies:
    """Contains and initializes all dependencies for the API"""

    invoker: Invoker = None

    @staticmethod
    def initialize(config: InvokeAIAppConfig, event_handler_id: int, logger: Logger = logger):
        logger.info(f"InvokeAI version {__version__}")
        logger.info(f"Root directory = {str(config.root_path)}")
        logger.debug(f"Internet connectivity is {config.internet_available}")

        events = FastAPIEventService(event_handler_id)

        output_folder = config.output_path

        # TODO: build a file/path manager?
        db_location = config.db_path
        db_location.parent.mkdir(parents=True, exist_ok=True)

        graph_execution_manager = SqliteItemStorage[GraphExecutionState](
            filename=db_location, table_name="graph_executions"
        )

        urls = LocalUrlService()
        image_record_storage = SqliteImageRecordStorage(db_location)
        image_file_storage = DiskImageFileStorage(f"{output_folder}/images")
        names = SimpleNameService()
        latents = ForwardCacheLatentsStorage(DiskLatentsStorage(f"{output_folder}/latents"))

        board_record_storage = SqliteBoardRecordStorage(db_location)
        board_image_record_storage = SqliteBoardImageRecordStorage(db_location)

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

        services = InvocationServices(
            model_manager=ModelManagerService(config, logger),
            events=events,
            latents=latents,
            images=images,
            boards=boards,
            board_images=board_images,
            queue=MemoryInvocationQueue(),
            graph_library=SqliteItemStorage[LibraryGraph](filename=db_location, table_name="graphs"),
            graph_execution_manager=graph_execution_manager,
            processor=DefaultInvocationProcessor(),
            configuration=config,
            logger=logger,
        )

        create_system_graphs(services.graph_library)

        ApiDependencies.invoker = Invoker(services)

    @staticmethod
    def shutdown():
        if ApiDependencies.invoker:
            ApiDependencies.invoker.stop()
