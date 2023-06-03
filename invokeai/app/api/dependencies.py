# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from logging import Logger
import os
from invokeai.app.services.image_record_storage import SqliteImageRecordStorage
from invokeai.app.services.images import ImageService
from invokeai.app.services.metadata import CoreMetadataService
from invokeai.app.services.resource_name import SimpleNameService
from invokeai.app.services.urls import LocalUrlService
from invokeai.backend.util.logging import InvokeAILogger

from ..services.default_graphs import create_system_graphs
from ..services.latent_storage import DiskLatentsStorage, ForwardCacheLatentsStorage
from ..services.model_manager_initializer import get_model_manager
from ..services.restoration_services import RestorationServices
from ..services.graph import GraphExecutionState, LibraryGraph
from ..services.image_file_storage import DiskImageFileStorage
from ..services.invocation_queue import MemoryInvocationQueue
from ..services.invocation_services import InvocationServices
from ..services.invoker import Invoker
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
    except:
        return False


logger = InvokeAILogger.getLogger()


class ApiDependencies:
    """Contains and initializes all dependencies for the API"""

    invoker: Invoker = None

    @staticmethod
    def initialize(config, event_handler_id: int, logger: Logger = logger):
        logger.info(f"Internet connectivity is {config.internet_available}")

        events = FastAPIEventService(event_handler_id)

        output_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../outputs")
        )

        # TODO: build a file/path manager?
        db_location = os.path.join(output_folder, "invokeai.db")

        graph_execution_manager = SqliteItemStorage[GraphExecutionState](
            filename=db_location, table_name="graph_executions"
        )

        urls = LocalUrlService()
        metadata = CoreMetadataService()
        image_record_storage = SqliteImageRecordStorage(db_location)
        image_file_storage = DiskImageFileStorage(f"{output_folder}/images")
        names = SimpleNameService()
        latents = ForwardCacheLatentsStorage(
            DiskLatentsStorage(f"{output_folder}/latents")
        )

        images = ImageService(
            image_record_storage=image_record_storage,
            image_file_storage=image_file_storage,
            metadata=metadata,
            url=urls,
            logger=logger,
            names=names,
            graph_execution_manager=graph_execution_manager,
        )

        services = InvocationServices(
            model_manager=get_model_manager(config, logger),
            events=events,
            latents=latents,
            images=images,
            queue=MemoryInvocationQueue(),
            graph_library=SqliteItemStorage[LibraryGraph](
                filename=db_location, table_name="graphs"
            ),
            graph_execution_manager=graph_execution_manager,
            processor=DefaultInvocationProcessor(),
            restoration=RestorationServices(config, logger),
            configuration=config,
            logger=logger,
        )

        create_system_graphs(services.graph_library)

        ApiDependencies.invoker = Invoker(services)

    @staticmethod
    def shutdown():
        if ApiDependencies.invoker:
            ApiDependencies.invoker.stop()
