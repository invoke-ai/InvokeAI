# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from logging import Logger

import torch

from invokeai.app.services.item_storage.item_storage_memory import ItemStorageMemory
from invokeai.app.services.object_serializer.object_serializer_ephemeral_disk import ObjectSerializerEphemeralDisk
from invokeai.app.services.object_serializer.object_serializer_forward_cache import ObjectSerializerForwardCache
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.backend.model_manager.metadata import ModelMetadataStore
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData
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
from ..services.model_install import ModelInstallService
from ..services.model_manager.model_manager_default import ModelManagerService
from ..services.model_records import ModelRecordServiceSQL
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
        if output_folder is None:
            raise ValueError("Output folder is not set")

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
        tensors = ObjectSerializerForwardCache(ObjectSerializerEphemeralDisk[torch.Tensor](output_folder / "tensors"))
        conditioning = ObjectSerializerForwardCache(
            ObjectSerializerEphemeralDisk[ConditioningFieldData](output_folder / "conditioning")
        )
        model_manager = ModelManagerService(config, logger)
        model_record_service = ModelRecordServiceSQL(db=db)
        download_queue_service = DownloadQueueService(event_bus=events)
        metadata_store = ModelMetadataStore(db=db)
        model_install_service = ModelInstallService(
            app_config=config,
            record_store=model_record_service,
            download_queue=download_queue_service,
            metadata_store=metadata_store,
            event_bus=events,
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
            logger=logger,
            model_manager=model_manager,
            model_records=model_record_service,
            download_queue=download_queue_service,
            model_install=model_install_service,
            names=names,
            performance_statistics=performance_statistics,
            processor=processor,
            queue=queue,
            session_processor=session_processor,
            session_queue=session_queue,
            urls=urls,
            workflow_records=workflow_records,
            tensors=tensors,
            conditioning=conditioning,
        )

        ApiDependencies.invoker = Invoker(services)
        db.clean()

    @staticmethod
    def shutdown() -> None:
        if ApiDependencies.invoker:
            ApiDependencies.invoker.stop()
