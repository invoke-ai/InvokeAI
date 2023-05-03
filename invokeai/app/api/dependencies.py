# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import os

import invokeai.backend.util.logging as logger
from typing import types

from ..services.default_graphs import create_system_graphs
from ..services.latent_storage import DiskLatentsStorage, ForwardCacheLatentsStorage
from ...backend import Globals
from ..services.model_manager_initializer import get_model_manager
from ..services.restoration_services import RestorationServices
from ..services.graph import GraphExecutionState, LibraryGraph
from ..services.image_storage import DiskImageStorage
from ..services.invocation_queue import MemoryInvocationQueue
from ..services.invocation_services import InvocationServices
from ..services.invoker import Invoker
from ..services.processor import DefaultInvocationProcessor
from ..services.sqlite import SqliteItemStorage
from ..services.metadata import PngMetadataService
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


class ApiDependencies:
    """Contains and initializes all dependencies for the API"""

    invoker: Invoker = None

    @staticmethod
    def initialize(config, event_handler_id: int, logger: types.ModuleType=logger):
        Globals.try_patchmatch = config.patchmatch
        Globals.always_use_cpu = config.always_use_cpu
        Globals.internet_available = config.internet_available and check_internet()
        Globals.disable_xformers = not config.xformers
        Globals.ckpt_convert = config.ckpt_convert

        # TO DO: Use the config to select the logger rather than use the default
        # invokeai logging module
        logger.info(f"Internet connectivity is {Globals.internet_available}")

        events = FastAPIEventService(event_handler_id)

        output_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../outputs")
        )

        latents = ForwardCacheLatentsStorage(DiskLatentsStorage(f'{output_folder}/latents'))

        metadata = PngMetadataService()

        images = DiskImageStorage(f'{output_folder}/images', metadata_service=metadata)

        # TODO: build a file/path manager?
        db_location = os.path.join(output_folder, "invokeai.db")

        services = InvocationServices(
            model_manager=get_model_manager(config,logger),
            events=events,
            logger=logger,
            latents=latents,
            images=images,
            metadata=metadata,
            queue=MemoryInvocationQueue(),
            graph_library=SqliteItemStorage[LibraryGraph](
                filename=db_location, table_name="graphs"
            ),
            graph_execution_manager=SqliteItemStorage[GraphExecutionState](
                filename=db_location, table_name="graph_executions"
            ),
            processor=DefaultInvocationProcessor(),
            restoration=RestorationServices(config,logger),
        )

        create_system_graphs(services.graph_library)

        ApiDependencies.invoker = Invoker(services)

    @staticmethod
    def shutdown():
        if ApiDependencies.invoker:
            ApiDependencies.invoker.stop()
