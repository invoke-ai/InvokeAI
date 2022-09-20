# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

"""Containers module."""

from dependency_injector import containers, providers
from ldm.dream.app.services.models import Signal
from ldm.dream.app.services.logging.services import LogService
from ldm.dream.app.services.storage.containers import StorageContainer
from ldm.dream.app.services.storage.services import SignalQueueService
from ldm.dream.app.services.generation.containers import GeneratorContainer

# An override signal service that does nothing but queue signals
# TODO: convert signaling to use events and just don't inject any socketio stuff by default
class SignalServiceOverride:
    __queue: SignalQueueService

    def __init__(self, queue: SignalQueueService):
        self.__queue = queue
        pass

    def get_signal(self) -> Signal:
        return self.__queue.get(block=True)

    def emit(self, signal: Signal):
        self.__queue.push(signal)


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=[
        "ldm.dream.app.services",
        "ldm.dream.app.console"
        ])

    config = providers.Configuration()

    # TODO: get locations from config
    log_service = providers.ThreadSafeSingleton(
        LogService, "./outputs/img-samples/", "dream_web_log.txt"
    )

    storage_package = providers.Container(StorageContainer)

    signal_service = providers.Singleton(
        SignalServiceOverride, queue=storage_package.signal_queue_service
    )

    generator_package = providers.Container(
        GeneratorContainer,
        config=config,
        signal_service=signal_service,
        generation_queue_service=storage_package.generation_queue_service,
        image_storage_service=storage_package.image_storage_service,
        image_intermediates_storage_service=storage_package.image_intermediates_storage_service,
        log_service=log_service,
    )
