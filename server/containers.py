# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

"""Containers module."""

from dependency_injector import containers, providers
from server.generation.containers import GeneratorContainer
from server.signaling.containers import SignalingContainer
from server.storage.containers import StorageContainer
from server.logging.services import LogService


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=["server"])

    config = providers.Configuration()

    # TODO: get locations from config
    log_service = providers.ThreadSafeSingleton(
        LogService, "./outputs/img-samples/", "dream_web_log.txt"
    )

    storage_package = providers.Container(StorageContainer)

    signaling_package = providers.Container(
        SignalingContainer, signal_queue_service=storage_package.signal_queue_service
    )

    generator_package = providers.Container(
        GeneratorContainer,
        config=config,
        signal_service=signaling_package.signal_service,
        generation_queue_service=storage_package.generation_queue_service,
        image_storage_service=storage_package.image_storage_service,
        image_intermediates_storage_service=storage_package.image_intermediates_storage_service,
        log_service=log_service,
    )
