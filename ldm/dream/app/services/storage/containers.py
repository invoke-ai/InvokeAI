# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

"""Containers module."""

from dependency_injector import containers, providers
from ldm.dream.app.services.storage.services import (
    ImageStorageService,
    JobQueueService,
    SignalQueueService,
)


class StorageContainer(containers.DeclarativeContainer):
    # TODO: get location from config
    image_storage_service = providers.ThreadSafeSingleton(
        ImageStorageService, "./outputs/img-samples/"
    )

    # TODO: get location from config
    image_intermediates_storage_service = providers.ThreadSafeSingleton(
        ImageStorageService, "./outputs/intermediates/"
    )

    # TODO: Move queues to their own container?
    signal_queue_service = providers.ThreadSafeSingleton(SignalQueueService)

    generation_queue_service = providers.ThreadSafeSingleton(JobQueueService)
