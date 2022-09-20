# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

"""Containers module."""

from dependency_injector import containers, providers
from ldm.generate import Generate
from ldm.dream.app.services.generation.services import GeneratorService


class GeneratorContainer(containers.DeclarativeContainer):
    # TODO: ideally we'd only load configuration at the parent, but it's not available
    # until after usage, and configuration passed as a Dependency is static
    # https://github.com/ets-labs/python-dependency-injector/issues/620
    config = providers.Configuration()

    signal_service = providers.Dependency()
    generation_queue_service = providers.Dependency()
    image_storage_service = providers.Dependency()
    image_intermediates_storage_service = providers.Dependency()
    log_service = providers.Dependency()

    # TODO: Add a model provider service that provides model(s) dynamically
    model_singleton = providers.ThreadSafeSingleton(
        Generate,
        model=config.model,
        sampler_name=config.sampler_name,
        embedding_path=config.embedding_path,
        full_precision=config.full_precision,
    )

    generator_service = providers.ThreadSafeSingleton(
        GeneratorService,
        model=model_singleton,
        queue=generation_queue_service,
        imageStorage=image_storage_service,
        intermediateStorage=image_intermediates_storage_service,
        log=log_service,
        signal_service=signal_service,
    )
