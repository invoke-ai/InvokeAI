# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from ..services.invocation_services import InvocationServices
from ..services.invoker import Invoker
from ....generate import Generate
from .events import FastAPIEventService


class ApiDependencies:
    """Contains and initializes all dependencies for the API"""
    invoker: Invoker = None

    @staticmethod
    def Initialize(config,
        event_handler_id: int
        ):
        # TODO: lazy-initialize this by wrapping it
        generate = Generate(
            model=config.model,
            sampler_name=config.sampler_name,
            embedding_path=config.embedding_path,
            full_precision=config.full_precision,
        )

        events = FastAPIEventService(event_handler_id)

        services = InvocationServices(
            generate = generate,
            events = events
            )
        ApiDependencies.invoker = Invoker(services)
