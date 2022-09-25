# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from ldm.dream.app.invocations.baseinvocation import InvocationServices
from ldm.dream.app.services.invoker import Invoker
from ldm.dream.args import Args
from ldm.generate import Generate


class ApiDependencies:
    """Contains and initializes all dependencies for the API"""
    invoker: Invoker = None

    @staticmethod
    def Initialize(config): # TODO: pass configuration to this method?
        # TODO: lazy-initialize this by wrapping it
        generate = Generate(
            model=config.model,
            sampler_name=config.sampler_name,
            embedding_path=config.embedding_path,
            full_precision=config.full_precision,
        )

        services = InvocationServices(generate = generate)
        ApiDependencies.invoker = Invoker(services)
