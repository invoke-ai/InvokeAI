# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from abc import ABC, abstractmethod

class InvocationABC(ABC):
    """A node to process inputs and produce outputs.
    May use dependency injection in __init__ to receive providers.
    """
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def invoke(self, **kwargs) -> dict:
        """Invoke with provided arguments and return outputs.
        **kwargs should be replaced with specific arguments on deriving classes.
        Deriving classes should additionally provide an InvocationSchema to define
        inputs and outputs.
        """
        pass

