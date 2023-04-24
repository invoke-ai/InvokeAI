"""
This file implements base class Inference Model for model selection
Implements abstract methods for inference related operations
"""
from abc import ABC, abstractmethod
class inferencePipeline(ABC):

    """
    Instantiation of Inference model class
    """

    @abstractmethod
    def prompt2image(self):
        pass

    @abstractmethod
    def getCompleter(self):
        pass
