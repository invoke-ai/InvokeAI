"""
This file implements base class for Inference pipeline
Implements abstract methods for inference related operations
"""
from abc import ABC, abstractmethod
class inferencePipeline(ABC):

    """
    Instantiation of Inference Pipeline class
    """

    @abstractmethod
    def prompt2image(self):
        pass

    @abstractmethod
    def getCompleter(self):
        pass
