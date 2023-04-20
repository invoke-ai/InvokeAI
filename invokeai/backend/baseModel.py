"""
This file implements base class Inference Model for model selection
Implements abstract methods for inference related operations
"""
from abc import ABC, abstractmethod
class inferenceModel(ABC):

    """
    Instantiation of Inference model class
    """
    def __init__(
        self,
        model_type = "Pytorch"
    ):
        self.model_type = model_type

    @abstractmethod
    def prompt2image(self):
        pass

    @abstractmethod
    def getCompleter(self):
        pass
