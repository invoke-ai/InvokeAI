"""
This file implements base class Inference Model for model selection
Implements abstract methods for inference related operations
"""

class inferenceModel:
    """
    Instantiation of Inference model class
    """
    def __init__(
        self,
        model_type = "Pytorch"
    ):
        self.model_type = model_type
