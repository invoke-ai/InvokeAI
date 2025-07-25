__version__ = "0.0.9"

from invokeai.backend.bria.controlnet_aux.canny import CannyDetector as CannyDetector
from invokeai.backend.bria.controlnet_aux.open_pose import OpenposeDetector as OpenposeDetector

__all__ = ["CannyDetector", "OpenposeDetector"]
