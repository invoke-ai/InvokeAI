"""Initialization file for model load service module."""

from invokeai.app.services.model_load.model_load_base import ModelLoadServiceBase
from invokeai.app.services.model_load.model_load_default import ModelLoadService

__all__ = ["ModelLoadServiceBase", "ModelLoadService"]
