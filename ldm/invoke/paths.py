"""
Filesystem paths
"""

from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass
import inspect
import os

@dataclass
class PathSpec:
    kind: str
    description: str
    location: Path

class InvokePaths:
    """
    Singleton class to manage paths
    """
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(InvokePaths, cls).__new__(cls)
        return cls.instance

    # def __init__(self, path: Union[str, Path] = None) -> None:

    #     rootdir = self.rootdir(path)

    # def set_outputs(self, path: Union[str, Path] = None) -> Path:
    #     self.outputs.location = self.root.location / "outputs" if path is None else Path(path)
    #     return self.outputs.location

    ### PathSpecs ###

    @property
    def rootdir(self) -> PathSpec:
        return PathSpec(
            kind = "directory",
            description = "InvokeAI application",
            location = Path("~/invokeai")
        )

    @rootdir.setter
    def rootdir(self, path: Union[str, Path] = None) -> None:
        if path is None:
            if os.environ.get("INVOKEAI_ROOT"):
                location = Path(os.environ.get("INVOKEAI_ROOT"))
            elif os.environ.get("VIRTUAL_ENV"):
                location = Path(os.environ.get("VIRTUAL_ENV")).parent
            else:
                location = Path("~/invokeai")

        self.rootdir.location = Path(location).expanduser().resolve()

    @property
    def models(self) -> PathSpec:
        return PathSpec(
            kind = "directory",
            description = "Model cache",
            location = self.root.location / "models"
        )

    @property
    def configs(self) -> PathSpec:
        return PathSpec(
            kind = "directory",
            description = "Common configuration files",
            location = self.root.location / "configs"
        )

    @property
    def outdir(self) -> PathSpec:
        return PathSpec(
            kind = "directory",
            description = "Image outputs",
            location = self.root.location / "outputs"
        )

    @property
    def default_weights(self) -> PathSpec:
        return PathSpec(
            kind = "directory",
            description = "Default SD weights",
            location = self.models.location / "ldm/stable-diffusion-v1"
        )

    @property
    def sd_configs(self) -> PathSpec:
        return PathSpec(
            kind = "directory",
            description = "SD model parameters",
            location = self.configs.location / "stable-diffusion"
        )

    @property
    def models_config(self) -> PathSpec:
        return PathSpec(
            kind = "file",
            description = "Active models configuration",
            location = self.configs.location / "models.yaml"
        )

    @property
    def initfile(self) -> PathSpec:
        return PathSpec(
            kind = "file",
            description = "Application init",
            location = self.root.location / "invokeai.init"
        )

    @property
    def init_models(self) -> PathSpec:
        return PathSpec(
            kind = "file",
            description = "Core supported models config",
            location = self.configs.location / "INITIAL_MODELS.yaml"
        )

    ## not yet in use - a unified config file is a proposed change
    # @property
    # def main_config(self) -> PathSpec:
    #     return PathSpec(
    #         kind = "file",
    #         description = "Application configuration",
    #         location = self.root.location / "invokeai.yaml"
    #     )

    def get(self) -> list[PathSpec]:
        """
        Returns a list of all defined PathSpecs
        """

        attrs = inspect.getmembers(self, predicate = lambda m: not (inspect.isroutine(m)))
        return [a[1] for a in attrs if isinstance(a[1], PathSpec)]