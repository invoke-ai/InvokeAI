"""
Filesystem paths
"""

from pathlib import Path
from typing import Union
from dataclasses import dataclass
import inspect
import os

@dataclass
class PathSpec:
    kind: str
    description: str
    location: Path

class Paths:
    def __init__(self, root_dir: Union[str, Path] = None) -> None:

        self.set_root(root_dir)

    def set_root(self, path: Union[str, Path] = None) -> Path:

        if path is None:
            if (path := os.getenv("INVOKEAI_ROOT")) is None:
                # Default location
                path = "~/invokeai"

        self.root_dir = Path(path)
        return path

    @property
    def root(self) -> PathSpec:
        return PathSpec(
            kind = "directory",
            description = "InvokeAI runtime (root)",
            location = self.root_dir
        )

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
    def outputs(self) -> PathSpec:
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
            description = "Known models configuration",
            location = self.configs.location / "models.yaml"
        )

    @property
    def initfile(self) -> PathSpec:
        return PathSpec(
            kind = "file",
            description = "Application init",
            location = self.root.location / "invokeai.init"
        )

    ## not yet in use
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