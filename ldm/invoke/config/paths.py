"""
Filesystem paths

Provides a singleton class

Configurable:
- path to runtime dir
- path and/or filename for models.yaml
- path and/or dirname for outputs

"""

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

DEFAULT_RUNTIME_DIR = "~/invokeai"


@dataclass
class PathSpec:
    """
    A PathSpec describes a location on the filesystem together with its metadata
    """

    kind: str
    description: str
    location: Path

    def __repr__(self) -> str:
        return str(self.location.expanduser().resolve())


class InvokePaths:
    """
    Singleton class to manage paths
    """

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(InvokePaths, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(
        self,
        root: Union[str, Path] = None,
        outdir: Union[str, Path] = None,
        config: Union[str, Path] = None,
    ) -> None:
        self.root = root
        self.outdir = outdir or self.root.location / "outputs"
        self.config = config or self.config_dir.location / "models.yaml"

    #### Runtime directory

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, path: Union[str, Path] = None) -> None:
        if path is None:
            if os.environ.get("INVOKEAI_ROOT"):
                location = Path(os.environ.get("INVOKEAI_ROOT"))
            elif os.environ.get("VIRTUAL_ENV"):
                # how to handle this in non-automated-install situations?
                # https://github.com/invoke-ai/InvokeAI/issues/2064
                location = Path(os.environ.get("VIRTUAL_ENV")).parent
            else:
                location = Path(DEFAULT_RUNTIME_DIR)
        else:
            location = Path(path)
        self._root = PathSpec(kind="directory", description="InvokeAI runtime", location=location)

    ### Image outputs

    @property
    def outdir(self) -> PathSpec:
        return self._outdir

    @outdir.setter
    def outdir(self, path: Union[str, Path] = None) -> None:
        path = "outputs" if path is None else path
        path = Path(path).expanduser()
        location = self.root.location / path if not path.is_absolute() else path
        self._outdir = PathSpec(kind="directory", description="Image outputs", location=location)

    ### Main config file (model configuration)

    @property
    def config(self) -> PathSpec:
        return self._config

    @config.setter
    def config(self, path: Union[str, Path] = None) -> None:
        path = "models.yaml" if path is None else path
        path = Path(path).expanduser()
        location = self.config_dir.location / path if not path.is_absolute() else path
        # should we verify this is indeed a YAML file? raise?
        self._config = PathSpec(kind="file", description="Main configuration", location=location)

    ### Model cache

    @property
    def models_dir(self) -> PathSpec:
        return PathSpec(
            kind="directory",
            description="Model cache",
            location=self.root.location / "models",
        )

    ### Configuration store

    @property
    def config_dir(self) -> PathSpec:
        return PathSpec(
            kind="directory",
            description="Common configuration files",
            location=self.root.location / "configs",
        )

    ### SD weights store

    @property
    def default_weights(self) -> PathSpec:
        return PathSpec(
            kind="directory",
            description="Stable Diffusion weights",
            location=self.models_dir.location / "ldm/stable-diffusion-v1",
        )

    ### SD specific config files

    @property
    def sd_configs_dir(self) -> PathSpec:
        return PathSpec(
            kind="directory",
            description="SD model parameters",
            location=self.config_dir.location / "stable-diffusion",
        )

    ### App initialization file (default CLI switches)

    @property
    def initfile(self) -> PathSpec:
        return PathSpec(
            kind="file",
            description="Application init",
            location=self.root.location / "invokeai.init",
        )

    ### Default model configs - initial source for the main config file

    @property
    def initial_models_config(self) -> PathSpec:
        return PathSpec(
            kind="file",
            description="Initial models configuration",
            location=self.config_dir.location / "INITIAL_MODELS.yaml",
        )

    def get(self) -> list[PathSpec]:
        """
        Returns a list of all defined PathSpecs
        """

        attrs = inspect.getmembers(self, predicate=lambda m: not (inspect.isroutine(m)))
        return [a[1] for a in attrs if ((isinstance(a[1], PathSpec)) and not (a[0].startswith("_")))]
