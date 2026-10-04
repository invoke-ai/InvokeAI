"""Class for simple diffusers model loading in InvokeAI."""

import importlib
import inspect
from pathlib import Path
from typing import Any, Optional

from diffusers.configuration_utils import ConfigMixin

from invokeai.backend.model_manager.configs.base import Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.T2IAdapter, format=ModelFormat.Diffusers)
class GenericDiffusersLoader(ModelLoader):
    """Class to load simple diffusers models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        model_path = Path(config.path)
        model_class = self.get_hf_load_class(model_path)
        if submodel_type is not None:
            raise Exception(f"There are no submodels in models of type {model_class}")
        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None
        try:
            result: AnyModel = model_class.from_pretrained(
                model_path, torch_dtype=self._torch_dtype, variant=variant, local_files_only=True
            )
        except OSError as e:
            if variant and "no file named" in str(
                e
            ):  # try without the variant, just in case user's preferences changed
                result = model_class.from_pretrained(model_path, torch_dtype=self._torch_dtype, local_files_only=True)
            else:
                raise e
        result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
        return result

    def get_hf_load_class(self, model_path: Path, submodel_type: Optional[SubModelType] = None) -> type:
        """Return the class declared by a Diffusers or Transformers config."""
        if submodel_type:
            config = self._load_diffusers_config(model_path, config_name="model_index.json")
            if not isinstance(config, dict):
                raise ValueError("model_index.json must contain a JSON object")
            definition = config.get(submodel_type.value)
            if not isinstance(definition, (list, tuple)) or len(definition) != 2:
                raise ValueError(f'The "{submodel_type}" submodel class metadata is missing or malformed.')
            module, class_name = definition
            if not isinstance(class_name, str) or not class_name.strip():
                raise ValueError(f'The "{submodel_type}" submodel class name must be a nonempty string.')
            if not isinstance(module, str) or not module:
                raise ValueError(f'The "{submodel_type}" submodel module must be a nonempty string.')
            return self._hf_definition_to_type(module=module, class_name=class_name)
        else:
            config = self._load_diffusers_config(model_path, config_name="config.json")
            if not isinstance(config, dict):
                raise ValueError("config.json must contain a JSON object")

            if "_class_name" in config:
                class_name = config["_class_name"]
                if not isinstance(class_name, str) or not class_name.strip():
                    raise ValueError("config.json _class_name must be a nonempty string")
                return self._hf_definition_to_type(module="diffusers", class_name=class_name)

            architectures = config.get("architectures")
            if (
                not isinstance(architectures, list)
                or not architectures
                or any(not isinstance(name, str) or not name.strip() for name in architectures)
            ):
                raise ValueError("config.json architectures must be a nonempty list of class names")
            return self._hf_definition_to_type(module="transformers", class_name=architectures[0])

    def _hf_definition_to_type(self, module: str, class_name: str) -> type:
        """Resolve only known model namespaces and reject symbols that cannot load local weights."""
        known_modules = {
            "diffusers",
            "transformers",
            "invokeai.backend.quantization.fast_quantized_transformers_model",
            "invokeai.backend.quantization.fast_quantized_diffusion_model",
        }
        try:
            if module in known_modules:
                namespace = importlib.import_module(module)
            elif module == "stable_diffusion":
                namespace = importlib.import_module("diffusers.pipelines.stable_diffusion")
            elif module == "diffusers.pipelines" or module.startswith("diffusers.pipelines."):
                # Diffusers indexes use both `diffusers` and pipeline module paths. The old resolver exposed
                # pipeline classes through the top-level pipelines namespace for either path.
                namespace = importlib.import_module("diffusers.pipelines")
            else:
                raise ValueError(f"Unsupported model class module {module!r}")
            result = getattr(namespace, class_name)
            if not inspect.isclass(result) or not callable(getattr(result, "from_pretrained", None)):
                raise TypeError(f"{class_name!r} is not a loadable class")
        except Exception as e:
            raise ValueError(f"Unable to resolve loadable model class {class_name!r} from {module!r}") from e
        return result

    def _load_diffusers_config(self, model_path: Path, config_name: str = "config.json") -> dict[str, Any]:
        return ConfigLoader.load_config(model_path, config_name=config_name)


class ConfigLoader(ConfigMixin):
    """Subclass of ConfigMixin for loading diffusers configuration files."""

    @classmethod
    def load_config(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:  # pyright: ignore [reportIncompatibleMethodOverride]
        """Load a diffusrs ConfigMixin configuration."""
        cls.config_name = kwargs.pop("config_name")
        # TODO(psyche): the types on this diffusers method are not correct
        return super().load_config(*args, **kwargs)  # type: ignore
