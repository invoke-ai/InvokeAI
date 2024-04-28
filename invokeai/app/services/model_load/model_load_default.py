# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Team
"""Implementation of model loader service."""

from pathlib import Path
from typing import Callable, Dict, Optional, Type

from picklescan.scanner import scan_file_path
from safetensors.torch import load_file as safetensors_load_file
from torch import Tensor
from torch import load as torch_load

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.invocation_context import InvocationContextData
from invokeai.backend.model_manager import AnyModel, AnyModelConfig, SubModelType
from invokeai.backend.model_manager.load import (
    LoadedModel,
    ModelLoaderRegistry,
    ModelLoaderRegistryBase,
)
from invokeai.backend.model_manager.load.convert_cache import ModelConvertCacheBase
from invokeai.backend.model_manager.load.model_cache.model_cache_base import ModelCacheBase
from invokeai.backend.util.logging import InvokeAILogger

from .model_load_base import ModelLoadServiceBase


class ModelLoadService(ModelLoadServiceBase):
    """Wrapper around ModelLoaderRegistry."""

    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        ram_cache: ModelCacheBase[AnyModel],
        convert_cache: ModelConvertCacheBase,
        registry: Optional[Type[ModelLoaderRegistryBase]] = ModelLoaderRegistry,
    ):
        """Initialize the model load service."""
        logger = InvokeAILogger.get_logger(self.__class__.__name__)
        logger.setLevel(app_config.log_level.upper())
        self._logger = logger
        self._app_config = app_config
        self._ram_cache = ram_cache
        self._convert_cache = convert_cache
        self._registry = registry

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    @property
    def ram_cache(self) -> ModelCacheBase[AnyModel]:
        """Return the RAM cache used by this loader."""
        return self._ram_cache

    @property
    def convert_cache(self) -> ModelConvertCacheBase:
        """Return the checkpoint convert cache used by this loader."""
        return self._convert_cache

    def load_model(
        self,
        model_config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
        context_data: Optional[InvocationContextData] = None,
    ) -> LoadedModel:
        """
        Given a model's configuration, load it and return the LoadedModel object.

        :param model_config: Model configuration record (as returned by ModelRecordBase.get_model())
        :param submodel: For main (pipeline models), the submodel to fetch.
        :param context: Invocation context used for event reporting
        """
        if context_data:
            self._emit_load_event(
                context_data=context_data,
                model_config=model_config,
                submodel_type=submodel_type,
            )

        implementation, model_config, submodel_type = self._registry.get_implementation(model_config, submodel_type)  # type: ignore
        loaded_model: LoadedModel = implementation(
            app_config=self._app_config,
            logger=self._logger,
            ram_cache=self._ram_cache,
            convert_cache=self._convert_cache,
        ).load_model(model_config, submodel_type)

        if context_data:
            self._emit_load_event(
                context_data=context_data,
                model_config=model_config,
                submodel_type=submodel_type,
                loaded=True,
            )
        return loaded_model

    def load_ckpt_from_path(
        self, model_path: Path, loader: Optional[Callable[[Path], Dict[str, Tensor]]] = None
    ) -> LoadedModel:
        """
        Load the checkpoint-format model file located at the indicated Path.

        This will load an arbitrary model file into the RAM cache. If the optional loader
        argument is provided, the loader will be invoked to load the model into
        memory. Otherwise the method will call safetensors.torch.load_file() or
        torch.load() as appropriate to the file suffix.

        Be aware that the LoadedModel object will have a `config` attribute of None.

        Args:
          model_path: A pathlib.Path to a checkpoint-style models file
          loader: A Callable that expects a Path and returns a Dict[str, Tensor]

        Returns:
          A LoadedModel object.
        """
        cache_key = str(model_path)
        ram_cache = self.ram_cache
        try:
            return LoadedModel(_locker=ram_cache.get(key=cache_key))
        except IndexError:
            pass

        def torch_load_file(checkpoint: Path) -> Dict[str, Tensor]:
            scan_result = scan_file_path(checkpoint)
            if scan_result.infected_files != 0:
                raise Exception("The model at {checkpoint} is potentially infected by malware. Aborting load.")
            result: Dict[str, Tensor] = torch_load(checkpoint, map_location="cpu")
            return result

        if loader is None:
            loader = (
                torch_load_file
                if model_path.suffix.endswith((".ckpt", ".pt", ".pth", ".bin"))
                else lambda path: safetensors_load_file(path, device="cpu")
            )

        raw_model = loader(model_path)
        ram_cache.put(key=cache_key, model=raw_model)
        return LoadedModel(_locker=ram_cache.get(key=cache_key))

    def _emit_load_event(
        self,
        context_data: InvocationContextData,
        model_config: AnyModelConfig,
        loaded: Optional[bool] = False,
        submodel_type: Optional[SubModelType] = None,
    ) -> None:
        if not self._invoker:
            return

        if not loaded:
            self._invoker.services.events.emit_model_load_started(
                queue_id=context_data.queue_item.queue_id,
                queue_item_id=context_data.queue_item.item_id,
                queue_batch_id=context_data.queue_item.batch_id,
                graph_execution_state_id=context_data.queue_item.session_id,
                model_config=model_config,
                submodel_type=submodel_type,
            )
        else:
            self._invoker.services.events.emit_model_load_completed(
                queue_id=context_data.queue_item.queue_id,
                queue_item_id=context_data.queue_item.item_id,
                queue_batch_id=context_data.queue_item.batch_id,
                graph_execution_state_id=context_data.queue_item.session_id,
                model_config=model_config,
                submodel_type=submodel_type,
            )
