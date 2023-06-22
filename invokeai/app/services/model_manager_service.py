# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Callable, List, Tuple, types, TYPE_CHECKING
from dataclasses import dataclass

from invokeai.backend.model_management.model_manager import (
    ModelManager,
    BaseModelType,
    ModelType,
    SubModelType,
    ModelInfo,
)
from invokeai.app.models.exceptions import CanceledException
from .config import InvokeAIAppConfig
from ...backend.util import choose_precision, choose_torch_device

if TYPE_CHECKING:
    from ..invocations.baseinvocation import BaseInvocation, InvocationContext


class ModelManagerServiceBase(ABC):
    """Responsible for managing models on disk and in memory"""

    @abstractmethod
    def __init__(
        self,
        config: InvokeAIAppConfig,
        logger: types.ModuleType,
    ):
        """
        Initialize with the path to the models.yaml config file. 
        Optional parameters are the torch device type, precision, max_models,
        and sequential_offload boolean. Note that the default device
        type and precision are set up for a CUDA system running at half precision.
        """
        pass
    
    @abstractmethod
    def get_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        submodel: Optional[SubModelType] = None,
        node: Optional[BaseInvocation] = None,
        context: Optional[InvocationContext] = None,
    ) -> ModelInfo:
        """Retrieve the indicated model with name and type. 
        submodel can be used to get a part (such as the vae) 
        of a diffusers pipeline."""
        pass

    @property
    @abstractmethod
    def logger(self):
        pass

    @abstractmethod
    def model_exists(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
    ) -> bool:
        pass

    @abstractmethod
    def model_info(self, model_name: str, base_model: BaseModelType, model_type: ModelType) -> dict:
        """
        Given a model name returns a dict-like (OmegaConf) object describing it.
        """
        pass

    @abstractmethod
    def model_names(self) -> List[Tuple[str, BaseModelType, ModelType]]:
        """
        Returns a list of all the model names known.
        """
        pass

    @abstractmethod
    def list_models(self, base_model: Optional[BaseModelType] = None, model_type: Optional[ModelType] = None) -> dict:
        """
        Return a dict of models in the format:
        { model_type1:
          { model_name1: {'status': 'active'|'cached'|'not loaded',
                         'model_name' : name,
                         'model_type' : SDModelType,
                         'description': description,
                         'format': 'folder'|'safetensors'|'ckpt'
                         },
            model_name2: { etc }
          },
          model_type2:
            { model_name_n: etc
        }
        """
        pass


    @abstractmethod
    def add_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        model_attributes: dict,
        clobber: bool = False
    ) -> None:
        """
        Update the named model with a dictionary of attributes. Will fail with an
        assertion error if the name already exists. Pass clobber=True to overwrite.
        On a successful update, the config will be changed in memory. Will fail 
        with an assertion error if provided attributes are incorrect or 
        the model name is missing. Call commit() to write changes to disk.
        """
        pass

    @abstractmethod
    def del_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
    ):
        """
        Delete the named model from configuration. If delete_files is true, 
        then the underlying weight file or diffusers directory will be deleted 
        as well. Call commit() to write to disk.
        """
        pass

    @abstractmethod
    def commit(self, conf_file: Path = None) -> None:
        """
        Write current configuration out to the indicated file.
        If no conf_file is provided, then replaces the
        original file/database used to initialize the object.
        """
        pass

# simple implementation
class ModelManagerService(ModelManagerServiceBase):
    """Responsible for managing models on disk and in memory"""
    def __init__(
        self,
        config: InvokeAIAppConfig,
        logger: types.ModuleType,
    ):
        """
        Initialize with the path to the models.yaml config file. 
        Optional parameters are the torch device type, precision, max_models,
        and sequential_offload boolean. Note that the default device
        type and precision are set up for a CUDA system running at half precision.
        """
        if config.model_conf_path and config.model_conf_path.exists():
            config_file = config.model_conf_path
        else:
            config_file = config.root_dir / "configs/models.yaml"
        if not config_file.exists():
            raise IOError(f"The file {config_file} could not be found.")

        logger.debug(f'config file={config_file}')

        device = torch.device(choose_torch_device())
        precision = config.precision
        if precision == "auto":
            precision = choose_precision(device)
        dtype = torch.float32 if precision == 'float32' else torch.float16

        # this is transitional backward compatibility
        # support for the deprecated `max_loaded_models`
        # configuration value. If present, then the
        # cache size is set to 2.5 GB times
        # the number of max_loaded_models. Otherwise
        # use new `max_cache_size` config setting
        max_cache_size = config.max_cache_size \
            if hasattr(config,'max_cache_size') \
               else config.max_loaded_models * 2.5

        sequential_offload = config.sequential_guidance

        self.mgr = ModelManager(
            config=config_file,
            device_type=device,
            precision=dtype,
            max_cache_size=max_cache_size,
            sequential_offload=sequential_offload,
            logger=logger,
        )
        logger.info('Model manager service initialized')

    def get_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        submodel: Optional[SubModelType] = None,
        node: Optional[BaseInvocation] = None,
        context: Optional[InvocationContext] = None,
    ) -> ModelInfo:
        """
        Retrieve the indicated model. submodel can be used to get a
        part (such as the vae) of a diffusers mode.
        """

        # if we are called from within a node, then we get to emit
        # load start and complete events
        if node and context:
            self._emit_load_event(
                node=node,
                context=context,
                model_name=model_name,
                base_model=base_model,
                model_type=model_type,
                submodel=submodel,
            )

        model_info = self.mgr.get_model(
            model_name,
            base_model,
            model_type,
            submodel,
        )

        if node and context:
            self._emit_load_event(
                node=node,
                context=context,
                model_name=model_name,
                base_model=base_model,
                model_type=model_type,
                submodel=submodel,
                model_info=model_info
            )
            
        return model_info

    def model_exists(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
    ) -> bool:
        """
        Given a model name, returns True if it is a valid
        identifier.
        """
        return self.mgr.model_exists(
            model_name,
            base_model,
            model_type,
        )

    def model_info(self, model_name: str, base_model: BaseModelType, model_type: ModelType) -> dict:
        """
        Given a model name returns a dict-like (OmegaConf) object describing it.
        """
        return self.mgr.model_info(model_name, base_model, model_type)

    def model_names(self) -> List[Tuple[str, BaseModelType, ModelType]]:
        """
        Returns a list of all the model names known.
        """
        return self.mgr.model_names()

    def list_models(
        self,
        base_model: Optional[BaseModelType] = None,
        model_type: Optional[ModelType] = None
    ) -> list[dict]:
    # ) -> dict:
        """
        Return a list of models.
        """
        return self.mgr.list_models(base_model, model_type)

    def add_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        model_attributes: dict,
        clobber: bool = False,
    )->None:
        """
        Update the named model with a dictionary of attributes. Will fail with an
        assertion error if the name already exists. Pass clobber=True to overwrite.
        On a successful update, the config will be changed in memory. Will fail 
        with an assertion error if provided attributes are incorrect or 
        the model name is missing. Call commit() to write changes to disk.
        """
        return self.mgr.add_model(model_name, base_model, model_type, model_attributes, clobber)


    def del_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
    ):
        """
        Delete the named model from configuration. If delete_files is true, 
        then the underlying weight file or diffusers directory will be deleted 
        as well. Call commit() to write to disk.
        """
        self.mgr.del_model(model_name, base_model, model_type)


    def commit(self, conf_file: Optional[Path]=None):
        """
        Write current configuration out to the indicated file.
        If no conf_file is provided, then replaces the
        original file/database used to initialize the object.
        """
        return self.mgr.commit(conf_file)

    def _emit_load_event(
        self,
        node,
        context,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        submodel: SubModelType,
        model_info: Optional[ModelInfo] = None,
    ):
        if context.services.queue.is_canceled(context.graph_execution_state_id):
            raise CanceledException()
        graph_execution_state = context.services.graph_execution_manager.get(context.graph_execution_state_id)
        source_node_id = graph_execution_state.prepared_source_mapping[node.id]
        if model_info:
            context.services.events.emit_model_load_completed(
                graph_execution_state_id=context.graph_execution_state_id,
                node=node.dict(),
                source_node_id=source_node_id,
                model_name=model_name,
                base_model=base_model,
                model_type=model_type,
                submodel=submodel,
                model_info=model_info
            )
        else:
            context.services.events.emit_model_load_started(
                graph_execution_state_id=context.graph_execution_state_id,
                node=node.dict(),
                source_node_id=source_node_id,
                model_name=model_name,
                base_model=base_model,
                model_type=model_type,
                submodel=submodel,
            )


    @property
    def logger(self):
        return self.mgr.logger
        
