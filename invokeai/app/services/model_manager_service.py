# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Callable, List, Tuple, types, TYPE_CHECKING
from dataclasses import dataclass

from invokeai.backend.model_management.model_manager import (
    ModelManager,
    SDModelType,
    SDModelInfo,
)
from invokeai.app.models.exceptions import CanceledException
from .config import InvokeAIAppConfig
from ...backend.util import choose_precision, choose_torch_device

if TYPE_CHECKING:
    from ..invocations.baseinvocation import BaseInvocation, InvocationContext

@dataclass
class LastUsedModel:
    model_name: str=None
    model_type: SDModelType=None

last_used_model = LastUsedModel()

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
        model_type: SDModelType = SDModelType.Diffusers,
        submodel: Optional[SDModelType] = None,
        node: Optional[BaseInvocation] = None,
        context: Optional[InvocationContext] = None,
    ) -> SDModelInfo:
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
        model_type: SDModelType,
    ) -> bool:
        pass

    @abstractmethod
    def default_model(self) -> Optional[Tuple[str, SDModelType]]:
        """
        Returns the name and typeof the default model, or None
        if none is defined.
        """
        pass

    @abstractmethod
    def set_default_model(self, model_name: str, model_type: SDModelType):
        """Sets the default model to the indicated name."""
        pass

    @abstractmethod
    def model_info(self, model_name: str, model_type: SDModelType) -> dict:
        """
        Given a model name returns a dict-like (OmegaConf) object describing it.
        """
        pass

    @abstractmethod
    def model_names(self) -> List[Tuple[str, SDModelType]]:
        """
        Returns a list of all the model names known.
        """
        pass

    @abstractmethod
    def list_models(self, model_type: SDModelType=None) -> dict:
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
        model_type: SDModelType,
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
        model_type: SDModelType,
        delete_files: bool = False,
    ):
        """
        Delete the named model from configuration. If delete_files is true, 
        then the underlying weight file or diffusers directory will be deleted 
        as well. Call commit() to write to disk.
        """
        pass

    @abstractmethod
    def import_diffuser_model(
        repo_or_path: Union[str, Path],
        model_name: Optional[str] = None,
        description: Optional[str] = None,
        vae: Optional[dict] = None,
    ) -> bool:
        """
        Install the indicated diffuser model and returns True if successful.

        "repo_or_path" can be either a repo-id or a path-like object corresponding to the
        top of a downloaded diffusers directory.

        You can optionally provide a model name and/or description. If not provided,
        then these will be derived from the repo name. Call commit() to write to disk.
        """
        pass
    
    @abstractmethod
    def import_lora(
        self,
        path: Path,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Creates an entry for the indicated lora file. Call
        mgr.commit() to write out the configuration to models.yaml
        """
        pass

    @abstractmethod
    def import_embedding(
        self,
        path: Path,
        model_name: str=None,
        description: str=None,
    ):
        """
        Creates an entry for the indicated textual inversion embedding file. 
        Call commit() to write out the configuration to models.yaml
        """
        pass

    @abstractmethod
    def heuristic_import(
        self,
        path_url_or_repo: str,
        model_name: str = None,
        description: str = None,
        model_config_file: Path = None,
        commit_to_conf: Path = None,
        config_file_callback: Callable[[Path], Path] = None,
    ) -> str:
        """Accept a string which could be:
           - a HF diffusers repo_id
           - a URL pointing to a legacy .ckpt or .safetensors file
           - a local path pointing to a legacy .ckpt or .safetensors file
           - a local directory containing .ckpt and .safetensors files
           - a local directory containing a diffusers model

        After determining the nature of the model and downloading it
        (if necessary), the file is probed to determine the correct
        configuration file (if needed) and it is imported.

        The model_name and/or description can be provided. If not, they will
        be generated automatically.

        If commit_to_conf is provided, the newly loaded model will be written
        to the `models.yaml` file at the indicated path. Otherwise, the changes
        will only remain in memory.

        The routine will do its best to figure out the config file
        needed to convert legacy checkpoint file, but if it can't it
        will call the config_file_callback routine, if provided. The
        callback accepts a single argument, the Path to the checkpoint
        file, and returns a Path to the config file to use.

        The (potentially derived) name of the model is returned on
        success, or None on failure. When multiple models are added
        from a directory, only the last imported one is returned.

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
        if config.precision == "auto":
            precision = choose_precision(device)
        dtype = torch.float32 if precision=='float32' \
                 else torch.float16

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
        model_type: SDModelType = SDModelType.Diffusers,
        submodel: Optional[SDModelType] = None,
        node: Optional[BaseInvocation] = None,
        context: Optional[InvocationContext] = None,
    ) -> SDModelInfo:
        """
        Retrieve the indicated model. submodel can be used to get a
        part (such as the vae) of a diffusers mode.
        """
        
        # Temporary hack here: we remember the last model fetched
        # so that when executing a graph, the first node called gets
        # to set default model for subsequent nodes in the event that
        # they do not set the model explicitly. This should be
        # displaced by model loader mechanism.
        # This is to work around lack of model loader at current time,
        # which was causing inconsistent model usage throughout graph.
        global last_used_model
        
        if not model_name:
            self.logger.debug('No model name provided, defaulting to last loaded model')
            model_name = last_used_model.model_name
            model_type = model_type or last_used_model.model_type
        else:
            last_used_model.model_name = model_name
            last_used_model.model_type = model_type

        # if we are called from within a node, then we get to emit
        # load start and complete events
        if node and context:
            self._emit_load_event(
                node=node,
                context=context,
                model_name=model_name,
                model_type=model_type,
                submodel=submodel
            )

        model_info = self.mgr.get_model(
            model_name,
            model_type,
            submodel,
        )

        if node and context:
            self._emit_load_event(
                node=node,
                context=context,
                model_name=model_name,
                model_type=model_type,
                submodel=submodel,
                model_info=model_info
            )
            
        return model_info

    def model_exists(
        self,
        model_name: str,
        model_type: SDModelType,
    ) -> bool:
        """
        Given a model name, returns True if it is a valid
        identifier.
        """
        return self.mgr.model_exists(
            model_name,
            model_type,
        )

    def default_model(self) -> Optional[Tuple[str, SDModelType]]:
        """
        Returns the name of the default model, or None
        if none is defined.
        """
        return self.mgr.default_model()

    def set_default_model(self, model_name: str, model_type: SDModelType):
        """Sets the default model to the indicated name."""
        self.mgr.set_default_model(model_name)

    def model_info(self, model_name: str, model_type: SDModelType) -> dict:
        """
        Given a model name returns a dict-like (OmegaConf) object describing it.
        """
        return self.mgr.model_info(model_name)

    def model_names(self) -> List[Tuple[str, SDModelType]]:
        """
        Returns a list of all the model names known.
        """
        return self.mgr.model_names()

    def list_models(self, model_type: SDModelType=None) -> dict:
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
        return self.mgr.list_models()

    def add_model(
        self,
        model_name: str,
        model_type: SDModelType,
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
        return self.mgr.add_model(model_name, model_type, model_attributes, clobber)


    def del_model(
        self,
        model_name: str,
        model_type: SDModelType = SDModelType.Diffusers,
        delete_files: bool = False,
    ):
        """
        Delete the named model from configuration. If delete_files is true, 
        then the underlying weight file or diffusers directory will be deleted 
        as well. Call commit() to write to disk.
        """
        self.mgr.del_model(model_name, model_type, delete_files)

    def import_diffuser_model(
        self,
        repo_or_path: Union[str, Path],
        model_name: Optional[str] = None,
        description: Optional[str] = None,
        vae: Optional[dict] = None,
    ) -> bool:
        """
        Install the indicated diffuser model and returns True if successful.

        "repo_or_path" can be either a repo-id or a path-like object corresponding to the
        top of a downloaded diffusers directory.

        You can optionally provide a model name and/or description. If not provided,
        then these will be derived from the repo name. Call commit() to write to disk.
        """
        return self.mgr.import_diffuser_model(repo_or_path, model_name, description, vae)
    
    def import_lora(
        self,
        path: Path,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Creates an entry for the indicated lora file. Call
        mgr.commit() to write out the configuration to models.yaml
        """
        self.mgr.import_lora(path, model_name, description)

    def import_embedding(
        self,
        path: Path,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Creates an entry for the indicated textual inversion embedding file. 
        Call commit() to write out the configuration to models.yaml
        """
        self.mgr.import_embedding(path, model_name, description)

    def heuristic_import(
        self,
        path_url_or_repo: str,
        model_name: str = None,
        description: str = None,
        model_config_file: Optional[Path] = None,
        commit_to_conf: Optional[Path] = None,
        config_file_callback: Optional[Callable[[Path], Path]] = None,
    ) -> str:
        """Accept a string which could be:
           - a HF diffusers repo_id
           - a URL pointing to a legacy .ckpt or .safetensors file
           - a local path pointing to a legacy .ckpt or .safetensors file
           - a local directory containing .ckpt and .safetensors files
           - a local directory containing a diffusers model

        After determining the nature of the model and downloading it
        (if necessary), the file is probed to determine the correct
        configuration file (if needed) and it is imported.

        The model_name and/or description can be provided. If not, they will
        be generated automatically.

        If commit_to_conf is provided, the newly loaded model will be written
        to the `models.yaml` file at the indicated path. Otherwise, the changes
        will only remain in memory.

        The routine will do its best to figure out the config file
        needed to convert legacy checkpoint file, but if it can't it
        will call the config_file_callback routine, if provided. The
        callback accepts a single argument, the Path to the checkpoint
        file, and returns a Path to the config file to use.

        The (potentially derived) name of the model is returned on
        success, or None on failure. When multiple models are added
        from a directory, only the last imported one is returned.

        """
        return self.mgr.heuristic_import(
            path_url_or_repo,
            model_name,
            description,
            model_config_file,
            commit_to_conf,
            config_file_callback
        )


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
        model_type: SDModelType,
        submodel: SDModelType,
        model_info: Optional[SDModelInfo] = None,
    ):
        if context.services.queue.is_canceled(context.graph_execution_state_id):
            raise CanceledException()
        graph_execution_state = context.services.graph_execution_manager.get(context.graph_execution_state_id)
        source_node_id = graph_execution_state.prepared_source_mapping[node.id]
        if context:
            context.services.events.emit_model_load_started(
                graph_execution_state_id=context.graph_execution_state_id,
                node=node.dict(),
                source_node_id=source_node_id,
                model_name=model_name,
                model_type=model_type,
                submodel=submodel,
            )
        else:
            context.services.events.emit_model_load_completed(
                graph_execution_state_id=context.graph_execution_state_id,
                node=node.dict(),
                source_node_id=source_node_id,
                model_name=model_name,
                model_type=model_type,
                submodel=submodel,
                model_info=model_info
            )

    @property
    def logger(self):
        return self.mgr.logger
        
