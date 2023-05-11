# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Callable

from invokeai.backend.util import CUDA_DEVICE
from invokeai.backend.model_management.model_manager import (
    ModelManager,
    SDModelType,
    SDModelInfo,
    DictConfig,
    MAX_CACHE_SIZE,
    types,
    torch,
    logger,
)

class ModelManagerServiceBase(ABC):
    """Responsible for managing models on disk and in memory"""

    @abstractmethod
    def get_model(self,
                  model_name: str,
                  model_type: SDModelType=SDModelType.diffusers,
                  submodel: SDModelType=None
                  )->SDModelInfo:
        """Retrieve the indicated model with name and type. 
        submodel can be used to get a part (such as the vae) 
        of a diffusers pipeline."""
        pass

    @property
    @abstractmethod
    def logger(self):
        pass

    @abstractmethod
    def valid_model(self, model_name: str) -> bool:
        """
        Given a model name, returns True if it is a valid
        identifier.
        """
        pass

    @abstractmethod
    def default_model(self) -> Union[str,None]:
        """
        Returns the name of the default model, or None
        if none is defined.
        """
        pass

    @abstractmethod
    def set_default_model(self, model_name:str):
        """Sets the default model to the indicated name."""
        pass

    @abstractmethod
    def model_info(self, model_name: str)->dict:
        """
        Given a model name returns a dict-like (OmegaConf) object describing it.
        """
        pass

    @abstractmethod
    def model_names(self)->list[str]:
        """
        Returns a list of all the model names known.
        """
        pass

    @abstractmethod
    def list_models(self)->dict:
        """
        Return a dict of models in the format:
        { model_name1: {'status': ('active'|'cached'|'not loaded'),
                        'description': description,
                        'format': ('ckpt'|'diffusers'|'vae'|'text_encoder'|'tokenizer'|'lora'...),
                       },
          model_name2: { etc }
        """
        pass


    @abstractmethod
    def add_model(
            self, model_name: str, model_attributes: dict, clobber: bool = False)->None:
        """
        Update the named model with a dictionary of attributes. Will fail with an
        assertion error if the name already exists. Pass clobber=True to overwrite.
        On a successful update, the config will be changed in memory. Will fail 
        with an assertion error if provided attributes are incorrect or 
        the model name is missing. Call commit() to write changes to disk.
        """
        pass

    @abstractmethod
    def del_model(self,
                  model_name: str,
                  model_type: SDModelType=SDModelType.diffusers,
                  delete_files: bool = False):
        """
        Delete the named model from configuration. If delete_files is true, 
        then the underlying weight file or diffusers directory will be deleted 
        as well. Call commit() to write to disk.
        """
        pass

    @abstractmethod
    def import_diffuser_model(
        repo_or_path: Union[str, Path],
        model_name: str = None,
        description: str = None,
        vae: dict = None,
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
            model_name: str=None,
            description: str=None,
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
    def commit(self, conf_file: Path=None) -> None:
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
            config: Union[Path, DictConfig, str],
            device_type: torch.device = CUDA_DEVICE,
            precision: torch.dtype = torch.float16,
            max_cache_size=MAX_CACHE_SIZE,
            sequential_offload=False,
            logger: types.ModuleType = logger,
    ):
        """
        Initialize with the path to the models.yaml config file. 
        Optional parameters are the torch device type, precision, max_models,
        and sequential_offload boolean. Note that the default device
        type and precision are set up for a CUDA system running at half precision.
        """
        self.mgr = ModelManager(config=config,
                                device_type=device_type,
                                precision=precision,
                                max_cache_size=max_cache_size,
                                sequential_offload=sequential_offload,
                                logger=logger
                                )

    def get_model(self,
                  model_name: str,
                  model_type: SDModelType=SDModelType.diffusers,
                  submodel: SDModelType=None,
                  )->SDModelInfo:
        """
        Retrieve the indicated model. submodel can be used to get a
        part (such as the vae) of a diffusers mode.
        """
        return self.mgr.get_model(
            model_name,
            model_type,
            submodel,
        )

    def valid_model(self, *args, **kwargs) -> bool:
        """
        Given a model name, returns True if it is a valid
        identifier.
        """
        return self.mgr.valid_model(*args, **kwargs)

    def default_model(self) -> Union[str,None]:
        """
        Returns the name of the default model, or None
        if none is defined.
        """
        return self.mgr.default_model()

    def set_default_model(self, model_name:str):
        """Sets the default model to the indicated name."""
        self.mgr.set_default_model(model_name)

    def model_info(self, model_name: str)->dict:
        """
        Given a model name returns a dict-like (OmegaConf) object describing it.
        """
        return self.mgr.model_info(model_name)

    def model_names(self)->list[str]:
        """
        Returns a list of all the model names known.
        """
        return self.mgr.model_names()

    def list_models(self)->dict:
        """
        Return a dict of models in the format:
        { model_name1: {'status': ('active'|'cached'|'not loaded'),
                        'description': description,
                        'format': ('ckpt'|'diffusers'|'vae'|'text_encoder'|'tokenizer'|'lora'...),
                       },
          model_name2: { etc }
        """
        return self.mgr.list_models()

    def add_model(
            self, model_name: str, model_attributes: dict, clobber: bool = False)->None:
        """
        Update the named model with a dictionary of attributes. Will fail with an
        assertion error if the name already exists. Pass clobber=True to overwrite.
        On a successful update, the config will be changed in memory. Will fail 
        with an assertion error if provided attributes are incorrect or 
        the model name is missing. Call commit() to write changes to disk.
        """
        return self.mgr.add_model(model_name, model_attributes, dict, clobber)


    def del_model(self,
                  model_name: str,
                  model_type: SDModelType=SDModelType.diffusers,
                  delete_files: bool = False
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
            model_name: str = None,
            description: str = None,
            vae: dict = None,
    ) -> bool:
        """
        Install the indicated diffuser model and returns True if successful.

        "repo_or_path" can be either a repo-id or a path-like object corresponding to the
        top of a downloaded diffusers directory.

        You can optionally provide a model name and/or description. If not provided,
        then these will be derived from the repo name. Call commit() to write to disk.
        """
        return  self.mgr.import_diffuser_model(repo_or_path, model_name, description, vae)
    
    def import_lora(
            self,
            path: Path,
            model_name: str=None,
            description: str=None,
    ):
        """
        Creates an entry for the indicated lora file. Call
        mgr.commit() to write out the configuration to models.yaml
        """
        self.mgr.import_lora(path, model_name, description)

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
        self.mgr(path, model_name, description)

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
        return self.mgr.heuristic_import(
            path_url_or_repo,
            model_name,
            description,
            model_config_file,
            commit_to_conf,
            config_file_callback
        )


    def commit(self, conf_file: Path=None):
        """
        Write current configuration out to the indicated file.
        If no conf_file is provided, then replaces the
        original file/database used to initialize the object.
        """
        return self.mgr.commit(conf_file)

    @property
    def logger(self):
        return self.mgr.logger
        
