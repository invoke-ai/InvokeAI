"""
Routines for downloading and installing models.
"""
import json
import safetensors
import safetensors.torch
import torch
import traceback
from dataclasses import dataclass
from diffusers import ModelMixin
from enum import Enum
from typing import Callable
from pathlib import Path

import invokeai.backend.util.logging as logger
from invokeai.app.services.config import InvokeAIAppConfig
from .models import BaseModelType, ModelType
from .model_probe import ModelProbe, ModelVariantInfo

class ModelInstall(object):
    '''
    This class is able to download and install several different kinds of 
    InvokeAI models. The helper function, if provided, is called on to distinguish
    between v2-base and v2-768 stable diffusion pipelines. This usually involves
    asking the user to select the proper type, as there is no way of distinguishing
    the two type of v2 file programmatically (as far as I know).
    '''
    def __init__(self,
                 config: InvokeAIAppConfig,
                 model_base_helper: Callable[[Path],BaseModelType]=None,
                 clobber:bool = False
                 ):
        '''
        :param config: InvokeAI configuration object
        :param model_base_helper: A function call that accepts the Path to a checkpoint model and returns a ModelType enum
        :param clobber: If true, models with colliding names will be overwritten
        '''
        self.config = config
        self.clogger = clobber
        self.helper = model_base_helper
        self.prober = ModelProbe()

    def install_checkpoint_file(self, checkpoint: Path)->dict:
        '''
        Install the checkpoint file at path and return a
        configuration entry that can be added to `models.yaml`.
        Model checkpoints and VAEs will be converted into 
        diffusers before installation. Note that the model manager
        does not hold entries for anything but diffusers pipelines,
        and the configuration file stanzas returned from such models
        can be safely ignored.
        '''
        model_info = self.prober.probe(checkpoint, self.helper)
        if not model_info:
            raise ValueError(f"Unable to determine type of checkpoint file {checkpoint}")
        
        # non-pipeline; no conversion needed, just copy into right place
        if model_info.model_type != ModelType.Pipeline:
            destination_path = self._dest_path(model_info) / checkpoint.name
            self._check_for_collision(destination_path)
            shutil.copyfile(checkpoint, destination_path)
            key = ModelManager.create_key(
                model_name = checkpoint.stem,
                base_model = model_info.base_type
                model_type = model_info.model_type
            )
            return {
                key: dict(
                    name = model_name,
                    description = f'{model_info.model_type} model {model_name}',
                    path = str(destination_path),
                    format = 'checkpoint',
                    base = str(base_model),
                    type = str(model_type),
                    variant = str(model_info.variant_type),
                )
            }
                                        
            
        destination_path = self._dest_path(model_info) / checkpoint.stem




    def _check_for_collision(self, path: Path):
        if not path.exists():
            return
        if self.clobber:
            shutil.rmtree(path)
        else:
            raise ValueError(f"Destination {path} already exists. Won't overwrite unless clobber=True.")

    def _staging_directory(self)->tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory(dir=self.config.root_path)

    
        
