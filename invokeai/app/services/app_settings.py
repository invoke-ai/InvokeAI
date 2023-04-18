# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team

from pydantic import Field
from pathlib import Path
from typing import Literal, List
from .config_management import InvokeAISettings, get_configuration

class InvokeAIWebConfig(InvokeAISettings):
    '''
    Web-specific settings
    '''
    #fmt: off
    type               : Literal["web"] = "web"
    allow_origins      : List = Field(default=[], description='Allowed CORS origins')
    allow_credentials  : bool = Field(default=True, description='Allow CORS credentials')
    allow_methods      : List = Field(default=["*"], description='Methods allowed for CORS')
    allow_headers      : List = Field(default=["*"], description='Headers allowed for CORS')
    #fmt: on

class InvokeAIAppConfig(InvokeAISettings):
    '''
    Application-wide settings, not associated with any Invocation
    '''
    #fmt: off
    type: Literal["app_settings"] = "app_settings"
    precision           : Literal[tuple(['auto','float16','float32','autocast'])] = 'float16'
    conf                : Path = Field(default='configs/models.yaml', description='Path to models definition file')
    outdir              : Path = Field(default='outputs', description='Default folder for output images')
    root                : Path = Field(default='~/invokeai', description='InvokeAI runtime root directory')
    embedding_dir       : Path = Field(default='embeddings', description='Path to InvokeAI embeddings directory')
    autoconvert_dir     : Path = Field(default=None, description='Path to a directory of ckpt files to be converted into diffusers and imported on startup.')
    gfpgan_model_dir    : Path = Field(default="./models/gfpgan/GFPGANv1.4.pth", description='Path to GFPGAN models directory.')
    embeddings          : bool = Field(default=True, description='Load contents of embeddings directory')
    xformers_enabled    : bool = Field(default=True, description="Enable/disable memory-efficient attention")
    sequential_guidance : bool = Field(default=False, description="Whether to calculate guidance in serial instead of in parallel, lowering memory requirements")
    max_loaded_models   : int = Field(default=2, gt=0, description="Maximum number of models to keep in memory for rapid switching")
    nsfw_checker        : bool = Field(default=True, description="Enable/disable the NSFW checker")
    restore             : bool = Field(default=True, description="Enable/disable face restoration code")
    esrgan              : bool = Field(default=True, description="Enable/disable upscaling code")
    #fmt: on

    @property
    def root_dir(self)->Path:
        return self.root.expanduser()

    def _resolve(self,partial_path:Path)->Path:
        return (self.root_dir / partial_path).resolve()

    @property
    def output_path(self)->Path:
        return self._resolve(self.outdir)

    @property
    def model_conf_path(self)->Path:
        return self._resolve(self.conf)

    @property
    def embedding_path(self)->Path:
        return self._resolve(self.embedding_dir) if self.embedding_dir else None

    @property
    def autoconvert_path(self)->Path:
        return self._resolve(self.autoconvert_dir) if self.autoconvert_dir else None

    @property
    def gfpgan_model_path(self)->Path:
        return self._resolve(self.gfpgan_model_dir) if self.gfpgan_model_dir else None

