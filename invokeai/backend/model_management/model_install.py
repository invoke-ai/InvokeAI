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
from picklescan.scanner import scan_file_path

import invokeai.backend.util.logging as logger
from .models import BaseModelType, ModelType

class CheckpointProbe(object):
    PROBES = dict()   # see below for redefinition
    
    def __init__(self,
                 checkpoint_path: Path,
                 checkpoint: dict = None,
                 helper: Callable[[Path], BaseModelType]=None
                 ):
        checkpoint = checkpoint or self._scan_and_load_checkpoint(self.checkpoint_path)
        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint_path
        self.helper = helper
    
    def probe(self) -> ModelVariantInfo:
        '''
        Probes the checkpoint at path `checkpoint_path` and return
        a ModelType object indicating the model base, model type and 
        model variant for the checkpoint.
        '''
        checkpoint = self.checkpoint
        state_dict = checkpoint.get("state_dict") or checkpoint

        model_info = None

        try:
            model_type = self.get_checkpoint_type(state_dict)
            if not model_type:
                if self.checkpoint_path.name == "learned_embeds.bin":
                    model_type = ModelType.TextualInversion
                else:
                    return None # we give up
            probe = self.PROBES[model_type]()
            base_type = probe.get_base_type(checkpoint, self.checkpoint_path, self.helper)
            variant_type = probe.get_variant_type(model_type, checkpoint)
            
            model_info = ModelVariantInfo(
                model_type = model_type,
                base_type = base_type,
                variant_type = variant_type,
            )
        except (KeyError, ValueError) as e:
            logger.error(f'An error occurred while probing {self.checkpoint_path}: {str(e)}')
            logger.error(traceback.format_exc())

        return model_info

    class CheckpointProbeBase(object):
        def get_base_type(self,
                          checkpoint: dict,
                          checkpoint_path: Path = None,
                          helper: Callable[[Path],BaseModelType] = None
                          )->BaseModelType:
            pass
        
        def get_variant_type(self,
                             model_type: ModelType,
                             checkpoint: dict,
                             )-> VariantType:
            if model_type != ModelType.Pipeline:
                return None
            state_dict = checkpoint.get('state_dict') or checkpoint
            in_channels = state_dict[
                "model.diffusion_model.input_blocks.0.0.weight"
            ].shape[1]
            if in_channels == 9:
                return VariantType.Inpaint
            elif in_channels == 5:
                return VariantType.depth
            else:
                return None

        
    class CheckpointProbe(CheckpointProbeBase):
        def get_base_type(self,
                          checkpoint: dict,
                          checkpoint_path: Path = None,
                          helper: Callable[[Path],BaseModelType] = None
                          )->BaseModelType:
            state_dict = checkpoint.get('state_dict') or checkpoint
            key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
            if key_name in state_dict and state_dict[key_name].shape[-1] == 768:
                return BaseModelType.StableDiffusion1_5
            if key_name in state_dict and state_dict[key_name].shape[-1] == 1024:
                if 'global_step' in checkpoint:
                    if checkpoint['global_step'] == 220000:
                        return BaseModelType.StableDiffusion2Base
                    elif checkpoint["global_step"] == 110000:
                        return BaseModelType.StableDiffusion2
                if checkpoint_path and helper:
                    return helper(checkpoint_path)
                else:
                    return None

    class VaeProbe(CheckpointProbeBase):
        def get_base_type(self,
                          checkpoint: dict,
                          checkpoint_path: Path = None,
                          helper: Callable[[Path],BaseModelType] = None
                          )->BaseModelType:
            # I can't find any standalone 2.X VAEs to test with!
            return BaseModelType.StableDiffusion1_5

    class LoRAProbe(CheckpointProbeBase):
        def get_base_type(self,
                          checkpoint: dict,
                          checkpoint_path: Path = None,
                          helper: Callable[[Path],BaseModelType] = None
                          )->BaseModelType:
            key1 = "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight"
            key2 = "lora_te_text_model_encoder_layers_0_self_attn_k_proj.hada_w1_a"
            lora_token_vector_length = (
                checkpoint[key1].shape[1]
                if key1 in checkpoint
                else checkpoint[key2].shape[0]
                if key2 in checkpoint
                else 768
            )
            if lora_token_vector_length == 768:
                return BaseModelType.StableDiffusion1_5
            elif lora_token_vector_length == 1024:
                return BaseModelType.StableDiffusion2
            else:
                return None
        
    class TextualInversionProbe(CheckpointProbeBase):
        def get_base_type(self,
                          checkpoint: dict,
                          checkpoint_path: Path = None,
                          helper: Callable[[Path],BaseModelType] = None
                          )->BaseModelType:
            
            if 'string_to_token' in checkpoint:
                token_dim = list(checkpoint['string_to_param'].values())[0].shape[-1]
            elif 'emb_params' in checkpoint:
                token_dim = checkpoint['emb_params'].shape[-1]
            else:
                token_dim = list(checkpoint.values())[0].shape[0]
            if token_dim == 768:
                return BaseModelType.StableDiffusion1_5
            elif token_dim == 1024:
                return BaseModelType.StableDiffusion2Base
            else:
                return None

    class ControlNetProbe(CheckpointProbeBase):
        def get_base_type(self,
                          checkpoint: dict,
                          checkpoint_path: Path = None,
                          helper: Callable[[Path],BaseModelType] = None
                          )->BaseModelType:
            for key_name in ('control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight',
                             'input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight'
                             ):
                if key_name not in checkpoint:
                    continue
                if checkpoint[key_name].shape[-1] == 768:
                    return BaseModelType.StableDiffusion1_5
                elif checkpoint_path and helper:
                    return helper(checkpoint_path)
    PROBES = {
        ModelType.Pipeline: CheckpointProbe,
        ModelType.Vae: VaeProbe,
        ModelType.Lora: LoRAProbe,
        ModelType.TextualInversion: TextualInversionProbe,
        ModelType.ControlNet: ControlNetProbe,
    }

    @classmethod
    def get_checkpoint_type(cls, state_dict: dict) -> ModelType:
        if any([x.startswith("model.diffusion_model") for x in state_dict.keys()]):
            return ModelType.Pipeline
        if any([x.startswith("encoder.conv_in") for x in state_dict.keys()]):
            return ModelType.Vae
        if "string_to_token" in state_dict or "emb_params" in state_dict:
            return ModelType.TextualInversion
        if any([x.startswith("lora") for x in state_dict.keys()]):
            return ModelType.Lora
        if any([x.startswith("control_model") for x in state_dict.keys()]):
            return ModelType.ControlNet
        if any([x.startswith("input_blocks") for x in state_dict.keys()]):
            return ModelType.ControlNet
        return None # give up

