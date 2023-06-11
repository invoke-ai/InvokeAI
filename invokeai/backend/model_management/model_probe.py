import json
import traceback
import torch
import safetensors.torch

from dataclasses import dataclass
from diffusers import ModelMixin
from pathlib import Path
from typing import Callable, Literal, Union, Dict
from picklescan.scanner import scan_file_path

import invokeai.backend.util.logging as logger
from .models import BaseModelType, ModelType, VariantType

@dataclass
class ModelVariantInfo(object):
    model_type: ModelType
    base_type: BaseModelType
    variant_type: VariantType

class ProbeBase(object):
    '''forward declaration'''
    pass

class ModelProbe(object):
    
    PROBES = {
        'folder': { },
        'file': { },
    }

    CLASS2TYPE = {
        "StableDiffusionPipeline" : ModelType.Pipeline,
        "AutoencoderKL": ModelType.Vae,
        "ControlNetModel" : ModelType.ControlNet,
        
    }
    
    @classmethod
    def register_probe(cls,
                       format: Literal['folder','file'],
                       model_type: ModelType,
                       probe_class: ProbeBase):
        cls.PROBES[format][model_type] = probe_class

    @classmethod
    def probe(cls,
              model_path: Path,
              model: Union[Dict, ModelMixin] = None,
              base_helper: Callable[[Path],BaseModelType] = None)->ModelVariantInfo:
        '''
        Probe the model at model_path and return sufficient information about it
        to place it somewhere in the models directory hierarchy. If the model is
        already loaded into memory, you may provide it as model in order to avoid
        opening it a second time. The base_helper callable is a function that receives
        the path to the model and returns the BaseModelType. It is called to distinguish
        between V2-Base and V2-768 SD models.
        '''
        format = 'folder' if model_path.is_dir() else 'file'
        model_info = None
        try:
            model_type = cls.get_model_type_from_folder(model_path, model) \
                if format == 'folder' \
                   else cls.get_model_type_from_checkpoint(model_path, model)
            probe_class = cls.PROBES[format].get(model_type)
            if not probe_class:
                return None
            probe = probe_class(model_path, model, base_helper)
            base_type = probe.get_base_type()
            variant_type = probe.get_variant_type()
            model_info = ModelVariantInfo(
                model_type = model_type,
                base_type = base_type,
                variant_type = variant_type,
            )
        except (KeyError, ValueError) as e:
            logger.error(f'An error occurred while probing {model_path}: {str(e)}')
            logger.error(traceback.format_exc())

        return model_info

    @classmethod
    def get_model_type_from_checkpoint(cls, model_path: Path, checkpoint: dict)->ModelType:
        checkpoint = checkpoint or cls._scan_and_load_checkpoint(model_path)
        state_dict = checkpoint.get("state_dict") or checkpoint
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

    @classmethod
    def get_model_type_from_folder(cls, folder_path: Path, model: ModelMixin)->ModelType:
        '''
        Get the model type of a hugging-face style folder.
        '''
        if (folder_path / 'learned_embeds.bin').exists():
            return ModelType.TextualInversion

        if (folder_path / 'pytorch_lora_weights.bin').exists():
            return ModelType.Lora

        i  = folder_path / 'model_index.json'
        c = folder_path / 'config.json'
        config_path = i if i.exists() else c if c.exists() else None
        
        if config_path:
            conf = json.loads(config_path)
            class_name = conf['_class_name']
            if type := cls.CLASS2TYPE.get(class_name):
                return type

        # give up
        raise ValueError("Unable to determine model type of {model_path}")

    @classmethod
    def _scan_and_load_checkpoint(cls,model_path: Path)->dict:
        if model_path.suffix.endswith((".ckpt", ".pt", ".bin")):
            cls._scan_model(model_path, model_path)
            return torch.load(model_path)
        else:
            return safetensors.torch.load_file(model_path)

    @classmethod
    def _scan_model(cls, model_name, checkpoint):
            """
            Apply picklescanner to the indicated checkpoint and issue a warning
            and option to exit if an infected file is identified.
            """
            # scan model
            scan_result = scan_file_path(checkpoint)
            if scan_result.infected_files != 0:
                raise "The model {model_name} is potentially infected by malware. Aborting import."

###################################################3
# Checkpoint probing
###################################################3
class ProbeBase(object):
    def get_base_type(self)->BaseModelType:
        pass

    def get_variant_type(self)->VariantType:
        pass
    
class CheckpointProbeBase(ProbeBase):
    def __init__(self,
                 checkpoint_path: Path,
                 checkpoint: dict,
                 helper: Callable[[Path],BaseModelType] = None
                 )->BaseModelType:
        self.checkpoint = checkpoint or ModelProbe._scan_and_load_checkpoint(checkpoint_path)
        self.checkpoint_path = checkpoint_path
        self.helper = helper

    def get_base_type(self)->BaseModelType:
        pass

    def get_variant_type(self)-> VariantType:
        model_type = ModelProbe.get_model_type_from_checkpoint(self.checkpoint_path,self.checkpoint)
        if model_type != ModelType.Pipeline:
            return VariantType.Normal
        state_dict = self.checkpoint.get('state_dict') or self.checkpoint
        in_channels = state_dict[
            "model.diffusion_model.input_blocks.0.0.weight"
        ].shape[1]
        if in_channels == 9:
            return VariantType.Inpaint
        elif in_channels == 5:
            return VariantType.Depth
        else:
            return None

class PipelineCheckpointProbe(CheckpointProbeBase):
    def get_base_type(self)->BaseModelType:
        checkpoint = self.checkpoint
        helper = self.helper
        state_dict = self.checkpoint.get('state_dict') or checkpoint
        key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        if key_name in state_dict and state_dict[key_name].shape[-1] == 768:
            return BaseModelType.StableDiffusion1_5
        if key_name in state_dict and state_dict[key_name].shape[-1] == 1024:
            if 'global_step' in checkpoint:
                if checkpoint['global_step'] == 220000:
                    return BaseModelType.StableDiffusion2Base
                elif checkpoint["global_step"] == 110000:
                    return BaseModelType.StableDiffusion2
            if self.checkpoint_path and helper:
                return helper(self.checkpoint_path)
            else:
                return None

class VaeCheckpointProbe(CheckpointProbeBase):
    def get_base_type(self)->BaseModelType:
        # I can't find any standalone 2.X VAEs to test with!
        return BaseModelType.StableDiffusion1_5

class LoRACheckpointProbe(CheckpointProbeBase):
    def get_base_type(self)->BaseModelType:
        checkpoint = self.checkpoint
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

class TextualInversionCheckpointProbe(CheckpointProbeBase):
    def get_base_type(self)->BaseModelType:
        checkpoint = self.checkpoint
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

class ControlNetCheckpointProbe(CheckpointProbeBase):
    def get_base_type(self)->BaseModelType:
        checkpoint = self.checkpoint
        for key_name in ('control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight',
                         'input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight'
                         ):
            if key_name not in checkpoint:
                continue
            if checkpoint[key_name].shape[-1] == 768:
                return BaseModelType.StableDiffusion1_5
            elif self.checkpoint_path and self.helper:
                return self.helper(self.checkpoint_path)

########################################################
# classes for probing folders
#######################################################
class FolderProbeBase(ProbeBase):
    def __init__(self,
                 model: ModelMixin,
                 folder_path: Path,
                 helper: Callable=None  # not used
                 ):
        self.model = model
        self.folder_path = folder_path

    def get_variant_type(self)->VariantType:

        # only works for pipelines
        config_file = self.folder_path / 'unet' / 'config.json'
        if not config_file.exists():
            return VariantType.Normal

        conf = json.loads(config_file)
        channels = conf['in_channels']
        if channels == 9:
            return VariantType.Inpainting
        elif channels == 5:
            return VariantType.Depth
        elif channels == 4:
            return VariantType.Normal
        else:
            return VariantType.Normal

class PipelineFolderProbe(FolderProbeBase):
    def get_base_type(self)->BaseModelType:
        config_file = self.folder_path / 'scheduler' / 'scheduler_config.json'
        if not config_file.exists():
            return None
        conf = json.load(config_file)
        if conf['prediction_type'] == "v_prediction":
            return BaseModelType.StableDiffusion2
        elif conf['prediction_type'] == 'epsilon':
            return BaseModelType.StableDiffusion2Base
        else:
            return BaseModelType.StableDiffusion2
        
class VaeFolderProbe(FolderProbeBase):
    def get_base_type(self)->BaseModelType:
        return BaseModelType.StableDiffusion1_5

class TextualInversionFolderProbe(FolderProbeBase):
    def get_base_type(self)->BaseModelType:
        path = self.folder_path / 'learned_embeds.bin'
        if not path.exists():
            return None
        checkpoint = ModelProbe._scan_and_load_checkpoint(path)
        return TextualInversionCheckpointProbe(checkpoint).get_base_type

class ControlNetFolderProbe(FolderProbeBase):
    def get_base_type(self)->BaseModelType:
        config_file = self.folder_path / 'scheduler_config.json'
        if not config_file.exists():
            return None
        config = json.load(config_file)
        # no obvious way to distinguish between sd2-base and sd2-768
        return BaseModelType.StableDiffusion1_5 \
            if config['cross_attention_dim']==768 \
               else BaseModelType.StableDiffusion2

class LoRAFolderProbe(FolderProbeBase):
    # I've never seen one of these in the wild, so this is a noop
    pass

############## register probe classes ######
ModelProbe.register_probe('folder', ModelType.Pipeline,  PipelineFolderProbe)
ModelProbe.register_probe('folder', ModelType.Vae, VaeFolderProbe)
ModelProbe.register_probe('folder', ModelType.Lora, LoRAFolderProbe)
ModelProbe.register_probe('folder', ModelType.TextualInversion, TextualInversionFolderProbe)
ModelProbe.register_probe('folder', ModelType.ControlNet, ControlNetFolderProbe)
ModelProbe.register_probe('file', ModelType.Pipeline, PipelineCheckpointProbe)
ModelProbe.register_probe('file', ModelType.Vae, VaeCheckpointProbe)
ModelProbe.register_probe('file', ModelType.Lora, LoRACheckpointProbe)
ModelProbe.register_probe('file', ModelType.TextualInversion, TextualInversionCheckpointProbe)
ModelProbe.register_probe('file', ModelType.ControlNet, ControlNetCheckpointProbe)
