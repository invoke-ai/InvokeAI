import json
import traceback
import torch
import safetensors.torch

from dataclasses import dataclass
from enum import Enum

from diffusers import ModelMixin, ConfigMixin, StableDiffusionPipeline, AutoencoderKL, ControlNetModel
from pathlib import Path
from typing import Callable, Literal, Union, Dict
from picklescan.scanner import scan_file_path

import invokeai.backend.util.logging as logger
from .models import BaseModelType, ModelType, ModelVariantType, SchedulerPredictionType
from .model_cache import SilenceWarnings

@dataclass
class ModelVariantInfo(object):
    model_type: ModelType
    base_type: BaseModelType
    variant_type: ModelVariantType
    prediction_type: SchedulerPredictionType
    format: Literal['folder','checkpoint']
    image_size: int

class ProbeBase(object):
    '''forward declaration'''
    pass

class ModelProbe(object):
    
    PROBES = {
        'folder': { },
        'checkpoint': { },
    }

    CLASS2TYPE = {
        'StableDiffusionPipeline' : ModelType.Pipeline,
        'AutoencoderKL' : ModelType.Vae,
        'ControlNetModel' : ModelType.ControlNet,
    }
    
    @classmethod
    def register_probe(cls,
                       format: Literal['folder','file'],
                       model_type: ModelType,
                       probe_class: ProbeBase):
        cls.PROBES[format][model_type] = probe_class

    @classmethod
    def heuristic_probe(cls,
                        model: Union[Dict, ModelMixin, Path],
                        prediction_type_helper: Callable[[Path],BaseModelType]=None,
                        )->ModelVariantInfo:
        if isinstance(model,Path):
            return cls.probe(model_path=model,prediction_type_helper=prediction_type_helper)
        elif isinstance(model,(dict,ModelMixin,ConfigMixin)):
            return cls.probe(model_path=None, model=model, prediction_type_helper=prediction_type_helper)
        else:
            raise Exception("model parameter {model} is neither a Path, nor a model")

    @classmethod
    def probe(cls,
              model_path: Path,
              model: Union[Dict, ModelMixin] = None,
              prediction_type_helper: Callable[[Path],BaseModelType] = None)->ModelVariantInfo:
        '''
        Probe the model at model_path and return sufficient information about it
        to place it somewhere in the models directory hierarchy. If the model is
        already loaded into memory, you may provide it as model in order to avoid
        opening it a second time. The prediction_type_helper callable is a function that receives
        the path to the model and returns the BaseModelType. It is called to distinguish
        between V2-Base and V2-768 SD models.
        '''
        if model_path:
            format = 'folder' if model_path.is_dir() else 'checkpoint'
        else:
            format = 'folder' if isinstance(model,(ConfigMixin,ModelMixin)) else 'checkpoint'

        model_info = None
        try:
            model_type = cls.get_model_type_from_folder(model_path, model) \
                if format == 'folder' \
                   else cls.get_model_type_from_checkpoint(model_path, model)
            probe_class = cls.PROBES[format].get(model_type)
            if not probe_class:
                return None
            probe = probe_class(model_path, model, prediction_type_helper)
            base_type = probe.get_base_type()
            variant_type = probe.get_variant_type()
            prediction_type = probe.get_scheduler_prediction_type()
            model_info = ModelVariantInfo(
                model_type = model_type,
                base_type = base_type,
                variant_type = variant_type,
                prediction_type = prediction_type,
                format = format,
                image_size = 768 if (base_type==BaseModelType.StableDiffusion2 \
                                     and prediction_type==SchedulerPredictionType.VPrediction \
                                     ) else 512
            )
        except Exception as e:
            return None

        return model_info

    @classmethod
    def get_model_type_from_checkpoint(cls, model_path: Path, checkpoint: dict)->ModelType:
        if model_path.suffix not in ('.bin','.pt','.ckpt','.safetensors'):
            return None
        if model_path.name=='learned_embeds.bin':
            return ModelType.TextualInversion
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
        class_name = None
        if model:
            class_name = model.__class__.__name__
        else:
            if (folder_path / 'learned_embeds.bin').exists():
                return ModelType.TextualInversion

            if (folder_path / 'pytorch_lora_weights.bin').exists():
                return ModelType.Lora

            i  = folder_path / 'model_index.json'
            c = folder_path / 'config.json'
            config_path = i if i.exists() else c if c.exists() else None

            if config_path:
                with open(config_path,'r') as file:
                    conf = json.load(file)
                class_name = conf['_class_name']

        if class_name and (type := cls.CLASS2TYPE.get(class_name)):
            return type

        # give up
        raise ValueError("Unable to determine model type")

    @classmethod
    def _scan_and_load_checkpoint(cls,model_path: Path)->dict:
        with SilenceWarnings():
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

    def get_variant_type(self)->ModelVariantType:
        pass
    
    def get_scheduler_prediction_type(self)->SchedulerPredictionType:
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

    def get_variant_type(self)-> ModelVariantType:
        model_type = ModelProbe.get_model_type_from_checkpoint(self.checkpoint_path,self.checkpoint)
        if model_type != ModelType.Pipeline:
            return ModelVariantType.Normal
        state_dict = self.checkpoint.get('state_dict') or self.checkpoint
        in_channels = state_dict[
            "model.diffusion_model.input_blocks.0.0.weight"
        ].shape[1]
        if in_channels == 9:
            return ModelVariantType.Inpaint
        elif in_channels == 5:
            return ModelVariantType.Depth
        elif in_channels == 4:
            return ModelVariantType.Normal
        else:
            raise Exception("Cannot determine variant type")

class PipelineCheckpointProbe(CheckpointProbeBase):
    def get_base_type(self)->BaseModelType:
        checkpoint = self.checkpoint
        state_dict = self.checkpoint.get('state_dict') or checkpoint
        key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        if key_name in state_dict and state_dict[key_name].shape[-1] == 768:
            return BaseModelType.StableDiffusion1
        if key_name in state_dict and state_dict[key_name].shape[-1] == 1024:
            return BaseModelType.StableDiffusion2
        raise Exception("Cannot determine base type")

    def get_scheduler_prediction_type(self)->SchedulerPredictionType:
        type = self.get_base_type()
        if type == BaseModelType.StableDiffusion1:
            return SchedulerPredictionType.Epsilon
        checkpoint = self.checkpoint
        state_dict = self.checkpoint.get('state_dict') or checkpoint
        key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        if key_name in state_dict and state_dict[key_name].shape[-1] == 1024:
            if 'global_step' in checkpoint:
                if checkpoint['global_step'] == 220000:
                    return SchedulerPredictionType.Epsilon
                elif checkpoint["global_step"] == 110000:
                    return SchedulerPredictionType.VPrediction
            if self.checkpoint_path and self.helper:
                return self.helper(self.checkpoint_path)
            else:
                return None

class VaeCheckpointProbe(CheckpointProbeBase):
    def get_base_type(self)->BaseModelType:
        # I can't find any standalone 2.X VAEs to test with!
        return BaseModelType.StableDiffusion1

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
            return BaseModelType.StableDiffusion1
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
            return BaseModelType.StableDiffusion1
        elif token_dim == 1024:
            return BaseModelType.StableDiffusion2
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
                return BaseModelType.StableDiffusion1
            elif checkpoint[key_name].shape[-1] == 1024:
                return BaseModelType.StableDiffusion2
            elif self.checkpoint_path and self.helper:
                return self.helper(self.checkpoint_path)
        raise Exception("Unable to determine base type for {self.checkpoint_path}")

########################################################
# classes for probing folders
#######################################################
class FolderProbeBase(ProbeBase):
    def __init__(self,
                 folder_path: Path,
                 model: ModelMixin = None,
                 helper: Callable=None  # not used
                 ):
        self.model = model
        self.folder_path = folder_path

    def get_variant_type(self)->ModelVariantType:
        return ModelVariantType.Normal

class PipelineFolderProbe(FolderProbeBase):
    def get_base_type(self)->BaseModelType:
        if self.model:
            unet_conf = self.model.unet.config
            scheduler_conf = self.model.scheduler.config
        else:
            with open(self.folder_path / 'unet' / 'config.json','r') as file:
                unet_conf = json.load(file)
            with open(self.folder_path / 'scheduler' / 'scheduler_config.json','r') as file:
                scheduler_conf = json.load(file)
            
        if unet_conf['cross_attention_dim'] == 768:
            return BaseModelType.StableDiffusion1  
        elif unet_conf['cross_attention_dim'] == 1024:
            return BaseModelType.StableDiffusion2
        else:
            raise ValueError(f'Unknown base model for {self.folder_path}')

    def get_scheduler_prediction_type(self)->SchedulerPredictionType:
        if self.model:
            scheduler_conf = self.model.scheduler.config
        else:
            with open(self.folder_path / 'scheduler' / 'scheduler_config.json','r') as file:
                scheduler_conf = json.load(file)
        if scheduler_conf['prediction_type'] == "v_prediction":
            return SchedulerPredictionType.VPrediction
        elif scheduler_conf['prediction_type'] == 'epsilon':
            return SchedulerPredictionType.Epsilon
        else:
            return None
        
    def get_variant_type(self)->ModelVariantType:
        # This only works for pipelines! Any kind of
        # exception results in our returning the
        # "normal" variant type
        try:
            if self.model:
                conf = self.model.unet.config
            else:
                config_file = self.folder_path / 'unet' / 'config.json'
                with open(config_file,'r') as file:
                    conf = json.load(file)
                
            in_channels = conf['in_channels']
            if in_channels == 9:
                return ModelVariantType.Inpainting
            elif in_channels == 5:
                return ModelVariantType.Depth
            elif in_channels == 4:
                return ModelVariantType.Normal
        except:
            pass
        return ModelVariantType.Normal

class VaeFolderProbe(FolderProbeBase):
    def get_base_type(self)->BaseModelType:
        return BaseModelType.StableDiffusion1

class TextualInversionFolderProbe(FolderProbeBase):
    def get_base_type(self)->BaseModelType:
        path = self.folder_path / 'learned_embeds.bin'
        if not path.exists():
            return None
        checkpoint = ModelProbe._scan_and_load_checkpoint(path)
        return TextualInversionCheckpointProbe(None,checkpoint=checkpoint).get_base_type()

class ControlNetFolderProbe(FolderProbeBase):
    def get_base_type(self)->BaseModelType:
        config_file = self.folder_path / 'config.json'
        if not config_file.exists():
            raise Exception(f"Cannot determine base type for {self.folder_path}")
        with open(config_file,'r') as file:
            config = json.load(file)
        # no obvious way to distinguish between sd2-base and sd2-768
        return BaseModelType.StableDiffusion1 \
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
ModelProbe.register_probe('checkpoint', ModelType.Pipeline, PipelineCheckpointProbe)
ModelProbe.register_probe('checkpoint', ModelType.Vae, VaeCheckpointProbe)
ModelProbe.register_probe('checkpoint', ModelType.Lora, LoRACheckpointProbe)
ModelProbe.register_probe('checkpoint', ModelType.TextualInversion, TextualInversionCheckpointProbe)
ModelProbe.register_probe('checkpoint', ModelType.ControlNet, ControlNetCheckpointProbe)
