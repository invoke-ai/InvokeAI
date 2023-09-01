import json
import torch
import safetensors.torch

from dataclasses import dataclass

from diffusers import ModelMixin, ConfigMixin
from pathlib import Path
from typing import Callable, Literal, Union, Dict, Optional
from picklescan.scanner import scan_file_path

from .models import (
    BaseModelType,
    ModelType,
    ModelVariantType,
    SchedulerPredictionType,
    SilenceWarnings,
    InvalidModelException,
)
from .util import lora_token_vector_length
from .models.base import read_checkpoint_meta


@dataclass
class ModelProbeInfo(object):
    model_type: ModelType
    base_type: BaseModelType
    variant_type: ModelVariantType
    prediction_type: SchedulerPredictionType
    upcast_attention: bool
    format: Literal["diffusers", "checkpoint", "lycoris", "olive", "onnx"]
    image_size: int


class ProbeBase(object):
    """forward declaration"""

    pass


class ModelProbe(object):
    PROBES = {
        "diffusers": {},
        "checkpoint": {},
        "onnx": {},
    }

    CLASS2TYPE = {
        "StableDiffusionPipeline": ModelType.Main,
        "StableDiffusionInpaintPipeline": ModelType.Main,
        "StableDiffusionXLPipeline": ModelType.Main,
        "StableDiffusionXLImg2ImgPipeline": ModelType.Main,
        "StableDiffusionXLInpaintPipeline": ModelType.Main,
        "AutoencoderKL": ModelType.Vae,
        "ControlNetModel": ModelType.ControlNet,
    }

    @classmethod
    def register_probe(
        cls, format: Literal["diffusers", "checkpoint", "onnx"], model_type: ModelType, probe_class: ProbeBase
    ):
        cls.PROBES[format][model_type] = probe_class

    @classmethod
    def heuristic_probe(
        cls,
        model: Union[Dict, ModelMixin, Path],
        prediction_type_helper: Callable[[Path], SchedulerPredictionType] = None,
    ) -> ModelProbeInfo:
        if isinstance(model, Path):
            return cls.probe(model_path=model, prediction_type_helper=prediction_type_helper)
        elif isinstance(model, (dict, ModelMixin, ConfigMixin)):
            return cls.probe(model_path=None, model=model, prediction_type_helper=prediction_type_helper)
        else:
            raise InvalidModelException("model parameter {model} is neither a Path, nor a model")

    @classmethod
    def probe(
        cls,
        model_path: Path,
        model: Optional[Union[Dict, ModelMixin]] = None,
        prediction_type_helper: Optional[Callable[[Path], SchedulerPredictionType]] = None,
    ) -> ModelProbeInfo:
        """
        Probe the model at model_path and return sufficient information about it
        to place it somewhere in the models directory hierarchy. If the model is
        already loaded into memory, you may provide it as model in order to avoid
        opening it a second time. The prediction_type_helper callable is a function that receives
        the path to the model and returns the BaseModelType. It is called to distinguish
        between V2-Base and V2-768 SD models.
        """
        if model_path:
            format_type = "diffusers" if model_path.is_dir() else "checkpoint"
        else:
            format_type = "diffusers" if isinstance(model, (ConfigMixin, ModelMixin)) else "checkpoint"
        model_info = None
        try:
            model_type = (
                cls.get_model_type_from_folder(model_path, model)
                if format_type == "diffusers"
                else cls.get_model_type_from_checkpoint(model_path, model)
            )
            format_type = "onnx" if model_type == ModelType.ONNX else format_type
            probe_class = cls.PROBES[format_type].get(model_type)
            if not probe_class:
                return None
            probe = probe_class(model_path, model, prediction_type_helper)
            base_type = probe.get_base_type()
            variant_type = probe.get_variant_type()
            prediction_type = probe.get_scheduler_prediction_type()
            format = probe.get_format()
            model_info = ModelProbeInfo(
                model_type=model_type,
                base_type=base_type,
                variant_type=variant_type,
                prediction_type=prediction_type,
                upcast_attention=(
                    base_type == BaseModelType.StableDiffusion2
                    and prediction_type == SchedulerPredictionType.VPrediction
                ),
                format=format,
                image_size=1024
                if (base_type in {BaseModelType.StableDiffusionXL, BaseModelType.StableDiffusionXLRefiner})
                else 768
                if (
                    base_type == BaseModelType.StableDiffusion2
                    and prediction_type == SchedulerPredictionType.VPrediction
                )
                else 512,
            )
        except Exception:
            raise

        return model_info

    @classmethod
    def get_model_type_from_checkpoint(cls, model_path: Path, checkpoint: dict) -> ModelType:
        if model_path.suffix not in (".bin", ".pt", ".ckpt", ".safetensors", ".pth"):
            return None

        if model_path.name == "learned_embeds.bin":
            return ModelType.TextualInversion

        ckpt = checkpoint if checkpoint else read_checkpoint_meta(model_path, scan=True)
        ckpt = ckpt.get("state_dict", ckpt)

        for key in ckpt.keys():
            if any(key.startswith(v) for v in {"cond_stage_model.", "first_stage_model.", "model.diffusion_model."}):
                return ModelType.Main
            elif any(key.startswith(v) for v in {"encoder.conv_in", "decoder.conv_in"}):
                return ModelType.Vae
            elif any(key.startswith(v) for v in {"lora_te_", "lora_unet_"}):
                return ModelType.Lora
            elif any(key.endswith(v) for v in {"to_k_lora.up.weight", "to_q_lora.down.weight"}):
                return ModelType.Lora
            elif any(key.startswith(v) for v in {"control_model", "input_blocks"}):
                return ModelType.ControlNet
            elif key in {"emb_params", "string_to_param"}:
                return ModelType.TextualInversion

        else:
            # diffusers-ti
            if len(ckpt) < 10 and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                return ModelType.TextualInversion

        raise InvalidModelException(f"Unable to determine model type for {model_path}")

    @classmethod
    def get_model_type_from_folder(cls, folder_path: Path, model: ModelMixin) -> ModelType:
        """
        Get the model type of a hugging-face style folder.
        """
        class_name = None
        if model:
            class_name = model.__class__.__name__
        else:
            if (folder_path / "unet/model.onnx").exists():
                return ModelType.ONNX
            if (folder_path / "learned_embeds.bin").exists():
                return ModelType.TextualInversion

            if (folder_path / "pytorch_lora_weights.bin").exists():
                return ModelType.Lora

            i = folder_path / "model_index.json"
            c = folder_path / "config.json"
            config_path = i if i.exists() else c if c.exists() else None

            if config_path:
                with open(config_path, "r") as file:
                    conf = json.load(file)
                class_name = conf["_class_name"]

        if class_name and (type := cls.CLASS2TYPE.get(class_name)):
            return type

        # give up
        raise InvalidModelException(f"Unable to determine model type for {folder_path}")

    @classmethod
    def _scan_and_load_checkpoint(cls, model_path: Path) -> dict:
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


# ##################################################3
# Checkpoint probing
# ##################################################3
class ProbeBase(object):
    def get_base_type(self) -> BaseModelType:
        pass

    def get_variant_type(self) -> ModelVariantType:
        pass

    def get_scheduler_prediction_type(self) -> SchedulerPredictionType:
        pass

    def get_format(self) -> str:
        pass


class CheckpointProbeBase(ProbeBase):
    def __init__(
        self, checkpoint_path: Path, checkpoint: dict, helper: Callable[[Path], SchedulerPredictionType] = None
    ) -> BaseModelType:
        self.checkpoint = checkpoint or ModelProbe._scan_and_load_checkpoint(checkpoint_path)
        self.checkpoint_path = checkpoint_path
        self.helper = helper

    def get_base_type(self) -> BaseModelType:
        pass

    def get_format(self) -> str:
        return "checkpoint"

    def get_variant_type(self) -> ModelVariantType:
        model_type = ModelProbe.get_model_type_from_checkpoint(self.checkpoint_path, self.checkpoint)
        if model_type != ModelType.Main:
            return ModelVariantType.Normal
        state_dict = self.checkpoint.get("state_dict") or self.checkpoint
        in_channels = state_dict["model.diffusion_model.input_blocks.0.0.weight"].shape[1]
        if in_channels == 9:
            return ModelVariantType.Inpaint
        elif in_channels == 5:
            return ModelVariantType.Depth
        elif in_channels == 4:
            return ModelVariantType.Normal
        else:
            raise InvalidModelException(
                f"Cannot determine variant type (in_channels={in_channels}) at {self.checkpoint_path}"
            )


class PipelineCheckpointProbe(CheckpointProbeBase):
    def get_base_type(self) -> BaseModelType:
        checkpoint = self.checkpoint
        state_dict = self.checkpoint.get("state_dict") or checkpoint
        key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        if key_name in state_dict and state_dict[key_name].shape[-1] == 768:
            return BaseModelType.StableDiffusion1
        if key_name in state_dict and state_dict[key_name].shape[-1] == 1024:
            return BaseModelType.StableDiffusion2
        key_name = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"
        if key_name in state_dict and state_dict[key_name].shape[-1] == 2048:
            return BaseModelType.StableDiffusionXL
        elif key_name in state_dict and state_dict[key_name].shape[-1] == 1280:
            return BaseModelType.StableDiffusionXLRefiner
        else:
            raise InvalidModelException("Cannot determine base type")

    def get_scheduler_prediction_type(self) -> SchedulerPredictionType:
        type = self.get_base_type()
        if type == BaseModelType.StableDiffusion1:
            return SchedulerPredictionType.Epsilon
        checkpoint = self.checkpoint
        state_dict = self.checkpoint.get("state_dict") or checkpoint
        key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        if key_name in state_dict and state_dict[key_name].shape[-1] == 1024:
            if "global_step" in checkpoint:
                if checkpoint["global_step"] == 220000:
                    return SchedulerPredictionType.Epsilon
                elif checkpoint["global_step"] == 110000:
                    return SchedulerPredictionType.VPrediction
            if (
                self.checkpoint_path and self.helper and not self.checkpoint_path.with_suffix(".yaml").exists()
            ):  # if a .yaml config file exists, then this step not needed
                return self.helper(self.checkpoint_path)
            else:
                return None


class VaeCheckpointProbe(CheckpointProbeBase):
    def get_base_type(self) -> BaseModelType:
        # I can't find any standalone 2.X VAEs to test with!
        return BaseModelType.StableDiffusion1


class LoRACheckpointProbe(CheckpointProbeBase):
    def get_format(self) -> str:
        return "lycoris"

    def get_base_type(self) -> BaseModelType:
        checkpoint = self.checkpoint
        token_vector_length = lora_token_vector_length(checkpoint)

        if token_vector_length == 768:
            return BaseModelType.StableDiffusion1
        elif token_vector_length == 1024:
            return BaseModelType.StableDiffusion2
        elif token_vector_length == 2048:
            return BaseModelType.StableDiffusionXL
        else:
            raise InvalidModelException(f"Unknown LoRA type: {self.checkpoint_path}")


class TextualInversionCheckpointProbe(CheckpointProbeBase):
    def get_format(self) -> str:
        return None

    def get_base_type(self) -> BaseModelType:
        checkpoint = self.checkpoint
        if "string_to_token" in checkpoint:
            token_dim = list(checkpoint["string_to_param"].values())[0].shape[-1]
        elif "emb_params" in checkpoint:
            token_dim = checkpoint["emb_params"].shape[-1]
        else:
            token_dim = list(checkpoint.values())[0].shape[0]
        if token_dim == 768:
            return BaseModelType.StableDiffusion1
        elif token_dim == 1024:
            return BaseModelType.StableDiffusion2
        else:
            return None


class ControlNetCheckpointProbe(CheckpointProbeBase):
    def get_base_type(self) -> BaseModelType:
        checkpoint = self.checkpoint
        for key_name in (
            "control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",
            "input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",
        ):
            if key_name not in checkpoint:
                continue
            if checkpoint[key_name].shape[-1] == 768:
                return BaseModelType.StableDiffusion1
            elif checkpoint[key_name].shape[-1] == 1024:
                return BaseModelType.StableDiffusion2
            elif self.checkpoint_path and self.helper:
                return self.helper(self.checkpoint_path)
        raise InvalidModelException("Unable to determine base type for {self.checkpoint_path}")


########################################################
# classes for probing folders
#######################################################
class FolderProbeBase(ProbeBase):
    def __init__(self, folder_path: Path, model: ModelMixin = None, helper: Callable = None):  # not used
        self.model = model
        self.folder_path = folder_path

    def get_variant_type(self) -> ModelVariantType:
        return ModelVariantType.Normal

    def get_format(self) -> str:
        return "diffusers"


class PipelineFolderProbe(FolderProbeBase):
    def get_base_type(self) -> BaseModelType:
        if self.model:
            unet_conf = self.model.unet.config
        else:
            with open(self.folder_path / "unet" / "config.json", "r") as file:
                unet_conf = json.load(file)
        if unet_conf["cross_attention_dim"] == 768:
            return BaseModelType.StableDiffusion1
        elif unet_conf["cross_attention_dim"] == 1024:
            return BaseModelType.StableDiffusion2
        elif unet_conf["cross_attention_dim"] == 1280:
            return BaseModelType.StableDiffusionXLRefiner
        elif unet_conf["cross_attention_dim"] == 2048:
            return BaseModelType.StableDiffusionXL
        else:
            raise InvalidModelException(f"Unknown base model for {self.folder_path}")

    def get_scheduler_prediction_type(self) -> SchedulerPredictionType:
        if self.model:
            scheduler_conf = self.model.scheduler.config
        else:
            with open(self.folder_path / "scheduler" / "scheduler_config.json", "r") as file:
                scheduler_conf = json.load(file)
        if scheduler_conf["prediction_type"] == "v_prediction":
            return SchedulerPredictionType.VPrediction
        elif scheduler_conf["prediction_type"] == "epsilon":
            return SchedulerPredictionType.Epsilon
        else:
            return None

    def get_variant_type(self) -> ModelVariantType:
        # This only works for pipelines! Any kind of
        # exception results in our returning the
        # "normal" variant type
        try:
            if self.model:
                conf = self.model.unet.config
            else:
                config_file = self.folder_path / "unet" / "config.json"
                with open(config_file, "r") as file:
                    conf = json.load(file)

            in_channels = conf["in_channels"]
            if in_channels == 9:
                return ModelVariantType.Inpaint
            elif in_channels == 5:
                return ModelVariantType.Depth
            elif in_channels == 4:
                return ModelVariantType.Normal
        except Exception:
            pass
        return ModelVariantType.Normal


class VaeFolderProbe(FolderProbeBase):
    def get_base_type(self) -> BaseModelType:
        config_file = self.folder_path / "config.json"
        if not config_file.exists():
            raise InvalidModelException(f"Cannot determine base type for {self.folder_path}")
        with open(config_file, "r") as file:
            config = json.load(file)
        return (
            BaseModelType.StableDiffusionXL
            if config.get("scaling_factor", 0) == 0.13025 and config.get("sample_size") in [512, 1024]
            else BaseModelType.StableDiffusion1
        )


class TextualInversionFolderProbe(FolderProbeBase):
    def get_format(self) -> str:
        return None

    def get_base_type(self) -> BaseModelType:
        path = self.folder_path / "learned_embeds.bin"
        if not path.exists():
            return None
        checkpoint = ModelProbe._scan_and_load_checkpoint(path)
        return TextualInversionCheckpointProbe(None, checkpoint=checkpoint).get_base_type()


class ONNXFolderProbe(FolderProbeBase):
    def get_format(self) -> str:
        return "onnx"

    def get_base_type(self) -> BaseModelType:
        return BaseModelType.StableDiffusion1

    def get_variant_type(self) -> ModelVariantType:
        return ModelVariantType.Normal


class ControlNetFolderProbe(FolderProbeBase):
    def get_base_type(self) -> BaseModelType:
        config_file = self.folder_path / "config.json"
        if not config_file.exists():
            raise InvalidModelException(f"Cannot determine base type for {self.folder_path}")
        with open(config_file, "r") as file:
            config = json.load(file)
        # no obvious way to distinguish between sd2-base and sd2-768
        dimension = config["cross_attention_dim"]
        base_model = (
            BaseModelType.StableDiffusion1
            if dimension == 768
            else BaseModelType.StableDiffusion2
            if dimension == 1024
            else BaseModelType.StableDiffusionXL
            if dimension == 2048
            else None
        )
        if not base_model:
            raise InvalidModelException(f"Unable to determine model base for {self.folder_path}")
        return base_model


class LoRAFolderProbe(FolderProbeBase):
    def get_base_type(self) -> BaseModelType:
        model_file = None
        for suffix in ["safetensors", "bin"]:
            base_file = self.folder_path / f"pytorch_lora_weights.{suffix}"
            if base_file.exists():
                model_file = base_file
                break
        if not model_file:
            raise InvalidModelException("Unknown LoRA format encountered")
        return LoRACheckpointProbe(model_file, None).get_base_type()


############## register probe classes ######
ModelProbe.register_probe("diffusers", ModelType.Main, PipelineFolderProbe)
ModelProbe.register_probe("diffusers", ModelType.Vae, VaeFolderProbe)
ModelProbe.register_probe("diffusers", ModelType.Lora, LoRAFolderProbe)
ModelProbe.register_probe("diffusers", ModelType.TextualInversion, TextualInversionFolderProbe)
ModelProbe.register_probe("diffusers", ModelType.ControlNet, ControlNetFolderProbe)
ModelProbe.register_probe("checkpoint", ModelType.Main, PipelineCheckpointProbe)
ModelProbe.register_probe("checkpoint", ModelType.Vae, VaeCheckpointProbe)
ModelProbe.register_probe("checkpoint", ModelType.Lora, LoRACheckpointProbe)
ModelProbe.register_probe("checkpoint", ModelType.TextualInversion, TextualInversionCheckpointProbe)
ModelProbe.register_probe("checkpoint", ModelType.ControlNet, ControlNetCheckpointProbe)
ModelProbe.register_probe("onnx", ModelType.ONNX, ONNXFolderProbe)
