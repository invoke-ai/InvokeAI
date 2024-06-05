from abc import ABC, abstractmethod
import einops
import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from torchvision.transforms.functional import resize as tv_resize
from dataclasses import dataclass
from invokeai.app.invocations.t2i_adapter import T2IAdapterField
from invokeai.app.invocations.ip_adapter import IPAdapterField
from .controlnet_image_processors import ControlField
from invokeai.invocation_api import (
    InvocationContext,
    ConditioningField,
    LatentsField,
    UNetField,
)

@dataclass
class DenoiseLatentsData:
    context: InvocationContext
    positive_conditioning: ConditioningField
    negative_conditioning: ConditioningField
    noise: LatentsField | None
    latents: LatentsField | None
    steps: int
    cfg_scale: float
    denoising_start: float
    denoising_end: float
    scheduler: SchedulerMixin
    unet: UNetField
    unet_model: UNet2DConditionModel
    control: ControlField | list[ControlField] | None
    ip_adapter: IPAdapterField | list[IPAdapterField] | None
    t2i_adapter: T2IAdapterField | list[T2IAdapterField] | None
    seed: int

    def copy(self):
        return DenoiseLatentsData(
            context=self.context,
            positive_conditioning=self.positive_conditioning,
            negative_conditioning=self.negative_conditioning,
            noise=self.noise,
            latents=self.latents,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            denoising_start=self.denoising_start,
            denoising_end=self.denoising_end,
            scheduler=self.scheduler,
            unet=self.unet,
            unet_model=self.unet_model,
            control=self.control,
            ip_adapter=self.ip_adapter,
            t2i_adapter=self.t2i_adapter,
            seed=self.seed
        )


class DenoiseExtensionSD12X(ABC):

    def __init__(self, denoise_latents_data: DenoiseLatentsData, priority: int, extension_kwargs: dict):
        """
        Do not modify: Use __post_init__ to handle extension-specific parameters
        During injection calls, extensions will be called in order of self.priority (ascending)
        self.denoise_latents_data exists in case you need to access the data from calling node
        """
        self.denoise_latents_data = denoise_latents_data
        self.priority = priority
        self.__post_init__(**extension_kwargs)

    def __post_init__(self):
        """
        Called after the object is created.
        Override this method to perform additional initialization steps.
        """
        pass

    def list_modifies(self) -> list[str]:
        """
        A list of all the modify methods that this extension provides.
        e.g. ['modify_latents_before_scaling', 'modify_latents_before_noise_prediction']
        The injection names must match the method names in this class.
        """
        return []
    
    def list_provides(self) -> list[str]:
        """
        A list of all the provide methods that this extension provides.
        e.g. ['provide_latents', 'provide_noise']
        The injection names must match the method names in this class.
        """
        return []
    
    def list_swaps(self) -> list[str]:
        """
        A list of all the swap methods that this extension provides.
        e.g. ['swap_latents', 'swap_noise']
        The injection names must match the method names in this class.
        """
        return []
        
    def modify_latents_before_scaling(self, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samplers apply scalar multiplication to the latents before predicting noise.
        This method allows you to modify the latents before this scaling is applied each step.
        Useful if the modifications need to align with image or color in the normal latent space.
        """
        return latents

    def modify_latents_before_noise_prediction(self, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Last chance to modify latents before noise is predicted.
        Additional channels for inpaint models are added here.
        """
        return latents

    def modify_result_before_callback(self, step_output, t) -> torch.Tensor:
        """
        step_output.prev_sample is the current latents that will be used in the next step.
        if step_output.pred_original_sample is provided/modified, it will be used in the image preview for the user.
        """
        return step_output

    def modify_latents_after_denoising(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Final result of the latents after all steps are complete.
        """
        return latents


class AddsMaskGuidance(DenoiseExtensionSD12X):

    def __post_init__(self, mask_name: str, masked_latents_name: str | None, gradient_mask: bool):
        """Align internal data and create noise if necessary"""
        context = self.denoise_latents_data.context
        if self.denoise_latents_data.latents is not None:
            self.orig_latents = context.tensors.load(self.denoise_latents_data.latents.latents_name)
        else:
            raise ValueError("Latents input is required for the denoise mask extension")
        if self.denoise_latents_data.noise is not None:
            self.noise = context.tensors.load(self.denoise_latents_data.noise.latents_name)
        else:
            self.noise = torch.randn(
                self.orig_latents.shape,
                dtype=torch.float32,
                device="cpu",
                generator=torch.Generator(device="cpu").manual_seed(self.denoise_latents_data.seed),
            ).to(device=self.orig_latents.device, dtype=self.orig_latents.dtype)

        self.mask: torch.Tensor = context.tensors.load(mask_name)
        self.masked_latents = None if masked_latents_name is None else context.tensors.load(masked_latents_name)
        self.scheduler: SchedulerMixin = self.denoise_latents_data.scheduler
        self.gradient_mask: bool = gradient_mask
        self.unet_type: str = self.denoise_latents_data.unet.unet.base
        self.inpaint_model = self.denoise_latents_data.unet_model.conv_in.in_channels == 9
        self.seed: int = self.denoise_latents_data.seed

        self.mask = tv_resize(self.mask, list(self.orig_latents.shape[-2:]))
        self.mask = self.mask.to(device=self.orig_latents.device, dtype=self.orig_latents.dtype)
    
    def list_injections(self) -> list[str]:
        return [
            "modify_latents_before_scaling",
            "modify_latents_before_noise_prediction",
            "modify_result_before_callback",
            "modify_latents_after_denoising"
            ]

    def mask_from_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """Create a mask based on the current timestep"""
        if self.inpaint_model:
            mask_bool = self.mask < 1
            floored_mask = torch.where(mask_bool, 0, 1)
            return floored_mask
        elif self.gradient_mask:
            threshhold = (t.item()) / self.scheduler.config.num_train_timesteps
            mask_bool = self.mask < 1 - threshhold
            timestep_mask = torch.where(mask_bool, 0, 1)
            return timestep_mask.to(device=self.mask.device)
        else:
            return self.mask.clone()

    def modify_latents_before_scaling(self, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Replace unmasked region with original latents. Called before the scheduler scales the latent values."""
        if self.inpaint_model:
            return latents # skip this stage

        #expand to match batch size if necessary
        batch_size = latents.size(0)
        mask = self.mask_from_timestep(t).to(device=latents.device, dtype=latents.dtype)
        mask = einops.repeat(mask, "b c h w -> (repeat b) c h w", repeat=batch_size)
        if t.dim() == 0:
            t = einops.repeat(t, "-> batch", batch=batch_size)

        # create noised version of the original latents
        noised_latents = self.scheduler.add_noise(self.orig_latents, self.noise, t)
        noised_latents = einops.repeat(noised_latents, "b c h w -> (repeat b) c h w", repeat=batch_size).to(device=latents.device, dtype=latents.dtype)
        mask = self.mask_from_timestep(t).to(device=latents.device, dtype=latents.dtype)
        masked_input = torch.lerp(latents, noised_latents, mask)
        return masked_input

    def shrink_mask(self, mask: torch.Tensor, n_operations: int) -> torch.Tensor:
        kernel = torch.ones(1, 1, 3, 3).to(device=mask.device, dtype=mask.dtype)
        for _ in range(n_operations):
            mask = torch.nn.functional.conv2d(mask, kernel, padding=1).clamp(0, 1)
        return mask

    def modify_latents_before_noise_prediction(self, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Expand latents with information needed by inpaint model"""
        if not self.inpaint_model:
            return latents # skip this stage

        mask = self.mask_from_timestep(t).to(device=latents.device, dtype=latents.dtype)
        if self.masked_latents is None:
            #latent values for a black region after VAE encode
            if self.unet_type == "sd-1":
                latent_zeros = [0.78857421875, -0.638671875, 0.576171875, 0.12213134765625]
            elif self.unet_type == "sd-2":
                latent_zeros = [0.7890625, -0.638671875, 0.576171875, 0.12213134765625]
                print("WARNING: SD-2 Inpaint Models are not yet supported")
            elif self.unet_type == "sdxl":
                latent_zeros = [-0.578125, 0.501953125, 0.59326171875, -0.393798828125]
            else:
                raise ValueError(f"Unet type {self.unet_type} not supported as an inpaint model. Where did you get this?")

            # replace masked region with specified values
            mask_values = torch.tensor(latent_zeros).view(1, 4, 1, 1).expand_as(latents).to(device=latents.device, dtype=latents.dtype)
            small_mask = self.shrink_mask(mask, 1) #make the synthetic mask fill in the masked_latents smaller than the mask channel
            self.masked_latents = torch.where(small_mask == 0, mask_values, self.orig_latents)

        masked_latents = self.scheduler.scale_model_input(self.masked_latents,t)
        masked_latents = einops.repeat(masked_latents, "b c h w -> (repeat b) c h w", repeat=latents.size(0))
        mask = einops.repeat(mask, "b c h w -> (repeat b) c h w", repeat=latents.size(0))
        model_input = torch.cat([latents, 1 - mask, masked_latents], dim=1).to(dtype=latents.dtype, device=latents.device)
        return model_input

    def modify_result_before_callback(self, step_output, t) -> torch.Tensor:
        """Fix preview images to show the original image in the unmasked region"""
        if hasattr(step_output, "denoised"): #LCM Sampler
            prediction = step_output.denoised
        elif hasattr(step_output, "pred_original_sample"): #Samplers with final predictions
            prediction = step_output.pred_original_sample
        else: #all other samplers (no prediction available)
            prediction = step_output.prev_sample

        mask = self.mask_from_timestep(t)
        mask = einops.repeat(self.mask, "b c h w -> (repeat b) c h w", repeat=prediction.size(0))
        step_output.pred_original_sample = torch.lerp(prediction, self.orig_latents.to(dtype=prediction.dtype), mask.to(dtype=prediction.dtype))

        return step_output

    def modify_latents_after_denoising(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply original unmasked to denoised latents"""
        if self.inpaint_model:
            if self.masked_latents is None:
                mask = self.shrink_mask(self.mask, 1)
            else:
                return latents
        else:
            mask = self.mask_from_timestep(torch.Tensor([0]))
        mask = einops.repeat(mask, "b c h w -> (repeat b) c h w", repeat=latents.size(0))
        latents = torch.lerp(latents, self.orig_latents.to(dtype=latents.dtype), mask.to(dtype=latents.dtype)).to(device=latents.device)
        return latents
