"""
invokeai.backend.generator.txt2img inherits from invokeai.backend.generator
"""
import PIL.Image
import torch

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.models.controlnet import ControlNetModel, ControlNetOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel

from ..stable_diffusion import (
    ConditioningData,
    PostprocessingSettings,
    StableDiffusionGeneratorPipeline,
)
from .base import Generator


class Txt2Img(Generator):
    def __init__(self, model, precision,
                 control_model: Optional[Union[ControlNetModel, List[ControlNetModel]]] = None,
                 **kwargs):
        self.control_model = control_model
        if isinstance(self.control_model, list):
            self.control_model = MultiControlNetModel(self.control_model)
        super().__init__(model, precision, **kwargs)

    @torch.no_grad()
    def get_make_image(
        self,
        prompt,
        sampler,
        steps,
        cfg_scale,
        ddim_eta,
        conditioning,
        width,
        height,
        step_callback=None,
        threshold=0.0,
        warmup=0.2,
        perlin=0.0,
        h_symmetry_time_pct=None,
        v_symmetry_time_pct=None,
        attention_maps_callback=None,
        **kwargs,
    ):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it
        kwargs are 'width' and 'height'
        """
        self.perlin = perlin
        control_image = kwargs.get("control_image", None)
        do_classifier_free_guidance = cfg_scale > 1.0

        # noinspection PyTypeChecker
        pipeline: StableDiffusionGeneratorPipeline = self.model
        pipeline.control_model = self.control_model
        pipeline.scheduler = sampler

        uc, c, extra_conditioning_info = conditioning
        conditioning_data = ConditioningData(
            uc,
            c,
            cfg_scale,
            extra_conditioning_info,
            postprocessing_settings=PostprocessingSettings(
                threshold=threshold,
                warmup=warmup,
                h_symmetry_time_pct=h_symmetry_time_pct,
                v_symmetry_time_pct=v_symmetry_time_pct,
            ),
        ).add_scheduler_args_if_applicable(pipeline.scheduler, eta=ddim_eta)

        # FIXME: still need to test with different widths, heights, devices, dtypes
        #        and add in batch_size, num_images_per_prompt?
        if control_image is not None:
            if isinstance(self.control_model, ControlNetModel):
                control_image = pipeline.prepare_control_image(
                    image=control_image,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    width=width,
                    height=height,
                    # batch_size=batch_size * num_images_per_prompt,
                    # num_images_per_prompt=num_images_per_prompt,
                    device=self.control_model.device,
                    dtype=self.control_model.dtype,
                )
            elif isinstance(self.control_model, MultiControlNetModel):
                images = []
                for image_ in control_image:
                    image_ = self.model.prepare_control_image(
                        image=image_,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        width=width,
                        height=height,
                        # batch_size=batch_size * num_images_per_prompt,
                        # num_images_per_prompt=num_images_per_prompt,
                        device=self.control_model.device,
                        dtype=self.control_model.dtype,
                    )
                    images.append(image_)
                control_image = images
            kwargs["control_image"] = control_image

        def make_image(x_T: torch.Tensor, _: int) -> PIL.Image.Image:
            pipeline_output = pipeline.image_from_embeddings(
                latents=torch.zeros_like(x_T, dtype=self.torch_dtype()),
                noise=x_T,
                num_inference_steps=steps,
                conditioning_data=conditioning_data,
                callback=step_callback,
                **kwargs,
            )

            if (
                pipeline_output.attention_map_saver is not None
                and attention_maps_callback is not None
            ):
                attention_maps_callback(pipeline_output.attention_map_saver)

            return pipeline.numpy_to_pil(pipeline_output.images)[0]

        return make_image
