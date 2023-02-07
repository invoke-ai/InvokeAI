'''
ldm.invoke.generator.txt2img inherits from ldm.invoke.generator
'''
import PIL.Image
import torch

from .base import Generator
from .diffusers_pipeline import StableDiffusionGeneratorPipeline, ConditioningData
from ...models.diffusion.shared_invokeai_diffusion import ThresholdSettings


class Txt2Img(Generator):
    def __init__(self, model, precision):
        super().__init__(model, precision)

    @torch.no_grad()
    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,width,height,step_callback=None,threshold=0.0,perlin=0.0,
                       attention_maps_callback=None,
                       **kwargs):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it
        kwargs are 'width' and 'height'
        """
        self.perlin = perlin

        # noinspection PyTypeChecker
        pipeline: StableDiffusionGeneratorPipeline = self.model
        pipeline.scheduler = sampler

        uc, c, extra_conditioning_info   = conditioning
        conditioning_data = (
            ConditioningData(
                uc, c, cfg_scale, extra_conditioning_info,
                threshold = ThresholdSettings(threshold, warmup=0.2) if threshold else None)
            .add_scheduler_args_if_applicable(pipeline.scheduler, eta=ddim_eta))

        def make_image(x_T) -> PIL.Image.Image:
            pipeline_output = pipeline.image_from_embeddings(
                latents=torch.zeros_like(x_T,dtype=self.torch_dtype()),
                noise=x_T,
                num_inference_steps=steps,
                conditioning_data=conditioning_data,
                callback=step_callback,
            )
            if pipeline_output.attention_map_saver is not None and attention_maps_callback is not None:
                attention_maps_callback(pipeline_output.attention_map_saver)
            return pipeline.numpy_to_pil(pipeline_output.images)[0]

        return make_image



