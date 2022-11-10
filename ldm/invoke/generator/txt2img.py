'''
ldm.invoke.generator.txt2img inherits from ldm.invoke.generator
'''
import PIL.Image
import torch

from .base import Generator


class Txt2Img(Generator):
    def __init__(self, model, precision):
        super().__init__(model, precision)

    @torch.no_grad()
    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,width,height,step_callback=None,threshold=0.0,perlin=0.0,
                       **kwargs):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it
        kwargs are 'width' and 'height'
        """
        self.perlin = perlin
        uc, c, extra_conditioning_info   = conditioning

        pipeline = self.model
        # TODO: customize a new pipeline for the given sampler (Scheduler)

        def make_image(x_T) -> PIL.Image.Image:
            # FIXME: restore free_gpu_mem functionality
            # if self.free_gpu_mem and self.model.model.device != self.model.device:
            #     self.model.model.to(self.model.device)

            # FIXME: how the embeddings are combined should be internal to the pipeline
            combined_text_embeddings = torch.cat([uc, c])

            pipeline_output = pipeline.image_from_embeddings(
                latents=x_T,
                num_inference_steps=steps,
                text_embeddings=combined_text_embeddings,
                guidance_scale=cfg_scale,
                callback=step_callback,
                # TODO: extra_conditioning_info = extra_conditioning_info,
                # TODO: eta = ddim_eta,
                # TODO: threshold = threshold,
            )

            # FIXME: restore free_gpu_mem functionality
            # if self.free_gpu_mem:
            #     self.model.model.to("cpu")

            return pipeline.numpy_to_pil(pipeline_output.images)[0]

        return make_image


    # returns a tensor filled with random numbers from a normal distribution
    def get_noise(self,width,height):
        device         = self.model.device
        if self.use_mps_noise or device.type == 'mps':
            x = torch.randn([1,
                                self.latent_channels,
                                height // self.downsampling_factor,
                                width  // self.downsampling_factor],
                               device='cpu').to(device)
        else:
            x = torch.randn([1,
                                self.latent_channels,
                                height // self.downsampling_factor,
                                width  // self.downsampling_factor],
                               device=device)
        if self.perlin > 0.0:
            x = (1-self.perlin)*x + self.perlin*self.get_perlin_noise(width  // self.downsampling_factor, height // self.downsampling_factor)
        return x

