"""
invokeai.backend.generator.img2img descends from .generator
"""

from .base import Generator


class Img2Img(Generator):
    def get_make_image(
        self,
        sampler,
        steps,
        cfg_scale,
        ddim_eta,
        conditioning,
        init_image,
        strength,
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
        Return value depends on the seed at the time you call it.
        """
        raise NotImplementedError("replaced by invokeai.app.invocations.latent.LatentsToLatentsInvocation")
