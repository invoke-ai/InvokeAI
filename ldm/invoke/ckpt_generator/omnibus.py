"""omnibus module to be used with the runwayml 9-channel custom inpainting model"""

import torch
import numpy as  np
from einops import repeat
from PIL import Image, ImageOps, ImageChops
from ldm.invoke.devices import choose_autocast
from ldm.invoke.ckpt_generator.base import downsampling
from ldm.invoke.ckpt_generator.img2img import CkptImg2Img
from ldm.invoke.ckpt_generator.txt2img import CkptTxt2Img

class CkptOmnibus(CkptImg2Img,CkptTxt2Img):
    def __init__(self, model, precision):
        super().__init__(model, precision)
        self.pil_mask = None
        self.pil_image = None

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
            init_image = None,
            mask_image = None,
            strength = None,
            step_callback=None,
            threshold=0.0,
            perlin=0.0,
            mask_blur_radius: int = 8,
            **kwargs):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it.
        """
        self.perlin = perlin
        num_samples = 1

        sampler.make_schedule(
            ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False
        )

        if isinstance(init_image, Image.Image):
            self.pil_image = init_image
            if init_image.mode != 'RGB':
                init_image = init_image.convert('RGB')
            init_image = self._image_to_tensor(init_image)

        if isinstance(mask_image, Image.Image):
            self.pil_mask = mask_image

            mask_image = ImageChops.multiply(mask_image.convert('L'), self.pil_image.split()[-1])
            mask_image = self._image_to_tensor(ImageOps.invert(mask_image), normalize=False)

        self.mask_blur_radius = mask_blur_radius

        t_enc = steps

        if init_image is not None and mask_image is not None: # inpainting
            masked_image = init_image * (1 - mask_image)  # masked image is the image masked by mask - masked regions zero

        elif init_image is not None: # img2img
            scope = choose_autocast(self.precision)

            with scope(self.model.device.type):
                self.init_latent = self.model.get_first_stage_encoding(
                    self.model.encode_first_stage(init_image)
                ) # move to latent space

            # create a completely black mask  (1s)
            mask_image = torch.ones(1, 1, init_image.shape[2], init_image.shape[3], device=self.model.device)
            # and the masked image is just a copy of the original
            masked_image = init_image

        else: # txt2img
            init_image = torch.zeros(1, 3, height, width, device=self.model.device)
            mask_image = torch.ones(1, 1, height, width, device=self.model.device)
            masked_image = init_image

        self.init_latent = init_image
        height = init_image.shape[2]
        width = init_image.shape[3]
        model = self.model

        def make_image(x_T):
            with torch.no_grad():
                scope = choose_autocast(self.precision)
                with scope(self.model.device.type):

                    batch = self.make_batch_sd(
                        init_image,
                        mask_image,
                        masked_image,
                        prompt=prompt,
                        device=model.device,
                        num_samples=num_samples,
                    )

                    c = model.cond_stage_model.encode(batch["txt"])
                    c_cat = list()
                    for ck in model.concat_keys:
                        cc = batch[ck].float()
                        if ck != model.masked_image_key:
                            bchw = [num_samples, 4, height//8, width//8]
                            cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                        else:
                            cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                        c_cat.append(cc)
                    c_cat = torch.cat(c_cat, dim=1)

                    # cond
                    cond={"c_concat": [c_cat], "c_crossattn": [c]}

                    # uncond cond
                    uc_cross = model.get_unconditional_conditioning(num_samples, "")
                    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
                    shape = [model.channels, height//8, width//8]

                    samples, _ = sampler.sample(
                        batch_size = 1,
                        S = steps,
                        x_T = x_T,
                        conditioning = cond,
                        shape = shape,
                        verbose = False,
                        unconditional_guidance_scale = cfg_scale,
                        unconditional_conditioning = uc_full,
                        eta = 1.0,
                        img_callback = step_callback,
                        threshold = threshold,
                    )
                    if self.free_gpu_mem:
                        self.model.model.to("cpu")
            return self.sample_to_image(samples)

        return make_image

    def make_batch_sd(
            self,
            image,
            mask,
            masked_image,
            prompt,
            device,
            num_samples=1):
        batch = {
                "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
                "txt": num_samples * [prompt],
                "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
                "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
                }
        return batch

    def get_noise(self, width:int, height:int):
        if self.init_latent is not None:
            height = self.init_latent.shape[2]
            width = self.init_latent.shape[3]
        return CkptTxt2Img.get_noise(self,width,height)


    def sample_to_image(self, samples)->Image.Image:
        gen_result = super().sample_to_image(samples).convert('RGB')

        if self.pil_image is None or self.pil_mask is None:
            return gen_result
        if self.pil_image.size != self.pil_mask.size:
            return gen_result

        corrected_result = super(CkptImg2Img, self).repaste_and_color_correct(gen_result, self.pil_image, self.pil_mask, self.mask_blur_radius)

        return corrected_result
