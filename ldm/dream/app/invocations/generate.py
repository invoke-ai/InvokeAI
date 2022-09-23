# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from marshmallow import fields
from marshmallow.validate import OneOf, Range
from PIL.Image import Image
from ldm.dream.app.services.schemas import ImageField, InvocationSchemaBase
from ldm.dream.app.invocations.invocationabc import InvocationABC
from ldm.generate import Generate


# Text to image
class InvokeTextToImage(InvocationABC):
    """Generates an image using text2img."""
    def __init__(
        self,
        generate: Generate,
        **kwargs # consume unused arguments
    ):
        self._generate: Generate = generate

    def invoke(self, prompt: str, **kwargs) -> dict:  # See args in schema below
        results = self._generate.prompt2image(
            prompt=prompt,
            **kwargs
        )

        # Results are image and seed, unwrap for now
        # TODO: can this return multiple results?
        return dict(image=results[0][0])


class TextToImageSchema(InvocationSchemaBase):
    """txt2img"""
    class Meta:
        type = "txt2img"
        outputs = {
            "image": ImageField()
        }
        invokes = InvokeTextToImage
        # TODO: output intermediates? That doesn't seem quite right, since they couldn't be used
        # Maybe send them to the job? Job context?

    # TODO: consider making this optional and just fail if not supplied? (to provide prompt building capability)
    prompt = fields.String(required=True)
    seed = fields.Integer(load_default=0)  # 0 is random
    steps = fields.Integer(load_default=10)
    width = fields.Integer(load_default=512)
    height = fields.Integer(load_default=512)
    cfg_scale = fields.Float(load_default=7.5)
    sampler_name = fields.String(
        load_default="k_lms",
        validate=OneOf(["ddim","plms","k_lms","k_dpm_2","k_dpm_2_a","k_euler","k_euler_a","k_heun"]),
    )
    seamless = fields.Boolean(load_default=False)
    model = fields.String(load_default="")  # currently unused
    embeddings = fields.Raw(load_default="")  # currently unused
    progress_images = fields.Boolean(load_default="false")


# Image to image
class InvokeImageToImage(InvocationABC):
    """Generates an image using img2img."""
    def __init__(
        self,
        generate: Generate,
        **kwargs # consume unused arguments
    ):
        self._generate: Generate = generate
        
    def invoke(self, image: Image, prompt: str, **kwargs) -> dict:  # See args in schema below
        results = self._generate.prompt2image(
            prompt=prompt,
            init_img=image,
            **kwargs
        )

        # Results are image and seed, unwrap for now
        # TODO: can this return multiple results?
        return dict(image=results[0][0])


class ImageToImageSchema(TextToImageSchema):
    """img2img, runs txt2img with a weighted initial image"""
    class Meta(TextToImageSchema.Meta):
        type = 'img2img'
        invokes = InvokeImageToImage

    image = ImageField()
    strength = fields.Float(load_default=0.75, validate=Range(0.0, 1.0, min_inclusive=False, max_inclusive=True))
    fit = fields.Boolean(load_default=True)
