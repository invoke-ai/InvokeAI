import pytest
import torch

from invokeai.backend.model_management.model_cache import ModelCache, SDModelType
from invokeai.backend.stable_diffusion import StableDiffusionGeneratorPipeline

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    SchedulerMixin,
)
from transformers import (
    CLIPTokenizer,
    CLIPFeatureExtractor,
    CLIPTextModel,
)    


cache = ModelCache()

def test_pipeline_fetch():
    model0 = cache.get_model('stabilityai/sd-vae-ft-mse',SDModelType.vae)
    model1 = cache.get_model('stabilityai/stable-diffusion-2-1',SDModelType.diffusion_pipeline)
    model1_2 = cache.get_model('stabilityai/stable-diffusion-2-1')
    assert model1==model1_2
    assert model1.device==torch.device('cuda')
    model2 = cache.get_model('runwayml/stable-diffusion-v1-5')
    assert model2.device==torch.device('cuda')
    assert model1.device==torch.device('cpu')
    model1 = cache.get_model('stabilityai/stable-diffusion-2-1')
    assert model1.device==torch.device('cuda')

def test_submodel_fetch():
    model1_vae = cache.get_submodel('stabilityai/stable-diffusion-2-1',SDModelType.vae)
    assert isinstance(model1_vae,AutoencoderKL)
    model1 = cache.get_model('stabilityai/stable-diffusion-2-1',SDModelType.diffusion_pipeline)
    assert model1_vae == model1.vae
    model1_vae_2 = cache.get_submodel('stabilityai/stable-diffusion-2-1')
    assert model1_vae == model1_vae_2

def test_transformer_fetch():
    model4 = cache.get_model('openai/clip-vit-large-patch14',SDModelType.tokenizer)
    assert isinstance(model4,CLIPTokenizer)

    model5 = cache.get_model('openai/clip-vit-large-patch14',SDModelType.text_encoder)
    assert isinstance(model5,CLIPTextModel)

def test_subfolder_fetch():
    model6 = cache.get_model('stabilityai/stable-diffusion-2',SDModelType.tokenizer,subfolder="tokenizer")
    assert isinstance(model6,CLIPTokenizer)

    model7 = cache.get_model('stabilityai/stable-diffusion-2',SDModelType.text_encoder,subfolder="text_encoder")
    assert isinstance(model7,CLIPTextModel)
