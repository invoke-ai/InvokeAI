"""What the wan architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import WAN21_16, WAN22_48, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.vae import VaeCompatibility, VaeFacet
from invokeai.backend.architectures.facets.variant import VariantFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, WanLoRAVariantType, WanVariantType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import WanConditioningInfo

# Two variants that model identity cannot tell apart: A14B denoises in the 16-channel Wan 2.1
# space at 8x, TI2V-5B in the 48-channel Wan2.2-VAE space at 16x. The loaded checkpoint
# decides, so the sample's channel count is what resolves it.
register(
    BaseModelType.Wan,
    LatentSpaceFacet(
        WAN21_16,
        alternates=(WAN22_48,),
        # Generation tells these apart by the sample's channel count; the served table has only a
        # model record, so the variant names it there.
        by_variant={WanVariantType.TI2V_5B: WAN22_48},
    ),
    ConditioningFacet(WanConditioningInfo),
    DefaultSettingsFacet(
        {
            WanVariantType.TI2V_5B: MainModelDefaultSettings(
                scheduler="euler", steps=30, cfg_scale=5.0, width=1024, height=1024
            ),
            # A14B, and whatever an unknown variant turns out to be.
            None: MainModelDefaultSettings(scheduler="euler", steps=40, cfg_scale=4.0, width=1024, height=1024),
        }
    ),
    # Wan generates images at num_frames=1 and video above that, from text or from one or two
    # given frames -- `interpolate` fills between two, `extend_video` continues an existing clip.
    ModalityFacet(
        frozenset({"txt2img", "img2img", "inpaint", "outpaint", "t2v", "i2v", "interpolate", "extend_video"}),
        metadata_slug="wan",
    ),
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=True, usage="always"),
        dimension_grid=16,
        # The transformer's patch_size=(1, 2, 2) doubles the VAE's spatial compression: 8x VAE for
        # A14B is a 16-pixel grid, the TI2V-5B Wan2.2-VAE's 16x is a 32-pixel one. `wan_denoise`
        # enforces the A14B number as a field constraint and the TI2V-5B one inside `invoke()`.
        dimension_grid_by_variant={WanVariantType.TI2V_5B: 32},
        guidance_label="Guidance",
        # wan_denoise.guidance_scale is ge=1.0. Its optional low-noise counterpart is a separate
        # field with its own ge=0.0 and its own control, not this slider.
        guidance_min=1.0,
        scheduler_set="flow",
    ),
    # A14B uses the 16-channel Wan 2.1 VAE; TI2V-5B needs the 48-channel Wan2.2-VAE, and
    # `wan_model_loader` rejects either the other way round. Both are registered under `wan`, so
    # the channel count is the only thing that tells them apart.
    VaeFacet(
        frozenset({VaeCompatibility(BaseModelType.Wan, latent_channels=16)}),
        by_variant={WanVariantType.TI2V_5B: frozenset({VaeCompatibility(BaseModelType.Wan, latent_channels=48)})},
    ),
    # The one architecture whose LoRAs carry a different variant enum from its mains. They are not
    # interchangeable: an A14B LoRA (inner_dim=5120) against a TI2V-5B main (3072) crashes in the
    # layer patcher, which is why the LoRA enum exists separately at all.
    VariantFacet(
        {
            ModelType.Main: WanVariantType,
            ModelType.LoRA: WanLoRAVariantType,
        }
    ),
)
