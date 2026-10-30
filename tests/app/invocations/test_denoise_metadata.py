from types import SimpleNamespace
from unittest.mock import patch

import pytest

from invokeai.app.invocations.anima.anima_denoise import AnimaDenoiseInvocation
from invokeai.app.invocations.anima.anima_lllite import AnimaLLLiteField
from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, InvocationRegistry
from invokeai.app.invocations.cogview4.cogview4_denoise import CogView4DenoiseInvocation
from invokeai.app.invocations.ernie_image.ernie_image_denoise import ErnieImageDenoiseInvocation
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    ImageField,
    LatentsField,
    MetadataField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.flux.flux_controlnet import FluxControlNetField
from invokeai.app.invocations.flux.flux_denoise import FluxDenoiseInvocation
from invokeai.app.invocations.flux2.flux2_control import Flux2ControlField
from invokeai.app.invocations.flux2.flux2_denoise import Flux2DenoiseInvocation
from invokeai.app.invocations.ideogram4.ideogram4_denoise import Ideogram4DenoiseInvocation
from invokeai.app.invocations.krea2.krea2_denoise import Krea2DenoiseInvocation
from invokeai.app.invocations.ltx2.ltx2_denoise import LTX2DenoiseInvocation, LTX2DenoiseOutput
from invokeai.app.invocations.metadata import (
    CORE_METADATA_VERSION,
    LatentsMetaOutput,
    MetadataInvocation,
    MetadataItemField,
    MetadataOutput,
    build_denoise_metadata,
    serialize_adapter_metadata,
    serialize_ip_adapter_ref_images,
    serialize_lora_metadata,
)
from invokeai.app.invocations.metadata_linked import (
    CORE_LABELS,
    CORE_LABELS_BOOL,
    CORE_LABELS_FLOAT,
    CORE_LABELS_INTEGER,
    CORE_LABELS_MODEL,
    CORE_LABELS_SCHEDULER,
    CORE_LABELS_STRING,
    CORE_LABELS_VAE,
    CUSTOM_LABEL,
    AnimaDenoiseMetaInvocation,
    DenoiseLatentsMetaInvocation,
    FluxDenoiseLatentsMetaInvocation,
    MetadataToControlnetsInvocation,
    MetadataToIPAdaptersInvocation,
    MetadataToT2IAdaptersInvocation,
    ZImageDenoiseMetaInvocation,
)
from invokeai.app.invocations.minimax_h3.minimax_h3_denoise import MiniMaxH3DenoiseInvocation, MiniMaxH3DenoiseOutput
from invokeai.app.invocations.model import (
    ControlLoRAField,
    FreeUConfig,
    LoRAField,
    LTX2TransformerField,
    MiniMaxH3AdaLNOverlayField,
    MiniMaxH3TransformerField,
    ModelIdentifierField,
    Qwen3EncoderField,
    TransformerField,
    UNetField,
    VAEField,
    WanTransformerField,
)
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.invocations.qwen_image.qwen_image_control import QwenImageControlField
from invokeai.app.invocations.qwen_image.qwen_image_denoise import QwenImageDenoiseInvocation
from invokeai.app.invocations.qwen_image_2_1.qwen_image_2_1_denoise import QwenImage21DenoiseInvocation
from invokeai.app.invocations.sd.controlnet import ControlField
from invokeai.app.invocations.sd.denoise_latents import DenoiseLatentsInvocation
from invokeai.app.invocations.sd.ip_adapter import IPAdapterField
from invokeai.app.invocations.sd.t2i_adapter import T2IAdapterField
from invokeai.app.invocations.sd.tiled_multi_diffusion_denoise_latents import TiledMultiDiffusionDenoiseLatents
from invokeai.app.invocations.sd3.sd3_denoise import SD3DenoiseInvocation
from invokeai.app.invocations.wan.wan_denoise import WanDenoiseInvocation
from invokeai.app.invocations.wan.wan_video_denoise import WanVideoDenoiseInvocation
from invokeai.app.invocations.z_image.z_image_control import ZImageControlField
from invokeai.app.invocations.z_image.z_image_denoise import ZImageDenoiseInvocation
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType
from invokeai.version.invokeai_version import __version__


def _make_model_id(
    key: str = "model-key-1",
    name: str = "test-model",
    base: BaseModelType = BaseModelType.Flux,
    model_type: ModelType = ModelType.Main,
    submodel_type: SubModelType | None = SubModelType.Transformer,
) -> ModelIdentifierField:
    return ModelIdentifierField(
        key=key,
        hash="blake3:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        name=name,
        base=base,
        type=model_type,
        submodel_type=submodel_type,
    )


def _discover_denoise_invocations() -> list[type[BaseInvocation]]:
    return [
        cls
        for cls in InvocationRegistry.get_invocation_classes()
        if "denoise" in cls.get_type() and not cls.get_type().endswith("_mask")
    ]


def test_all_denoise_invocations_conform_to_metadata_contract() -> None:
    denoise_classes = _discover_denoise_invocations()
    assert len(denoise_classes) >= 21  # 17 active denoise nodes + 4 deprecated *_meta wrappers

    for cls in denoise_classes:
        node_type = cls.get_type()
        assert issubclass(cls, WithMetadata), f"{node_type} must inherit WithMetadata"
        assert "metadata" in cls.model_fields, f"{node_type} must expose a 'metadata' input field"
        assert not issubclass(cls, WithBoard), f"{node_type} must not inherit WithBoard"
        assert "board" not in cls.model_fields, f"{node_type} must not expose a 'board' input field"

        output_cls = cls.get_output_annotation()
        assert issubclass(output_cls, MetadataOutput), (
            f"{node_type} output {output_cls.__name__} must subclass MetadataOutput"
        )
        assert "metadata" in output_cls.model_fields, (
            f"{node_type} output {output_cls.__name__} must expose a 'metadata' output field"
        )

        if node_type.endswith(("_denoise_meta", "_latents_meta")):
            assert cls.UIConfig.classification == Classification.Deprecated, (
                f"{node_type} wrapper must be marked Classification.Deprecated"
            )


def test_build_denoise_metadata_does_not_mutate_existing_metadata() -> None:
    original_dict = {"positive_prompt": "a scenic mountain", "steps": 10, "custom_tag": "keep-me"}
    existing = MetadataField.model_validate(original_dict.copy())

    result = build_denoise_metadata(existing, {"steps": 30, "width": 1024, "height": 768})

    # Input MetadataField.root must remain untouched
    assert existing.root == original_dict
    assert existing.root["steps"] == 10
    assert "width" not in existing.root

    # Output MetadataField has merged values + version stamps
    assert result.root["positive_prompt"] == "a scenic mountain"
    assert result.root["custom_tag"] == "keep-me"
    assert result.root["steps"] == 30
    assert result.root["width"] == 1024
    assert result.root["height"] == 768
    assert result.root["app_version"] == __version__
    assert result.root["metadata_version"] == CORE_METADATA_VERSION


def test_serialize_lora_metadata_deduplicates_across_lists() -> None:
    lora_a = _make_model_id(key="lora-a", name="LoRA A", model_type=ModelType.LoRA, submodel_type=None)
    lora_b = _make_model_id(key="lora-b", name="LoRA B", model_type=ModelType.LoRA, submodel_type=None)

    list_1 = [LoRAField(lora=lora_a, weight=0.75), LoRAField(lora=lora_b, weight=0.5)]
    # lora_a with same weight is deduplicated; lora_a with different weight is kept
    list_2 = [LoRAField(lora=lora_a, weight=0.75), LoRAField(lora=lora_a, weight=1.0)]

    serialized = serialize_lora_metadata(list_1, None, list_2)
    assert len(serialized) == 3
    assert serialized[0]["model"]["key"] == "lora-a"
    assert serialized[0]["weight"] == 0.75
    assert serialized[1]["model"]["key"] == "lora-b"
    assert serialized[1]["weight"] == 0.5
    assert serialized[2]["model"]["key"] == "lora-a"
    assert serialized[2]["weight"] == 1.0


def test_serialize_adapter_metadata_normalizes_single_and_list() -> None:
    assert serialize_adapter_metadata(None) == []

    ctrl_model = _make_model_id(key="ctrl-1", model_type=ModelType.ControlNet, submodel_type=None)
    field = ZImageControlField(
        image_name="ctrl.png",
        control_model=ctrl_model,
        control_context_scale=0.8,
        begin_step_percent=0.0,
        end_step_percent=1.0,
    )
    single = serialize_adapter_metadata(field)
    many = serialize_adapter_metadata([field])
    assert len(single) == 1
    assert single == many
    assert single[0]["image_name"] == "ctrl.png"
    assert single[0]["control_model"]["key"] == "ctrl-1"


def test_sd_and_tiled_denoise_build_metadata() -> None:
    unet_id = _make_model_id(key="sdxl-unet", base=BaseModelType.StableDiffusionXL, submodel_type=SubModelType.UNet)
    sched_id = _make_model_id(
        key="sdxl-sched", base=BaseModelType.StableDiffusionXL, submodel_type=SubModelType.Scheduler
    )
    lora_id = _make_model_id(key="sdxl-lora", base=BaseModelType.StableDiffusionXL, model_type=ModelType.LoRA)
    ctrl_id = _make_model_id(key="sdxl-ctrl", base=BaseModelType.StableDiffusionXL, model_type=ModelType.ControlNet)
    ipa_id = _make_model_id(key="sdxl-ipa", base=BaseModelType.StableDiffusionXL, model_type=ModelType.IPAdapter)
    t2i_id = _make_model_id(key="sdxl-t2i", base=BaseModelType.StableDiffusionXL, model_type=ModelType.T2IAdapter)

    unet_field = UNetField(
        unet=unet_id,
        scheduler=sched_id,
        loras=[LoRAField(lora=lora_id, weight=0.8)],
        seamless_axes=["x"],
        freeu_config=FreeUConfig(s1=0.9, s2=0.2, b1=1.2, b2=1.4),
    )
    control = ControlField(
        image=ImageField(image_name="cn.png"),
        control_model=ctrl_id,
        control_weight=0.75,
        begin_step_percent=0.0,
        end_step_percent=1.0,
        control_mode="balanced",
        resize_mode="just_resize",
    )
    ipa = IPAdapterField(
        image=ImageField(image_name="ipa.png"),
        ip_adapter_model=ipa_id,
        image_encoder_model=ipa_id,
        weight=0.6,
        method="full",
        begin_step_percent=0.0,
        end_step_percent=1.0,
    )
    t2i = T2IAdapterField(
        image=ImageField(image_name="t2i.png"),
        t2i_adapter_model=t2i_id,
        weight=0.5,
        begin_step_percent=0.0,
        end_step_percent=1.0,
        resize_mode="just_resize",
    )
    upstream = MetadataField.model_validate({"positive_prompt": "castle"})

    sd_node = DenoiseLatentsInvocation.model_construct(
        unet=unet_field,
        cfg_scale=7.5,
        cfg_rescale_multiplier=0.1,
        steps=25,
        scheduler="dpmpp_2m_k",
        denoising_start=0.1,
        denoising_end=0.9,
        control=control,
        ip_adapter=ipa,
        t2i_adapter=t2i,
        noise=LatentsField(latents_name="n", seed=4242),
        latents=None,
        metadata=upstream,
    )
    md = sd_node._build_metadata(1024, 768).root
    assert upstream.root == {"positive_prompt": "castle"}
    assert md["positive_prompt"] == "castle"
    assert md["width"] == 1024
    assert md["height"] == 768
    assert md["cfg_scale"] == 7.5
    assert md["cfg_rescale_multiplier"] == 0.1
    assert md["steps"] == 25
    assert md["scheduler"] == "dpmpp_2m_k"
    assert md["denoising_start"] == 0.1
    assert md["denoising_end"] == 0.9
    assert md["model"] == unet_id
    assert md["seamless_x"] is True
    assert md["seamless_y"] is False
    assert (md["freeu_s1"], md["freeu_s2"], md["freeu_b1"], md["freeu_b2"]) == (0.9, 0.2, 1.2, 1.4)
    assert len(md["loras"]) == 1 and md["loras"][0]["weight"] == 0.8
    assert len(md["controlnets"]) == 1
    assert len(md["ref_images"]) == 1
    assert "ipAdapters" not in md
    assert len(md["t2iAdapters"]) == 1
    assert md["seed"] == 4242

    tiled_node = TiledMultiDiffusionDenoiseLatents.model_construct(
        unet=unet_field,
        cfg_scale=6.0,
        cfg_rescale_multiplier=0.0,
        steps=18,
        scheduler="euler",
        denoising_start=0.0,
        denoising_end=1.0,
        tile_width=1024,
        tile_height=1024,
        tile_overlap=64,
        control=control,
        noise=None,
        latents=LatentsField(latents_name="l", seed=777),
        metadata=None,
    )
    tiled_md = tiled_node._build_metadata(2048, 2048).root
    assert tiled_md["tile_width"] == 1024
    assert tiled_md["tile_height"] == 1024
    assert tiled_md["tile_overlap"] == 64
    assert tiled_md["seed"] == 777
    assert len(tiled_md["controlnets"]) == 1


def test_flux_and_flux2_denoise_build_metadata() -> None:
    tf_id = _make_model_id(key="flux-tf", base=BaseModelType.Flux)
    lora_id = _make_model_id(key="flux-lora", base=BaseModelType.Flux, model_type=ModelType.LoRA)
    ctrl_lora_id = _make_model_id(key="flux-ctrl-lora", base=BaseModelType.Flux, model_type=ModelType.ControlLoRa)
    ctrl_id = _make_model_id(key="flux-cn", base=BaseModelType.Flux, model_type=ModelType.ControlNet)
    ipa_id = _make_model_id(key="flux-ipa", base=BaseModelType.Flux, model_type=ModelType.IPAdapter)
    flux_cn_vae_id = _make_model_id(key="flux-cn-vae", base=BaseModelType.Flux, model_type=ModelType.VAE)
    vae_id = _make_model_id(key="flux2-vae", base=BaseModelType.Flux2, model_type=ModelType.VAE)

    flux_node = FluxDenoiseInvocation.model_construct(
        latents=None,
        noise=None,
        add_noise=True,
        denoising_start=0.0,
        denoising_end=1.0,
        width=1024,
        height=768,
        num_steps=20,
        scheduler="euler",
        guidance=3.5,
        cfg_scale=1.5,
        cfg_scale_start_step=1,
        cfg_scale_end_step=15,
        seed=101,
        dype_preset="area",
        dype_scale=2.5,
        dype_exponent=1.8,
        transformer=TransformerField(transformer=tf_id, loras=[LoRAField(lora=lora_id, weight=0.9)]),
        control_lora=ControlLoRAField(lora=ctrl_lora_id, weight=0.7, img=ImageField(image_name="cl.png")),
        control=FluxControlNetField(
            image=ImageField(image_name="fc.png"),
            control_model=ctrl_id,
            control_weight=0.8,
            begin_step_percent=0.0,
            end_step_percent=1.0,
            resize_mode="just_resize",
        ),
        controlnet_vae=VAEField(vae=flux_cn_vae_id),
        ip_adapter=IPAdapterField(
            image=ImageField(image_name="fipa.png"),
            ip_adapter_model=ipa_id,
            image_encoder_model=ipa_id,
            weight=0.5,
            method="full",
            begin_step_percent=0.0,
            end_step_percent=1.0,
        ),
        metadata=None,
    )
    flux_md = flux_node._build_metadata().root
    assert flux_md["width"] == 1024
    assert flux_md["height"] == 768
    assert flux_md["steps"] == 20
    assert flux_md["scheduler"] == "euler"
    assert flux_md["guidance"] == 3.5
    assert flux_md["cfg_scale"] == 1.5
    assert flux_md["cfg_scale_start_step"] == 1
    assert flux_md["cfg_scale_end_step"] == 15
    assert flux_md["dype_preset"] == "area"
    assert flux_md["dype_scale"] == 2.5
    assert flux_md["dype_exponent"] == 1.8
    assert flux_md["vae"] == flux_cn_vae_id
    assert len(flux_md["loras"]) == 2
    assert len(flux_md["controlnets"]) == 1
    assert len(flux_md["ref_images"]) == 1
    assert "ipAdapters" not in flux_md
    assert flux_md["seed"] == 101

    # When upstream metadata already contains "vae", controlnet_vae does not overwrite it
    upstream_vae_id = _make_model_id(key="flux-upstream-vae", base=BaseModelType.Flux, model_type=ModelType.VAE)
    flux_with_upstream_vae = flux_node.model_copy(
        update={"metadata": MetadataField.model_validate({"vae": upstream_vae_id})}
    )
    assert flux_with_upstream_vae._build_metadata().root["vae"] == upstream_vae_id

    flux2_node = Flux2DenoiseInvocation.model_construct(
        latents=None,
        noise=None,
        add_noise=True,
        denoising_start=0.0,
        denoising_end=1.0,
        width=1024,
        height=768,
        num_steps=8,
        scheduler="heun",
        guidance=4.0,
        cfg_scale=2.0,
        seed=202,
        transformer=TransformerField(transformer=tf_id, loras=[LoRAField(lora=lora_id, weight=0.6)]),
        vae=VAEField(vae=vae_id),
        control=Flux2ControlField(
            control_model=ctrl_id,
            control_image=ImageField(image_name="f2c.png"),
            control_context_scale=0.75,
            begin_step_percent=0.0,
            end_step_percent=1.0,
        ),
        metadata=None,
    )
    flux2_md = flux2_node._build_metadata().root
    assert flux2_md["width"] == 1024
    assert flux2_md["height"] == 768
    assert flux2_md["steps"] == 8
    assert flux2_md["scheduler"] == "heun"
    assert flux2_md["guidance"] == 4.0
    assert flux2_md["cfg_scale"] == 2.0
    assert flux2_md["vae"] == vae_id
    assert len(flux2_md["loras"]) == 1
    assert len(flux2_md["controlnets"]) == 1
    assert flux2_md["seed"] == 202


def test_sd3_cogview4_zimage_anima_denoise_build_metadata() -> None:
    tf_id = _make_model_id(key="tf-1")
    lora_id = _make_model_id(key="lora-1", model_type=ModelType.LoRA)
    vae_id = _make_model_id(key="vae-1", model_type=ModelType.VAE)
    ctrl_id = _make_model_id(key="ctrl-1", model_type=ModelType.ControlNet)
    tf_field = TransformerField(transformer=tf_id, loras=[LoRAField(lora=lora_id, weight=0.5)])

    sd3_md = (
        SD3DenoiseInvocation.model_construct(
            latents=None,
            noise=None,
            denoising_start=0.0,
            denoising_end=1.0,
            width=1024,
            height=768,
            transformer=tf_field,
            cfg_scale=3.5,
            steps=28,
            seed=11,
            metadata=None,
        )
        ._build_metadata()
        .root
    )
    assert sd3_md["width"] == 1024
    assert sd3_md["height"] == 768
    assert sd3_md["steps"] == 28
    assert sd3_md["cfg_scale"] == 3.5
    assert sd3_md["seed"] == 11
    assert len(sd3_md["loras"]) == 1

    cog_md = (
        CogView4DenoiseInvocation.model_construct(
            latents=None,
            noise=None,
            denoising_start=0.0,
            denoising_end=1.0,
            width=1024,
            height=768,
            transformer=tf_field,
            cfg_scale=4.0,
            steps=25,
            seed=22,
            metadata=None,
        )
        ._build_metadata()
        .root
    )
    assert cog_md["width"] == 1024
    assert cog_md["height"] == 768
    assert cog_md["steps"] == 25
    assert cog_md["cfg_scale"] == 4.0
    assert cog_md["seed"] == 22

    z_node = ZImageDenoiseInvocation.model_construct(
        latents=None,
        noise=None,
        add_noise=True,
        denoising_start=0.0,
        denoising_end=1.0,
        width=1024,
        height=768,
        transformer=tf_field,
        guidance_scale=1.5,
        steps=8,
        seed=33,
        shift=2.5,
        scheduler="euler",
        vae=VAEField(vae=vae_id),
        control=ZImageControlField(
            image_name="zc.png",
            control_model=ctrl_id,
            control_context_scale=0.9,
            begin_step_percent=0.0,
            end_step_percent=1.0,
        ),
        metadata=None,
    )
    z_md = z_node._build_metadata().root
    assert z_md["width"] == 1024
    assert z_md["height"] == 768
    assert z_md["cfg_scale"] == 1.5
    assert z_md["guidance"] == 1.5
    assert z_md["z_image_shift"] == 2.5
    assert z_md["vae"] == vae_id
    assert len(z_md["controlnets"]) == 1
    assert z_md["seed"] == 33
    assert z_node.model_copy(update={"shift": None})._build_metadata().root["z_image_shift"] == "auto"

    anima_md = (
        AnimaDenoiseInvocation.model_construct(
            latents=None,
            noise=None,
            add_noise=True,
            denoising_start=0.0,
            denoising_end=1.0,
            width=1024,
            height=768,
            transformer=tf_field,
            guidance_scale=4.5,
            steps=30,
            seed=44,
            scheduler="euler",
            control_lllite=AnimaLLLiteField(
                control_model=ctrl_id,
                image_name="ac.png",
                weight=0.8,
                begin_step_percent=0.0,
                end_step_percent=1.0,
            ),
            metadata=None,
        )
        ._build_metadata()
        .root
    )
    assert anima_md["width"] == 1024
    assert anima_md["height"] == 768
    assert anima_md["cfg_scale"] == 4.5
    assert anima_md["steps"] == 30
    assert len(anima_md["controlnets"]) == 1
    assert anima_md["seed"] == 44


def test_ernie_ideogram4_krea2_qwen_denoise_build_metadata() -> None:
    tf_id = _make_model_id(key="tf-main")
    uncond_tf_id = _make_model_id(key="tf-uncond")
    lora_1 = _make_model_id(key="lora-1", model_type=ModelType.LoRA)
    lora_2 = _make_model_id(key="lora-2", model_type=ModelType.LoRA)
    vae_id = _make_model_id(key="vae-1", model_type=ModelType.VAE)
    ctrl_id = _make_model_id(key="ctrl-1", model_type=ModelType.ControlNet)

    ernie_md = (
        ErnieImageDenoiseInvocation.model_construct(
            transformer=TransformerField(transformer=tf_id, loras=[LoRAField(lora=lora_1, weight=0.7)]),
            latents=None,
            add_noise=True,
            width=1024,
            height=768,
            steps=8,
            guidance_scale=1.0,
            denoising_start=0.0,
            denoising_end=1.0,
            seed=55,
            scheduler="euler",
            metadata=None,
        )
        ._build_metadata()
        .root
    )
    assert ernie_md["width"] == 1024
    assert ernie_md["height"] == 768
    assert ernie_md["steps"] == 8
    assert ernie_md["cfg_scale"] == 1.0
    assert ernie_md["seed"] == 55
    assert len(ernie_md["loras"]) == 1

    # Ideogram 4: preset fallback + explicit overrides + dual-branch LoRA deduplication
    ideo_node = Ideogram4DenoiseInvocation.model_construct(
        transformer=TransformerField(
            transformer=tf_id,
            loras=[LoRAField(lora=lora_1, weight=0.5), LoRAField(lora=lora_2, weight=0.4)],
        ),
        unconditional_transformer=TransformerField(
            transformer=uncond_tf_id,
            loras=[LoRAField(lora=lora_1, weight=0.5)],
        ),
        sampler_preset="V4_DEFAULT_20",
        steps=None,
        guidance_scale=None,
        mu=None,
        width=1024,
        height=768,
        seed=66,
        metadata=None,
    )
    ideo_md = ideo_node._build_metadata().root
    assert ideo_md["width"] == 1024
    assert ideo_md["height"] == 768
    assert ideo_md["steps"] == 20
    assert ideo_md["cfg_scale"] == 7.0
    assert ideo_md["guidance"] == 7.0
    assert ideo_md["ideogram4_sampler_preset"] == "V4_DEFAULT_20"
    assert "ideogram4_steps" not in ideo_md
    assert "ideogram4_guidance_scale" not in ideo_md
    assert "ideogram4_mu" not in ideo_md
    assert ideo_md["ideogram4_unconditional_model"] == uncond_tf_id
    assert len(ideo_md["loras"]) == 2

    ideo_overrides_md = (
        ideo_node.model_copy(update={"steps": 14, "guidance_scale": 5.5, "mu": 0.25})._build_metadata().root
    )
    assert ideo_overrides_md["steps"] == 14
    assert ideo_overrides_md["cfg_scale"] == 5.5
    assert ideo_overrides_md["guidance"] == 5.5
    assert ideo_overrides_md["ideogram4_steps"] == 14
    assert ideo_overrides_md["ideogram4_guidance_scale"] == 5.5
    assert ideo_overrides_md["ideogram4_mu"] == 0.25

    krea_md = (
        Krea2DenoiseInvocation.model_construct(
            transformer=TransformerField(transformer=tf_id, loras=[LoRAField(lora=lora_1, weight=1.0)]),
            cfg_scale=1.0,
            width=1280,
            height=720,
            steps=8,
            denoising_start=0.0,
            denoising_end=1.0,
            seed=77,
            shift=1.15,
            style_reference=SimpleNamespace(style_strength=0.65),
            metadata=None,
        )
        ._build_metadata()
        .root
    )
    assert krea_md["width"] == 1280
    assert krea_md["height"] == 720
    assert krea_md["shift"] == 1.15
    assert krea_md["krea2_style_strength"] == 0.65
    assert krea_md["seed"] == 77
    assert len(krea_md["loras"]) == 1

    qwen_node = QwenImageDenoiseInvocation.model_construct(
        transformer=TransformerField(transformer=tf_id, loras=[LoRAField(lora=lora_1, weight=0.9)]),
        cfg_scale=4.0,
        width=1360,
        height=768,
        steps=40,
        denoising_start=0.0,
        denoising_end=1.0,
        seed=88,
        shift=3.0,
        vae=VAEField(vae=vae_id),
        control=QwenImageControlField(
            control_model=ctrl_id,
            control_image=ImageField(image_name="qc.png"),
            control_context_scale=0.8,
            begin_step_percent=0.0,
            end_step_percent=1.0,
        ),
        metadata=None,
    )
    qwen_md = qwen_node._build_metadata().root
    assert qwen_md["width"] == 1360
    assert qwen_md["height"] == 768
    assert qwen_md["qwen_image_shift"] == 3.0
    assert qwen_md["vae"] == vae_id
    assert qwen_md["qwen_image_vae"] == vae_id
    assert len(qwen_md["controlnets"]) == 1
    assert qwen_md["seed"] == 88
    assert qwen_node.model_copy(update={"shift": None})._build_metadata().root["qwen_image_shift"] == "auto"

    qwen21_md = (
        QwenImage21DenoiseInvocation.model_construct(
            transformer=TransformerField(transformer=tf_id, loras=[LoRAField(lora=lora_1, weight=0.9)]),
            cfg_scale=1.0,
            width=1024,
            height=768,
            steps=8,
            denoising_start=0.0,
            denoising_end=1.0,
            seed=99,
            metadata=None,
        )
        ._build_metadata()
        .root
    )
    assert qwen21_md["width"] == 1024
    assert qwen21_md["height"] == 768
    assert qwen21_md["steps"] == 8
    assert qwen21_md["seed"] == 99
    assert len(qwen21_md["loras"]) == 1


def test_wan_ltx2_minimax_h3_denoise_build_metadata() -> None:
    high_tf = _make_model_id(key="wan-high", base=BaseModelType.Wan)
    low_tf = _make_model_id(key="wan-low", base=BaseModelType.Wan)
    lora_shared = _make_model_id(key="wan-lora-1", base=BaseModelType.Wan, model_type=ModelType.LoRA)
    lora_low = _make_model_id(key="wan-lora-2", base=BaseModelType.Wan, model_type=ModelType.LoRA)

    wan_tf_field = WanTransformerField(
        transformer=high_tf,
        transformer_low_noise=low_tf,
        loras=[LoRAField(lora=lora_shared, weight=0.8)],
        loras_low_noise=[LoRAField(lora=lora_shared, weight=0.8), LoRAField(lora=lora_low, weight=0.6)],
    )

    wan_md = (
        WanDenoiseInvocation.model_construct(
            transformer=wan_tf_field,
            guidance_scale=4.0,
            guidance_scale_low_noise=3.0,
            denoising_start=0.0,
            denoising_end=1.0,
            add_noise=True,
            width=1024,
            height=768,
            steps=40,
            seed=111,
            metadata=None,
        )
        ._build_metadata()
        .root
    )
    assert wan_md["width"] == 1024
    assert wan_md["height"] == 768
    assert wan_md["cfg_scale"] == 4.0
    assert wan_md["guidance"] == 4.0
    assert wan_md["wan_guidance_scale_low_noise"] == 3.0
    assert wan_md["wan_transformer_low_noise"] == low_tf
    assert len(wan_md["loras"]) == 2

    # guidance_scale_low_noise < 1.0 falls back to guidance_scale and is omitted from metadata
    wan_fallback_md = (
        WanDenoiseInvocation.model_construct(
            transformer=wan_tf_field,
            guidance_scale=4.0,
            guidance_scale_low_noise=0.0,
            denoising_start=0.0,
            denoising_end=1.0,
            add_noise=True,
            width=1024,
            height=768,
            steps=40,
            seed=111,
            metadata=None,
        )
        ._build_metadata()
        .root
    )
    assert "wan_guidance_scale_low_noise" not in wan_fallback_md

    wan_video_md = (
        WanVideoDenoiseInvocation.model_construct(
            transformer=wan_tf_field,
            guidance_scale=5.0,
            guidance_scale_low_noise=4.0,
            width=832,
            height=480,
            num_frames=81,
            steps=40,
            seed=222,
            metadata=None,
        )
        ._build_metadata()
        .root
    )
    assert wan_video_md["width"] == 832
    assert wan_video_md["height"] == 480
    assert wan_video_md["num_frames"] == 81
    assert wan_video_md["cfg_scale"] == 5.0
    assert wan_video_md["wan_guidance_scale_low_noise"] == 4.0
    assert len(wan_video_md["loras"]) == 2

    # LTX-2: base pass (no latents -> no ltx2_noise_scale) vs refine pass (with latents -> includes ltx2_noise_scale)
    ltx_tf = _make_model_id(key="ltx2-tf", base=BaseModelType.LTX2)
    ltx_base = LTX2DenoiseInvocation.model_construct(
        transformer=LTX2TransformerField(transformer=ltx_tf, loras=[LoRAField(lora=lora_shared, weight=0.5)]),
        width=1248,
        height=704,
        num_frames=121,
        fps=23.976,
        steps=30,
        cfg_scale=3.0,
        audio_cfg_scale=7.0,
        stg_scale=1.0,
        modality_scale=3.0,
        guidance_rescale=0.7,
        schedule="dev",
        seed=333,
        latents=None,
        noise_scale=0.5,
        metadata=None,
    )
    ltx_base_md = ltx_base._build_metadata().root
    assert ltx_base_md["width"] == 1248
    assert ltx_base_md["height"] == 704
    assert ltx_base_md["num_frames"] == 121
    assert ltx_base_md["fps"] == 24
    assert isinstance(ltx_base_md["fps"], int)
    assert ltx_base_md["ltx2_audio_cfg_scale"] == 7.0
    assert ltx_base_md["ltx2_stg_scale"] == 1.0
    assert ltx_base_md["ltx2_modality_scale"] == 3.0
    assert ltx_base_md["ltx2_guidance_rescale"] == 0.7
    assert ltx_base_md["ltx2_schedule"] == "dev"
    assert "ltx2_noise_scale" not in ltx_base_md

    ltx_refine = ltx_base.model_copy(update={"latents": LatentsField(latents_name="stage1")})
    ltx_refine_md = ltx_refine._build_metadata().root
    assert ltx_refine_md["ltx2_noise_scale"] == 0.5
    assert issubclass(LTX2DenoiseOutput, MetadataOutput)

    # MiniMax H3: includes fixed 24 fps, int num_frames, and hybrid vs non-hybrid model swap
    h3_fl2va = _make_model_id(key="h3-fl2va-base", base=BaseModelType.MiniMaxH3)
    h3_ref2va = _make_model_id(key="h3-ref2va-main", base=BaseModelType.MiniMaxH3)
    h3_hybrid_node = MiniMaxH3DenoiseInvocation.model_construct(
        transformer=MiniMaxH3TransformerField(
            transformer=h3_fl2va,
            loras=[LoRAField(lora=lora_shared, weight=0.75)],
            adaln_overlay=MiniMaxH3AdaLNOverlayField(
                overlay=h3_ref2va,
                start_block=20,
                end_block=30,
                include_final_layer=True,
            ),
        ),
        width=1344,
        height=768,
        num_frames="124",
        steps=50,
        seed=444,
        metadata=None,
    )
    h3_hybrid_md = h3_hybrid_node._build_metadata().root
    assert h3_hybrid_md["width"] == 1344
    assert h3_hybrid_md["height"] == 768
    assert h3_hybrid_md["num_frames"] == 124
    assert h3_hybrid_md["fps"] == 24
    assert h3_hybrid_md["model"] == h3_ref2va
    assert h3_hybrid_md["minimax_h3_hybrid_base_model"] == h3_fl2va
    assert h3_hybrid_md["minimax_h3_hybrid_start_block"] == 20
    assert len(h3_hybrid_md["loras"]) == 1
    assert issubclass(MiniMaxH3DenoiseOutput, MetadataOutput)

    h3_plain_node = MiniMaxH3DenoiseInvocation.model_construct(
        transformer=MiniMaxH3TransformerField(
            transformer=h3_fl2va,
            loras=[LoRAField(lora=lora_shared, weight=0.75)],
            adaln_overlay=None,
        ),
        width=1344,
        height=768,
        num_frames="124",
        steps=50,
        seed=444,
        metadata=None,
    )
    h3_plain_md = h3_plain_node._build_metadata().root
    assert h3_plain_md["model"] == h3_fl2va
    assert "minimax_h3_hybrid_base_model" not in h3_plain_md


@pytest.mark.parametrize(
    ("wrapper_cls", "parent_cls"),
    [
        (DenoiseLatentsMetaInvocation, DenoiseLatentsInvocation),
        (FluxDenoiseLatentsMetaInvocation, FluxDenoiseInvocation),
        (ZImageDenoiseMetaInvocation, ZImageDenoiseInvocation),
        (AnimaDenoiseMetaInvocation, AnimaDenoiseInvocation),
    ],
)
def test_deprecated_meta_wrappers_delegate_to_parent(
    wrapper_cls: type[BaseInvocation],
    parent_cls: type[BaseInvocation],
) -> None:
    assert wrapper_cls.UIConfig.classification == Classification.Deprecated
    assert issubclass(wrapper_cls, parent_cls)

    parent_metadata = MetadataField.model_validate({"steps": 12, "seed": 999, "app_version": __version__})
    parent_output = LatentsMetaOutput(
        latents=LatentsField(latents_name="latents-1", seed=999),
        width=512,
        height=512,
        metadata=parent_metadata,
    )
    node = wrapper_cls.model_construct(vae=None, qwen3_encoder=None, metadata=None)
    with patch.object(parent_cls, "invoke", return_value=parent_output) as mock_super:
        result = node.invoke(SimpleNamespace())
        mock_super.assert_called_once()

    assert isinstance(result, LatentsMetaOutput)
    assert result.metadata.root["steps"] == 12
    assert result.metadata.root["seed"] == 999


def test_anima_denoise_meta_merges_optional_vae_and_encoder() -> None:
    vae_id = _make_model_id(key="anima-vae", base=BaseModelType.Anima, model_type=ModelType.VAE)
    enc_id = _make_model_id(
        key="anima-qwen3",
        base=BaseModelType.Any,
        model_type=ModelType.Qwen3Encoder,
        submodel_type=SubModelType.TextEncoder,
    )
    tok_id = _make_model_id(
        key="anima-qwen3-tok",
        base=BaseModelType.Any,
        model_type=ModelType.Qwen3Encoder,
        submodel_type=SubModelType.Tokenizer,
    )
    parent_output = LatentsOutput(
        latents=LatentsField(latents_name="anima-lat", seed=None),
        width=1024,
        height=1024,
    )
    node = AnimaDenoiseMetaInvocation.model_construct(
        latents=None,
        noise=None,
        denoise_mask=None,
        add_noise=True,
        denoising_start=0.0,
        denoising_end=1.0,
        width=1024,
        height=1024,
        steps=25,
        guidance_scale=4.5,
        scheduler="euler",
        transformer=SimpleNamespace(transformer="anima-tf", loras=[]),
        control_lllite=None,
        metadata=MetadataField.model_validate({"positive_prompt": "anime hero"}),
        seed=555,
        vae=VAEField(vae=vae_id),
        qwen3_encoder=Qwen3EncoderField(tokenizer=tok_id, text_encoder=enc_id, loras=[]),
    )
    with patch.object(AnimaDenoiseInvocation, "invoke", return_value=parent_output):
        result = node.invoke(SimpleNamespace())

    assert result.metadata.root["positive_prompt"] == "anime hero"
    assert result.metadata.root["vae"] == vae_id
    assert result.metadata.root["qwen3_encoder"] == enc_id
    assert result.metadata.root["seed"] == 555


def test_denoise_invoke_preserves_width_height_for_5d_and_16x_latents() -> None:
    import torch

    tf_id = _make_model_id(key="tf-1")
    ctx = SimpleNamespace(tensors=SimpleNamespace(save=lambda tensor: "saved-latents"))

    # 1. QwenImageDenoiseInvocation returns 5D latents (B, C, 1, H // 8, W // 8)
    qwen_node = QwenImageDenoiseInvocation.model_construct(
        transformer=TransformerField(transformer=tf_id, loras=[]),
        cfg_scale=4.0,
        width=1360,
        height=768,
        steps=20,
        denoising_start=0.0,
        denoising_end=1.0,
        seed=88,
        shift=None,
        vae=None,
        control=None,
        metadata=None,
    )
    with patch.object(QwenImageDenoiseInvocation, "_run_diffusion", return_value=torch.zeros(1, 16, 1, 96, 170)):
        qwen_out = qwen_node.invoke(ctx)
    assert qwen_out.width == 1360
    assert qwen_out.height == 768
    assert qwen_out.metadata.root["width"] == 1360
    assert qwen_out.metadata.root["height"] == 768

    # 2. Krea2DenoiseInvocation returns 5D latents (B, C, 1, H // 8, W // 8)
    krea_node = Krea2DenoiseInvocation.model_construct(
        transformer=TransformerField(transformer=tf_id, loras=[]),
        cfg_scale=1.0,
        width=1280,
        height=720,
        steps=8,
        denoising_start=0.0,
        denoising_end=1.0,
        seed=77,
        shift=None,
        style_reference=None,
        metadata=None,
    )
    with patch.object(Krea2DenoiseInvocation, "_run_diffusion", return_value=torch.zeros(1, 16, 1, 90, 160)):
        krea_out = krea_node.invoke(ctx)
    assert krea_out.width == 1280
    assert krea_out.height == 720
    assert krea_out.metadata.root["width"] == 1280
    assert krea_out.metadata.root["height"] == 720

    # 3. WanDenoiseInvocation with TI2V-5B 16x spatial compression (1, 48, H // 16, W // 16)
    wan_node = WanDenoiseInvocation.model_construct(
        transformer=WanTransformerField(transformer=tf_id, transformer_low_noise=None, loras=[], loras_low_noise=[]),
        guidance_scale=4.0,
        guidance_scale_low_noise=0.0,
        denoising_start=0.0,
        denoising_end=1.0,
        add_noise=True,
        width=1280,
        height=704,
        steps=40,
        seed=111,
        metadata=None,
    )
    with patch.object(WanDenoiseInvocation, "_run_diffusion", return_value=torch.zeros(1, 48, 44, 80)):
        wan_out = wan_node.invoke(ctx)
    assert wan_out.width == 1280
    assert wan_out.height == 704
    assert wan_out.metadata.root["width"] == 1280
    assert wan_out.metadata.root["height"] == 704

    # 4. LatentsMetaOutput.build fallback without explicit width/height uses shape[-1] and shape[-2]
    fallback_out = LatentsMetaOutput.build(
        latents_name="lat-5d",
        latents=torch.zeros(1, 16, 1, 96, 170),
    )
    assert fallback_out.width == 1360
    assert fallback_out.height == 768


def test_core_labels_order_and_alignment_with_typed_literals() -> None:
    from typing import get_args

    all_labels = get_args(CORE_LABELS)
    typed_literals = (
        CORE_LABELS_STRING,
        CORE_LABELS_INTEGER,
        CORE_LABELS_FLOAT,
        CORE_LABELS_BOOL,
        CORE_LABELS_SCHEDULER,
        CORE_LABELS_MODEL,
        CORE_LABELS_VAE,
    )
    assert all_labels[0] == CUSTOM_LABEL
    typed_union: set[str] = set()
    for lit in typed_literals:
        args = get_args(lit)
        assert args[0] == CUSTOM_LABEL
        typed_union.update(args)

    # Every label in CORE_LABELS must be backed by at least one typed CORE_LABELS_* literal, and vice versa
    assert set(all_labels) == typed_union


def test_metadata_invocation_stamps_metadata_version() -> None:
    node = MetadataInvocation(items=MetadataItemField(label="positive_prompt", value="a castle"))
    out = node.invoke(SimpleNamespace())
    assert out.metadata.root["positive_prompt"] == "a castle"
    assert out.metadata.root["app_version"] == __version__
    assert out.metadata.root["metadata_version"] == CORE_METADATA_VERSION


def test_generation_mode_inference_and_upstream_preservation() -> None:
    tf_id = _make_model_id(key="tf-1")
    tf_field = TransformerField(transformer=tf_id, loras=[])
    lat = LatentsField(latents_name="lat-1")
    mask = DenoiseMaskField(mask_name="mask-1")

    # 1. SD1/SD2 vs SDXL on DenoiseLatentsInvocation and TiledMultiDiffusionDenoiseLatents
    sd1_unet = UNetField(
        unet=_make_model_id(key="sd1", base=BaseModelType.StableDiffusion1, submodel_type=SubModelType.UNet),
        scheduler=_make_model_id(
            key="sd1-s", base=BaseModelType.StableDiffusion1, submodel_type=SubModelType.Scheduler
        ),
        loras=[],
    )
    sdxl_unet = UNetField(
        unet=_make_model_id(key="sdxl", base=BaseModelType.StableDiffusionXL, submodel_type=SubModelType.UNet),
        scheduler=_make_model_id(
            key="sdxl-s", base=BaseModelType.StableDiffusionXL, submodel_type=SubModelType.Scheduler
        ),
        loras=[],
    )
    sd_node = DenoiseLatentsInvocation.model_construct(
        unet=sd1_unet,
        latents=None,
        denoise_mask=None,
        noise=None,
        cfg_scale=7.5,
        steps=20,
        scheduler="euler",
        denoising_start=0.0,
        denoising_end=1.0,
        cfg_rescale_multiplier=0.0,
        control=None,
        ip_adapter=None,
        t2i_adapter=None,
        hidiffusion=False,
        hidiffusion_raunet=True,
        hidiffusion_window_attn=True,
        hidiffusion_t1_ratio=0.4,
        hidiffusion_t2_ratio=0.8,
        metadata=None,
    )
    assert sd_node._build_metadata(512, 512).root["generation_mode"] == "txt2img"
    assert sd_node.model_copy(update={"latents": lat})._build_metadata(512, 512).root["generation_mode"] == "img2img"
    assert (
        sd_node.model_copy(update={"latents": lat, "denoise_mask": mask})
        ._build_metadata(512, 512)
        .root["generation_mode"]
        == "inpaint"
    )
    assert (
        sd_node.model_copy(update={"unet": sdxl_unet})._build_metadata(1024, 1024).root["generation_mode"]
        == "sdxl_txt2img"
    )
    assert (
        sd_node.model_copy(update={"unet": sdxl_unet, "latents": lat})
        ._build_metadata(1024, 1024)
        .root["generation_mode"]
        == "sdxl_img2img"
    )
    assert (
        sd_node.model_copy(update={"unet": sdxl_unet, "latents": lat, "denoise_mask": mask})
        ._build_metadata(1024, 1024)
        .root["generation_mode"]
        == "sdxl_inpaint"
    )

    tiled_node = TiledMultiDiffusionDenoiseLatents.model_construct(
        unet=sdxl_unet,
        latents=lat,
        noise=None,
        cfg_scale=7.0,
        steps=20,
        scheduler="euler",
        denoising_start=0.2,
        denoising_end=1.0,
        cfg_rescale_multiplier=0.0,
        tile_width=1024,
        tile_height=1024,
        tile_overlap=128,
        control=None,
        metadata=None,
    )
    assert tiled_node._build_metadata(2048, 2048).root["generation_mode"] == "sdxl_img2img"

    # 2. Upstream generation_mode (e.g. outpaint) is preserved when already set
    upstream_outpaint = MetadataField.model_validate({"generation_mode": "flux_outpaint"})
    flux_node = FluxDenoiseInvocation.model_construct(
        latents=lat,
        denoise_mask=mask,
        noise=None,
        add_noise=True,
        denoising_start=0.0,
        denoising_end=1.0,
        width=1024,
        height=1024,
        num_steps=20,
        scheduler="euler",
        guidance=4.0,
        cfg_scale=1.0,
        cfg_scale_start_step=0,
        cfg_scale_end_step=-1,
        transformer=tf_field,
        control_lora=None,
        control=None,
        controlnet_vae=None,
        ip_adapter=None,
        seed=1,
        dype_preset="off",
        dype_scale=None,
        dype_exponent=None,
        metadata=None,
    )
    assert flux_node._build_metadata().root["generation_mode"] == "flux_inpaint"
    assert (
        flux_node.model_copy(update={"metadata": upstream_outpaint})._build_metadata().root["generation_mode"]
        == "flux_outpaint"
    )

    # 3. Video & specialized generation_mode inference (Wan Video, LTX-2, MiniMax H3)
    wan_tf = WanTransformerField(transformer=tf_id, transformer_low_noise=None, loras=[], loras_low_noise=[])
    wan_vid = WanVideoDenoiseInvocation.model_construct(
        transformer=wan_tf,
        ref_image=None,
        guidance_scale=5.0,
        guidance_scale_low_noise=0.0,
        width=832,
        height=480,
        num_frames=81,
        steps=40,
        seed=1,
        metadata=None,
    )
    assert wan_vid._build_metadata().root["generation_mode"] == "wan_t2v"
    assert (
        wan_vid.model_copy(update={"ref_image": SimpleNamespace()})._build_metadata().root["generation_mode"]
        == "wan_i2v"
    )

    ltx_node = LTX2DenoiseInvocation.model_construct(
        transformer=LTX2TransformerField(transformer=tf_id, loras=[]),
        width=1248,
        height=704,
        num_frames=121,
        fps=24.0,
        steps=30,
        cfg_scale=3.0,
        audio_cfg_scale=7.0,
        stg_scale=1.0,
        modality_scale=3.0,
        guidance_rescale=0.7,
        schedule="dev",
        seed=1,
        latents=None,
        noise_scale=0.5,
        video_conditioning=None,
        keyframe_conditioning=None,
        audio_conditioning=None,
        full_video_conditioning=None,
        metadata=None,
    )
    assert ltx_node._build_metadata().root["generation_mode"] == "ltx2_t2v"
    assert (
        ltx_node.model_copy(update={"video_conditioning": SimpleNamespace()})._build_metadata().root["generation_mode"]
        == "ltx2_i2v"
    )
    assert (
        ltx_node.model_copy(update={"keyframe_conditioning": SimpleNamespace()})
        ._build_metadata()
        .root["generation_mode"]
        == "ltx2_lf2v"
    )
    assert (
        ltx_node.model_copy(
            update={"video_conditioning": SimpleNamespace(), "keyframe_conditioning": SimpleNamespace()}
        )
        ._build_metadata()
        .root["generation_mode"]
        == "ltx2_flf2v"
    )
    assert (
        ltx_node.model_copy(update={"audio_conditioning": SimpleNamespace()})._build_metadata().root["generation_mode"]
        == "ltx2_a2v"
    )
    assert (
        ltx_node.model_copy(update={"full_video_conditioning": SimpleNamespace()})
        ._build_metadata()
        .root["generation_mode"]
        == "ltx2_v2a"
    )

    h3_node = MiniMaxH3DenoiseInvocation.model_construct(
        transformer=MiniMaxH3TransformerField(transformer=tf_id, loras=[], adaln_overlay=None),
        width=1344,
        height=768,
        num_frames="124",
        steps=50,
        seed=1,
        frame_conditioning=None,
        reference_conditioning=None,
        metadata=None,
    )
    assert h3_node._build_metadata().root["generation_mode"] == "minimax_h3_t2v"
    assert (
        h3_node.model_copy(update={"frame_conditioning": SimpleNamespace()})._build_metadata().root["generation_mode"]
        == "minimax_h3_i2v"
    )
    assert (
        h3_node.model_copy(update={"reference_conditioning": SimpleNamespace()})
        ._build_metadata()
        .root["generation_mode"]
        == "minimax_h3_ref2v"
    )
    assert (
        h3_node.model_copy(update={"num_frames": "5"})._build_metadata().root["generation_mode"] == "minimax_h3_txt2img"
    )


def test_ref_images_serialization_and_metadata_to_adapters_multi_format() -> None:
    ipa_model = _make_model_id(
        key="ipa-key-1", name="ipa-sdxl", base=BaseModelType.StableDiffusionXL, model_type=ModelType.IPAdapter
    )
    enc_model = _make_model_id(
        key="enc-key-1",
        name="clip-vit-large-patch14",
        base=BaseModelType.Any,
        model_type=ModelType.CLIPVision,
    )
    ctrl_model = _make_model_id(
        key="ctrl-key-1",
        name="cn-canny",
        base=BaseModelType.StableDiffusion1,
        model_type=ModelType.ControlNet,
    )
    t2i_model = _make_model_id(
        key="t2i-key-1",
        name="t2i-canny",
        base=BaseModelType.StableDiffusion1,
        model_type=ModelType.T2IAdapter,
    )
    ipa_field = IPAdapterField(
        image=ImageField(image_name="ref-1.png"),
        ip_adapter_model=ipa_model,
        image_encoder_model=enc_model,
        weight=0.75,
        method="style_precise",
        begin_step_percent=0.1,
        end_step_percent=0.9,
    )

    from pydantic import ConfigDict

    from invokeai.backend.model_manager.configs.ip_adapter import IPAdapter_Checkpoint_Config_Base

    class _MockIPAConfig(IPAdapter_Checkpoint_Config_Base):
        model_config = ConfigDict(extra="allow")

    mock_ctx = SimpleNamespace(
        images=SimpleNamespace(get_dto=lambda name: SimpleNamespace(width=640, height=480)),
        models=SimpleNamespace(
            exists=lambda key: key in {"ipa-key-1", "enc-key-1", "ctrl-key-1", "t2i-key-1"},
            get_config=lambda key: {
                "ipa-key-1": _MockIPAConfig.model_validate(
                    {
                        "key": "ipa-key-1",
                        "hash": ipa_model.hash,
                        "name": ipa_model.name,
                        "base": ipa_model.base,
                        "type": ModelType.IPAdapter,
                    }
                ),
                "ctrl-key-1": SimpleNamespace(
                    key="ctrl-key-1",
                    hash=ctrl_model.hash,
                    name=ctrl_model.name,
                    base=ctrl_model.base,
                    type=ModelType.ControlNet,
                ),
                "t2i-key-1": SimpleNamespace(
                    key="t2i-key-1",
                    hash=t2i_model.hash,
                    name=t2i_model.name,
                    base=t2i_model.base,
                    type=ModelType.T2IAdapter,
                ),
            }[key],
            search_by_attrs=lambda **kwargs: [
                SimpleNamespace(
                    key="enc-key-1",
                    hash=enc_model.hash,
                    name=kwargs.get("name", enc_model.name),
                    base=BaseModelType.Any,
                    type=ModelType.CLIPVision,
                )
            ],
        ),
    )

    refs = serialize_ip_adapter_ref_images(ipa_field, context=mock_ctx, default_width=1024, default_height=1024)
    assert len(refs) == 1
    assert refs[0]["id"] == "ip_adapter_1"
    assert refs[0]["isEnabled"] is True
    cfg = refs[0]["config"]
    assert cfg["type"] == "ip_adapter"
    assert cfg["image"]["original"]["image"] == {"image_name": "ref-1.png", "width": 640, "height": 480}
    assert cfg["model"]["key"] == "ipa-key-1"
    assert cfg["model"]["type"] == "ip_adapter"
    assert cfg["clipVisionModel"] == "ViT-L"
    assert cfg["method"] == "style_precise"
    assert cfg["weight"] == 0.75
    assert cfg["beginEndStepPct"] == [0.1, 0.9]

    # Merging upstream ref_images with denoise ref_images deduplicates identical (type, image_name, model_key)
    flux_ref = {
        "id": "kontext_1",
        "isEnabled": True,
        "config": {
            "type": "flux_kontext",
            "image": {"original": {"image": {"image_name": "kontext.png", "width": 1024, "height": 1024}}},
            "model": {"key": "flux-kontext", "type": "main"},
        },
    }
    existing_md = MetadataField.model_validate({"ref_images": [flux_ref, refs[0]]})
    merged_md = build_denoise_metadata(existing_md, {"ref_images": refs})
    assert len(merged_md.root["ref_images"]) == 2

    # MetadataToIPAdaptersInvocation: Format 1 (v7 ref_images)
    md_to_ipa_v7 = MetadataToIPAdaptersInvocation.model_construct(
        metadata=MetadataField.model_validate({"ref_images": refs}),
        ip_adapter_list=None,
    )
    out_v7 = md_to_ipa_v7.invoke(mock_ctx)
    assert isinstance(out_v7.ip_adapter_list, list)
    assert len(out_v7.ip_adapter_list) == 1
    assert out_v7.ip_adapter_list[0].image.image_name == "ref-1.png"
    assert out_v7.ip_adapter_list[0].weight == 0.75
    assert out_v7.ip_adapter_list[0].method == "style_precise"
    assert out_v7.ip_adapter_list[0].begin_step_percent == 0.1
    assert out_v7.ip_adapter_list[0].end_step_percent == 0.9

    # MetadataToIPAdaptersInvocation: Format 2 (v5 canvas_v2_metadata.referenceImages.entities)
    v5_md = MetadataField.model_validate(
        {
            "canvas_v2_metadata": {
                "referenceImages": {
                    "entities": [
                        {
                            "id": "ref-v5",
                            "isEnabled": True,
                            "ipAdapter": {
                                "image": {"original": {"image": {"image_name": "v5-ref.png"}}},
                                "model": {"key": "ipa-key-1"},
                                "clipVisionModel": "ViT-L",
                                "method": "composition",
                                "weight": 0.6,
                                "beginEndStepPct": [0.2, 0.8],
                            },
                        }
                    ]
                }
            }
        }
    )
    out_v5 = MetadataToIPAdaptersInvocation.model_construct(metadata=v5_md, ip_adapter_list=None).invoke(mock_ctx)
    assert len(out_v5.ip_adapter_list) == 1
    assert out_v5.ip_adapter_list[0].image.image_name == "v5-ref.png"
    assert out_v5.ip_adapter_list[0].method == "composition"
    assert out_v5.ip_adapter_list[0].weight == 0.6

    # MetadataToControlnetsInvocation & MetadataToT2IAdaptersInvocation: safe defaults & v5 fallback
    cn_md = MetadataField.model_validate(
        {
            "controlnets": [
                {"control_model": {"key": "ctrl-key-1"}, "weight": 0.8},  # skipped (no image)
                {"image_name": "z-ctrl.png", "control_model": {"key": "ctrl-key-1"}, "weight": 0.55},
            ]
        }
    )
    out_cn = MetadataToControlnetsInvocation.model_construct(metadata=cn_md, control_list=None).invoke(mock_ctx)
    assert len(out_cn.control_list) == 1
    assert out_cn.control_list[0].image.image_name == "z-ctrl.png"
    assert out_cn.control_list[0].control_weight == 0.55
    assert out_cn.control_list[0].control_mode == "balanced"
    assert out_cn.control_list[0].resize_mode == "just_resize"

    t2i_v5_md = MetadataField.model_validate(
        {
            "canvas_v2_metadata": {
                "controlLayers": {
                    "entities": [
                        {
                            "id": "cl-1",
                            "isEnabled": True,
                            "controlAdapter": {
                                "type": "t2i_adapter",
                                "model": {"key": "t2i-key-1"},
                                "weight": 0.85,
                                "beginEndStepPct": [0.0, 0.75],
                            },
                            "objects": [{"type": "image", "image": {"image_name": "t2i-v5.png"}}],
                        }
                    ]
                }
            }
        }
    )
    out_t2i = MetadataToT2IAdaptersInvocation.model_construct(metadata=t2i_v5_md, t2i_adapter_list=None).invoke(
        mock_ctx
    )
    assert len(out_t2i.t2i_adapter_list) == 1
    assert out_t2i.t2i_adapter_list[0].image.image_name == "t2i-v5.png"
    assert out_t2i.t2i_adapter_list[0].weight == 0.85
    assert out_t2i.t2i_adapter_list[0].end_step_percent == 0.75

    # Missing/uninstalled models and non-SD models (e.g. FLUX ControlNet / FLUX IP-Adapter) are safely skipped
    flux_ipa_model = _make_model_id(
        key="flux-ipa-1", name="flux-ipa", base=BaseModelType.Flux, model_type=ModelType.IPAdapter
    )
    flux_ctrl_model = _make_model_id(
        key="flux-ctrl-1", name="flux-cn", base=BaseModelType.Flux, model_type=ModelType.ControlNet
    )
    flux_t2i_model = _make_model_id(
        key="flux-t2i-1", name="flux-t2i", base=BaseModelType.Flux, model_type=ModelType.T2IAdapter
    )
    mixed_ctx = SimpleNamespace(
        images=mock_ctx.images,
        models=SimpleNamespace(
            exists=lambda key: key
            in {"ipa-key-1", "enc-key-1", "ctrl-key-1", "t2i-key-1", "flux-ipa-1", "flux-ctrl-1", "flux-t2i-1"},
            get_config=lambda key: {
                "ipa-key-1": mock_ctx.models.get_config("ipa-key-1"),
                "ctrl-key-1": mock_ctx.models.get_config("ctrl-key-1"),
                "t2i-key-1": mock_ctx.models.get_config("t2i-key-1"),
                "flux-ipa-1": _MockIPAConfig.model_validate(
                    {
                        "key": "flux-ipa-1",
                        "hash": flux_ipa_model.hash,
                        "name": flux_ipa_model.name,
                        "base": BaseModelType.Flux,
                        "type": ModelType.IPAdapter,
                    }
                ),
                "flux-ctrl-1": SimpleNamespace(
                    key="flux-ctrl-1",
                    hash=flux_ctrl_model.hash,
                    name=flux_ctrl_model.name,
                    base=BaseModelType.Flux,
                    type=ModelType.ControlNet,
                ),
                "flux-t2i-1": SimpleNamespace(
                    key="flux-t2i-1",
                    hash=flux_t2i_model.hash,
                    name=flux_t2i_model.name,
                    base=BaseModelType.Flux,
                    type=ModelType.T2IAdapter,
                ),
            }[key],
            search_by_attrs=lambda **kwargs: (
                mock_ctx.models.search_by_attrs(**kwargs) if kwargs.get("type") == ModelType.CLIPVision else []
            ),
        ),
    )

    mixed_cn_md = MetadataField.model_validate(
        {
            "controlnets": [
                {"image": {"image_name": "missing.png"}, "control_model": {"key": "uninstalled-ctrl"}},
                {"image": {"image_name": "flux-cn.png"}, "control_model": {"key": "flux-ctrl-1"}},
                {"image": {"image_name": "sd-cn.png"}, "control_model": {"key": "ctrl-key-1"}},
            ]
        }
    )
    out_mixed_cn = MetadataToControlnetsInvocation.model_construct(metadata=mixed_cn_md, control_list=None).invoke(
        mixed_ctx
    )
    assert len(out_mixed_cn.control_list) == 1
    assert out_mixed_cn.control_list[0].image.image_name == "sd-cn.png"

    mixed_ipa_md = MetadataField.model_validate(
        {
            "ipAdapters": [
                {"image": {"image_name": "missing-ipa.png"}, "ip_adapter_model": {"key": "uninstalled-ipa"}},
                {"image": {"image_name": "flux-ipa.png"}, "ip_adapter_model": {"key": "flux-ipa-1"}},
                {"image": {"image_name": "sdxl-ipa.png"}, "ip_adapter_model": {"key": "ipa-key-1"}},
            ]
        }
    )
    out_mixed_ipa = MetadataToIPAdaptersInvocation.model_construct(metadata=mixed_ipa_md, ip_adapter_list=None).invoke(
        mixed_ctx
    )
    assert len(out_mixed_ipa.ip_adapter_list) == 1
    assert out_mixed_ipa.ip_adapter_list[0].image.image_name == "sdxl-ipa.png"

    mixed_t2i_md = MetadataField.model_validate(
        {
            "t2iAdapters": [
                {"image": {"image_name": "missing-t2i.png"}, "t2i_adapter_model": {"key": "uninstalled-t2i"}},
                {"image": {"image_name": "flux-t2i.png"}, "t2i_adapter_model": {"key": "flux-t2i-1"}},
                {"image": {"image_name": "sd-t2i.png"}, "t2i_adapter_model": {"key": "t2i-key-1"}},
            ]
        }
    )
    out_mixed_t2i = MetadataToT2IAdaptersInvocation.model_construct(
        metadata=mixed_t2i_md, t2i_adapter_list=None
    ).invoke(mixed_ctx)
    assert len(out_mixed_t2i.t2i_adapter_list) == 1
    assert out_mixed_t2i.t2i_adapter_list[0].image.image_name == "sd-t2i.png"
