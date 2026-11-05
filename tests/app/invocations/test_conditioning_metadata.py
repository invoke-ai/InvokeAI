from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from invokeai.app.invocations.fields import (
    AnimaConditioningField,
    CogView4ConditioningField,
    ConditioningField,
    ErnieImageConditioningField,
    Flux3ConditioningField,
    FluxConditioningField,
    Ideogram4ConditioningField,
    Krea2ConditioningField,
    LatentsField,
    LoRAMetadataField,
    LTX2ConditioningField,
    MetadataField,
    MiniMaxH3ConditioningField,
    ModelIdentifierField,
    MossImageConditioningField,
    QwenImage21ConditioningField,
    QwenImageConditioningField,
    SD3ConditioningField,
    TensorField,
    WanConditioningField,
    ZImageConditioningField,
)
from invokeai.app.invocations.krea2.krea2_conditioning_rebalance import (
    _NUM_TEXT_LAYERS,
    Krea2ConditioningRebalanceInvocation,
)
from invokeai.app.invocations.krea2.krea2_denoise import Krea2DenoiseInvocation
from invokeai.app.invocations.krea2.krea2_seed_variance import Krea2SeedVarianceInvocation
from invokeai.app.invocations.ltx2.ltx2_denoise import LTX2DenoiseInvocation
from invokeai.app.invocations.metadata import (
    extract_conditioning_metadata,
    extract_encoder_model,
    extract_loras_metadata,
    serialize_lora_metadata,
)
from invokeai.app.invocations.metadata_linked import MetadataToStringCollectionInvocation
from invokeai.app.invocations.model import (
    CLIPField,
    GlmEncoderField,
    LoRAField,
    LTX2TextEncoderField,
    LTX2TransformerField,
    MiniMaxH3TextEncoderField,
    Mistral3EncoderField,
    MistralEncoderField,
    Qwen3EncoderField,
    Qwen3VLEncoderField,
    Qwen35EncoderField,
    QwenVLEncoderField,
    T5EncoderField,
    TransformerField,
    UNetField,
    WanT5EncoderField,
)
from invokeai.app.invocations.sd.denoise_latents import DenoiseLatentsInvocation
from invokeai.app.invocations.text_encoder.anima_text_encoder import AnimaTextEncoderInvocation
from invokeai.app.invocations.text_encoder.cogview4_text_encoder import CogView4TextEncoderInvocation
from invokeai.app.invocations.text_encoder.compel import (
    CompelInvocation,
    SDXLCompelPromptInvocation,
    SDXLPromptInvocationBase,
    SDXLRefinerCompelPromptInvocation,
)
from invokeai.app.invocations.text_encoder.ernie_image_text_encoder import ErnieImageTextEncoderInvocation
from invokeai.app.invocations.text_encoder.flux2_dev_text_encoder import Flux2DevTextEncoderInvocation
from invokeai.app.invocations.text_encoder.flux2_klein_text_encoder import Flux2KleinTextEncoderInvocation
from invokeai.app.invocations.text_encoder.flux3_text_encoder import Flux3TextEncoderInvocation
from invokeai.app.invocations.text_encoder.flux_text_encoder import FluxTextEncoderInvocation
from invokeai.app.invocations.text_encoder.ideogram4_text_encoder import Ideogram4TextEncoderInvocation
from invokeai.app.invocations.text_encoder.krea2_text_encoder import Krea2TextEncoderInvocation
from invokeai.app.invocations.text_encoder.ltx2_text_encoder import LTX2TextEncoderInvocation
from invokeai.app.invocations.text_encoder.minimax_h3_text_encoder import MiniMaxH3TextEncoderInvocation
from invokeai.app.invocations.text_encoder.moss_image_text_encoder import MossImageTextEncoderInvocation
from invokeai.app.invocations.text_encoder.qwen_image_2_1_text_encoder import QwenImage21TextEncoderInvocation
from invokeai.app.invocations.text_encoder.qwen_image_text_encoder import QwenImageTextEncoderInvocation
from invokeai.app.invocations.text_encoder.sd3_text_encoder import Sd3TextEncoderInvocation
from invokeai.app.invocations.text_encoder.wan_text_encoder import WanTextEncoderInvocation
from invokeai.app.invocations.text_encoder.z_image_text_encoder import ZImageTextEncoderInvocation
from invokeai.app.invocations.z_image.z_image_denoise import ZImageDenoiseInvocation
from invokeai.app.invocations.z_image.z_image_seed_variance_enhancer import ZImageSeedVarianceEnhancerInvocation
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    Krea2ConditioningInfo,
    ZImageConditioningInfo,
)


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


def _mock_context(saved_names: list[str] | None = None) -> SimpleNamespace:
    storage: dict[str, ConditioningFieldData] = {}
    names = saved_names if saved_names is not None else []

    def _save(data: ConditioningFieldData) -> str:
        name = f"cond_{len(names)}"
        names.append(name)
        storage[name] = data
        return name

    def _load(name: str) -> ConditioningFieldData:
        return storage[name]

    @contextmanager
    def _model_on_device(*args, **kwargs):
        mock_text_encoder = MagicMock(spec=CLIPTextModel)
        mock_text_encoder.dtype = torch.float32
        yield ({}, mock_text_encoder)

    def _load_model(model_id):
        mock_loaded = MagicMock()
        mock_loaded.config = SimpleNamespace(base=BaseModelType.StableDiffusion1)
        mock_loaded.model = MagicMock(spec=ModelPatchRaw)
        mock_loaded.compute_device = torch.device("cpu")
        mock_loaded.model_in_ram.return_value = False
        mock_loaded.model_on_device = _model_on_device
        mock_loaded.__enter__.return_value = MagicMock(spec=CLIPTokenizer)
        return mock_loaded

    return SimpleNamespace(
        conditioning=SimpleNamespace(save=_save, load=_load, _storage=storage),
        util=SimpleNamespace(signal_progress=lambda *_: None),
        models=SimpleNamespace(load=_load_model),
        config=SimpleNamespace(get=lambda: SimpleNamespace(log_tokenization=False)),
    )


# -------------------------------------------------------------------------------------------------
# 1. Text Encoder Output Metadata
# -------------------------------------------------------------------------------------------------


def test_compel_output_metadata(monkeypatch) -> None:
    clip_id = _make_model_id(
        key="clip-sd1", name="CLIP SD1", base=BaseModelType.StableDiffusion1, model_type=ModelType.CLIPEmbed
    )
    lora_id = _make_model_id(key="lora-1", name="LoRA 1", model_type=ModelType.LoRA, submodel_type=None)
    lora_field = LoRAField(lora=lora_id, weight=0.85)

    node = CompelInvocation(
        prompt="a mystical forest at dusk",
        clip=CLIPField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            text_encoder=clip_id,
            skipped_layers=2,
            loras=[lora_field],
        ),
    )

    mock_compel_instance = MagicMock()
    mock_compel_instance.build_conditioning_tensor_for_conjunction.return_value = (torch.zeros(1, 77, 768), None)
    mock_compel_cls = MagicMock(return_value=mock_compel_instance)
    mock_compel_cls.parse_prompt_string.return_value = MagicMock()
    monkeypatch.setattr("invokeai.app.invocations.text_encoder.compel.Compel", mock_compel_cls)
    monkeypatch.setattr("invokeai.app.invocations.text_encoder.compel.generate_ti_list", lambda *_: [])

    @contextmanager
    def _dummy_ctx(*args, **kwargs):
        yield

    @contextmanager
    def _dummy_ti(*args, **kwargs):
        yield (MagicMock(), MagicMock())

    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.compel.LayerPatcher.apply_smart_model_patches", _dummy_ctx
    )
    monkeypatch.setattr("invokeai.app.invocations.text_encoder.compel.ModelPatcher.apply_clip_skip", _dummy_ctx)
    monkeypatch.setattr("invokeai.app.invocations.text_encoder.compel.ModelPatcher.apply_ti", _dummy_ti)

    ctx = _mock_context()
    output = node.invoke(ctx)

    cond = output.conditioning
    assert isinstance(cond, ConditioningField)
    assert cond.prompt == "a mystical forest at dusk"
    assert cond.skipped_layers == 2
    assert cond.loras is not None
    assert len(cond.loras) == 1
    assert cond.loras[0].model.key == "lora-1"
    assert cond.loras[0].weight == 0.85


def test_sdxl_compel_prompt_output_metadata(monkeypatch) -> None:
    clip1_id = _make_model_id(
        key="clip-l", name="CLIP-L", base=BaseModelType.StableDiffusionXL, model_type=ModelType.CLIPEmbed
    )
    clip2_id = _make_model_id(
        key="clip-g", name="CLIP-G", base=BaseModelType.StableDiffusionXL, model_type=ModelType.CLIPEmbed
    )
    lora1 = LoRAField(lora=_make_model_id(key="lora-sdxl-1", model_type=ModelType.LoRA, submodel_type=None), weight=0.6)
    lora2 = LoRAField(lora=_make_model_id(key="lora-sdxl-2", model_type=ModelType.LoRA, submodel_type=None), weight=0.4)

    node = SDXLCompelPromptInvocation(
        prompt="snowy mountain peak",
        style="cinematic photograph, sharp focus",
        clip=CLIPField(
            tokenizer=_make_model_id(key="tok1", model_type=ModelType.Main),
            text_encoder=clip1_id,
            skipped_layers=1,
            loras=[lora1],
        ),
        clip2=CLIPField(
            tokenizer=_make_model_id(key="tok2", model_type=ModelType.Main),
            text_encoder=clip2_id,
            skipped_layers=0,
            loras=[lora2],
        ),
    )

    monkeypatch.setattr(
        SDXLPromptInvocationBase,
        "run_clip_compel",
        lambda _self, _ctx, _clip, _prompt, _is_style, _empty_style, zero_on_empty=False: (
            torch.zeros(1, 77, 768),
            torch.zeros(1, 1280),
        ),
    )

    ctx = _mock_context()
    output = node.invoke(ctx)

    cond = output.conditioning
    assert isinstance(cond, ConditioningField)
    assert cond.prompt == "snowy mountain peak"
    assert cond.style_prompt == "cinematic photograph, sharp focus"
    assert cond.skipped_layers == 1
    assert cond.loras is not None
    assert len(cond.loras) == 2
    assert cond.loras[0].model.key == "lora-sdxl-1"
    assert cond.loras[0].weight == 0.6
    assert cond.loras[1].model.key == "lora-sdxl-2"
    assert cond.loras[1].weight == 0.4


def test_sdxl_refiner_compel_prompt_output_metadata(monkeypatch) -> None:
    clip2_id = _make_model_id(
        key="clip-g-refiner",
        name="CLIP-G Refiner",
        base=BaseModelType.StableDiffusionXLRefiner,
        model_type=ModelType.CLIPEmbed,
    )
    lora1 = LoRAField(
        lora=_make_model_id(key="lora-refiner", model_type=ModelType.LoRA, submodel_type=None), weight=0.7
    )

    node = SDXLRefinerCompelPromptInvocation(
        style="hyperrealistic portrait",
        aesthetic_score=7.8,
        clip2=CLIPField(
            tokenizer=_make_model_id(key="tok-ref", model_type=ModelType.Main),
            text_encoder=clip2_id,
            skipped_layers=0,
            loras=[lora1],
        ),
    )

    monkeypatch.setattr(
        SDXLPromptInvocationBase,
        "run_clip_compel",
        lambda _self, _ctx, _clip, _prompt, _is_style, _empty_style, zero_on_empty=False: (
            torch.zeros(1, 77, 1280),
            torch.zeros(1, 1280),
        ),
    )

    ctx = _mock_context()
    output = node.invoke(ctx)

    cond = output.conditioning
    assert isinstance(cond, ConditioningField)
    assert cond.style_prompt == "hyperrealistic portrait"
    assert cond.aesthetic_score == 7.8
    assert cond.loras is not None
    assert len(cond.loras) == 1
    assert cond.loras[0].model.key == "lora-refiner"
    assert cond.loras[0].weight == 0.7


def test_ideogram4_text_encoder_output_metadata(monkeypatch) -> None:
    encoder_id = _make_model_id(
        key="ideo-qwen", name="Qwen3 4B", base=BaseModelType.Ideogram4, model_type=ModelType.CLIPEmbed
    )
    lora1 = LoRAField(lora=_make_model_id(key="lora-ideo", model_type=ModelType.LoRA, submodel_type=None), weight=0.9)

    node = Ideogram4TextEncoderInvocation(
        prompt="sunset beach with neon clouds",
        qwen3_encoder=Qwen3EncoderField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            text_encoder=encoder_id,
            loras=[lora1],
        ),
    )

    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.ideogram4_text_encoder.peak_dequant_transient_bytes",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.ideogram4_text_encoder.encode_qwen3vl_prompt",
        lambda *args, **kwargs: torch.zeros(1, 128, 4096),
    )

    ctx = _mock_context()
    output = node.invoke(ctx)

    cond = output.conditioning
    assert isinstance(cond, Ideogram4ConditioningField)
    assert cond.prompt == "sunset beach with neon clouds"
    assert cond.qwen3_vl_encoder == encoder_id
    assert cond.loras is not None
    assert len(cond.loras) == 1
    assert cond.loras[0].model.key == "lora-ideo"
    assert cond.loras[0].weight == 0.9

    # Verify color_palette attribute preservation and extraction
    cond_with_palette = cond.model_copy(update={"color_palette": "coral, teal, amber"})
    extracted = extract_conditioning_metadata(cond_with_palette)
    assert extracted["positive_prompt"] == "sunset beach with neon clouds"
    assert extracted["ideogram4_color_palette"] == "coral, teal, amber"


def test_flux3_text_encoder_output_metadata(monkeypatch) -> None:
    encoder_id = _make_model_id(
        key="flux3-qwen-vl", name="Qwen3-VL 4B", base=BaseModelType.Flux3, model_type=ModelType.CLIPVision
    )
    lora1 = LoRAField(lora=_make_model_id(key="lora-flux3", model_type=ModelType.LoRA, submodel_type=None), weight=0.5)

    node = Flux3TextEncoderInvocation(
        prompt="isometric room filled with plants",
        qwen3_vl_encoder=Qwen3VLEncoderField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            text_encoder=encoder_id,
            loras=[lora1],
        ),
    )

    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.flux3_text_encoder.peak_dequant_transient_bytes",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.flux3_text_encoder.encode_flux3_prompt",
        lambda *args, **kwargs: torch.zeros(1, 256, 4096),
    )

    ctx = _mock_context()
    output = node.invoke(ctx)

    cond = output.conditioning
    assert isinstance(cond, Flux3ConditioningField)
    assert cond.prompt == "isometric room filled with plants"
    assert cond.qwen3_vl_encoder == encoder_id
    assert cond.loras is not None
    assert len(cond.loras) == 1
    assert cond.loras[0].model.key == "lora-flux3"

    # Verify flux3_caption attribute preservation and extraction
    cond_with_caption = cond.model_copy(update={"flux3_caption": "Detailed spatial layout of plants"})
    extracted = extract_conditioning_metadata(cond_with_caption)
    assert extracted["positive_prompt"] == "isometric room filled with plants"
    assert extracted["flux3_caption"] == "Detailed spatial layout of plants"


def test_moss_image_text_encoder_output_metadata(monkeypatch) -> None:
    encoder_id = _make_model_id(
        key="moss-qwen-35", name="Qwen3.5 9B", base=BaseModelType.MossImage, model_type=ModelType.CLIPEmbed
    )
    lora1 = LoRAField(lora=_make_model_id(key="lora-moss", model_type=ModelType.LoRA, submodel_type=None), weight=0.65)

    node = MossImageTextEncoderInvocation(
        prompt="cyberpunk alleyway",
        qwen3_5_encoder=Qwen35EncoderField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            text_encoder=encoder_id,
            loras=[lora1],
        ),
    )

    monkeypatch.setattr(MossImageTextEncoderInvocation, "_encode", lambda *args, **kwargs: torch.zeros(1, 128, 4096))

    ctx = _mock_context()
    output = node.invoke(ctx)

    cond = output.conditioning
    assert isinstance(cond, MossImageConditioningField)
    assert cond.prompt == "cyberpunk alleyway"
    assert cond.qwen3_5_encoder == encoder_id
    assert cond.loras is None

    # Verify layout_prompt and loras attribute preservation and extraction
    lora_meta = LoRAMetadataField(
        model=_make_model_id(key="lora-moss", model_type=ModelType.LoRA, submodel_type=None), weight=0.65
    )
    cond_with_layout = cond.model_copy(
        update={
            "layout_prompt": "left: neon ramen sign, right: vending machine",
            "loras": [lora_meta],
        }
    )
    extracted = extract_conditioning_metadata(cond_with_layout)
    assert extracted["positive_prompt"] == "cyberpunk alleyway"
    assert extracted["moss_image_layout_prompt"] == "left: neon ramen sign, right: vending machine"
    assert extracted["loras"] is not None
    assert extracted["loras"][0]["weight"] == 0.65
    assert extracted["loras"][0]["model"]["key"] == "lora-moss"


def test_qwen_image_text_encoder_output_metadata(monkeypatch) -> None:
    encoder_id = _make_model_id(
        key="qwen-vl", name="Qwen-VL", base=BaseModelType.QwenImage, model_type=ModelType.CLIPVision
    )
    source_id = _make_model_id(
        key="qwen-src", name="Qwen Base", base=BaseModelType.QwenImage, model_type=ModelType.Main
    )

    node = QwenImageTextEncoderInvocation(
        prompt="origami bird taking flight",
        quantization="int8",
        qwen_vl_encoder=QwenVLEncoderField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            text_encoder=encoder_id,
        ),
        component_source=source_id,
    )

    monkeypatch.setattr(
        QwenImageTextEncoderInvocation,
        "_encode",
        lambda *args, **kwargs: (torch.zeros(1, 100, 3584), torch.ones(1, 100)),
    )

    ctx = _mock_context()
    output = node.invoke(ctx)

    cond = output.conditioning
    assert isinstance(cond, QwenImageConditioningField)
    assert cond.prompt == "origami bird taking flight"
    assert cond.quantization == "int8"
    assert cond.qwen_vl_encoder == encoder_id

    # Verify component_source preservation and extraction
    cond_with_src = cond.model_copy(update={"component_source": source_id})
    assert cond_with_src.component_source == source_id
    extracted = extract_conditioning_metadata(cond_with_src)
    assert extracted["positive_prompt"] == "origami bird taking flight"
    assert extracted["qwen_image_quantization"] == "int8"
    assert extracted["qwen_vl_encoder"].key == "qwen-vl"
    assert extracted["component_source"].key == "qwen-src"


def test_all_text_encoder_invocations_output_metadata(monkeypatch) -> None:
    """Verify each remaining standalone text encoder populates prompt, encoders, and LoRAs."""
    ctx = _mock_context()

    # 1. AnimaTextEncoderInvocation
    qwen3_id = _make_model_id(key="anima-qwen3", base=BaseModelType.Anima, model_type=ModelType.CLIPEmbed)
    qwen35_id = _make_model_id(key="anima-qwen35", base=BaseModelType.Anima, model_type=ModelType.CLIPEmbed)
    lora_a = LoRAField(lora=_make_model_id(key="lora-a", model_type=ModelType.LoRA, submodel_type=None), weight=0.5)
    anima_node = AnimaTextEncoderInvocation(
        prompt="anima character",
        qwen3_encoder=Qwen3EncoderField(
            tokenizer=_make_model_id(key="tok1", model_type=ModelType.Main),
            text_encoder=qwen3_id,
            loras=[lora_a],
        ),
        qwen3_5_encoder=Qwen35EncoderField(
            tokenizer=_make_model_id(key="tok2", model_type=ModelType.Main),
            text_encoder=qwen35_id,
        ),
    )
    monkeypatch.setattr(
        AnimaTextEncoderInvocation,
        "_encode_prompt",
        lambda *args, **kwargs: (torch.zeros(1, 64, 4096), torch.zeros(1, 10, dtype=torch.long), None),
    )
    monkeypatch.setattr(AnimaTextEncoderInvocation, "_encode_qwen3_5", lambda *args, **kwargs: (None, None))
    anima_out = anima_node.invoke(ctx)
    assert isinstance(anima_out.conditioning, AnimaConditioningField)
    assert anima_out.conditioning.prompt == "anima character"
    assert anima_out.conditioning.qwen3_encoder == qwen3_id
    assert anima_out.conditioning.qwen3_5_encoder == qwen35_id
    assert anima_out.conditioning.loras is not None

    # 2. CogView4TextEncoderInvocation
    glm_id = _make_model_id(key="cog-glm", base=BaseModelType.CogView4, model_type=ModelType.CLIPEmbed)
    cog_node = CogView4TextEncoderInvocation(
        prompt="cogview landscape",
        glm_encoder=GlmEncoderField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            text_encoder=glm_id,
        ),
    )
    monkeypatch.setattr(CogView4TextEncoderInvocation, "_glm_encode", lambda *args, **kwargs: torch.zeros(1, 64, 4096))
    cog_out = cog_node.invoke(ctx)
    assert isinstance(cog_out.conditioning, CogView4ConditioningField)
    assert cog_out.conditioning.prompt == "cogview landscape"
    assert cog_out.conditioning.glm_encoder == glm_id

    # 3. ErnieImageTextEncoderInvocation
    mistral_id = _make_model_id(key="ernie-mistral", base=BaseModelType.ErnieImage, model_type=ModelType.CLIPEmbed)
    ernie_node = ErnieImageTextEncoderInvocation(
        prompt="ernie ancient pagoda",
        text_encoder=Mistral3EncoderField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            text_encoder=mistral_id,
        ),
    )
    monkeypatch.setattr(
        ErnieImageTextEncoderInvocation, "_encode_prompt", lambda *args, **kwargs: torch.zeros(1, 64, 4096)
    )
    ernie_out = ernie_node.invoke(ctx)
    assert isinstance(ernie_out.conditioning, ErnieImageConditioningField)
    assert ernie_out.conditioning.prompt == "ernie ancient pagoda"
    assert ernie_out.conditioning.mistral_encoder == mistral_id

    # 4. FluxTextEncoderInvocation
    t5_id = _make_model_id(key="flux-t5", base=BaseModelType.Flux, model_type=ModelType.T5Encoder)
    clip_id = _make_model_id(key="flux-clip", base=BaseModelType.Flux, model_type=ModelType.CLIPEmbed)
    lora_flux = LoRAField(
        lora=_make_model_id(key="lora-flux", model_type=ModelType.LoRA, submodel_type=None), weight=0.8
    )
    flux_node = FluxTextEncoderInvocation(
        prompt="flux futuristic car",
        t5_encoder=T5EncoderField(
            tokenizer=_make_model_id(key="tok1", model_type=ModelType.Main),
            text_encoder=t5_id,
            loras=[lora_flux],
        ),
        clip=CLIPField(
            tokenizer=_make_model_id(key="tok2", model_type=ModelType.Main),
            text_encoder=clip_id,
            skipped_layers=0,
            loras=[],
        ),
    )
    monkeypatch.setattr(FluxTextEncoderInvocation, "_clip_encode", lambda *args, **kwargs: torch.zeros(1, 768))
    monkeypatch.setattr(FluxTextEncoderInvocation, "_t5_encode", lambda *args, **kwargs: torch.zeros(1, 256, 4096))
    flux_out = flux_node.invoke(ctx)
    assert isinstance(flux_out.conditioning, FluxConditioningField)
    assert flux_out.conditioning.prompt == "flux futuristic car"
    assert flux_out.conditioning.t5_encoder == t5_id
    assert flux_out.conditioning.clip_embed_model == clip_id
    assert flux_out.conditioning.loras is not None

    # 5. Flux2DevTextEncoderInvocation
    f2_dev_id = _make_model_id(key="f2-dev-mistral", base=BaseModelType.Flux2, model_type=ModelType.CLIPEmbed)
    f2_dev_node = Flux2DevTextEncoderInvocation(
        prompt="flux2 dev portrait",
        mistral_encoder=MistralEncoderField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            text_encoder=f2_dev_id,
            loras=[lora_flux],
        ),
    )
    monkeypatch.setattr(
        Flux2DevTextEncoderInvocation, "_encode_prompt", lambda *args, **kwargs: torch.zeros(1, 128, 4096)
    )
    f2_dev_out = f2_dev_node.invoke(ctx)
    assert isinstance(f2_dev_out.conditioning, FluxConditioningField)
    assert f2_dev_out.conditioning.prompt == "flux2 dev portrait"
    assert f2_dev_out.conditioning.mistral_encoder == f2_dev_id
    assert f2_dev_out.conditioning.loras is not None

    # 6. Flux2KleinTextEncoderInvocation
    f2_klein_id = _make_model_id(key="f2-klein-qwen", base=BaseModelType.Flux2, model_type=ModelType.CLIPEmbed)
    f2_klein_node = Flux2KleinTextEncoderInvocation(
        prompt="flux2 klein scene",
        qwen3_encoder=Qwen3EncoderField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            text_encoder=f2_klein_id,
            loras=[lora_flux],
        ),
    )
    monkeypatch.setattr(
        Flux2KleinTextEncoderInvocation,
        "_encode_prompt",
        lambda *args, **kwargs: (torch.zeros(1, 128, 4096), torch.ones(1, 128, dtype=torch.bool)),
    )
    f2_klein_out = f2_klein_node.invoke(ctx)
    assert isinstance(f2_klein_out.conditioning, FluxConditioningField)
    assert f2_klein_out.conditioning.prompt == "flux2 klein scene"
    assert f2_klein_out.conditioning.qwen3_encoder == f2_klein_id

    # 7. LTX2TextEncoderInvocation
    ltx2_enc_id = _make_model_id(key="ltx2-enc", base=BaseModelType.LTX2, model_type=ModelType.T5Encoder)
    ltx2_conn_id = _make_model_id(key="ltx2-conn", base=BaseModelType.LTX2, model_type=ModelType.Main)
    ltx2_src_id = _make_model_id(key="ltx2-src", base=BaseModelType.LTX2, model_type=ModelType.Main)
    ltx2_node = LTX2TextEncoderInvocation(
        prompt="ltx2 cinematic river",
        negative_prompt="blurry",
        text_encoder=LTX2TextEncoderField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            text_encoder=ltx2_enc_id,
            connectors=ltx2_conn_id,
        ),
        ltx2_component_source=ltx2_src_id,
    )
    monkeypatch.setattr(
        LTX2TextEncoderInvocation,
        "_encode_with_tower",
        lambda *args, **kwargs: [torch.zeros(1, 128, 2048), torch.zeros(1, 32, 2048)],
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.ltx2_text_encoder.estimate_connector_working_memory",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.ltx2_text_encoder.apply_connectors",
        lambda *args, **kwargs: MagicMock(),
    )
    ltx2_out = ltx2_node.invoke(ctx)
    assert isinstance(ltx2_out.conditioning, LTX2ConditioningField)
    assert ltx2_out.conditioning.prompt == "ltx2 cinematic river"
    assert ltx2_out.conditioning.ltx2_text_encoder_model == ltx2_enc_id
    ltx2_with_src = ltx2_out.conditioning.model_copy(update={"ltx2_component_source": ltx2_src_id})
    assert ltx2_with_src.ltx2_component_source == ltx2_src_id
    assert ltx2_out.negative_conditioning is not None
    assert ltx2_out.negative_conditioning.prompt == "blurry"

    # 8. MiniMaxH3TextEncoderInvocation
    h3_enc_id = _make_model_id(key="h3-enc", base=BaseModelType.MiniMaxH3, model_type=ModelType.T5Encoder)
    h3_proc_id = _make_model_id(key="h3-proc", base=BaseModelType.MiniMaxH3, model_type=ModelType.Main)
    h3_src_id = _make_model_id(key="h3-src", base=BaseModelType.MiniMaxH3, model_type=ModelType.Main)
    h3_node = MiniMaxH3TextEncoderInvocation(
        prompt="minimax animated dragon",
        text_encoder=MiniMaxH3TextEncoderField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            processor=h3_proc_id,
            text_encoder=h3_enc_id,
        ),
        minimax_h3_component_source=h3_src_id,
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.minimax_h3_text_encoder.encode_prompt",
        lambda *args, **kwargs: (torch.zeros(1, 128, 4096), torch.zeros(1, 128, dtype=torch.long)),
    )
    h3_out = h3_node.invoke(ctx)
    assert isinstance(h3_out.conditioning, MiniMaxH3ConditioningField)
    assert h3_out.conditioning.prompt == "minimax animated dragon"
    assert h3_out.conditioning.minimax_h3_text_encoder_model == h3_enc_id
    h3_with_src = h3_out.conditioning.model_copy(update={"minimax_h3_component_source": h3_src_id})
    assert h3_with_src.minimax_h3_component_source == h3_src_id

    # 9. QwenImage21TextEncoderInvocation
    qwen21_id = _make_model_id(key="qwen21-enc", base=BaseModelType.QwenImage21, model_type=ModelType.CLIPVision)
    qwen21_node = QwenImage21TextEncoderInvocation(
        prompt="qwen 2.1 high tech city",
        qwen3_vl_encoder=Qwen3VLEncoderField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            text_encoder=qwen21_id,
        ),
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.qwen_image_2_1_text_encoder.sequence_length",
        lambda *args, **kwargs: 10,
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.qwen_image_2_1_text_encoder.peak_dequant_transient_bytes",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.qwen_image_2_1_text_encoder.working_memory_bytes",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.qwen_image_2_1_text_encoder.encode_prompt",
        lambda *args, **kwargs: SimpleNamespace(
            embeds=torch.zeros(1, 64, 2048), image_pad_mask=None, reference_grids=[]
        ),
    )
    qwen21_out = qwen21_node.invoke(ctx)
    assert isinstance(qwen21_out.conditioning, QwenImage21ConditioningField)
    assert qwen21_out.conditioning.prompt == "qwen 2.1 high tech city"
    assert qwen21_out.conditioning.qwen3_vl_encoder == qwen21_id

    # 10. Sd3TextEncoderInvocation
    sd3_t5_id = _make_model_id(key="sd3-t5", base=BaseModelType.StableDiffusion3, model_type=ModelType.T5Encoder)
    sd3_l_id = _make_model_id(key="sd3-l", base=BaseModelType.StableDiffusion3, model_type=ModelType.CLIPEmbed)
    sd3_g_id = _make_model_id(key="sd3-g", base=BaseModelType.StableDiffusion3, model_type=ModelType.CLIPEmbed)
    sd3_node = Sd3TextEncoderInvocation(
        prompt="sd3 crystal palace",
        clip_l=CLIPField(
            tokenizer=_make_model_id(key="tok1", model_type=ModelType.Main),
            text_encoder=sd3_l_id,
            skipped_layers=0,
            loras=[],
        ),
        clip_g=CLIPField(
            tokenizer=_make_model_id(key="tok2", model_type=ModelType.Main),
            text_encoder=sd3_g_id,
            skipped_layers=0,
            loras=[],
        ),
        t5_encoder=T5EncoderField(
            tokenizer=_make_model_id(key="tok3", model_type=ModelType.Main),
            text_encoder=sd3_t5_id,
            loras=[lora_flux],
        ),
    )
    monkeypatch.setattr(
        Sd3TextEncoderInvocation, "_clip_encode", lambda *args, **kwargs: (torch.zeros(1, 77, 768), torch.zeros(1, 768))
    )
    monkeypatch.setattr(Sd3TextEncoderInvocation, "_t5_encode", lambda *args, **kwargs: torch.zeros(1, 256, 4096))
    sd3_out = sd3_node.invoke(ctx)
    assert isinstance(sd3_out.conditioning, SD3ConditioningField)
    assert sd3_out.conditioning.prompt == "sd3 crystal palace"
    assert sd3_out.conditioning.t5_encoder == sd3_t5_id
    assert sd3_out.conditioning.clip_l_model == sd3_l_id
    assert sd3_out.conditioning.clip_g_model == sd3_g_id
    assert sd3_out.conditioning.loras is not None

    # 11. WanTextEncoderInvocation
    wan_t5_id = _make_model_id(key="wan-t5", base=BaseModelType.Wan, model_type=ModelType.WanT5Encoder)
    wan_src_id = _make_model_id(key="wan-src", base=BaseModelType.Wan, model_type=ModelType.Main)
    wan_node = WanTextEncoderInvocation(
        prompt="wan flowing river",
        wan_t5_encoder=WanT5EncoderField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            text_encoder=wan_t5_id,
        ),
        wan_component_source=wan_src_id,
    )
    monkeypatch.setattr(WanTextEncoderInvocation, "_encode", lambda *args, **kwargs: (torch.zeros(1, 128, 4096), None))
    wan_out = wan_node.invoke(ctx)
    assert isinstance(wan_out.conditioning, WanConditioningField)
    assert wan_out.conditioning.prompt == "wan flowing river"
    assert wan_out.conditioning.wan_t5_encoder_model == wan_t5_id
    wan_with_src = wan_out.conditioning.model_copy(update={"wan_component_source": wan_src_id})
    assert wan_with_src.wan_component_source == wan_src_id

    # 12. ZImageTextEncoderInvocation
    z_qwen_id = _make_model_id(key="z-qwen", base=BaseModelType.ZImage, model_type=ModelType.CLIPEmbed)
    z_node = ZImageTextEncoderInvocation(
        prompt="z-image celestial warrior",
        qwen3_encoder=Qwen3EncoderField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            text_encoder=z_qwen_id,
            loras=[lora_a],
        ),
    )
    monkeypatch.setattr(
        ZImageTextEncoderInvocation, "_encode_prompt", lambda *args, **kwargs: torch.zeros(1, 128, 2560)
    )
    z_out = z_node.invoke(ctx)
    assert isinstance(z_out.conditioning, ZImageConditioningField)
    assert z_out.conditioning.prompt == "z-image celestial warrior"
    assert z_out.conditioning.qwen3_encoder == z_qwen_id
    assert z_out.conditioning.loras is not None


# -------------------------------------------------------------------------------------------------
# 2. Modifier Chaining & Preservation
# -------------------------------------------------------------------------------------------------


def test_krea2_conditioning_rebalance_active_and_bypass() -> None:
    encoder_id = _make_model_id(key="krea-vlm", base=BaseModelType.Krea2, model_type=ModelType.CLIPVision)
    lora_meta = LoRAMetadataField(
        model=_make_model_id(key="lora-krea", model_type=ModelType.LoRA, submodel_type=None), weight=0.75
    )
    regional_mask = TensorField(tensor_name="test_mask")

    ctx = _mock_context()
    dummy_data = ConditioningFieldData(
        conditionings=[
            Krea2ConditioningInfo(
                prompt_embeds=torch.zeros(1, 10, _NUM_TEXT_LAYERS, 64),
                prompt_embeds_mask=torch.ones(1, 10, dtype=torch.bool),
            )
        ]
    )
    initial_name = ctx.conditioning.save(dummy_data)

    upstream_cond = Krea2ConditioningField(
        conditioning_name=initial_name,
        mask=regional_mask,
        prompt="cyberpunk street scene",
        qwen3_vl_encoder=encoder_id,
        loras=[lora_meta],
    )

    # Active rebalance
    per_layer = ",".join(str(float(i) * 0.1 + 0.5) for i in range(_NUM_TEXT_LAYERS))
    active_node = Krea2ConditioningRebalanceInvocation(
        conditioning=upstream_cond,
        multiplier=1.2,
        per_layer_weights=per_layer,
    )
    active_out = active_node.invoke(ctx).conditioning
    assert active_out.conditioning_name != initial_name
    assert active_out.prompt == "cyberpunk street scene"
    assert active_out.qwen3_vl_encoder == encoder_id
    assert active_out.loras == [lora_meta]
    assert active_out.mask == regional_mask
    assert active_out.rebalance_enabled is True
    assert active_out.rebalance_multiplier == 1.2
    assert len(active_out.rebalance_weights or []) == _NUM_TEXT_LAYERS

    # Bypass (multiplier = 0.0)
    bypass_node = Krea2ConditioningRebalanceInvocation(
        conditioning=upstream_cond,
        multiplier=0.0,
        per_layer_weights=per_layer,
    )
    bypass_out = bypass_node.invoke(ctx).conditioning
    assert bypass_out.prompt == "cyberpunk street scene"
    assert bypass_out.qwen3_vl_encoder == encoder_id
    assert bypass_out.loras == [lora_meta]
    assert bypass_out.mask == regional_mask
    assert bypass_out.rebalance_enabled is False
    assert bypass_out.rebalance_multiplier == 0.0


def test_krea2_seed_variance_active_and_bypass() -> None:
    encoder_id = _make_model_id(key="krea-vlm", base=BaseModelType.Krea2, model_type=ModelType.CLIPVision)
    ctx = _mock_context()
    dummy_data = ConditioningFieldData(
        conditionings=[
            Krea2ConditioningInfo(
                prompt_embeds=torch.randn(1, 10, _NUM_TEXT_LAYERS, 64),
                prompt_embeds_mask=torch.ones(1, 10, dtype=torch.bool),
            )
        ]
    )
    initial_name = ctx.conditioning.save(dummy_data)

    upstream_cond = Krea2ConditioningField(
        conditioning_name=initial_name,
        prompt="steampunk mechanical owl",
        qwen3_vl_encoder=encoder_id,
        rebalance_enabled=True,
        rebalance_multiplier=1.1,
    )

    # Active variance
    active_node = Krea2SeedVarianceInvocation(
        conditioning=upstream_cond,
        seed=42,
        strength=0.35,
        randomize_percent=25.0,
    )
    active_out = active_node.invoke(ctx).conditioning
    assert active_out.conditioning_name != initial_name
    assert active_out.prompt == "steampunk mechanical owl"
    assert active_out.qwen3_vl_encoder == encoder_id
    assert active_out.rebalance_enabled is True
    assert active_out.rebalance_multiplier == 1.1
    assert active_out.seed_variance_enabled is True
    assert active_out.seed_variance_strength == 0.35
    assert active_out.seed_variance_randomize_percent == 25

    # Bypass (strength = 0.0)
    bypass_node = Krea2SeedVarianceInvocation(
        conditioning=upstream_cond,
        seed=42,
        strength=0.0,
        randomize_percent=25.0,
    )
    bypass_out = bypass_node.invoke(ctx).conditioning
    assert bypass_out.conditioning_name == initial_name  # preserved without re-saving
    assert bypass_out.prompt == "steampunk mechanical owl"
    assert bypass_out.qwen3_vl_encoder == encoder_id
    assert bypass_out.seed_variance_enabled is False
    assert bypass_out.seed_variance_strength == 0.0
    assert bypass_out.seed_variance_randomize_percent == 25


def test_z_image_seed_variance_enhancer_active_and_bypass() -> None:
    encoder_id = _make_model_id(key="z-qwen", base=BaseModelType.ZImage, model_type=ModelType.CLIPEmbed)
    ctx = _mock_context()
    dummy_data = ConditioningFieldData(conditionings=[ZImageConditioningInfo(prompt_embeds=torch.randn(1, 20, 2560))])
    initial_name = ctx.conditioning.save(dummy_data)

    upstream_cond = ZImageConditioningField(
        conditioning_name=initial_name,
        prompt="glowing crystals inside cavern",
        qwen3_encoder=encoder_id,
    )

    # Active enhancer
    active_node = ZImageSeedVarianceEnhancerInvocation(
        conditioning=upstream_cond,
        seed=99,
        strength=0.45,
        randomize_percent=30.0,
    )
    active_out = active_node.invoke(ctx).conditioning
    assert active_out.conditioning_name != initial_name
    assert active_out.prompt == "glowing crystals inside cavern"
    assert active_out.qwen3_encoder == encoder_id
    assert active_out.seed_variance_enabled is True
    assert active_out.seed_variance_strength == 0.45
    assert active_out.seed_variance_randomize_percent == 30

    # Bypass (strength = 0.0)
    bypass_node = ZImageSeedVarianceEnhancerInvocation(
        conditioning=upstream_cond,
        seed=99,
        strength=0.0,
        randomize_percent=30.0,
    )
    bypass_out = bypass_node.invoke(ctx).conditioning
    assert bypass_out.conditioning_name == initial_name
    assert bypass_out.prompt == "glowing crystals inside cavern"
    assert bypass_out.qwen3_encoder == encoder_id
    assert bypass_out.seed_variance_enabled is False
    assert bypass_out.seed_variance_strength == 0.0
    assert bypass_out.seed_variance_randomize_percent == 30


def test_krea2_end_to_end_chain(monkeypatch) -> None:
    """End-to-End Chain: Krea2TextEncoder -> Rebalance -> Seed Variance -> Krea2DenoiseInvocation._build_metadata()"""
    encoder_id = _make_model_id(key="krea-vlm", base=BaseModelType.Krea2, model_type=ModelType.CLIPVision)
    lora1 = LoRAField(lora=_make_model_id(key="lora-krea-1", model_type=ModelType.LoRA, submodel_type=None), weight=0.8)
    tf_id = _make_model_id(key="krea-transformer", base=BaseModelType.Krea2, model_type=ModelType.Main)

    ctx = _mock_context()

    # Step 1: Text Encoder
    text_encoder_node = Krea2TextEncoderInvocation(
        prompt="majestic cyber phoenix rising",
        qwen3_vl_encoder=Qwen3VLEncoderField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            text_encoder=encoder_id,
            loras=[lora1],
        ),
    )
    monkeypatch.setattr(
        Krea2TextEncoderInvocation,
        "_encode",
        lambda *args, **kwargs: (torch.randn(1, 16, _NUM_TEXT_LAYERS, 64), torch.ones(1, 16, dtype=torch.bool)),
    )
    encoded_cond = text_encoder_node.invoke(ctx).conditioning

    # Step 2: Rebalance
    per_layer = ",".join("1.0" for _ in range(_NUM_TEXT_LAYERS))
    rebalance_node = Krea2ConditioningRebalanceInvocation(
        conditioning=encoded_cond,
        multiplier=1.15,
        per_layer_weights=per_layer,
    )
    rebalanced_cond = rebalance_node.invoke(ctx).conditioning

    # Step 3: Seed Variance
    variance_node = Krea2SeedVarianceInvocation(
        conditioning=rebalanced_cond,
        seed=1234,
        strength=0.2,
        randomize_percent=15.0,
    )
    final_cond = variance_node.invoke(ctx).conditioning

    # Step 4: Denoise _build_metadata()
    denoise_node = Krea2DenoiseInvocation.model_construct(
        transformer=TransformerField(transformer=tf_id, loras=[]),
        positive_conditioning=final_cond,
        steps=25,
        cfg_scale=4.5,
    )

    md = denoise_node._build_metadata().root
    assert md["positive_prompt"] == "majestic cyber phoenix rising"
    assert md["qwen3_vl_encoder"] == encoder_id
    assert len(md["loras"]) == 1
    assert md["loras"][0]["model"]["key"] == "lora-krea-1"
    assert md["krea2_rebalance_enabled"] is True
    assert md["krea2_rebalance_multiplier"] == 1.15
    assert len(md["krea2_rebalance_weights"]) == _NUM_TEXT_LAYERS
    assert md["krea2_seed_variance_enabled"] is True
    assert md["krea2_seed_variance_strength"] == 0.2
    assert md["krea2_seed_variance_randomize_percent"] == 15


def test_z_image_end_to_end_chain(monkeypatch) -> None:
    """End-to-End Chain for Z-Image: ZImageTextEncoder -> Seed Variance -> ZImageDenoiseInvocation._build_metadata()"""
    encoder_id = _make_model_id(key="z-qwen", base=BaseModelType.ZImage, model_type=ModelType.CLIPEmbed)
    lora1 = LoRAField(lora=_make_model_id(key="lora-z-1", model_type=ModelType.LoRA, submodel_type=None), weight=0.7)
    tf_id = _make_model_id(key="z-transformer", base=BaseModelType.ZImage, model_type=ModelType.Main)

    ctx = _mock_context()

    # Step 1: Text Encoder
    text_encoder_node = ZImageTextEncoderInvocation(
        prompt="mythical water spirit in lotus pond",
        qwen3_encoder=Qwen3EncoderField(
            tokenizer=_make_model_id(key="tok", model_type=ModelType.Main),
            text_encoder=encoder_id,
            loras=[lora1],
        ),
    )
    monkeypatch.setattr(ZImageTextEncoderInvocation, "_encode_prompt", lambda *args, **kwargs: torch.randn(1, 24, 2560))
    encoded_cond = text_encoder_node.invoke(ctx).conditioning

    # Step 2: Seed Variance Enhancer
    variance_node = ZImageSeedVarianceEnhancerInvocation(
        conditioning=encoded_cond,
        seed=5678,
        strength=0.3,
        randomize_percent=20.0,
    )
    final_cond = variance_node.invoke(ctx).conditioning

    # Step 3: Denoise _build_metadata()
    denoise_node = ZImageDenoiseInvocation.model_construct(
        transformer=TransformerField(transformer=tf_id, loras=[]),
        positive_conditioning=final_cond,
        steps=30,
        guidance=3.5,
    )

    md = denoise_node._build_metadata().root
    assert md["positive_prompt"] == "mythical water spirit in lotus pond"
    assert md["qwen3_encoder"] == encoder_id
    assert len(md["loras"]) == 1
    assert md["loras"][0]["model"]["key"] == "lora-z-1"
    assert md["z_image_seed_variance_enabled"] is True
    assert md["z_image_seed_variance_strength"] == 0.3
    assert md["z_image_seed_variance_randomize_percent"] == 20


# -------------------------------------------------------------------------------------------------
# 3. Conditioning Collections & Multi-Prompt Handling
# -------------------------------------------------------------------------------------------------


def test_single_conditioning_positive_and_negative() -> None:
    pos = ConditioningField(conditioning_name="c1", prompt="a solitary lighthouse")
    neg = ConditioningField(conditioning_name="c2", prompt="foggy, blurry")

    extracted = extract_conditioning_metadata(positive_conditioning=pos, negative_conditioning=neg)

    assert extracted["positive_prompt"] == "a solitary lighthouse"
    assert extracted["negative_prompt"] == "foggy, blurry"
    assert "positive_prompts" not in extracted
    assert "negative_prompts" not in extracted
    assert "regions" not in extracted


def test_conditioning_collection_unmasked_primary_and_raw_lists() -> None:
    mask1 = TensorField(tensor_name="m1")
    mask2 = TensorField(tensor_name="m2")

    pos1 = ConditioningField(conditioning_name="c1", prompt="left side: cherry blossoms", mask=mask1)
    pos2 = ConditioningField(conditioning_name="c2", prompt="central park panorama", mask=None)  # unmasked primary!
    pos3 = ConditioningField(conditioning_name="c3", prompt="right side: pond with koi", mask=mask2)

    neg1 = ConditioningField(conditioning_name="n1", prompt="overexposed")
    neg2 = ConditioningField(conditioning_name="n2", prompt="watermark, text")

    extracted = extract_conditioning_metadata(
        positive_conditioning=[pos1, pos2, pos3], negative_conditioning=[neg1, neg2]
    )

    # Primary unmasked prompt
    assert extracted["positive_prompt"] == "central park panorama"
    # Exact raw lists
    assert extracted["positive_prompts"] == [
        "left side: cherry blossoms",
        "central park panorama",
        "right side: pond with koi",
    ]
    assert extracted["negative_prompt"] == "overexposed"
    assert extracted["negative_prompts"] == ["overexposed", "watermark, text"]
    # v7 structured regions
    assert extracted["regions"] == [
        {"positive_prompt": "left side: cherry blossoms"},
        {"positive_prompt": "central park panorama"},
        {"positive_prompt": "right side: pond with koi"},
    ]


def test_conditioning_collection_all_masked_falls_back_to_first() -> None:
    mask1 = TensorField(tensor_name="m1")
    mask2 = TensorField(tensor_name="m2")

    pos1 = ConditioningField(conditioning_name="c1", prompt="upper region: aurora borealis", mask=mask1)
    pos2 = ConditioningField(conditioning_name="c2", prompt="lower region: snowy cabin", mask=mask2)

    extracted = extract_conditioning_metadata(positive_conditioning=[pos1, pos2])

    assert extracted["positive_prompt"] == "upper region: aurora borealis"
    assert extracted["positive_prompts"] == [
        "upper region: aurora borealis",
        "lower region: snowy cabin",
    ]
    assert len(extracted["regions"]) == 2


def test_prompts_containing_pipe_delimiter_are_not_mangled() -> None:
    pipe_prompt = "portrait of an astronaut | hyperrealistic | 35mm lens, f/1.8 | octane render"
    pos = ConditioningField(conditioning_name="c1", prompt=pipe_prompt)

    extracted = extract_conditioning_metadata(positive_conditioning=pos)
    assert extracted["positive_prompt"] == pipe_prompt

    # In a collection
    pos2 = ConditioningField(conditioning_name="c2", prompt="background: deep space nebula | stars | dark void")
    extracted_col = extract_conditioning_metadata(positive_conditioning=[pos, pos2])
    assert extracted_col["positive_prompts"] == [
        pipe_prompt,
        "background: deep space nebula | stars | dark void",
    ]


# -------------------------------------------------------------------------------------------------
# 4. LoRA Merging & Deduplication
# -------------------------------------------------------------------------------------------------


def test_lora_merging_text_encoder_and_transformer() -> None:
    lora_a = _make_model_id(key="lora-a", name="LoRA A", model_type=ModelType.LoRA, submodel_type=None)
    lora_b = _make_model_id(key="lora-b", name="LoRA B", model_type=ModelType.LoRA, submodel_type=None)
    tf_id = _make_model_id(key="transformer", base=BaseModelType.Flux, model_type=ModelType.Main)

    pos_cond = FluxConditioningField(
        conditioning_name="c1",
        prompt="a futuristic hovercar",
        loras=[LoRAMetadataField(model=lora_a, weight=0.75)],
    )

    denoise_node = Krea2DenoiseInvocation.model_construct(
        transformer=TransformerField(transformer=tf_id, loras=[LoRAField(lora=lora_b, weight=0.5)]),
        positive_conditioning=pos_cond,
    )

    md = denoise_node._build_metadata().root
    assert "loras" in md
    assert len(md["loras"]) == 2
    keys = [item["model"]["key"] for item in md["loras"]]
    weights = [item["weight"] for item in md["loras"]]
    assert "lora-a" in keys
    assert "lora-b" in keys
    assert 0.75 in weights
    assert 0.5 in weights


def test_lora_deduplication_identical_text_encoder_and_transformer() -> None:
    lora_a = _make_model_id(key="lora-shared", name="Shared LoRA", model_type=ModelType.LoRA, submodel_type=None)
    tf_id = _make_model_id(key="transformer", base=BaseModelType.Flux, model_type=ModelType.Main)

    pos_cond = FluxConditioningField(
        conditioning_name="c1",
        prompt="a beautiful landscape",
        loras=[LoRAMetadataField(model=lora_a, weight=0.8)],
    )

    denoise_node = Krea2DenoiseInvocation.model_construct(
        transformer=TransformerField(transformer=tf_id, loras=[LoRAField(lora=lora_a, weight=0.8)]),
        positive_conditioning=pos_cond,
    )

    md = denoise_node._build_metadata().root
    assert "loras" in md
    assert len(md["loras"]) == 1
    assert md["loras"][0]["model"]["key"] == "lora-shared"
    assert md["loras"][0]["weight"] == 0.8


def test_lora_same_key_different_weight_preserved() -> None:
    lora_a = _make_model_id(key="lora-scaled", name="Scaled LoRA", model_type=ModelType.LoRA, submodel_type=None)

    list1 = [LoRAField(lora=lora_a, weight=0.8)]
    list2 = [LoRAMetadataField(model=lora_a, weight=0.4)]

    merged = serialize_lora_metadata(list1, list2)
    assert len(merged) == 2
    assert merged[0]["weight"] == 0.8
    assert merged[1]["weight"] == 0.4


# -------------------------------------------------------------------------------------------------
# 5. Workflow Extraction Invocations (MetadataToStringCollectionInvocation)
# -------------------------------------------------------------------------------------------------


def test_metadata_to_string_collection_positive_prompts() -> None:
    ctx = MagicMock()
    md = MetadataField.model_validate(
        {
            "positive_prompts": ["a medieval castle", "dragon in background", "dramatic sunset"],
        }
    )

    node = MetadataToStringCollectionInvocation(
        metadata=md,
        label="positive_prompts",
    )
    output = node.invoke(ctx)
    assert output.collection == ["a medieval castle", "dragon in background", "dramatic sunset"]


def test_metadata_to_string_collection_fallback_single_prompt() -> None:
    ctx = MagicMock()
    md = MetadataField.model_validate(
        {
            "positive_prompt": "a solitary wanderer",
        }
    )

    # When querying positive_prompts but only positive_prompt is present:
    node = MetadataToStringCollectionInvocation(
        metadata=md,
        label="positive_prompt",
    )
    output = node.invoke(ctx)
    # String automatically wrapped in single-element list
    assert output.collection == ["a solitary wanderer"]


def test_metadata_to_string_collection_default_fallback_when_absent() -> None:
    ctx = MagicMock()
    md = MetadataField.model_validate({})

    node = MetadataToStringCollectionInvocation(
        metadata=md,
        label="positive_prompts",
        default_value=["default fallback prompt"],
    )
    output = node.invoke(ctx)
    assert output.collection == ["default fallback prompt"]


# -------------------------------------------------------------------------------------------------
# 6. Multi-Pass Denoise Metadata
# -------------------------------------------------------------------------------------------------


def test_sdxl_refiner_multi_pass_denoise_metadata() -> None:
    base_model = _make_model_id(
        key="sdxl-base", name="SDXL Base", base=BaseModelType.StableDiffusionXL, model_type=ModelType.Main
    )
    refiner_model = _make_model_id(
        key="sdxl-refiner",
        name="SDXL Refiner",
        base=BaseModelType.StableDiffusionXLRefiner,
        model_type=ModelType.Main,
    )
    scheduler_model = _make_model_id(key="euler", model_type=ModelType.Main)

    base_md = MetadataField.model_validate(
        {
            "model": base_model.model_dump(),
            "cfg_scale": 7.5,
            "steps": 40,
            "scheduler": "euler",
            "positive_prompt": "an ancient Greek temple on cliff",
            "negative_prompt": "modern buildings",
        }
    )

    refiner_pos_cond = ConditioningField(
        conditioning_name="ref_pos",
        prompt="an ancient Greek temple on cliff",
        aesthetic_score=7.0,
    )
    refiner_neg_cond = ConditioningField(
        conditioning_name="ref_neg",
        prompt="modern buildings",
        aesthetic_score=2.5,
    )

    refiner_node = DenoiseLatentsInvocation.model_construct(
        unet=UNetField(unet=refiner_model, scheduler=scheduler_model, scheduler_options=None, loras=[]),
        positive_conditioning=refiner_pos_cond,
        negative_conditioning=refiner_neg_cond,
        steps=20,
        cfg_scale=6.0,
        scheduler="dpmpp_2m",
        denoising_start=0.7,
        denoising_end=1.0,
        metadata=base_md,
    )

    md = refiner_node._build_metadata(width=1024, height=1024).root

    # Base values must be strictly preserved
    assert md["model"] == base_model.model_dump()
    assert md["cfg_scale"] == 7.5
    assert md["steps"] == 40
    assert md["scheduler"] == "euler"
    assert md["positive_prompt"] == "an ancient Greek temple on cliff"
    assert md["negative_prompt"] == "modern buildings"

    # Refiner pass recorded without collision
    assert md["refiner_model"] == refiner_model
    assert md["refiner_cfg_scale"] == 6.0
    assert md["refiner_steps"] == 20
    assert md["refiner_scheduler"] == "dpmpp_2m"
    assert md["refiner_start"] == 0.7
    assert md["refiner_positive_aesthetic_score"] == 7.0
    assert md["refiner_negative_aesthetic_score"] == 2.5


def test_ltx2_two_stage_multi_pass_metadata() -> None:
    ltx_tf = _make_model_id(key="ltx2-tf", base=BaseModelType.LTX2)
    stage1_md = MetadataField.model_validate(
        {
            "width": 768,
            "height": 512,
            "num_frames": 97,
            "fps": 24,
            "positive_prompt": "a stormy ocean with giant waves",
        }
    )

    pos_cond = SimpleNamespace(prompt="a stormy ocean with giant waves")

    # Second pass with larger dimensions and input latents connected -> two-stage
    two_stage_node = LTX2DenoiseInvocation.model_construct(
        transformer=LTX2TransformerField(transformer=ltx_tf, loras=[]),
        positive_conditioning=pos_cond,
        width=1280,
        height=768,
        num_frames=97,
        fps=24.0,
        steps=30,
        cfg_scale=3.0,
        latents=LatentsField(latents_name="stage1_latents"),
        metadata=stage1_md,
    )

    md = two_stage_node._build_metadata().root
    assert md["width"] == 1280
    assert md["height"] == 768
    assert md["ltx2_two_stage"] is True
    assert md["ltx2_base_width"] == 768
    assert md["ltx2_base_height"] == 512
    assert md["positive_prompt"] == "a stormy ocean with giant waves"

    # Single pass (same dimensions) -> ltx2_two_stage not set
    single_pass_node = LTX2DenoiseInvocation.model_construct(
        transformer=LTX2TransformerField(transformer=ltx_tf, loras=[]),
        positive_conditioning=pos_cond,
        width=768,
        height=512,
        num_frames=97,
        fps=24.0,
        steps=30,
        cfg_scale=3.0,
        latents=LatentsField(latents_name="stage1_latents"),
        metadata=stage1_md,
    )

    single_md = single_pass_node._build_metadata().root
    assert "ltx2_two_stage" not in single_md


def test_text_encoders_tolerate_mock_inputs_without_pydantic_validation_error(monkeypatch) -> None:
    """Construct text encoders with MagicMock() inputs and verify invoke() outputs fields with None
    rather than raising ValidationError."""
    context = MagicMock()
    context.conditioning.save.return_value = "cond-id"

    # 1. WanTextEncoderInvocation
    wan_node = WanTextEncoderInvocation.model_construct(prompt="test", wan_t5_encoder=MagicMock())
    monkeypatch.setattr(wan_node, "_encode", lambda *args, **kwargs: (torch.zeros(1, 2, 4), None))
    out = wan_node.invoke(context)
    assert out.conditioning.wan_t5_encoder_model is None

    # 2. Sd3TextEncoderInvocation
    sd3_node = Sd3TextEncoderInvocation.model_construct(
        prompt="test", clip_l=MagicMock(), clip_g=MagicMock(), t5_encoder=MagicMock()
    )
    monkeypatch.setattr(sd3_node, "_clip_encode", lambda *args, **kwargs: (torch.zeros(1, 2, 4), torch.zeros(1, 4)))
    monkeypatch.setattr(sd3_node, "_t5_encode", lambda *args, **kwargs: torch.zeros(1, 2, 4))
    out = sd3_node.invoke(context)
    assert out.conditioning.clip_l_model is None
    assert out.conditioning.clip_g_model is None
    assert out.conditioning.t5_encoder is None
    assert out.conditioning.loras is None

    # 3. FluxTextEncoderInvocation
    flux_node = FluxTextEncoderInvocation.model_construct(prompt="test", clip=MagicMock(), t5_encoder=MagicMock())
    monkeypatch.setattr(flux_node, "_clip_encode", lambda *args, **kwargs: torch.zeros(1, 4))
    monkeypatch.setattr(flux_node, "_t5_encode", lambda *args, **kwargs: torch.zeros(1, 2, 4))
    out = flux_node.invoke(context)
    assert out.conditioning.clip_embed_model is None
    assert out.conditioning.t5_encoder is None
    assert out.conditioning.loras is None

    # 4. Flux2KleinTextEncoderInvocation
    klein_node = Flux2KleinTextEncoderInvocation.model_construct(prompt="test", qwen3_encoder=MagicMock())
    monkeypatch.setattr(klein_node, "_encode_prompt", lambda *args, **kwargs: (torch.zeros(1, 2, 4), torch.zeros(1, 4)))
    out = klein_node.invoke(context)
    assert out.conditioning.qwen3_encoder is None
    assert out.conditioning.loras is None

    # 5. Flux2DevTextEncoderInvocation
    dev_node = Flux2DevTextEncoderInvocation.model_construct(prompt="test", mistral_encoder=MagicMock())
    monkeypatch.setattr(dev_node, "_encode_prompt", lambda *args, **kwargs: torch.zeros(1, 2, 4))
    out = dev_node.invoke(context)
    assert out.conditioning.mistral_encoder is None
    assert out.conditioning.loras is None

    # 6. ZImageTextEncoderInvocation
    z_node = ZImageTextEncoderInvocation.model_construct(prompt="test", qwen3_encoder=MagicMock())
    monkeypatch.setattr(z_node, "_encode_prompt", lambda *args, **kwargs: torch.zeros(1, 2, 4))
    out = z_node.invoke(context)
    assert out.conditioning.qwen3_encoder is None
    assert out.conditioning.loras is None

    # 7. Krea2TextEncoderInvocation
    krea_node = Krea2TextEncoderInvocation.model_construct(
        prompt="test", qwen3_vl_encoder=MagicMock(), reference_images=[]
    )
    monkeypatch.setattr(krea_node, "_encode", lambda *args, **kwargs: (torch.zeros(1, 2, 4), None))
    out = krea_node.invoke(context)
    assert out.conditioning.qwen3_vl_encoder is None
    assert out.conditioning.loras is None

    # 8. Ideogram4TextEncoderInvocation
    import invokeai.app.invocations.text_encoder.ideogram4_text_encoder as ideogram4_module

    ideo_node = Ideogram4TextEncoderInvocation.model_construct(prompt="test", qwen3_encoder=MagicMock())
    context.models.load.return_value.model_on_device.return_value.__enter__.return_value = (None, MagicMock())
    monkeypatch.setattr(ideogram4_module, "encode_qwen3vl_prompt", lambda *args, **kwargs: torch.zeros(1, 2, 4))
    out = ideo_node.invoke(context)
    assert out.conditioning.qwen3_vl_encoder is None
    assert out.conditioning.loras is None

    # 9. ErnieImageTextEncoderInvocation
    ernie_node = ErnieImageTextEncoderInvocation.model_construct(prompt="test", text_encoder=MagicMock())
    monkeypatch.setattr(ernie_node, "_encode_prompt", lambda *args, **kwargs: torch.zeros(1, 2, 4))
    out = ernie_node.invoke(context)
    assert out.conditioning.mistral_encoder is None

    # 10. MossImageTextEncoderInvocation
    moss_node = MossImageTextEncoderInvocation.model_construct(prompt="test", qwen3_5_encoder=MagicMock())
    monkeypatch.setattr(moss_node, "_encode", lambda *args, **kwargs: torch.zeros(1, 2, 4))
    out = moss_node.invoke(context)
    assert out.conditioning.qwen3_5_encoder is None
    assert out.conditioning.loras is None


def test_text_encoders_tolerate_mock_lora_strings() -> None:
    """Verify mock LoRA objects with string models don't crash."""
    # 1. extract_loras_metadata ignores string lora models and mock objects without ModelIdentifierField
    mock_strings = [
        SimpleNamespace(lora="mock-lora-string", weight=0.75),
        MagicMock(lora=MagicMock(), weight=1.0),
    ]
    assert extract_loras_metadata(mock_strings) is None

    # 2. Valid ModelIdentifierField and dict-based keys are preserved
    valid_id = _make_model_id(key="lora-valid-1")
    valid_dict = _make_model_id(key="lora-valid-2").model_dump()
    mixed = [
        SimpleNamespace(lora=valid_id, weight=0.6),
        SimpleNamespace(lora="skip-me-string", weight=1.0),
        SimpleNamespace(lora=valid_dict, weight=0.9),
        MagicMock(lora="another-string", weight=0.5),
    ]
    extracted = extract_loras_metadata(mixed)
    assert extracted is not None
    assert len(extracted) == 2
    assert extracted[0].model == valid_id
    assert extracted[0].weight == 0.6
    assert extracted[1].model.key == "lora-valid-2"
    assert extracted[1].weight == 0.9

    # 3. Invocation level test with mock string LoRA
    context = MagicMock()
    context.conditioning.save.return_value = "cond-id"
    flux_node = FluxTextEncoderInvocation.model_construct(
        prompt="test",
        clip=MagicMock(loras=[SimpleNamespace(lora="mock-lora-string", weight=1.0)]),
        t5_encoder=MagicMock(loras=[]),
    )
    flux_node._clip_encode = MagicMock(return_value=torch.zeros(1, 4))
    flux_node._t5_encode = MagicMock(return_value=torch.zeros(1, 2, 4))
    out = flux_node.invoke(context)
    assert out.conditioning.loras is None

    # 4. extract_encoder_model tolerates mocks and strings
    assert extract_encoder_model(None) is None
    assert extract_encoder_model("string-not-model") is None
    assert extract_encoder_model(MagicMock()) is None
    assert extract_encoder_model(SimpleNamespace(text_encoder=MagicMock())) is None
    assert extract_encoder_model(valid_id) == valid_id
    assert extract_encoder_model(SimpleNamespace(text_encoder=valid_id)) == valid_id
    assert extract_encoder_model({"key": "invalid-incomplete-dict"}) is None
    dict_model = extract_encoder_model(_make_model_id(key="test-key").model_dump())
    assert dict_model is not None
    assert dict_model.key == "test-key"


def test_build_denoise_metadata_clears_stale_prompt_collections_and_regions() -> None:
    """Verify build_denoise_metadata clears stale multi-prompt and regional metadata
    when transitioning to single-prompt / non-regional workflows."""
    from invokeai.app.invocations.metadata import build_denoise_metadata

    existing = MetadataField.model_validate(
        {
            "positive_prompt": "old single",
            "positive_prompts": ["old 1", "old 2"],
            "negative_prompts": ["neg 1"],
            "regions": [{"positive_prompt": "old 1"}],
        }
    )
    updates = {"positive_prompt": "new single", "negative_prompt": "new neg"}
    result = build_denoise_metadata(existing, updates)

    assert result.root["positive_prompt"] == "new single"
    assert result.root["negative_prompt"] == "new neg"
    assert "positive_prompts" not in result.root
    assert "negative_prompts" not in result.root
    assert "regions" not in result.root

    # When updates explicitly provides positive_prompts, it is preserved
    updates_with_prompts = {
        "positive_prompt": "new single",
        "positive_prompts": ["new 1", "new 2"],
    }
    result_multi = build_denoise_metadata(existing, updates_with_prompts)
    assert result_multi.root["positive_prompts"] == ["new 1", "new 2"]


def test_unwrap_metadata_value_stage2_core_model_labels() -> None:
    """Finding A: Verify unwrap_metadata_value unwraps submodel wrappers with LoRAs
    when passed labels in CORE_MODEL_LABELS, returning the inner ModelIdentifierField."""
    from invokeai.app.invocations.metadata import CORE_MODEL_LABELS, CUSTOM_LABEL, unwrap_metadata_value

    model_id = _make_model_id(key="encoder-submodel-key")
    lora_id = _make_model_id(key="lora-test-key", model_type=ModelType.LoRA)
    lora_entry = LoRAField(lora=lora_id, weight=0.75)

    test_labels = (
        "wan_t5_encoder_model",
        "qwen_vl_encoder",
        "qwen_image_qwen_vl_encoder",
        "ltx2_text_encoder_model",
        "minimax_h3_text_encoder_model",
        "glm_encoder",
    )

    for label in test_labels:
        assert label in CORE_MODEL_LABELS

    # 1. WanT5EncoderField with LoRAs
    wan_field = WanT5EncoderField.model_construct(
        tokenizer=model_id,
        text_encoder=model_id,
        loras=[lora_entry],
    )
    unwrapped_wan = unwrap_metadata_value(wan_field, label="wan_t5_encoder_model")
    assert isinstance(unwrapped_wan, ModelIdentifierField)
    assert unwrapped_wan.key == "encoder-submodel-key"

    custom_wan = unwrap_metadata_value(wan_field, label=CUSTOM_LABEL)
    assert isinstance(custom_wan, WanT5EncoderField)
    assert len(custom_wan.loras) == 1

    # 2. QwenVLEncoderField with LoRAs
    qwen_field = QwenVLEncoderField.model_construct(
        tokenizer=model_id,
        text_encoder=model_id,
        loras=[lora_entry],
    )
    for q_label in ("qwen_vl_encoder", "qwen_image_qwen_vl_encoder"):
        unwrapped_qwen = unwrap_metadata_value(qwen_field, label=q_label)
        assert isinstance(unwrapped_qwen, ModelIdentifierField)
        assert unwrapped_qwen.key == "encoder-submodel-key"

    custom_qwen = unwrap_metadata_value(qwen_field, label="custom_qwen_label")
    assert isinstance(custom_qwen, QwenVLEncoderField)
    assert len(custom_qwen.loras) == 1

    # 3. LTX2TextEncoderField (or dict with loras)
    ltx2_dict = {
        "text_encoder": model_id.model_dump(),
        "tokenizer": model_id.model_dump(),
        "loras": [{"lora": lora_id.model_dump(), "weight": 0.5}],
    }
    unwrapped_ltx2 = unwrap_metadata_value(ltx2_dict, label="ltx2_text_encoder_model")
    assert isinstance(unwrapped_ltx2, dict)
    assert unwrapped_ltx2["key"] == "encoder-submodel-key"

    custom_ltx2 = unwrap_metadata_value(ltx2_dict, label=CUSTOM_LABEL)
    assert "loras" in custom_ltx2

    # 4. MiniMaxH3TextEncoderField and GlmEncoderField
    minimax_field = MiniMaxH3TextEncoderField.model_construct(
        tokenizer=model_id,
        processor=model_id,
        text_encoder=model_id,
    )
    unwrapped_minimax = unwrap_metadata_value(minimax_field, label="minimax_h3_text_encoder_model")
    assert isinstance(unwrapped_minimax, ModelIdentifierField)
    assert unwrapped_minimax.key == "encoder-submodel-key"

    glm_field = GlmEncoderField.model_construct(
        tokenizer=model_id,
        text_encoder=model_id,
    )
    unwrapped_glm = unwrap_metadata_value(glm_field, label="glm_encoder")
    assert isinstance(unwrapped_glm, ModelIdentifierField)
    assert unwrapped_glm.key == "encoder-submodel-key"
