from types import SimpleNamespace
from typing import get_args
from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter, ValidationError

from invokeai.app.invocations.fields import ImageField, MetadataField, VideoField
from invokeai.app.invocations.metadata import CORE_METADATA_VERSION, CoreMetadataInvocation, unwrap_metadata_value
from invokeai.app.invocations.metadata_constants import (
    CORE_LABELS,
    CORE_LABELS_BOOL,
    CORE_LABELS_FLOAT,
    CORE_LABELS_INTEGER,
    CORE_LABELS_MAIN_MODEL,
    CORE_LABELS_MODEL,
    CORE_LABELS_SCHEDULER,
    CORE_LABELS_SDXL_MODEL,
    CORE_LABELS_STRING,
    CORE_LABELS_VAE,
    CORE_METADATA_REGISTRY,
    CORE_MODEL_LABELS,
    CUSTOM_LABEL,
    MetadataType,
)
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, TransformerField
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.version import __version__


def test_core_metadata_stamps_the_record_version_beside_the_app_version() -> None:
    output = CoreMetadataInvocation(id="m", positive_prompt="a fox", seed=3).invoke(MagicMock())

    record = output.metadata.root
    assert record["metadata_version"] == CORE_METADATA_VERSION
    assert record["app_version"] == __version__
    assert record["positive_prompt"] == "a fox"
    assert "negative_prompt" not in record  # unset fields are omitted, not written as null


def test_core_metadata_records_the_video_profile_fields_by_their_canonical_names() -> None:
    encoder = ModelIdentifierField(key="k", hash="h", name="UMT5", base="any", type="wan_t5_encoder")
    output = CoreMetadataInvocation(
        id="m",
        generation_mode="wan_extend_video",
        num_frames=81,
        fps=16,
        wan_guidance_scale_low_noise=3.5,
        wan_t5_encoder_model=encoder,
        source_video=VideoField(video_name="clip.mp4"),
        source_video_start_frame=0,
        source_video_end_frame=40,
        last_frame_image=ImageField(image_name="end.png"),
        minimax_h3_hybrid_start_block=12,
    ).invoke(MagicMock())

    record = output.metadata.root
    assert record["fps"] == 16
    assert record["wan_guidance_scale_low_noise"] == 3.5
    assert record["wan_t5_encoder_model"]["hash"] == "h"
    assert record["source_video"] == {"video_name": "clip.mp4"}
    assert (record["source_video_start_frame"], record["source_video_end_frame"]) == (0, 40)
    assert record["minimax_h3_hybrid_start_block"] == 12


def test_core_metadata_records_the_image_profile_fields_by_their_canonical_names() -> None:
    source = ModelIdentifierField(key="s", hash="h", name="FLUX.2 dev", base="flux2", type="main")
    output = CoreMetadataInvocation(
        id="m", mistral_source=source, vae_precision="fp16", color_compensation=True
    ).invoke(MagicMock())

    record = output.metadata.root
    assert record["mistral_source"] == {"key": "s", "hash": "h", "name": "FLUX.2 dev", "base": "flux2", "type": "main"}
    assert record["vae_precision"] == "fp16"
    assert record["color_compensation"] is True
    # Recall reads an explicit off as "the encode ran without it"; it must survive the record's None filter.
    off = CoreMetadataInvocation(id="m", color_compensation=False).invoke(MagicMock()).metadata.root
    assert off["color_compensation"] is False
    with pytest.raises(ValidationError):
        CoreMetadataInvocation(id="m", vae_precision="bf16")


def test_core_metadata_registry_uniqueness_and_enum_validation() -> None:
    labels = [label for label, _ in CORE_METADATA_REGISTRY]
    assert len(labels) == len(set(labels)), "Duplicate keys found in CORE_METADATA_REGISTRY"
    assert CUSTOM_LABEL not in labels, "CUSTOM_LABEL should not be in CORE_METADATA_REGISTRY directly"

    for label, types in CORE_METADATA_REGISTRY:
        type_tuple = types if isinstance(types, tuple) else (types,)
        assert len(type_tuple) > 0, f"Empty types for label {label}"
        for t in type_tuple:
            assert isinstance(t, MetadataType), f"Invalid MetadataType {t!r} for label {label}"


def test_core_metadata_custom_label_at_index_zero_across_all_literals() -> None:
    all_literals = (
        CORE_LABELS,
        CORE_LABELS_STRING,
        CORE_LABELS_INTEGER,
        CORE_LABELS_FLOAT,
        CORE_LABELS_BOOL,
        CORE_LABELS_SCHEDULER,
        CORE_LABELS_MODEL,
        CORE_LABELS_MAIN_MODEL,
        CORE_LABELS_SDXL_MODEL,
        CORE_LABELS_VAE,
    )
    assert len(all_literals) == 10
    for lit in all_literals:
        args = get_args(lit)
        assert args[0] == CUSTOM_LABEL, f"CUSTOM_LABEL must be at index 0 in {lit}"
        assert len(args) == len(set(args)), f"Duplicates found in literal {lit}"


def test_core_model_labels_parity_with_model_and_vae_literals() -> None:
    model_labels = set(get_args(CORE_LABELS_MODEL)) - {CUSTOM_LABEL}
    vae_labels = set(get_args(CORE_LABELS_VAE)) - {CUSTOM_LABEL}
    expected_model_labels = model_labels | vae_labels
    assert CORE_MODEL_LABELS == expected_model_labels


def test_core_metadata_schema_export_preserves_flat_enum_lists() -> None:
    from invokeai.app.invocations.metadata_linked import MetadataItemLinkedInvocation

    schema = MetadataItemLinkedInvocation.model_json_schema()
    label_prop = schema["properties"]["label"]
    assert "enum" in label_prop
    assert label_prop["type"] == "string"
    assert label_prop["enum"][0] == CUSTOM_LABEL
    assert len(label_prop["enum"]) == len(get_args(CORE_LABELS))

    for lit in (
        CORE_LABELS,
        CORE_LABELS_STRING,
        CORE_LABELS_INTEGER,
        CORE_LABELS_FLOAT,
        CORE_LABELS_BOOL,
        CORE_LABELS_SCHEDULER,
        CORE_LABELS_MODEL,
        CORE_LABELS_MAIN_MODEL,
        CORE_LABELS_SDXL_MODEL,
        CORE_LABELS_VAE,
    ):
        ta_schema = TypeAdapter(lit).json_schema()
        assert ta_schema["type"] == "string"
        assert "enum" in ta_schema
        assert ta_schema["enum"][0] == CUSTOM_LABEL
        assert ta_schema["enum"] == list(get_args(lit))


def test_unwrap_metadata_value_for_new_core_model_labels() -> None:
    model_id = ModelIdentifierField(
        key="model-key",
        hash="model-hash",
        name="Test Model",
        base=BaseModelType.Any,
        type=ModelType.Main,
    )
    lora_id = ModelIdentifierField(
        key="lora-key",
        hash="lora-hash",
        name="Test LoRA",
        base=BaseModelType.Any,
        type=ModelType.LoRA,
    )

    wrapper_field = TransformerField(transformer=model_id, loras=[LoRAField(lora=lora_id, weight=0.75)])
    wrapper_dict = {
        "text_encoder": model_id.model_dump(),
        "tokenizer": model_id.model_dump(),
        "loras": [{"lora": lora_id.model_dump(), "weight": 0.75}],
    }

    for core_label in ("anima_inpaint_adapter", "qwen3_vl_vision_tower", "flux3_prompt_upsampler"):
        assert core_label in CORE_MODEL_LABELS
        unwrapped_field = unwrap_metadata_value(wrapper_field, label=core_label)
        assert isinstance(unwrapped_field, ModelIdentifierField)
        assert unwrapped_field.key == model_id.key

        unwrapped_dict = unwrap_metadata_value(wrapper_dict, label=core_label)
        assert isinstance(unwrapped_dict, dict)
        assert unwrapped_dict["key"] == model_id.key
        assert "loras" not in unwrapped_dict

    for custom_label in (CUSTOM_LABEL, "user_custom_model_label"):
        kept_field = unwrap_metadata_value(wrapper_field, label=custom_label)
        assert isinstance(kept_field, TransformerField)
        assert len(kept_field.loras) == 1
        assert kept_field.transformer.key == model_id.key

        kept_dict = unwrap_metadata_value(wrapper_dict, label=custom_label)
        assert isinstance(kept_dict, dict)
        assert len(kept_dict["loras"]) == 1
        assert kept_dict["text_encoder"]["key"] == model_id.key


def test_metadata_linked_and_extractors_support_new_core_keys() -> None:
    import pytest

    from invokeai.app.invocations.metadata_linked import (
        MetadataItemLinkedInvocation,
        MetadataToFloatInvocation,
        MetadataToModelInvocation,
        MetadataToSDXLModelInvocation,
    )

    ctx = SimpleNamespace()
    weight_linked = MetadataItemLinkedInvocation(
        id="link-weight",
        label="anima_inpaint_adapter_weight",
        value=0.65,
    ).invoke(ctx)
    assert weight_linked.metadata.root["anima_inpaint_adapter_weight"] == 0.65

    weight_metadata = MetadataField.model_validate(weight_linked.metadata.model_dump())
    weight_extracted = MetadataToFloatInvocation(
        id="extract-weight",
        label="anima_inpaint_adapter_weight",
        default_value=0.0,
        metadata=weight_metadata,
    ).invoke(ctx)
    assert weight_extracted.value == 0.65

    component_cases = (
        ("anima_inpaint_adapter", "anima-adapter-key", "Anima Inpaint Adapter", BaseModelType.Anima, ModelType.Main),
        (
            "qwen3_vl_vision_tower",
            "qwen3-vl-tower-key",
            "Qwen3 VL Vision Tower",
            BaseModelType.Any,
            ModelType.Qwen3VLVisionTower,
        ),
        (
            "flux3_prompt_upsampler",
            "flux3-upsampler-key",
            "FLUX.3 Prompt Upsampler",
            BaseModelType.Flux3,
            ModelType.PromptEnhancer,
        ),
        (
            "wan_t5_encoder_model",
            "wan-t5-key",
            "Wan UMT5-XXL",
            BaseModelType.Any,
            ModelType.T5Encoder,
        ),
    )

    default_model = ModelIdentifierField(
        key="default-key",
        hash="hash-default",
        name="Default",
        base=BaseModelType.StableDiffusionXL,
        type=ModelType.Main,
    )

    for label, model_key, model_name, base, model_type in component_cases:
        model_id = ModelIdentifierField(
            key=model_key,
            hash=f"hash-{model_key}",
            name=model_name,
            base=base,
            type=model_type,
        )
        lora_id = ModelIdentifierField(
            key="lora-key",
            hash="hash-lora",
            name="Test LoRA",
            base=base,
            type=ModelType.LoRA,
        )
        wrapped = TransformerField(transformer=model_id, loras=[LoRAField(lora=lora_id, weight=0.5)])

        linked = MetadataItemLinkedInvocation(id=f"link-{label}", label=label, value=wrapped).invoke(ctx)
        assert isinstance(linked.metadata.root[label], ModelIdentifierField)
        assert linked.metadata.root[label].key == model_key

        metadata = MetadataField.model_validate(linked.metadata.model_dump())

        # JP-R6-2: Component/encoder/adapter labels must NOT appear on MetadataToModel or MetadataToSDXLModel
        with pytest.raises(ValidationError):
            MetadataToModelInvocation(
                id=f"extract-model-{label}",
                label=label,  # type: ignore[arg-type]
                default_value=default_model,
                metadata=metadata,
            )
        with pytest.raises(ValidationError):
            MetadataToSDXLModelInvocation(
                id=f"extract-sdxl-{label}",
                label=label,  # type: ignore[arg-type]
                default_value=default_model,
                metadata=metadata,
            )

    # Main model labels still extract cleanly
    main_id = ModelIdentifierField(
        key="main-key",
        hash="hash-main-key",
        name="Main Model",
        base=BaseModelType.StableDiffusionXL,
        type=ModelType.Main,
    )
    main_cfg = SimpleNamespace(
        key="main-key",
        hash="hash-main-key",
        name="Main Model",
        base=BaseModelType.StableDiffusionXL,
        type=ModelType.Main,
    )
    main_ctx = SimpleNamespace(
        models=SimpleNamespace(
            exists=lambda key: key == "main-key",
            get_config=lambda key: main_cfg,
        )
    )
    main_md = MetadataField.model_validate({"model": main_id.model_dump(), "refiner_model": main_id.model_dump()})
    assert (
        MetadataToModelInvocation(id="extract-main", label="model", default_value=default_model, metadata=main_md)
        .invoke(main_ctx)
        .model
        == main_id
    )
    assert (
        MetadataToModelInvocation(
            id="extract-refiner", label="refiner_model", default_value=default_model, metadata=main_md
        )
        .invoke(main_ctx)
        .model
        == main_id
    )
    assert (
        MetadataToSDXLModelInvocation(id="extract-sdxl", label="model", default_value=default_model, metadata=main_md)
        .invoke(main_ctx)
        .model
        == main_id
    )
    with pytest.raises(ValidationError):
        MetadataToSDXLModelInvocation(
            id="extract-sdxl-refiner",
            label="refiner_model",  # type: ignore[arg-type]
            default_value=default_model,
            metadata=main_md,
        )


def test_metadata_to_model_rejects_mismatched_component_model_type_via_custom_label() -> None:
    """JP-R6-2: Passing wan_t5_encoder_model via CUSTOM_LABEL (both key-present and name-only)
    into MetadataToModelInvocation / MetadataToSDXLModelInvocation must fall back to default_value
    instead of returning a ModelType.T5Encoder model as ModelType.Main."""
    from invokeai.app.invocations.metadata_linked import MetadataToModelInvocation, MetadataToSDXLModelInvocation

    default_main = ModelIdentifierField(
        key="default-main-key",
        hash="hash-default-main",
        name="Default Main",
        base=BaseModelType.StableDiffusionXL,
        type=ModelType.Main,
    )
    default_cfg = SimpleNamespace(
        key=default_main.key,
        hash=default_main.hash,
        name=default_main.name,
        base=default_main.base,
        type=ModelType.Main,
    )
    t5_cfg = SimpleNamespace(
        key="wan-t5-key",
        hash="hash-wan-t5",
        name="UMT5-XXL",
        base=BaseModelType.Any,
        type=ModelType.T5Encoder,
    )
    configs = {default_main.key: default_cfg, t5_cfg.key: t5_cfg}
    mock_logger = MagicMock()
    ctx = SimpleNamespace(
        logger=mock_logger,
        models=SimpleNamespace(
            exists=lambda k: k in configs,
            get_config=lambda k: configs[k],
            search_by_attrs=lambda name=None, type=None: [
                c for c in configs.values() if c.name == name and c.type == type
            ],
        ),
    )

    # 1. Key-present: points to an existing T5Encoder key -> warns and falls back to default_main
    md_with_key = MetadataField.model_validate(
        {"wan_t5_encoder_model": {"key": t5_cfg.key, "name": t5_cfg.name, "type": ModelType.T5Encoder.value}}
    )
    out_model_key = MetadataToModelInvocation(
        id="to-model-key",
        label=CUSTOM_LABEL,
        custom_label="wan_t5_encoder_model",
        default_value=default_main,
        metadata=md_with_key,
    ).invoke(ctx)
    assert out_model_key.model == default_main
    assert mock_logger.warning.called

    out_sdxl_key = MetadataToSDXLModelInvocation(
        id="to-sdxl-key",
        label=CUSTOM_LABEL,
        custom_label="wan_t5_encoder_model",
        default_value=default_main,
        metadata=md_with_key,
    ).invoke(ctx)
    assert out_sdxl_key.model == default_main

    # 2. Name-only: searches with type=ModelType.Main -> finds nothing -> falls back to default_main
    md_name_only = MetadataField.model_validate({"wan_t5_encoder_model": {"name": t5_cfg.name}})
    out_model_name = MetadataToModelInvocation(
        id="to-model-name",
        label=CUSTOM_LABEL,
        custom_label="wan_t5_encoder_model",
        default_value=default_main,
        metadata=md_name_only,
    ).invoke(ctx)
    assert out_model_name.model == default_main

    out_sdxl_name = MetadataToSDXLModelInvocation(
        id="to-sdxl-name",
        label=CUSTOM_LABEL,
        custom_label="wan_t5_encoder_model",
        default_value=default_main,
        metadata=md_name_only,
    ).invoke(ctx)
    assert out_sdxl_name.model == default_main


def test_core_metadata_cfg_scale_schedule_and_hidiffusion_auto_ratios_parity() -> None:
    from invokeai.app.invocations.metadata_constants import CORE_BOOL_LABELS
    from invokeai.app.invocations.metadata_linked import (
        MetadataToBoolInvocation,
        MetadataToFloatCollectionInvocation,
        MetadataToFloatInvocation,
    )

    assert CORE_METADATA_VERSION == "1.4.0"
    assert "hidiffusion_auto_ratios" in get_args(CORE_LABELS_BOOL)
    assert "hidiffusion_auto_ratios" in CORE_BOOL_LABELS

    ctx = SimpleNamespace()
    schedule = [7.0, 5.5, 3.0]
    core_out = CoreMetadataInvocation(
        id="m-sched",
        cfg_scale=schedule,
        hidiffusion_auto_ratios=True,
    ).invoke(ctx)
    assert core_out.metadata.root["cfg_scale"] == schedule
    assert core_out.metadata.root["hidiffusion_auto_ratios"] is True

    md = MetadataField.model_validate(core_out.metadata.model_dump())
    assert (
        MetadataToFloatCollectionInvocation(id="fcoll", label="cfg_scale", default_value=[1.0], metadata=md)
        .invoke(ctx)
        .collection
        == schedule
    )
    assert (
        MetadataToFloatInvocation(id="fscalar", label="cfg_scale", default_value=1.0, metadata=md).invoke(ctx).value
        == 7.0
    )
    assert (
        MetadataToBoolInvocation(id="b-hidiff", label="hidiffusion_auto_ratios", default_value=False, metadata=md)
        .invoke(ctx)
        .value
        is True
    )


def test_metadata_scalar_and_collection_extractors_handle_scalars_lists_and_bool_guards() -> None:
    from invokeai.app.invocations.metadata_linked import (
        MetadataToBoolCollectionInvocation,
        MetadataToBoolInvocation,
        MetadataToFloatCollectionInvocation,
        MetadataToFloatInvocation,
        MetadataToIntegerCollectionInvocation,
        MetadataToIntegerInvocation,
        MetadataToStringCollectionInvocation,
        MetadataToStringInvocation,
    )

    ctx = SimpleNamespace()
    md = MetadataField.model_validate(
        {
            "cfg_scale": 6.5,
            "seed": 42,
            "moss_image_uncond_skip_layers": [22, 24],
            "positive_prompt": "alpine lake",
            "positive_prompts": ["first prompt", "second prompt"],
            "hidiffusion_auto_ratios": True,
            "empty_list": [],
            "bool_list": [True, False],
            "bool_for_num": True,
            "none_val": None,
        }
    )

    # Float scalar <-> list
    assert MetadataToFloatCollectionInvocation(id="fc-1", label="cfg_scale", default_value=[1.0], metadata=md).invoke(
        ctx
    ).collection == [6.5]
    assert (
        MetadataToFloatInvocation(
            id="fs-list", label=CUSTOM_LABEL, custom_label="bool_list", default_value=9.9, metadata=md
        )
        .invoke(ctx)
        .value
        == 9.9
    )
    assert (
        MetadataToFloatInvocation(
            id="fs-empty", label=CUSTOM_LABEL, custom_label="empty_list", default_value=2.5, metadata=md
        )
        .invoke(ctx)
        .value
        == 2.5
    )
    assert (
        MetadataToFloatCollectionInvocation(
            id="fc-empty", label=CUSTOM_LABEL, custom_label="empty_list", default_value=[2.5], metadata=md
        )
        .invoke(ctx)
        .collection
        == []
    )
    assert MetadataToFloatCollectionInvocation(
        id="fc-bool", label=CUSTOM_LABEL, custom_label="bool_for_num", default_value=[2.5], metadata=md
    ).invoke(ctx).collection == [2.5]

    # Integer scalar <-> list
    assert (
        MetadataToIntegerInvocation(id="is-list", label="moss_image_uncond_skip_layers", default_value=0, metadata=md)
        .invoke(ctx)
        .value
        == 22
    )
    assert MetadataToIntegerCollectionInvocation(
        id="ic-list", label="moss_image_uncond_skip_layers", default_value=[0], metadata=md
    ).invoke(ctx).collection == [22, 24]
    assert MetadataToIntegerCollectionInvocation(id="ic-scalar", label="seed", default_value=[0], metadata=md).invoke(
        ctx
    ).collection == [42]
    assert (
        MetadataToIntegerInvocation(
            id="is-empty", label=CUSTOM_LABEL, custom_label="empty_list", default_value=7, metadata=md
        )
        .invoke(ctx)
        .value
        == 7
    )
    assert (
        MetadataToIntegerInvocation(
            id="is-bool", label=CUSTOM_LABEL, custom_label="bool_for_num", default_value=7, metadata=md
        )
        .invoke(ctx)
        .value
        == 7
    )
    assert MetadataToIntegerCollectionInvocation(
        id="ic-bool", label=CUSTOM_LABEL, custom_label="bool_for_num", default_value=[7], metadata=md
    ).invoke(ctx).collection == [7]

    # String scalar <-> list
    assert (
        MetadataToStringInvocation(id="ss-list", label="positive_prompts", default_value="fallback", metadata=md)
        .invoke(ctx)
        .value
        == "first prompt"
    )
    assert MetadataToStringCollectionInvocation(
        id="sc-list", label="positive_prompts", default_value=["fallback"], metadata=md
    ).invoke(ctx).collection == ["first prompt", "second prompt"]
    assert MetadataToStringCollectionInvocation(
        id="sc-scalar", label="positive_prompt", default_value=["fallback"], metadata=md
    ).invoke(ctx).collection == ["alpine lake"]
    assert (
        MetadataToStringInvocation(
            id="ss-empty", label=CUSTOM_LABEL, custom_label="empty_list", default_value="fallback", metadata=md
        )
        .invoke(ctx)
        .value
        == "fallback"
    )

    # Bool scalar <-> list
    assert (
        MetadataToBoolInvocation(
            id="bs-list", label=CUSTOM_LABEL, custom_label="bool_list", default_value=False, metadata=md
        )
        .invoke(ctx)
        .value
        is True
    )
    assert MetadataToBoolCollectionInvocation(
        id="bc-list", label=CUSTOM_LABEL, custom_label="bool_list", default_value=[False], metadata=md
    ).invoke(ctx).collection == [True, False]
    assert MetadataToBoolCollectionInvocation(
        id="bc-scalar", label="hidiffusion_auto_ratios", default_value=[False], metadata=md
    ).invoke(ctx).collection == [True]
    assert (
        MetadataToBoolInvocation(
            id="bs-empty", label=CUSTOM_LABEL, custom_label="empty_list", default_value=True, metadata=md
        )
        .invoke(ctx)
        .value
        is True
    )
    assert (
        MetadataToBoolInvocation(
            id="bs-none", label=CUSTOM_LABEL, custom_label="none_val", default_value=True, metadata=md
        )
        .invoke(ctx)
        .value
        is True
    )


def test_unwrap_metadata_value_custom_label_preserves_extra_wrapper_fields_without_loras() -> None:
    from invokeai.app.invocations.model import LTX2TransformerField, VAEField, WanTransformerField
    from invokeai.backend.model_manager.taxonomy import SubModelType

    wan_high = ModelIdentifierField(key="wh", hash="hh", name="wh", base=BaseModelType.Wan, type=ModelType.Main)
    wan_low = ModelIdentifierField(key="wl", hash="hl", name="wl", base=BaseModelType.Wan, type=ModelType.Main)
    wan_field = WanTransformerField(transformer=wan_high, transformer_low_noise=wan_low, boundary_ratio=0.8)
    assert unwrap_metadata_value(wan_field, label="model") == wan_high
    assert unwrap_metadata_value(wan_field, label="wan_transformer_low_noise") == wan_low
    assert isinstance(unwrap_metadata_value(wan_field, label=CUSTOM_LABEL), WanTransformerField)
    assert unwrap_metadata_value(wan_field.model_dump(), label=CUSTOM_LABEL)["boundary_ratio"] == 0.8

    ltx_tf = ModelIdentifierField(
        key="lt",
        hash="hlt",
        name="lt",
        base=BaseModelType.LTX2,
        type=ModelType.Main,
        submodel_type=SubModelType.Transformer,
    )
    ltx_field = LTX2TransformerField(transformer=ltx_tf, loras=[], variant="ltx2_dev")
    assert unwrap_metadata_value(ltx_field, label="model") == ltx_tf
    assert isinstance(unwrap_metadata_value(ltx_field, label=CUSTOM_LABEL), LTX2TransformerField)
    assert unwrap_metadata_value(ltx_field.model_dump(), label=CUSTOM_LABEL)["variant"] == "ltx2_dev"

    vae_id = ModelIdentifierField(
        key="vk", hash="hv", name="vk", base=BaseModelType.StableDiffusionXL, type=ModelType.VAE
    )
    vae_seamless = VAEField(vae=vae_id, seamless_axes=["x", "y"])
    assert unwrap_metadata_value(vae_seamless, label="vae") == vae_id
    assert isinstance(unwrap_metadata_value(vae_seamless, label=CUSTOM_LABEL), VAEField)
    assert unwrap_metadata_value(vae_seamless.model_dump(), label=CUSTOM_LABEL)["seamless_axes"] == ["x", "y"]
