from types import SimpleNamespace
from typing import get_args
from unittest.mock import MagicMock

from pydantic import TypeAdapter

from invokeai.app.invocations.fields import ImageField, MetadataField, VideoField
from invokeai.app.invocations.metadata import CORE_METADATA_VERSION, CoreMetadataInvocation, unwrap_metadata_value
from invokeai.app.invocations.metadata_constants import (
    CORE_LABELS,
    CORE_LABELS_BOOL,
    CORE_LABELS_FLOAT,
    CORE_LABELS_INTEGER,
    CORE_LABELS_MODEL,
    CORE_LABELS_SCHEDULER,
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
        CORE_LABELS_VAE,
    )
    assert len(all_literals) == 8
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

    for core_label in ("anima_inpaint_adapter", "qwen3_vl_vision_tower", "flux3_prompt_upsampler", "component_source"):
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
    from invokeai.app.invocations.metadata_linked import (
        MetadataItemLinkedInvocation,
        MetadataToFloatInvocation,
        MetadataToModelInvocation,
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

    model_cases = (
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
            "component_source",
            "component-source-key",
            "Qwen Image Component Source",
            BaseModelType.QwenImage,
            ModelType.Main,
        ),
    )

    for label, model_key, model_name, base, model_type in model_cases:
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
        config = SimpleNamespace(
            key=model_key,
            hash=f"hash-{model_key}",
            name=model_name,
            base=base,
            type=model_type,
        )
        configs = {model_key: config}
        model_ctx = SimpleNamespace(
            models=SimpleNamespace(
                exists=lambda key, c=configs: key in c,
                get_config=lambda key, c=configs: c[key],
            )
        )
        default_model = ModelIdentifierField(
            key="default-key",
            hash="hash-default",
            name="Default",
            base=base,
            type=ModelType.Main,
        )

        extracted = MetadataToModelInvocation(
            id=f"extract-{label}",
            label=label,
            default_value=default_model,
            metadata=metadata,
        ).invoke(model_ctx)
        assert extracted.model == model_id
