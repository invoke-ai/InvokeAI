from typing import get_args
from unittest.mock import MagicMock

from pydantic import TypeAdapter

from invokeai.app.invocations.fields import ImageField, VideoField
from invokeai.app.invocations.metadata import CORE_METADATA_VERSION, CoreMetadataInvocation
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
from invokeai.app.invocations.model import ModelIdentifierField
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
