from unittest.mock import MagicMock

from invokeai.app.invocations.fields import ImageField, VideoField
from invokeai.app.invocations.metadata import CORE_METADATA_VERSION, CoreMetadataInvocation
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
