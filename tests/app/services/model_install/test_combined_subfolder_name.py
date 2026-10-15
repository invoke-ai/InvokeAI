"""The folder name an explicit-file Hugging Face source installs into stays a legal path component."""

from invokeai.app.services.model_install.model_install_default import _combined_subfolder_name

LTX2_COMPONENT_STEMS = [
    "ltx-2.5-22b_video_vae_bf16",
    "ltx-2.5-22b_audio_vae_bf16",
    "ltx-2.5-22b_vocoder_bf16",
    "ltx-2.5-22b_text_embedding_projection_bf16",
    "ltx-2.5-22b_video_embeddings_connector_bf16",
    "ltx-2.5-22b_audio_embeddings_connector_bf16",
    "ltx-2.5-spatial-upscaler-x2-1.0_bf16",
    "ltx-2.5-temporal-upscaler-x2-1.0_bf16",
]


def test_short_lists_keep_the_full_joined_name() -> None:
    assert (
        _combined_subfolder_name(["tokenizer", "processor", "vae", "audio_vae"]) == "tokenizer_processor_vae_audio_vae"
    )


def test_a_long_list_is_cut_to_a_path_component_that_still_says_what_it_is() -> None:
    name = _combined_subfolder_name(LTX2_COMPONENT_STEMS)
    assert len(name) < 120, "eight full stems would exceed the 255-byte component limit once prefixed"
    assert name.startswith("ltx-2.5-22b_video_vae_bf16_ltx-2.5-22b_audio_vae_bf16")
    assert name.endswith("_and_5_more")
