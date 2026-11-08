"""LTX-2.5 starter models.

Sourced from the ``DeepBeepMeep/LTX-2`` mirror: an ungated, per-component repack of Lightricks'
LTX-2.5 release (tensor values unchanged, connectors split out of the transformer files) that the
installer can fetch without a Hugging Face login. The official ``Lightricks/LTX-2.5`` repo is
license-gated and packs the connectors and text projection differently; only the mirror's split is
a complete component source in this version.

Two entries are exceptions and come from Lightricks' gated repos, so they need a Hugging Face login
and an accepted licence: the duration head, which exists only in ``Lightricks/LTX-2.5-Diffusers``,
and the nvfp4 distilled transformer. The mirror's nvfp4 repack names no quantized layer in a marker
or header, which the nvfp4 reader requires to know the block-scale layout, so it is refused at load
time; the official file lists its layers in ``_quantization_metadata``. Both are optional, so the
cost of the exception falls only on users who choose them.

The two GGUF transformers come from a community conversion, ``vantagewithai/LTX-2.5-GGUF``, and
the GGUF text encoder from ``Rootport/Nz-Gemma4-12B-LTX25``; both are ungated as well.

The preview decoder is madebyollin's (MIT, from GitHub), not part of the LTX-2.5 release.
"""

from invokeai.backend.model_manager.starter_models.common import TINY_AUTOENCODER_NOTE
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    LTX2VariantType,
    ModelFormat,
    ModelType,
)

_LICENSE_NOTE = (
    "NOTE: LTX-2.5 is distributed under the LTX-2.x Community License (free for entities under "
    "$10M annual revenue; see https://huggingface.co/Lightricks/LTX-2.5)."
)

_MIRROR = "DeepBeepMeep/LTX-2"

ltx2_5_components = StarterModel(
    name="LTX-2.5 Components",
    base=BaseModelType.LTX2,
    source=(
        f"{_MIRROR}::ltx-2.5-22b_video_vae_bf16.safetensors+ltx-2.5-22b_audio_vae_bf16.safetensors"
        "+ltx-2.5-22b_vocoder_bf16.safetensors+ltx-2.5-22b_text_embedding_projection_bf16.safetensors"
        "+ltx-2.5-22b_video_embeddings_connector_bf16.safetensors+ltx-2.5-22b_audio_embeddings_connector_bf16.safetensors"
        "+ltx-2.5-spatial-upscaler-x2-1.0_bf16.safetensors+ltx-2.5-temporal-upscaler-x2-1.0_bf16.safetensors"
    ),
    description="LTX-2.5 shared components: video and audio VAEs, 48 kHz vocoder, text connectors and the "
    f"x2 latent upsamplers (~9.5 GB). Used together with an LTX-2.5 transformer and the Gemma-4 text encoder. {_LICENSE_NOTE}",
    type=ModelType.Main,
    format=ModelFormat.Diffusers,
)

ltx2_5_prompt_enhancer = StarterModel(
    name="LTX-2.5 Prompt Enhancer (Gemma-4 E2B)",
    base=BaseModelType.Any,
    source="google/gemma-4-E2B-it",
    description=(
        "The prompt enhancer LTX-2.5 was released with, for Expand Prompt. Pair it with the seeded LTX-2.5 "
        "system prompts; the Image-to-Video one also reads the first frame. Optional. (~10 GB)"
    ),
    type=ModelType.TextLLM,
)

ltx2_5_text_encoder_int8 = StarterModel(
    name="LTX-2.5 Text Encoder (Gemma-4 12B, int8)",
    base=BaseModelType.LTX2,
    source=(
        f"{_MIRROR}::gemma4-12b-ltx-v1/config.json+gemma4-12b-ltx-v1/tokenizer.json"
        "+gemma4-12b-ltx-v1/tokenizer_config.json+gemma4-12b-ltx-v1/chat_template.jinja"
        "+gemma4-12b-ltx-v1/gemma4-12b-ltx-v1_int8_convrot.safetensors"
    ),
    description="Lightricks' LTX-tuned Gemma-4-12B text encoder with its tokenizer, int8 quantized (~13 GB). "
    f"The text encoder every LTX-2.5 generation uses. {_LICENSE_NOTE}",
    type=ModelType.Gemma4Encoder,
    format=ModelFormat.Gemma4Encoder,
)

ltx2_5_text_encoder_bf16 = StarterModel(
    name="LTX-2.5 Text Encoder (Gemma-4 12B, bf16)",
    base=BaseModelType.LTX2,
    source=(
        f"{_MIRROR}::gemma4-12b-ltx-v1/config.json+gemma4-12b-ltx-v1/tokenizer.json"
        "+gemma4-12b-ltx-v1/tokenizer_config.json+gemma4-12b-ltx-v1/chat_template.jinja"
        "+gemma4-12b-ltx-v1/gemma4-12b-ltx-v1_bf16.safetensors"
    ),
    description="Lightricks' LTX-tuned Gemma-4-12B text encoder with its tokenizer, unquantized bf16 (~24 GB). "
    f"The text encoder every LTX-2.5 generation uses. {_LICENSE_NOTE}",
    type=ModelType.Gemma4Encoder,
    format=ModelFormat.Gemma4Encoder,
)

# A community conversion of the release's encoder (ungated, like the mirror), with the tokenizer and config packed in.
# Its text projection and vision stubs are left unread.
ltx2_5_text_encoder_gguf_q4_k_m = StarterModel(
    name="LTX-2.5 Text Encoder (Gemma-4 12B, GGUF Q4_K_M)",
    base=BaseModelType.LTX2,
    source="Rootport/Nz-Gemma4-12B-LTX25::TextEncoder/LTX-2.5-gemma4-12b-text-encoder-Q4_K_M.gguf",
    description="Lightricks' LTX-tuned Gemma-4-12B text encoder as a community GGUF Q4_K_M build with its tokenizer "
    "(~9.2 GB; ~7.6 GiB on the card against ~12 GiB for int8). It drifts further from the unquantized encoder than "
    f"int8 does: the same scene, framed a little differently at the same seed. {_LICENSE_NOTE}",
    type=ModelType.Gemma4Encoder,
    format=ModelFormat.GGUFQuantized,
)

ltx2_5_dev_transformer_int8 = StarterModel(
    name="LTX-2.5 Dev Transformer (int8)",
    base=BaseModelType.LTX2,
    source=f"{_MIRROR}::ltx-2.5-22b-dev_diffusion_model_int8_convrot.safetensors",
    description="LTX-2.5 22B guided ('dev') video+audio transformer, int8 quantized single file (~19.5 GB). "
    f"Runs with the LTX-2.5 Components and the Gemma-4 text encoder. Total size with dependencies: ~42 GB. "
    f"{_LICENSE_NOTE}",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    variant=LTX2VariantType.Dev,
    dependencies=[ltx2_5_components, ltx2_5_text_encoder_int8],
)

ltx2_5_distilled_transformer_int8 = StarterModel(
    name="LTX-2.5 Distilled Transformer (int8)",
    base=BaseModelType.LTX2,
    source=f"{_MIRROR}::ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors",
    description="LTX-2.5 22B step-distilled video+audio transformer (8 steps, no guidance), int8 quantized "
    f"single file (~19.5 GB). Runs with the LTX-2.5 Components and the Gemma-4 text encoder. Total size with "
    f"dependencies: ~42 GB. {_LICENSE_NOTE}",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    variant=LTX2VariantType.Distilled,
    dependencies=[ltx2_5_components, ltx2_5_text_encoder_int8],
)

# From the gated official repo (see the module docstring). The bundled connectors it carries are
# dropped at load; the components entry still supplies them.
ltx2_5_distilled_transformer_nvfp4 = StarterModel(
    name="LTX-2.5 Distilled Transformer (NVFP4)",
    base=BaseModelType.LTX2,
    source="Lightricks/LTX-2.5::diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
    description="LTX-2.5 22B step-distilled video+audio transformer (8 steps, no guidance), NVFP4 quantized "
    "single file (~18.7 GB; the last six blocks and some projections stay bf16). Faster than the int8 build on "
    "cards that cannot hold either fully, such as 16 GB ones. Fetched from Lightricks' gated "
    "repo, so it needs a Hugging Face login whose account has accepted the licence. Runs with the LTX-2.5 "
    f"Components and the Gemma-4 text encoder. Total size with dependencies: ~41 GB. {_LICENSE_NOTE}",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    variant=LTX2VariantType.Distilled,
    dependencies=[ltx2_5_components, ltx2_5_text_encoder_int8],
)

ltx2_5_distilled_lora = StarterModel(
    name="LTX-2.5 Distilled LoRA",
    base=BaseModelType.LTX2,
    source=f"{_MIRROR}::ltx-2.5-22b-distilled-lora-450_bf16.safetensors",
    description="Step-distillation LoRA for the LTX-2.5 Dev transformer: renders in 8 steps without "
    "guidance instead of ~30 with it. Rank 450 over all 1660 attention and feed-forward projections, "
    f"bf16 (~8.9 GB) — large for a LoRA because the distillation is not a light touch. {_LICENSE_NOTE}",
    type=ModelType.LoRA,
    format=ModelFormat.LyCORIS,
)

# The connectors these builds bundle beside the transformer are left unread; the components entry supplies them.
_GGUF_REPO = "vantagewithai/LTX-2.5-GGUF"

ltx2_5_dev_transformer_gguf_q4_k_m = StarterModel(
    name="LTX-2.5 Dev Transformer (GGUF Q4_K_M)",
    base=BaseModelType.LTX2,
    source=f"{_GGUF_REPO}::dev/ltx-2.5-22b-dev-transformer-Q4_K_M.gguf",
    description="LTX-2.5 22B guided ('dev') video+audio transformer, community GGUF Q4_K_M build (~15.7 GB; "
    "~12.5 GiB on the card against ~18 GiB for int8). Runs with the LTX-2.5 Components and the Gemma-4 text "
    f"encoder. Total size with dependencies: ~38 GB. {_LICENSE_NOTE}",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=LTX2VariantType.Dev,
    dependencies=[ltx2_5_components, ltx2_5_text_encoder_int8],
)

ltx2_5_distilled_transformer_gguf_q4_k_m = StarterModel(
    name="LTX-2.5 Distilled Transformer (GGUF Q4_K_M)",
    base=BaseModelType.LTX2,
    source=f"{_GGUF_REPO}::distilled/ltx-2.5-22b-distilled-transformer-Q4_K_M.gguf",
    description="LTX-2.5 22B step-distilled video+audio transformer (8 steps, no guidance), community GGUF Q4_K_M "
    "build (~15.7 GB; ~12.5 GiB on the card against ~18 GiB for int8). Runs with the LTX-2.5 Components and the "
    f"Gemma-4 text encoder. Total size with dependencies: ~38 GB. {_LICENSE_NOTE}",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=LTX2VariantType.Distilled,
    dependencies=[ltx2_5_components, ltx2_5_text_encoder_int8],
)

ltx2_5_dev_transformer_bf16 = StarterModel(
    name="LTX-2.5 Dev Transformer (bf16)",
    base=BaseModelType.LTX2,
    source=f"{_MIRROR}::ltx-2.5-22b-dev_diffusion_model_bf16.safetensors",
    description="LTX-2.5 22B guided ('dev') video+audio transformer, unquantized bf16 single file (~38 GB; "
    f"needs a 48 GB+ card or partial loading). Total size with dependencies: ~61 GB. {_LICENSE_NOTE}",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    variant=LTX2VariantType.Dev,
    dependencies=[ltx2_5_components, ltx2_5_text_encoder_int8],
)


# Not from the ungated mirror: the head ships solely in Lightricks' gated repo. Installing it
# therefore requires `huggingface-cli login` with an account that has accepted the LTX-2.5 licence,
# which is why the description says so outright rather than letting the install fail with a 403.
ltx2_5_duration_head = StarterModel(
    name="LTX-2.5 Duration Head",
    base=BaseModelType.LTX2,
    source="Lightricks/LTX-2.5-Diffusers::duration_head/diffusion_pytorch_model.safetensors",
    description="Predicts how long the shot a prompt describes should run, so a generation can choose its own "
    "length (~3.6 MB). Optional. Unlike the other LTX-2.5 starter models this one is fetched from Lightricks' "
    f"gated repo, so it needs a Hugging Face login whose account has accepted the licence. {_LICENSE_NOTE}",
    type=ModelType.LTX2DurationHead,
    format=ModelFormat.Checkpoint,
)


# taeltx2_3 is upstream's decoder for LTX-2.3 and later. taeltx_2 (LTX-2) has the same network but
# scored 11.0 dB against the VAE on LTX-2.5 latents, below the linear preview's 26.6 (taeltx2_3: 37.8).
taeltx2_3 = StarterModel(
    name="TAELTX2.3 (LTX-2.5 preview decoder)",
    base=BaseModelType.LTX2,
    source="https://github.com/madebyollin/taehv/raw/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeltx2_3.safetensors",
    description=f"Sharp live previews for LTX-2.5. {TINY_AUTOENCODER_NOTE} ~24MB",
    type=ModelType.TinyAutoencoder,
    format=ModelFormat.Checkpoint,
)
