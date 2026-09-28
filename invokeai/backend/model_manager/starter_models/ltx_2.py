"""LTX-2.5 starter models.

Sourced from the ``DeepBeepMeep/LTX-2`` mirror: an ungated, per-component repack of Lightricks'
LTX-2.5 release (tensor values unchanged, connectors split out of the transformer files) that the
installer can fetch without a Hugging Face login. The official ``Lightricks/LTX-2.5`` repo is
license-gated and packs the connectors and text projection differently; only the mirror's split is
a complete component source in this version.

The mirror's nvfp4 transformer is deliberately not listed: its file names no quantized layer in a
marker or header, which the nvfp4 reader requires to know the block-scale layout, so it is refused
at load time.
"""

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
