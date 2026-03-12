# Comprehensive Plan: Adding Anima Model Support to InvokeAI

## 1. Executive Summary

**Anima** is a 2-billion-parameter anime-focused text-to-image model created by CircleStone Labs and Comfy Org, built on top of NVIDIA's Cosmos Predict2 architecture. It uses a **Cosmos DiT backbone** (`MiniTrainDIT`), a **Qwen3 0.6B text encoder**, a custom **LLM Adapter** (6-layer cross-attention transformer that fuses Qwen3 hidden states with learned T5-XXL token embeddings), and a **Qwen-Image VAE** (`AutoencoderKLQwenImage` — a fine-tuned Wan 2.1 VAE with 16 latent channels).

The model uses **rectified flow** sampling (shift=3.0, multiplier=1000) — the same `CONST` + `ModelSamplingDiscreteFlow` formulation used by Flux and Z-Image, meaning the existing `FlowMatchEulerDiscreteScheduler` and `FlowMatchHeunDiscreteScheduler` can be reused. The initial implementation targets **basic text-to-image generation only** — LoRA, ControlNet, inpainting, regional prompting, and img2img will come later.

**Key architectural difference from all existing InvokeAI models**: The LLM Adapter is a custom component embedded inside the diffusion model that cross-attends between Qwen3 encoder hidden states and learned T5-XXL token ID embeddings. This means the text encoding pipeline produces *two* outputs (Qwen3 hidden states + T5 token IDs) that are both fed into the transformer.

---

## 2. Model Architecture Reference

### 2.1 Components Overview

| Component | Architecture | Source | Size |
|-----------|-------------|--------|------|
| **Diffusion Transformer** | `MiniTrainDIT` (Cosmos Predict2 DiT) + `LLMAdapter` | Single-file checkpoint (`anima-preview2.safetensors`) | ~2B params |
| **Text Encoder** | Qwen3 0.6B (causal LM, hidden states extracted) | Single-file (`qwen_3_06b_base.safetensors`) | ~0.6B params |
| **T5-XXL Tokenizer** | SentencePiece tokenizer only (no T5 model weights needed) | Bundled with transformers library | ~2MB |
| **VAE** | `AutoencoderKLQwenImage` (fine-tuned Wan 2.1 VAE) | Single-file (`qwen_image_vae.safetensors`) | ~100M params |

### 2.2 Text Conditioning Pipeline

```
User Prompt
    ├──> Qwen3 0.6B Tokenizer (Qwen2Tokenizer)
    │       └──> Qwen3 0.6B Model → second-to-last hidden states [seq_len, 1024]
    │
    ├──> T5-XXL Tokenizer (T5TokenizerFast) → token IDs [seq_len] (no T5 model needed)
    │
    └──> LLM Adapter (inside transformer)
            ├── Embed T5 token IDs via learned Embedding(32128, 1024)
            ├── Cross-attend T5 embeddings ← Qwen3 hidden states (6 transformer layers with RoPE)
            └── Output: [512, 1024] conditioning tensor (zero-padded if < 512 tokens)
                    └──> Fed to Cosmos DiT cross-attention layers
```

### 2.3 Latent Space

- **Channels**: 16
- **Spatial compression**: 8× (VAE downsamples by 2^3)
- **Dimensions**: 3D (`[B, C, T, H, W]`) — temporal dim is 1 for single images
- **Normalization**: Mean/std normalization using Wan 2.1 constants (not simple scaling)
  - `process_in(latent) = (latent - latents_mean) / latents_std`
  - `process_out(latent) = latent * latents_std + latents_mean`

### 2.4 Sampling / Noise Schedule

- **Type**: Rectified Flow (`CONST` model — `denoised = input - output * sigma`)
- **Shift**: 3.0 (via `time_snr_shift(alpha=3.0, t)` — same formula as Flux)
- **Multiplier**: 1000
- **Sigma range**: 0.0 (clean) to 1.0 (noise), shifted by factor 3.0
- **Compatible schedulers**: `FlowMatchEulerDiscreteScheduler`, `FlowMatchHeunDiscreteScheduler` (already in InvokeAI for Z-Image/Flux)
- **Recommended settings**: CFG 4–5, 30–50 steps

### 2.5 Default Model Configuration (from ComfyUI)

```python
# MiniTrainDIT default constructor args for Anima:
model_channels = 2048      # Transformer hidden dim
num_blocks = 28            # Number of DiT blocks
num_heads = 32             # Attention heads
crossattn_emb_channels = 1024  # Must match LLM Adapter output dim
patch_spatial = 2          # Spatial patch size
patch_temporal = 1         # Temporal patch size (1 for images)
in_channels = 16           # Latent channels
out_channels = 16          # Output channels
max_img_h = 240            # Max height in patches (240 * 2 * 8 = 3840px)
max_img_w = 240            # Max width in patches
max_frames = 1             # Single image
```

---

## 3. ComfyUI Reference Implementation

The following ComfyUI source files contain the complete Anima implementation and should be reverse-engineered:

| File | URL | Purpose |
|------|-----|---------|
| **Anima model** | [comfy/ldm/anima/model.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ldm/anima/model.py) | `Anima(MiniTrainDIT)` + `LLMAdapter` + `RotaryEmbedding` + `TransformerBlock` + `Attention` |
| **Cosmos DiT base** | [comfy/ldm/cosmos/predict2.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ldm/cosmos/predict2.py) | `MiniTrainDIT` — the Cosmos Predict2 backbone that Anima extends |
| **Text encoder** | [comfy/text_encoders/anima.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/text_encoders/anima.py) | Dual tokenizer (Qwen3 + T5-XXL), `AnimaTEModel` that outputs Qwen3 hidden states + T5 token IDs |
| **Model registration** | [comfy/supported_models.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/supported_models.py) | `Anima` config class: shift=3.0, `Wan21` latent format, dtype support |
| **Model base** | [comfy/model_base.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_base.py) | `Anima(BaseModel)` — `ModelType.FLOW`, pre-processes text embeds via LLM adapter in `extra_conds()` |
| **Latent format** | [comfy/latent_formats.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py) | `Wan21` — 16ch, 3D, mean/std normalization constants |
| **Sampling** | [comfy/model_sampling.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_sampling.py) | `CONST` + `ModelSamplingDiscreteFlow` — rectified flow with shift |

### 3.1 LLM Adapter Architecture (from ComfyUI source)

The `LLMAdapter` is the critical custom component. From [comfy/ldm/anima/model.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ldm/anima/model.py):

- **Input**: `source_hidden_states` (Qwen3 output, dim=1024) + `target_input_ids` (T5-XXL token IDs)
- **Embedding**: `Embedding(32128, 1024)` — maps T5 token IDs to dense vectors
- **Projection**: `in_proj` (identity when `model_dim == target_dim`)
- **Positional encoding**: `RotaryEmbedding(head_dim=64)` — applied separately to query (target) and key (source) sequences
- **Blocks**: 6 × `TransformerBlock` each containing:
  - Self-attention on the target (T5 embedding) sequence
  - Cross-attention: target queries attend to source (Qwen3) keys/values
  - MLP with GELU activation (4× expansion)
  - RMSNorm (eps=1e-6) before each sub-layer
- **Output**: `norm(out_proj(x))` → `[batch, seq_len, 1024]`, zero-padded to 512 tokens

### 3.2 Key Insight: LLM Adapter Lives Inside the Checkpoint

The LLM Adapter weights are stored under the `llm_adapter.*` prefix in the main checkpoint file (`anima-preview2.safetensors`). They are **not** a separate file. The Anima class's `forward()` calls `preprocess_text_embeds()` which runs the adapter before passing to `MiniTrainDIT.forward()`.

### 3.3 Full ComfyUI Anima Model Source

```python
# comfy/ldm/anima/model.py — FULL SOURCE
from comfy.ldm.cosmos.predict2 import MiniTrainDIT
import torch
from torch import nn
import torch.nn.functional as F


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim, device=None, dtype=None, operations=None):
        super().__init__()

        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = operations.Linear(query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = operations.RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)

        self.k_proj = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = operations.RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)

        self.v_proj = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)

        self.o_proj = operations.Linear(inner_dim, query_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            assert position_embeddings_context is not None
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states = apply_rotary_pos_emb(key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

    def init_weights(self):
        torch.nn.init.zeros_(self.o_proj.weight)


class TransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, use_self_attn=False, layer_norm=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.use_self_attn = use_self_attn

        if self.use_self_attn:
            self.norm_self_attn = operations.LayerNorm(model_dim, device=device, dtype=dtype) if layer_norm else operations.RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
            self.self_attn = Attention(
                query_dim=model_dim,
                context_dim=model_dim,
                n_heads=num_heads,
                head_dim=model_dim//num_heads,
                device=device,
                dtype=dtype,
                operations=operations,
            )

        self.norm_cross_attn = operations.LayerNorm(model_dim, device=device, dtype=dtype) if layer_norm else operations.RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
        self.cross_attn = Attention(
            query_dim=model_dim,
            context_dim=source_dim,
            n_heads=num_heads,
            head_dim=model_dim//num_heads,
            device=device,
            dtype=dtype,
            operations=operations,
        )

        self.norm_mlp = operations.LayerNorm(model_dim, device=device, dtype=dtype) if layer_norm else operations.RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            operations.Linear(model_dim, int(model_dim * mlp_ratio), device=device, dtype=dtype),
            nn.GELU(),
            operations.Linear(int(model_dim * mlp_ratio), model_dim, device=device, dtype=dtype)
        )

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None, position_embeddings=None, position_embeddings_context=None):
        if self.use_self_attn:
            normed = self.norm_self_attn(x)
            attn_out = self.self_attn(normed, mask=target_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings)
            x = x + attn_out

        normed = self.norm_cross_attn(x)
        attn_out = self.cross_attn(normed, mask=source_attention_mask, context=context, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        x = x + attn_out

        x = x + self.mlp(self.norm_mlp(x))
        return x

    def init_weights(self):
        torch.nn.init.zeros_(self.mlp[2].weight)
        self.cross_attn.init_weights()


class LLMAdapter(nn.Module):
    def __init__(
            self,
            source_dim=1024,
            target_dim=1024,
            model_dim=1024,
            num_layers=6,
            num_heads=16,
            use_self_attn=True,
            layer_norm=False,
            device=None,
            dtype=None,
            operations=None,
        ):
        super().__init__()

        self.embed = operations.Embedding(32128, target_dim, device=device, dtype=dtype)
        if model_dim != target_dim:
            self.in_proj = operations.Linear(target_dim, model_dim, device=device, dtype=dtype)
        else:
            self.in_proj = nn.Identity()
        self.rotary_emb = RotaryEmbedding(model_dim//num_heads)
        self.blocks = nn.ModuleList([
            TransformerBlock(source_dim, model_dim, num_heads=num_heads, use_self_attn=use_self_attn, layer_norm=layer_norm, device=device, dtype=dtype, operations=operations)
            for _ in range(num_layers)
        ])
        self.out_proj = operations.Linear(model_dim, target_dim, device=device, dtype=dtype)
        self.norm = operations.RMSNorm(target_dim, eps=1e-6, device=device, dtype=dtype)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)

        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        context = source_hidden_states
        x = self.in_proj(self.embed(target_input_ids, out_dtype=context.dtype))
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(x, context,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings_context)
        return self.norm(self.out_proj(x))


class Anima(MiniTrainDIT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_adapter = LLMAdapter(device=kwargs.get("device"), dtype=kwargs.get("dtype"), operations=kwargs.get("operations"))

    def preprocess_text_embeds(self, text_embeds, text_ids, t5xxl_weights=None):
        if text_ids is not None:
            out = self.llm_adapter(text_embeds, text_ids)
            if t5xxl_weights is not None:
                out = out * t5xxl_weights

            if out.shape[1] < 512:
                out = torch.nn.functional.pad(out, (0, 0, 0, 512 - out.shape[1]))
            return out
        else:
            return text_embeds

    def forward(self, x, timesteps, context, **kwargs):
        t5xxl_ids = kwargs.pop("t5xxl_ids", None)
        if t5xxl_ids is not None:
            context = self.preprocess_text_embeds(context, t5xxl_ids, t5xxl_weights=kwargs.pop("t5xxl_weights", None))
        return super().forward(x, timesteps, context, **kwargs)
```

### 3.4 ComfyUI Text Encoder Source

```python
# comfy/text_encoders/anima.py — FULL SOURCE
from transformers import Qwen2Tokenizer, T5TokenizerFast
import comfy.text_encoders.llama
from comfy import sd1_clip
import os
import torch


class Qwen3Tokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(tokenizer_path,
            pad_with_end=False, embedding_directory=embedding_directory, embedding_size=1024,
            embedding_key='qwen3_06b', tokenizer_class=Qwen2Tokenizer,
            has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999,
            min_length=1, pad_token=151643, tokenizer_data=tokenizer_data)

class T5XXLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_tokenizer")
        super().__init__(tokenizer_path,
            embedding_directory=embedding_directory, pad_with_end=False, embedding_size=4096,
            embedding_key='t5xxl', tokenizer_class=T5TokenizerFast, has_start_token=False,
            pad_to_max_length=False, max_length=99999999, min_length=1,
            tokenizer_data=tokenizer_data)

class AnimaTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.qwen3_06b = Qwen3Tokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = {}
        qwen_ids = self.qwen3_06b.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["qwen3_06b"] = [[(k[0], 1.0, k[2]) if return_word_ids else (k[0], 1.0) for k in inner_list] for inner_list in qwen_ids]
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        return self.t5xxl.untokenize(token_weight_pair)

    def state_dict(self):
        return {}

    def decode(self, token_ids, **kwargs):
        return self.qwen3_06b.decode(token_ids, **kwargs)

class Qwen3_06BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={},
            dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False,
            model_class=comfy.text_encoders.llama.Qwen3_06B, enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask, model_options=model_options)


class AnimaTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype,
            name="qwen3_06b", clip_model=Qwen3_06BModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        out = super().encode_token_weights(token_weight_pairs)
        out[2]["t5xxl_ids"] = torch.tensor(list(map(lambda a: a[0], token_weight_pairs["t5xxl"][0])), dtype=torch.int)
        out[2]["t5xxl_weights"] = torch.tensor(list(map(lambda a: a[1], token_weight_pairs["t5xxl"][0])))
        return out

def te(dtype_llama=None, llama_quantization_metadata=None):
    class AnimaTEModel_(AnimaTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return AnimaTEModel_
```

### 3.5 ComfyUI Model Registration and Base

```python
# comfy/supported_models.py — Anima class (excerpt)
class Anima(supported_models_base.BASE):
    unet_config = {
        "image_model": "anima",
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 3.0,
    }

    unet_extra_config = {}
    latent_format = latent_formats.Wan21

    memory_usage_factor = 1.0

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Anima(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict,
            "{}qwen3_06b.transformer.".format(pref))
        return supported_models_base.ClipTarget(
            comfy.text_encoders.anima.AnimaTokenizer,
            comfy.text_encoders.anima.te(**detect))

    def set_inference_dtype(self, dtype, manual_cast_dtype, **kwargs):
        self.memory_usage_factor = (self.unet_config.get("model_channels", 2048) / 2048) * 0.95
        if dtype is torch.float16:
            self.memory_usage_factor *= 1.4
        return super().set_inference_dtype(dtype, manual_cast_dtype, **kwargs)
```

```python
# comfy/model_base.py — Anima class (excerpt)
class Anima(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device,
            unet_model=comfy.ldm.anima.model.Anima)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        t5xxl_ids = kwargs.get("t5xxl_ids", None)
        t5xxl_weights = kwargs.get("t5xxl_weights", None)
        device = kwargs["device"]
        if cross_attn is not None:
            if t5xxl_ids is not None:
                if t5xxl_weights is not None:
                    t5xxl_weights = t5xxl_weights.unsqueeze(0).unsqueeze(-1).to(cross_attn)
                t5xxl_ids = t5xxl_ids.unsqueeze(0)

                if torch.is_inference_mode_enabled():  # if not we are training
                    cross_attn = self.diffusion_model.preprocess_text_embeds(
                        cross_attn.to(device=device, dtype=self.get_dtype_inference()),
                        t5xxl_ids.to(device=device),
                        t5xxl_weights=t5xxl_weights.to(device=device, dtype=self.get_dtype_inference()))
                else:
                    out['t5xxl_ids'] = comfy.conds.CONDRegular(t5xxl_ids)
                    out['t5xxl_weights'] = comfy.conds.CONDRegular(t5xxl_weights)

            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out
```

### 3.6 ComfyUI Latent Format (Wan21)

```python
# comfy/latent_formats.py — Wan21 class (used by Anima)
class Wan21(LatentFormat):
    latent_channels = 16
    latent_dimensions = 3

    latent_rgb_factors = [
            [-0.1299, -0.1692,  0.2932],
            [ 0.0671,  0.0406,  0.0442],
            [ 0.3568,  0.2548,  0.1747],
            [ 0.0372,  0.2344,  0.1420],
            [ 0.0313,  0.0189, -0.0328],
            [ 0.0296, -0.0956, -0.0665],
            [-0.3477, -0.4059, -0.2925],
            [ 0.0166,  0.1902,  0.1975],
            [-0.0412,  0.0267, -0.1364],
            [-0.1293,  0.0740,  0.1636],
            [ 0.0680,  0.3019,  0.1128],
            [ 0.0032,  0.0581,  0.0639],
            [-0.1251,  0.0927,  0.1699],
            [ 0.0060, -0.0633,  0.0005],
            [ 0.3477,  0.2275,  0.2950],
            [ 0.1984,  0.0913,  0.1861]
        ]

    latent_rgb_factors_bias = [-0.1835, -0.0868, -0.3360]

    def __init__(self):
        self.scale_factor = 1.0
        self.latents_mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
             0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]).view(1, 16, 1, 1, 1)
        self.latents_std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]).view(1, 16, 1, 1, 1)

        self.taesd_decoder_name = "lighttaew2_1"

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean
```

### 3.7 ComfyUI Sampling Constants

```python
# comfy/model_sampling.py — relevant classes

def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)

class CONST:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = reshape_sigma(sigma, model_output.ndim)
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        sigma = reshape_sigma(sigma, noise.ndim)
        return sigma * noise + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(self, sigma, latent):
        sigma = reshape_sigma(sigma, latent.ndim)
        return latent / (1.0 - sigma)

class ModelSamplingDiscreteFlow(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}
        self.set_parameters(shift=sampling_settings.get("shift", 1.0), multiplier=sampling_settings.get("multiplier", 1000))

    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000):
        self.shift = shift
        self.multiplier = multiplier
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier)
        self.register_buffer('sigmas', ts)

    def timestep(self, sigma):
        return sigma * self.multiplier

    def sigma(self, timestep):
        return time_snr_shift(self.shift, timestep / self.multiplier)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return time_snr_shift(self.shift, 1.0 - percent)
```

---

## 4. Existing InvokeAI Patterns (Z-Image as Template)

The Z-Image integration is the closest architectural template. Here are the key files and patterns:

### 4.1 Taxonomy / Enums

- `invokeai/backend/model_manager/config/enums.py` — `BaseModelType` enum (Z-Image at line ~32), `ModelType` enum, `Qwen3Variant` enum (line ~75, has `Qwen3_4B` and `Qwen3_8B`)

### 4.2 Model Configs

- `invokeai/backend/model_manager/config/configs/main.py` — Z-Image configs at lines 1079–1167: `Main_Diffusers_ZImage_Config`, `Main_Checkpoint_ZImage_Config`, `Main_GGUFQuantized_ZImage_Config`, each with `probe()` and `_validate_z_image_checkpoint()` helper
- `invokeai/backend/model_manager/config/configs/qwen3_encoder.py` — Qwen3 encoder configs (lines 19–269): directory, checkpoint, and GGUF formats, with variant detection by `hidden_size`
- `invokeai/backend/model_manager/config/configs/factory.py` — `AnyModelConfig` discriminated union (lines 149–255)

### 4.3 Model Loaders

- `invokeai/backend/model_manager/load/model_loaders/z_image.py` — 1063 lines: `ZImageDiffusersLoader`, `ZImageCheckpointLoader`, `ZImageGGUFLoader`, plus Qwen3 encoder loaders and ControlNet loader

### 4.4 Invocation Nodes

| File | Node | Purpose |
|------|------|---------|
| `invokeai/app/invocations/z_image_model_loader.py` | `ZImageModelLoaderInvocation` | Loads transformer + Qwen3 encoder + VAE |
| `invokeai/app/invocations/z_image_text_encoder.py` | `ZImageTextEncoderInvocation` | Qwen3 with chat template → second-to-last hidden state |
| `invokeai/app/invocations/z_image_denoise.py` | `ZImageDenoiseInvocation` | Full denoising loop (771 lines) with flow matching |
| `invokeai/app/invocations/z_image_latents_to_image.py` | `ZImageLatentsToImageInvocation` | VAE decode (supports both AutoencoderKL and FluxAutoEncoder) |
| `invokeai/app/invocations/z_image_image_to_latents.py` | `ZImageImageToLatentsInvocation` | VAE encode |

### 4.5 Backend Module

- `invokeai/backend/z_image/` — `conditioning_data.py`, `z_image_patchify.py`, `z_image_regional_prompting.py`, control extensions, etc.

### 4.6 Frontend

- `frontend/web/src/features/nodes/util/graph/generation/buildZImageGraph.ts` — Graph builder for Z-Image generation
- `frontend/web/src/features/nodes/types/constants.ts` — UI constants (color, display name, grid size, features per base)
- `frontend/web/src/features/nodes/util/graph/generation/buildGraph.ts` — Main dispatch switch

### 4.7 Schedulers

- `invokeai/backend/flux/flow_match_schedulers.py` — `FLOW_MATCH_SCHEDULER_MAP`, `FLOW_MATCH_SCHEDULER_LABEL_MAP`: Euler, Heun, LCM — shared by Flux and Z-Image

### 4.8 Starter Models

- `invokeai/app/services/model_install/model_install_default.py` — Z-Image starter models at lines 803–860, Qwen3 encoder starters at lines 1017+

---

## 5. Diffusers Compatibility

**Critical finding**: All needed diffusers classes exist in the pinned **v0.36.0**:

| Class | Module | Purpose |
|-------|--------|---------|
| `CosmosTransformer3DModel` | `diffusers.models.transformers.transformer_cosmos` | Backbone transformer (but see note below) |
| `AutoencoderKLQwenImage` | `diffusers.models.autoencoders.autoencoder_kl_qwenimage` | VAE (fine-tuned Wan 2.1) |
| `AutoencoderKLWan` | `diffusers.models.autoencoders.autoencoder_kl_wan` | Alternative VAE class |

**⚠️ Important caveat**: The diffusers `CosmosTransformer3DModel` is the *vanilla* Cosmos Predict2 model. Anima extends it with the custom `LLMAdapter`. We have two options:

1. **Don't use diffusers' `CosmosTransformer3DModel` at all** — implement the full `MiniTrainDIT` + `LLMAdapter` as custom PyTorch modules (reverse-engineered from ComfyUI). This is the safer approach since the ComfyUI implementation is the reference.
2. **Use diffusers' `CosmosTransformer3DModel` for the backbone** and bolt on the `LLMAdapter` separately — requires key remapping between ComfyUI checkpoint format and diffusers' expected format.

**Recommendation**: Option 1 (custom implementation) is recommended for the initial version. The checkpoint is in ComfyUI format and guaranteed to load. Key remapping is error-prone and the model is not officially in diffusers anyway. Diffusers compatibility can be added later as a second format option.

---

## 6. Implementation Steps

### Step 1: Register Anima Base Type and Qwen3 0.6B Variant

**Files to modify:**

- `invokeai/backend/model_manager/config/enums.py`
  - Add `Anima = "anima"` to `BaseModelType` enum (after `ZImage`)
  - Add `Qwen3_06B = "qwen3-0.6b"` to `Qwen3Variant` enum

- `invokeai/backend/model_manager/config/configs/qwen3_encoder.py`
  - Update the variant detection logic to recognize hidden_size ~1024 → `Qwen3_06B`
  - The existing logic maps 2560 → `Qwen3_4B` and 4096 → `Qwen3_8B`; add 1024 → `Qwen3_06B`

### Step 2: Create Model Config Classes

**Files to modify:**

- `invokeai/backend/model_manager/config/configs/main.py`
  - Add `Main_Checkpoint_Anima_Config` class with:
    - `base = BaseModelType.Anima`, `type = ModelType.Main`, `format = ModelFormat.Checkpoint`
    - `probe()` method that validates state dict keys: look for `llm_adapter.` prefix (unique to Anima) plus Cosmos-style keys (`blocks.`, `t_embedder.`, `x_embedder.`, `final_layer.`)
    - Default generation settings: `width=1024`, `height=1024`, `steps=35`, `cfg_scale=4.5`

- `invokeai/backend/model_manager/config/configs/vae.py`
  - Add a config class for the QwenImage VAE (if needed as a standalone model type), or handle it within the main loader

- `invokeai/backend/model_manager/config/configs/factory.py`
  - Add `Main_Checkpoint_Anima_Config` (and any VAE configs) to the `AnyModelConfig` union

### Step 3: Create Backend Module

**New directory**: `invokeai/backend/anima/`

**New files:**

- `invokeai/backend/anima/__init__.py`

- `invokeai/backend/anima/llm_adapter.py`
  - Port the `LLMAdapter`, `TransformerBlock`, `Attention`, and `RotaryEmbedding` classes from [comfy/ldm/anima/model.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ldm/anima/model.py)
  - These are standard PyTorch `nn.Module` classes using `nn.Linear`, `nn.Embedding`, `nn.RMSNorm`, `F.scaled_dot_product_attention`
  - Replace ComfyUI's `operations.Linear` / `operations.RMSNorm` / `operations.Embedding` / `operations.LayerNorm` with standard `torch.nn` equivalents
  - Key architecture: `Embedding(32128, 1024)` → `in_proj` → 6 × `TransformerBlock(source_dim=1024, model_dim=1024, num_heads=16, use_self_attn=True)` → `out_proj` → `RMSNorm`

- `invokeai/backend/anima/anima_transformer.py`
  - Two approaches (see Section 5 recommendation):
    - **Option A (recommended)**: Port `MiniTrainDIT` from [comfy/ldm/cosmos/predict2.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ldm/cosmos/predict2.py), create `AnimaTransformer` that extends it and adds the `LLMAdapter`
    - **Option B**: Use `CosmosTransformer3DModel` from diffusers as backbone, wrap it with the `LLMAdapter`, implement key remapping
  - Whichever approach: the `forward()` must accept `(x, timesteps, context, t5xxl_ids=None, t5xxl_weights=None)` and run `preprocess_text_embeds()` before the DiT forward pass

- `invokeai/backend/anima/conditioning_data.py`
  - Define `AnimaConditioningData` dataclass holding:
    - `qwen3_embeds: torch.Tensor` — shape `[seq_len, 1024]`
    - `t5xxl_ids: torch.Tensor` — shape `[seq_len]` (T5 token IDs)
    - `t5xxl_weights: Optional[torch.Tensor]` — shape `[seq_len]` (token weights for prompt weighting)
  - Follow the pattern in `invokeai/backend/z_image/conditioning_data.py`

### Step 4: Create Model Loader

**New file**: `invokeai/backend/model_manager/load/model_loaders/anima.py`

- Register `AnimaCheckpointLoader` via `@ModelLoaderRegistry.register(base=BaseModelType.Anima, type=ModelType.Main, format=ModelFormat.Checkpoint)`
- **Loading logic**:
  1. Load the safetensors state dict
  2. Separate keys into two groups by prefix:
     - `llm_adapter.*` → `LLMAdapter` weights
     - Everything else (`blocks.*`, `t_embedder.*`, `x_embedder.*`, `final_layer.*`, etc.) → `MiniTrainDIT` / `CosmosTransformer3DModel` weights
  3. Instantiate the `AnimaTransformer` (which contains both components)
  4. Load state dict
- **VAE loading**: Register a loader for `AutoencoderKLQwenImage` from diffusers
  - Load from single-file safetensors
  - The VAE is a 3D causal conv VAE (processes single images as `[B, C, 1, H, W]`)
  - Latent normalization uses the Wan 2.1 `latents_mean` / `latents_std` constants
- **Qwen3 0.6B**: Reuse the existing `Qwen3EncoderCheckpointLoader` from the Z-Image loader — it already handles single-file Qwen3 encoders via `Qwen3ForCausalLM`. Just ensure the config detection maps `hidden_size=1024` to the new `Qwen3_06B` variant.

### Step 5: Create Invocation Nodes

**New files in `invokeai/app/invocations/`:**

- **`anima_model_loader.py`** — `AnimaModelLoaderInvocation`
  - Inputs: `model` (Anima main model identifier), optional `qwen3_encoder` (standalone Qwen3 0.6B), optional `vae` (standalone QwenImage VAE)
  - Outputs: `AnimaModelLoaderOutput` with `transformer: TransformerField`, `qwen3_encoder: Qwen3EncoderField`, `vae: VAEField`
  - Follow pattern of `invokeai/app/invocations/z_image_model_loader.py`

- **`anima_text_encoder.py`** — `AnimaTextEncoderInvocation`
  - Inputs: `prompt` (string), `qwen3_encoder` (Qwen3EncoderField)
  - Processing:
    1. Tokenize prompt with Qwen3 tokenizer (using chat template: `[{"role": "user", "content": prompt}]`)
    2. Run Qwen3 0.6B model → extract second-to-last hidden state → filter by attention mask
    3. Tokenize same prompt with T5-XXL tokenizer → get token IDs (no T5 model needed)
    4. Store both as conditioning tensors
  - Output: conditioning info containing `qwen3_embeds`, `t5xxl_ids`, `t5xxl_weights`
  - Follow pattern of `invokeai/app/invocations/z_image_text_encoder.py` for Qwen3 encoding
  - **New aspect**: Must also produce T5 token IDs. Need to bundle `T5TokenizerFast` — the `sentencepiece` dependency is already in `pyproject.toml` (line 46), and `T5TokenizerFast` is used elsewhere in InvokeAI (for Flux/SD3 text encoding)

- **`anima_denoise.py`** — `AnimaDenoiseInvocation`
  - Inputs: `transformer`, `positive_conditioning`, `negative_conditioning`, `width`, `height`, `num_steps`, `guidance_scale`, `seed`, `scheduler` (Euler/Heun from existing flow match scheduler map)
  - Processing:
    1. Generate random noise in latent space: `[1, 16, 1, H//8, W//8]` (note: 3D latents with T=1)
    2. Apply Wan 2.1 `process_in()` normalization if doing img2img (for txt2img, start from pure noise)
    3. Create sigma schedule using rectified flow with shift=3.0 (same `time_snr_shift` as Flux/Z-Image)
    4. Denoising loop: for each timestep, run transformer forward with conditioning, compute `denoised = input - output * sigma`
    5. CFG: when `guidance_scale > 1.0`, run both conditional and unconditional forward passes, blend: `output = uncond + guidance * (cond - uncond)`
    6. Apply scheduler step (Euler or Heun)
  - Output: latents tensor
  - Follow the flow-matching denoising pattern from `invokeai/app/invocations/z_image_denoise.py` (simplified: no regional prompting, no ControlNet, no inpainting for initial version)
  - **Key difference from Z-Image**: The transformer expects `[B, C, T, H, W]` 5D input (Cosmos format), not `[B, C, H, W]` 4D. Temporal dim = 1 for images.

- **`anima_latents_to_image.py`** — `AnimaLatentsToImageInvocation`
  - Inputs: `latents`, `vae` (VAEField)
  - Processing:
    1. Load `AutoencoderKLQwenImage` from diffusers
    2. Apply Wan 2.1 `process_out()` denormalization: `latent * latents_std + latents_mean`
    3. Decode: VAE expects `[B, C, T, H, W]` → outputs `[B, C, T, H, W]` → squeeze temporal dim → convert to image
  - Output: PIL Image
  - Follow pattern of `invokeai/app/invocations/z_image_latents_to_image.py`, but adapted for `AutoencoderKLQwenImage` instead of `FluxAutoEncoder`/`AutoencoderKL`

### Step 6: Update Frontend

**Files to modify:**

- `frontend/web/src/features/nodes/types/constants.ts`
  - Add `'anima'` to `BASE_COLOR_MAP` (suggest a unique color, e.g., `'pink'` or `'rose'` for anime association)
  - Add `'anima'` to `BASE_LABEL_MAP` with display name `'Anima'`
  - Add `'anima'` to feature support arrays (only `SUPPORTS_CFG_RESCALE_BASE_MODELS` and similar that apply; omit from LoRA/ControlNet/IP-Adapter arrays initially)

- `frontend/web/src/features/parameters/hooks/useMainModelDefaultSettings.ts` (or equivalent)
  - Add `'anima'` with defaults: width=1024, height=1024, steps=35, cfg_scale=4.5

- **New file**: `frontend/web/src/features/nodes/util/graph/generation/buildAnimaGraph.ts`
  - Create the graph builder function that assembles: `anima_model_loader` → `anima_text_encoder` (positive + negative) → `anima_denoise` → `anima_latents_to_image`
  - Follow pattern of `frontend/web/src/features/nodes/util/graph/generation/buildZImageGraph.ts` but simplified (no ControlNet, no regional prompting, no img2img initially)

- `frontend/web/src/features/nodes/util/graph/generation/buildGraph.ts`
  - Add `case 'anima': return await buildAnimaGraph(arg);` to the dispatch switch

- Various Zod schema files and node type union files — these will need `'anima'` added wherever `'z-image'` appears, following the same pattern

- Grid size / scale factor mappings:
  - Grid size: **8** (spatial compression is 8×, unlike Flux/Z-Image's 16×)
  - Default dimensions: 1024×1024

### Step 7: Register Starter Models

**File to modify:**

- `invokeai/app/services/model_install/model_install_default.py`
  - Add starter model entries for:
    - Anima Preview2 transformer: `circlestone-labs/Anima` → `anima-preview2.safetensors`
    - Qwen3 0.6B text encoder: `circlestone-labs/Anima` → `qwen_3_06b_base.safetensors`
    - QwenImage VAE: `circlestone-labs/Anima` → `qwen_image_vae.safetensors`
  - Follow the pattern of Z-Image starter models at lines 803–860

### Step 8: Regenerate OpenAPI Schema

- After all backend changes, run the schema generation script to update the auto-generated OpenAPI schema that the frontend consumes
- This is typically done via `python scripts/generate_openapi_schema.py`

---

## 7. Key Technical Challenges & Decisions

### 7.1 Cosmos DiT Implementation Strategy

**Decision needed**: Use diffusers' `CosmosTransformer3DModel` or port ComfyUI's `MiniTrainDIT`?

| Approach | Pros | Cons |
|----------|------|------|
| **Port MiniTrainDIT from ComfyUI** | Exact checkpoint compatibility, no key remapping, reference implementation | More code to maintain, must port supporting classes (`Block`, `PatchEmbed`, `FinalLayer`, `Timesteps`, etc.) |
| **Use diffusers CosmosTransformer3DModel** | Less custom code, maintained by diffusers team | Key names may differ from checkpoint, needs investigation, may have subtle behavioral differences |

**Recommendation**: Start with porting from ComfyUI. The checkpoint is in ComfyUI format and guaranteed to load. Diffusers compatibility can be added later as a second format option.

### 7.2 T5 Tokenizer Handling

The LLM Adapter needs T5-XXL token IDs but *not* the T5-XXL model. InvokeAI already has `T5TokenizerFast` usage for Flux/SD3 (see `invokeai/backend/flux/text_conditioning.py`). The tokenizer files are small (~2MB) and can be loaded from the `transformers` library cache.

**Approach**: Load `T5TokenizerFast` in the text encoder invocation using `T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")` (or bundle tokenizer files). No T5 model weights are needed.

### 7.3 VAE 3D Tensor Handling

The `AutoencoderKLQwenImage` is a 3D causal conv VAE that expects `[B, C, T, H, W]` tensors. For single images, `T=1`. The encode/decode calls must:
- **Encode**: `image_tensor.unsqueeze(2)` → `[B, C, 1, H, W]` → VAE encode → latents `[B, 16, 1, H//8, W//8]`
- **Decode**: latents `[B, 16, 1, H//8, W//8]` → VAE decode → `[B, C, 1, H, W]` → `.squeeze(2)` → `[B, C, H, W]`

Apply Wan 2.1 mean/std normalization (not simple scaling):
```python
latents_mean = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                              0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921])
latents_std = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                             3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160])
```

### 7.4 Noise Schedule: Rectified Flow with Shift=3.0

The sigma schedule uses the same `time_snr_shift` formula as Flux:
```python
def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)
```

With `alpha=3.0` and `multiplier=1000`. The existing `FlowMatchEulerDiscreteScheduler` in `invokeai/backend/flux/flow_match_schedulers.py` should work, but may need the shift parameter exposed or configured. Check if the scheduler's `shift` parameter matches Anima's 3.0 (Flux uses a different shift value).

### 7.5 Qwen3 0.6B vs 4B/8B Differences

The Qwen3 0.6B model has `hidden_size=1024` compared to 4B's 2560 and 8B's 4096. The existing Qwen3 encoder infrastructure in InvokeAI handles 4B and 8B. Adding 0.6B requires:
- New variant enum value
- Updated variant detection (hidden_size → variant mapping)
- The model class (`Qwen3ForCausalLM` from transformers) should work for any size — it's architecture-agnostic

### 7.6 State Dict Key Mapping (Checkpoint → Model)

The Anima checkpoint likely uses keys like:
```
llm_adapter.embed.weight
llm_adapter.blocks.0.self_attn.q_proj.weight
llm_adapter.blocks.0.cross_attn.k_proj.weight
llm_adapter.blocks.0.mlp.0.weight
llm_adapter.out_proj.weight
llm_adapter.norm.weight
llm_adapter.rotary_emb.inv_freq
blocks.0.attn.to_q.weight       (Cosmos DiT attention)
blocks.0.attn.to_k.weight
blocks.0.crossattn.to_q.weight  (Cosmos DiT cross-attention)
t_embedder.0.freqs               (Timestep embedding)
t_embedder.1.linear_1.weight
x_embedder.proj.weight           (Patch embedding)
final_layer.linear.weight
```

**This key structure must be verified by inspecting the actual checkpoint file.** The loader must correctly instantiate the model architecture and load these keys. If using the ComfyUI `MiniTrainDIT` port, keys should match directly. If using diffusers' `CosmosTransformer3DModel`, a key remapping function will be needed.

---

## 8. File Change Summary

### New Files (Backend — Python)

| File | Purpose |
|------|---------|
| `invokeai/backend/anima/__init__.py` | Package init |
| `invokeai/backend/anima/llm_adapter.py` | `LLMAdapter`, `TransformerBlock`, `Attention`, `RotaryEmbedding` |
| `invokeai/backend/anima/anima_transformer.py` | `AnimaTransformer` (MiniTrainDIT + LLMAdapter) or wrapper around `CosmosTransformer3DModel` |
| `invokeai/backend/anima/conditioning_data.py` | `AnimaConditioningData` dataclass |
| `invokeai/backend/model_manager/load/model_loaders/anima.py` | `AnimaCheckpointLoader`, VAE loader |
| `invokeai/app/invocations/anima_model_loader.py` | `AnimaModelLoaderInvocation` |
| `invokeai/app/invocations/anima_text_encoder.py` | `AnimaTextEncoderInvocation` |
| `invokeai/app/invocations/anima_denoise.py` | `AnimaDenoiseInvocation` |
| `invokeai/app/invocations/anima_latents_to_image.py` | `AnimaLatentsToImageInvocation` |

### Modified Files (Backend — Python)

| File | Change |
|------|--------|
| `invokeai/backend/model_manager/config/enums.py` | Add `Anima` to `BaseModelType`, `Qwen3_06B` to `Qwen3Variant` |
| `invokeai/backend/model_manager/config/configs/main.py` | Add `Main_Checkpoint_Anima_Config` |
| `invokeai/backend/model_manager/config/configs/qwen3_encoder.py` | Add hidden_size=1024 → `Qwen3_06B` detection |
| `invokeai/backend/model_manager/config/configs/factory.py` | Add Anima configs to `AnyModelConfig` union |
| `invokeai/app/services/model_install/model_install_default.py` | Add Anima starter models |

### New Files (Frontend — TypeScript)

| File | Purpose |
|------|---------|
| `frontend/web/src/features/nodes/util/graph/generation/buildAnimaGraph.ts` | Anima graph builder |

### Modified Files (Frontend — TypeScript)

| File | Change |
|------|--------|
| `frontend/web/src/features/nodes/types/constants.ts` | Add `'anima'` to all base model maps |
| `frontend/web/src/features/nodes/util/graph/generation/buildGraph.ts` | Add `'anima'` case to dispatch switch |
| Default settings hook | Add Anima defaults (1024×1024, CFG 4.5, 35 steps) |
| Zod schemas / node type unions | Add `'anima'` entries |

---

## 9. Out of Scope (Future Work)

The following features are explicitly deferred to follow-up implementations:

- **LoRA support** — requires LoRA config classes, patcher logic, and a loader node
- **ControlNet** — requires Cosmos ControlNet support (available in diffusers 0.37.0 as `CosmosControlNetModel`)
- **Inpainting / Outpainting** — requires latent masking and noise injection logic
- **Image-to-Image** — requires VAE encode path + denoising from partial noise
- **Regional Prompting** — requires mask-based attention manipulation
- **IP Adapter** — architecture-specific, if even applicable to Cosmos-based models
- **GGUF / Quantized model support** — can be added later following Z-Image's GGUF loader pattern
- **Diffusers format loading** — if/when an official Anima diffusers pipeline is created
