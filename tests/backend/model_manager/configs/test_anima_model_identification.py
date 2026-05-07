import pytest

from invokeai.backend.model_manager.configs.main import _has_anima_keys


def _make_state_dict(prefixes: list[str], keys: list[str]) -> dict[str, object]:
    """Build a minimal fake state dict with the given prefixes applied to the given keys."""
    return {f"{prefix}{key}": None for prefix in prefixes for key in keys}


# Minimal keys that satisfy both llm_adapter and cosmos DiT requirements
ANIMA_LLM_ADAPTER_KEYS = ["llm_adapter.blocks.0.cross_attn.k_norm.weight"]
ANIMA_COSMOS_DIT_KEYS = [
    "blocks.0.adaln_modulation_cross_attn.1.weight",
    "t_embedder.1.linear_1.weight",
    "x_embedder.proj.1.weight",
    "final_layer.adaln_modulation.1.weight",
]


class TestHasAnimaKeys:
    """Tests for _has_anima_keys heuristic used during model identification."""

    def test_bare_keys(self):
        """Bare keys (no prefix) should be recognized."""
        sd = _make_state_dict([""], ANIMA_LLM_ADAPTER_KEYS + ANIMA_COSMOS_DIT_KEYS)
        assert _has_anima_keys(sd) is True

    def test_net_prefix(self):
        """Official format with `net.` prefix should be recognized."""
        sd = _make_state_dict(["net."], ANIMA_LLM_ADAPTER_KEYS + ANIMA_COSMOS_DIT_KEYS)
        assert _has_anima_keys(sd) is True

    def test_comfyui_bundled_prefix(self):
        """ComfyUI bundled format with `model.diffusion_model.` prefix should be recognized."""
        sd = _make_state_dict(["model.diffusion_model."], ANIMA_LLM_ADAPTER_KEYS + ANIMA_COSMOS_DIT_KEYS)
        assert _has_anima_keys(sd) is True

    def test_comfyui_bundled_with_extra_keys(self):
        """Bundled checkpoint with VAE and text encoder keys should still be recognized."""
        sd = _make_state_dict(["model.diffusion_model."], ANIMA_LLM_ADAPTER_KEYS + ANIMA_COSMOS_DIT_KEYS)
        # Add bundled VAE and text encoder keys (should not interfere)
        sd["first_stage_model.conv1.weight"] = None
        sd["first_stage_model.encoder.downsamples.0.weight"] = None
        sd["cond_stage_model.qwen3_06b.transformer.model.embed_tokens.weight"] = None
        assert _has_anima_keys(sd) is True

    def test_missing_llm_adapter_keys(self):
        """Should not match if llm_adapter keys are absent."""
        sd = _make_state_dict([""], ANIMA_COSMOS_DIT_KEYS)
        assert _has_anima_keys(sd) is False

    def test_missing_cosmos_dit_keys(self):
        """Should not match if Cosmos DiT keys are absent."""
        sd = _make_state_dict([""], ANIMA_LLM_ADAPTER_KEYS)
        assert _has_anima_keys(sd) is False

    def test_empty_state_dict(self):
        """Empty state dict should not match."""
        assert _has_anima_keys({}) is False

    def test_unrelated_keys(self):
        """State dict with unrelated keys should not match."""
        sd = {
            "model.diffusion_model.input_blocks.0.0.weight": None,
            "model.diffusion_model.output_blocks.0.0.weight": None,
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": None,
        }
        assert _has_anima_keys(sd) is False

    @pytest.mark.parametrize(
        "prefix",
        ["", "net.", "model.diffusion_model."],
    )
    def test_all_prefixes_parametrized(self, prefix: str):
        """All supported prefix formats should be recognized."""
        sd = _make_state_dict([prefix], ANIMA_LLM_ADAPTER_KEYS + ANIMA_COSMOS_DIT_KEYS)
        assert _has_anima_keys(sd) is True


class TestAnimaDoesNotConflictWithOtherModels:
    """Verify that _has_anima_keys does not false-positive on similar model architectures."""

    def test_flux_bundled_checkpoint(self):
        """FLUX bundled checkpoints use double_blocks/single_blocks, not blocks — should not match."""
        sd = {
            "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale": None,
            "model.diffusion_model.double_blocks.0.img_attn.proj.weight": None,
            "model.diffusion_model.single_blocks.0.linear1.weight": None,
            "model.diffusion_model.context_embedder.weight": None,
            "model.diffusion_model.img_in.weight": None,
        }
        assert _has_anima_keys(sd) is False

    def test_sd1_bundled_checkpoint(self):
        """SD1/SD2/SDXL bundled checkpoints use input_blocks/output_blocks — should not match."""
        sd = {
            "model.diffusion_model.input_blocks.0.0.weight": None,
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": None,
            "model.diffusion_model.output_blocks.0.0.weight": None,
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight": None,
            "first_stage_model.encoder.down.0.block.0.conv1.weight": None,
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": None,
        }
        assert _has_anima_keys(sd) is False

    def test_raw_cosmos_dit_without_llm_adapter(self):
        """A raw Cosmos Predict2 DiT (without Anima's LLM adapter) should not match."""
        sd = {
            "blocks.0.adaln_modulation_cross_attn.1.weight": None,
            "blocks.0.self_attn.q_proj.weight": None,
            "t_embedder.1.linear_1.weight": None,
            "x_embedder.proj.1.weight": None,
            "final_layer.adaln_modulation.1.weight": None,
        }
        assert _has_anima_keys(sd) is False

    def test_z_image_checkpoint(self):
        """Z-Image uses blocks.* but with cap_embedder/context_refiner — should not match."""
        sd = {
            "model.diffusion_model.blocks.0.attn.to_q.weight": None,
            "model.diffusion_model.blocks.0.attn.to_k.weight": None,
            "model.diffusion_model.cap_embedder.0.weight": None,
            "model.diffusion_model.context_refiner.blocks.0.weight": None,
            "model.diffusion_model.t_embedder.mlp.0.weight": None,
            "model.diffusion_model.x_embedder.proj.weight": None,
        }
        # Z-Image has blocks/t_embedder/x_embedder but NOT llm_adapter
        assert _has_anima_keys(sd) is False

    def test_qwen_image_checkpoint(self):
        """QwenImage uses txt_in/txt_norm/img_in — should not match."""
        sd = {
            "txt_in.weight": None,
            "txt_norm.weight": None,
            "img_in.weight": None,
            "double_blocks.0.img_attn.proj.weight": None,
            "single_blocks.0.linear1.weight": None,
        }
        assert _has_anima_keys(sd) is False

    def test_flux_lora_does_not_match(self):
        """FLUX LoRA weights should not match as Anima."""
        sd = {
            "double_blocks.0.img_attn.proj.lora_down.weight": None,
            "double_blocks.0.img_attn.proj.lora_up.weight": None,
            "single_blocks.0.linear1.lora_down.weight": None,
        }
        assert _has_anima_keys(sd) is False

    def test_cosmos_dit_bundled_without_llm_adapter(self):
        """Bundled Cosmos DiT (model.diffusion_model. prefix) but no llm_adapter — should not match."""
        sd = {
            "model.diffusion_model.blocks.0.self_attn.q_proj.weight": None,
            "model.diffusion_model.t_embedder.1.linear_1.weight": None,
            "model.diffusion_model.x_embedder.proj.1.weight": None,
            "model.diffusion_model.final_layer.adaln_modulation.1.weight": None,
            "first_stage_model.encoder.downsamples.0.weight": None,
            "cond_stage_model.transformer.model.embed_tokens.weight": None,
        }
        # Has all the Cosmos DiT keys but missing llm_adapter — not Anima
        assert _has_anima_keys(sd) is False
