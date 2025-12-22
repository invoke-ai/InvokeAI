# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Z-Image ControlNet Extension for spatial conditioning.

This module provides an extension-based approach to Z-Image ControlNet,
similar to how FLUX ControlNet works. Instead of duplicating the entire
transformer, we compute control hints separately and inject them into
the base transformer's forward pass.
"""

from typing import List, Optional, Tuple

import torch
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from torch.nn.utils.rnn import pad_sequence

from invokeai.backend.z_image.z_image_control_adapter import ZImageControlAdapter
from invokeai.backend.z_image.z_image_patchify_utils import SEQ_MULTI_OF, patchify_control_context


class ZImageControlNetExtension:
    """Extension for Z-Image ControlNet - computes control hints without duplicating the transformer.

    This class follows the same pattern as FLUX ControlNet extensions:
    - The control adapter is loaded separately
    - Control hints are computed per step
    - Hints are injected into the transformer's layer outputs

    Attributes:
        control_adapter: The Z-Image control adapter model
        control_cond: VAE-encoded control image latents
        weight: Control strength (recommended: 0.65-0.80)
        begin_step_percent: When to start applying control (0.0 = start)
        end_step_percent: When to stop applying control (1.0 = end)
    """

    def __init__(
        self,
        control_adapter: ZImageControlAdapter,
        control_cond: torch.Tensor,
        weight: float = 0.75,
        begin_step_percent: float = 0.0,
        end_step_percent: float = 1.0,
        skip_layers: int = 0,  # Skip first N control injection layers
    ):
        self._adapter = control_adapter
        self._control_cond = control_cond
        self._weight = weight
        self._begin_step_percent = begin_step_percent
        self._end_step_percent = end_step_percent
        self._skip_layers = skip_layers

        # Get actual number of control blocks from loaded model (not config!)
        # The safetensors may have more blocks than the config suggests
        self._num_control_blocks = len(control_adapter.control_layers)

        # Control layers are applied at every other layer (0, 2, 4, ...)
        # This matches the default configuration in the original implementation
        self._control_places = [i * 2 for i in range(self._num_control_blocks)]

        # DEBUG: Print control configuration
        print(f"[DEBUG] Actual num_control_blocks: {self._num_control_blocks}")
        print(f"[DEBUG] control_places: {self._control_places}")

        # DEBUG: Check if control_layers have non-zero weights
        first_layer = control_adapter.control_layers[0]
        if hasattr(first_layer, "after_proj"):
            after_proj_norm = first_layer.after_proj.weight.norm().item()
            print(f"[DEBUG] First control layer after_proj weight norm: {after_proj_norm}")
            if after_proj_norm < 1e-6:
                print("[WARNING] after_proj weights are near-zero! Weights may not be loaded correctly.")

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def control_places(self) -> List[int]:
        return self._control_places

    def should_apply(self, step_index: int, total_steps: int) -> bool:
        """Check if control should be applied at this step."""
        if total_steps == 0:
            return True
        step_percent = step_index / total_steps
        return self._begin_step_percent <= step_percent <= self._end_step_percent

    def prepare_control_state(
        self,
        base_transformer: ZImageTransformer2DModel,
        cap_feats: torch.Tensor,
        timestep_emb: torch.Tensor,
        x_item_seqlens: List[int],
        cap_item_seqlens: List[int],
        x_freqs_cis: torch.Tensor,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ) -> torch.Tensor:
        """Prepare control state (control_unified) for incremental hint computation.

        This processes the control condition through patchify and noise_refiner,
        returning the control_unified tensor that will be used incrementally.
        """
        bsz = 1
        device = self._control_cond.device

        # Patchify control context
        control_context = [self._control_cond]
        (
            control_patches,
            _,
            _control_pos_ids,
            control_pad_mask,
        ) = patchify_control_context(
            control_context,
            patch_size,
            f_patch_size,
            cap_feats.size(1),
        )

        # Embed control context
        ctrl_item_seqlens = [len(p) for p in control_patches]
        ctrl_max_seqlen = max(ctrl_item_seqlens)

        control_cat = torch.cat(control_patches, dim=0)
        embedder_key = f"{patch_size}-{f_patch_size}"
        control_cat = self._adapter.control_all_x_embedder[embedder_key](control_cat)

        # Apply padding token
        adaln_input = timestep_emb.type_as(control_cat)
        x_pad_token = self._adapter.x_pad_token.to(dtype=control_cat.dtype)
        control_cat[torch.cat(control_pad_mask)] = x_pad_token

        control_list = list(control_cat.split(ctrl_item_seqlens, dim=0))
        control_padded = pad_sequence(control_list, batch_first=True, padding_value=0.0)

        # Use x_freqs_cis from main path for aligned position encoding
        ctrl_freqs_cis_for_refiner = x_freqs_cis[:, : control_padded.shape[1]]

        ctrl_attn_mask = torch.zeros((bsz, ctrl_max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(ctrl_item_seqlens):
            ctrl_attn_mask[i, :seq_len] = 1

        # Refine control context through control_noise_refiner
        for layer in self._adapter.control_noise_refiner:
            control_padded = layer(control_padded, ctrl_attn_mask, ctrl_freqs_cis_for_refiner, adaln_input)

        # Store these for compute_single_hint
        self._ctrl_item_seqlens = ctrl_item_seqlens
        self._adaln_input = adaln_input

        # Unify control with caption features
        control_unified = []
        for i in range(bsz):
            ctrl_len = ctrl_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            control_unified.append(torch.cat([control_padded[i][:ctrl_len], cap_feats[i][:cap_len]]))

        control_unified = pad_sequence(control_unified, batch_first=True, padding_value=0.0)

        # DEBUG (only once)
        if not hasattr(self, "_prepare_printed"):
            self._prepare_printed = True
            print(f"[DEBUG] Control state prepared: shape {control_unified.shape}")

        return control_unified

    def compute_single_hint(
        self,
        control_layer_idx: int,
        control_state: torch.Tensor,
        unified_hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute a single hint from one control layer.

        Args:
            control_layer_idx: Which control layer to use (0, 1, 2, ...)
            control_state: Current control state (stacked tensor from previous layers)
            unified_hidden_states: Current unified hidden states from main transformer
            attn_mask: Attention mask
            freqs_cis: RoPE frequencies
            adaln_input: Timestep embedding

        Returns:
            Tuple of (hint tensor, updated control_state)
        """
        layer = self._adapter.control_layers[control_layer_idx]

        # Run control layer with CURRENT unified_hidden_states
        control_state = layer(
            control_state,
            x=unified_hidden_states,
            attn_mask=attn_mask,
            freqs_cis=freqs_cis,
            adaln_input=adaln_input,
        )

        # Extract hint from stacked state
        # After control layer, control_state is stacked: [skip_0, ..., skip_n, running_state]
        # We want the latest skip (second to last element)
        unbinded = torch.unbind(control_state)
        hint = unbinded[-2]  # Latest skip connection

        return hint, control_state

    def compute_hints(
        self,
        base_transformer: ZImageTransformer2DModel,
        unified_hidden_states: torch.Tensor,
        cap_feats: torch.Tensor,
        timestep_emb: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        x_item_seqlens: List[int],
        cap_item_seqlens: List[int],
        x_freqs_cis: torch.Tensor,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ) -> Tuple[torch.Tensor, ...]:
        """Compute control hints using the adapter.

        This method processes the control condition through the adapter's
        control_noise_refiner and control_layers to produce hints that
        will be added to the transformer's hidden states.

        Args:
            base_transformer: The base Z-Image transformer (for rope_embedder)
            unified_hidden_states: Combined image+caption hidden states
            cap_feats: Caption feature embeddings (padded)
            timestep_emb: Timestep embeddings (adaln_input)
            attn_mask: Unified attention mask
            freqs_cis: RoPE frequencies
            x_item_seqlens: Image sequence lengths per batch item
            cap_item_seqlens: Caption sequence lengths per batch item
            patch_size: Spatial patch size
            f_patch_size: Frame patch size

        Returns:
            Tuple of hint tensors to add at each control layer position
        """
        # control_cond is always [C, F, H, W] format (single control image)
        # where C = control_in_dim (16 for V1, 33 for V2.0), F = 1 frame
        bsz = 1
        device = self._control_cond.device

        # Wrap control_cond in a list for patchify_control_context
        # Expected input: List of [C, F, H, W] tensors
        control_context = [self._control_cond]

        # Patchify control context
        # Note: We don't use control_pos_ids anymore - we use x_freqs_cis from main path instead
        (
            control_patches,
            _,
            _control_pos_ids,  # Not used - we use main path's position encoding
            control_pad_mask,
        ) = patchify_control_context(
            control_context,
            patch_size,
            f_patch_size,
            cap_feats.size(1),
        )

        # Embed control context
        ctrl_item_seqlens = [len(p) for p in control_patches]
        assert all(s % SEQ_MULTI_OF == 0 for s in ctrl_item_seqlens)
        ctrl_max_seqlen = max(ctrl_item_seqlens)

        control_cat = torch.cat(control_patches, dim=0)
        embedder_key = f"{patch_size}-{f_patch_size}"
        control_cat = self._adapter.control_all_x_embedder[embedder_key](control_cat)

        # Apply padding token (ensure dtype matches)
        adaln_input = timestep_emb.type_as(control_cat)
        x_pad_token = self._adapter.x_pad_token.to(dtype=control_cat.dtype)
        control_cat[torch.cat(control_pad_mask)] = x_pad_token

        control_list = list(control_cat.split(ctrl_item_seqlens, dim=0))

        control_padded = pad_sequence(control_list, batch_first=True, padding_value=0.0)

        # Use x_freqs_cis from main path for control patches (same spatial structure)
        # This ensures control and image have aligned position encodings
        ctrl_freqs_cis_for_refiner = x_freqs_cis[:, : control_padded.shape[1]]

        ctrl_attn_mask = torch.zeros((bsz, ctrl_max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(ctrl_item_seqlens):
            ctrl_attn_mask[i, :seq_len] = 1

        # Refine control context through control_noise_refiner
        # Using x_freqs_cis to match main path's position encoding
        for layer in self._adapter.control_noise_refiner:
            control_padded = layer(control_padded, ctrl_attn_mask, ctrl_freqs_cis_for_refiner, adaln_input)

        # Unify control with caption features
        control_unified = []
        for i in range(bsz):
            ctrl_len = ctrl_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            control_unified.append(torch.cat([control_padded[i][:ctrl_len], cap_feats[i][:cap_len]]))

        control_unified = pad_sequence(control_unified, batch_first=True, padding_value=0.0)
        c = control_unified

        # Process through control_layers to generate hints
        # DEBUG: Print shapes before control_layers (only on first call)
        if not hasattr(self, "_debug_printed"):
            self._debug_printed = True
            print(f"[DEBUG] control_unified shape: {control_unified.shape}")
            print(f"[DEBUG] unified_hidden_states shape: {unified_hidden_states.shape}")
            print(f"[DEBUG] ctrl_item_seqlens: {ctrl_item_seqlens}, x_item_seqlens: {x_item_seqlens}")

            # Check weight norms of critical layers
            layer0 = self._adapter.control_layers[0]
            if hasattr(layer0, "before_proj"):
                print(f"[DEBUG] before_proj weight norm: {layer0.before_proj.weight.norm().item():.6f}")
            if hasattr(layer0, "after_proj"):
                print(f"[DEBUG] after_proj weight norm: {layer0.after_proj.weight.norm().item():.6f}")

            # Check control_noise_refiner weights
            if len(self._adapter.control_noise_refiner) > 0:
                refiner0 = self._adapter.control_noise_refiner[0]
                if hasattr(refiner0, "attn"):
                    print(f"[DEBUG] noise_refiner[0] attn.wq norm: {refiner0.attn.wq.weight.norm().item():.6f}")

        for layer in self._adapter.control_layers:
            c = layer(
                c,
                x=unified_hidden_states,
                attn_mask=attn_mask,
                freqs_cis=freqs_cis,
                adaln_input=adaln_input,
            )

        # Extract hints (all but the last element which is the running state)
        hints = tuple(torch.unbind(c)[:-1])

        # DEBUG: Print hint shapes (only on first call)
        if not hasattr(self, "_hints_printed"):
            self._hints_printed = True
            print(f"[DEBUG] Number of hints: {len(hints)}")
            if hints:
                print(f"[DEBUG] First hint shape: {hints[0].shape}")
                # Also check hint statistics for each hint
                for i, h in enumerate(hints[:3]):  # First 3 hints
                    print(
                        f"[DEBUG] Hint[{i}] mean: {h.mean().item():.6f}, std: {h.std().item():.6f}, min: {h.min().item():.6f}, max: {h.max().item():.6f}"
                    )

        return hints


def z_image_forward_with_control(
    transformer: ZImageTransformer2DModel,
    x: List[torch.Tensor],
    t: torch.Tensor,
    cap_feats: List[torch.Tensor],
    control_extension: Optional[ZImageControlNetExtension] = None,
    patch_size: int = 2,
    f_patch_size: int = 1,
) -> Tuple[List[torch.Tensor], dict]:
    """Forward pass through Z-Image transformer with optional control injection.

    This function replicates the base transformer's forward pass but allows
    injecting control hints at specific layer positions. It uses the base
    transformer's weights directly without duplicating them.

    Args:
        transformer: The base Z-Image transformer model
        x: List of image tensors [C, F, H, W]
        t: Timestep tensor
        cap_feats: List of caption feature tensors
        control_extension: Optional control extension for hint injection
        patch_size: Spatial patch size (default: 2)
        f_patch_size: Frame patch size (default: 1)

    Returns:
        Tuple of (output tensors list, empty dict for compatibility)
    """
    assert patch_size in transformer.all_patch_size
    assert f_patch_size in transformer.all_f_patch_size

    bsz = len(x)
    device = x[0].device
    t_scaled = t * transformer.t_scale
    t_emb = transformer.t_embedder(t_scaled)

    # Patchify and embed using base transformer's method
    (
        x_patches,
        cap_feats_patches,
        x_size,
        x_pos_ids,
        cap_pos_ids,
        x_inner_pad_mask,
        cap_inner_pad_mask,
    ) = transformer.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

    # === X embed & refine ===
    x_item_seqlens = [len(p) for p in x_patches]
    assert all(s % SEQ_MULTI_OF == 0 for s in x_item_seqlens)
    x_max_item_seqlen = max(x_item_seqlens)

    embedder_key = f"{patch_size}-{f_patch_size}"
    x_cat = torch.cat(x_patches, dim=0)
    x_cat = transformer.all_x_embedder[embedder_key](x_cat)

    adaln_input = t_emb.type_as(x_cat)
    x_cat[torch.cat(x_inner_pad_mask)] = transformer.x_pad_token

    x_list = list(x_cat.split(x_item_seqlens, dim=0))
    x_freqs_cis = list(transformer.rope_embedder(torch.cat(x_pos_ids, dim=0)).split([len(p) for p in x_pos_ids], dim=0))

    x_padded = pad_sequence(x_list, batch_first=True, padding_value=0.0)
    x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
    x_freqs_cis = x_freqs_cis[:, : x_padded.shape[1]]

    x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(x_item_seqlens):
        x_attn_mask[i, :seq_len] = 1

    # Noise refiner
    for layer in transformer.noise_refiner:
        x_padded = layer(x_padded, x_attn_mask, x_freqs_cis, adaln_input)

    # === Cap embed & refine ===
    cap_item_seqlens = [len(p) for p in cap_feats_patches]
    cap_max_item_seqlen = max(cap_item_seqlens)

    cap_cat = torch.cat(cap_feats_patches, dim=0)
    cap_cat = transformer.cap_embedder(cap_cat)
    cap_cat[torch.cat(cap_inner_pad_mask)] = transformer.cap_pad_token

    cap_list = list(cap_cat.split(cap_item_seqlens, dim=0))
    cap_freqs_cis = list(
        transformer.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split([len(p) for p in cap_pos_ids], dim=0)
    )

    cap_padded = pad_sequence(cap_list, batch_first=True, padding_value=0.0)
    cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
    cap_freqs_cis = cap_freqs_cis[:, : cap_padded.shape[1]]

    cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(cap_item_seqlens):
        cap_attn_mask[i, :seq_len] = 1

    # Context refiner
    for layer in transformer.context_refiner:
        cap_padded = layer(cap_padded, cap_attn_mask, cap_freqs_cis)

    # === Unified ===
    unified = []
    unified_freqs_cis = []
    for i in range(bsz):
        x_len = x_item_seqlens[i]
        cap_len = cap_item_seqlens[i]
        unified.append(torch.cat([x_padded[i][:x_len], cap_padded[i][:cap_len]]))
        unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))

    unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens, strict=False)]
    unified_max_item_seqlen = max(unified_item_seqlens)

    unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
    unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)

    unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(unified_item_seqlens):
        unified_attn_mask[i, :seq_len] = 1

    # === Compute control hints if extension provided ===
    # IMPORTANT: Hints are computed ONCE using the INITIAL unified state (before main layers)
    # This matches the original VideoX-Fun architecture
    control_places: List[int] = []
    control_weight: float = 1.0
    hints: Optional[Tuple[torch.Tensor, ...]] = None

    # DEBUG: Print number of transformer layers (only once per session)
    if not hasattr(z_image_forward_with_control, "_layers_printed"):
        z_image_forward_with_control._layers_printed = True
        print(f"[DEBUG] Base transformer has {len(transformer.layers)} layers")

    if control_extension is not None:
        # Compute ALL hints at once using the INITIAL unified state (before main layers run)
        hints = control_extension.compute_hints(
            base_transformer=transformer,
            unified_hidden_states=unified,  # INITIAL unified state!
            cap_feats=cap_padded,
            timestep_emb=adaln_input,
            attn_mask=unified_attn_mask,
            freqs_cis=unified_freqs_cis,
            x_item_seqlens=x_item_seqlens,
            cap_item_seqlens=cap_item_seqlens,
            x_freqs_cis=x_freqs_cis,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
        )
        control_places = control_extension.control_places
        control_weight = control_extension.weight

    # === Main transformer layers with pre-computed hint injection ===
    skip_layers = control_extension._skip_layers if control_extension is not None else 0
    control_layer_idx = 0
    for layer_idx, layer in enumerate(transformer.layers):
        unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

        # Inject pre-computed control hint at designated positions
        if hints is not None and layer_idx in control_places and control_layer_idx < len(hints):
            # Skip first N hints if configured
            if control_layer_idx >= skip_layers:
                hint = hints[control_layer_idx]

                # DEBUG: Print on first injection
                if not hasattr(z_image_forward_with_control, "_injection_printed"):
                    z_image_forward_with_control._injection_printed = True
                    print(f"[DEBUG] Injection at layer {layer_idx} (control_layer {control_layer_idx})")
                    print(f"[DEBUG] Hint mean: {hint.mean().item():.6f}, std: {hint.std().item():.6f}")
                    print(f"[DEBUG] Unified mean: {unified.mean().item():.6f}, std: {unified.std().item():.6f}")
                    print(f"[DEBUG] control_weight: {control_weight}, skip_layers: {skip_layers}")

                unified = unified + hint * control_weight

            control_layer_idx += 1

    # === Final layer and unpatchify ===
    unified = transformer.all_final_layer[embedder_key](unified, adaln_input)
    unified = list(unified.unbind(dim=0))
    output = transformer.unpatchify(unified, x_size, patch_size, f_patch_size)

    return output, {}
