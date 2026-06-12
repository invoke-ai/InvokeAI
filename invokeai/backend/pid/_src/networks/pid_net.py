# PidNet — Super-resolution variant of PixDiT_T2I.
#
# Extends the text-to-image PixDiT model with LQ (low-quality) image/latent
# conditioning for image super-resolution. The base T2I architecture is unchanged;
# LQ information is injected via per-block gated injection between transformer
# blocks ("controlnet" mode — the only mode supported in this inference subset).
# Gate: sigma_aware_per_token_per_dim (sigma-conditioned LQ injection).
#
# All LQ modules are zero-initialized by default (zero_init_lq=True) so the network
# starts identical to the pretrained T2I model.
#
# Loading pretrained T2I checkpoint: use strict=False to ignore missing LQ keys.
#
# Reference:
#   - PixDiT_T2I: pid/_src/networks/pixeldit_official.py
#   - LQ projection: pid/_src/networks/lq_projection_2d.py

from typing import Optional

import torch

from invokeai.backend.pid._ext.imaginaire.utils import log
from invokeai.backend.pid._src.networks.lq_projection_2d import LQProjection2D
from invokeai.backend.pid._src.networks.pixeldit_official import PixDiT_T2I
from invokeai.backend.pid._src.utils.context_parallel import cat_outputs_cp_with_grad, split_inputs_cp


class PidNet(PixDiT_T2I):
    """PixDiT T2I with LQ condition injection for super-resolution.

    Inherits all PixDiT_T2I functionality (MMDiT patch blocks, PiT pixel blocks,
    text conditioning, RoPE, encoder-decoder compression, REPA). Adds LQ projection
    module and controlnet-style gated injection logic.

    Args (in addition to PixDiT_T2I args):
        lq_inject_mode: kept as a parameter for config compatibility — only
            "controlnet" is supported in this inference subset.
        lq_in_channels: LQ image channels (3 for RGB, 0 to disable image branch).
        lq_latent_channels: LQ latent channels (e.g. 16 for Wan VAE, 0 to disable).
        lq_hidden_dim: internal projection hidden dimension.
        lq_num_res_blocks: number of ResBlocks per branch for deeper feature extraction.
        lq_gate_type: "sigma_aware_per_token_per_dim" only.
        lq_interval: inject every N blocks.
        zero_init_lq: zero-init all LQ projections for safe pretrained start.
        train_lq_proj_only: freeze base T2I, train only LQ projection modules.
        sr_scale: super-resolution scale factor (default 4).
        latent_spatial_down_factor: VAE spatial downscale factor (default 8).
    """

    def __init__(
        self,
        # --- PixDiT_T2I base args ---
        in_channels=3,
        num_groups=16,
        hidden_size=1152,
        pixel_hidden_size=64,
        pixel_attn_hidden_size=None,
        pixel_num_groups=None,
        patch_depth=26,
        pixel_depth=2,
        num_text_blocks=4,
        patch_size=16,
        txt_embed_dim=4096,
        txt_max_length=1024,
        use_text_rope: bool = True,
        text_rope_theta: float = 10000.0,
        rope_mode: str = "ntk_aware",
        rope_ref_h: int = 1024,
        rope_ref_w: int = 1024,
        repa_encoder_index: int = -1,
        enable_ed: bool = False,
        ed_compress_ratio: int = 1,
        ed_depth_per_stage: int = 1,
        ed_window_size: int = 2,
        ed_num_heads: Optional[int] = None,
        ed_hidden_size: Optional[int] = None,
        ed_use_token_shuffle: bool = True,
        # --- SR-specific args ---
        lq_inject_mode: str = "controlnet",
        lq_in_channels: int = 3,
        lq_latent_channels: int = 0,
        lq_hidden_dim: int = 512,
        lq_num_res_blocks: int = 4,
        lq_gate_type: str = "sigma_aware_per_token_per_dim",
        lq_interval: int = 1,
        zero_init_lq: bool = True,
        train_lq_proj_only: bool = False,
        sr_scale: int = 4,
        latent_spatial_down_factor: int = 8,
        # --- PiT LQ injection args ---
        # Inject LQ features into PiT pixel blocks via a dedicated output head
        # from the same LQ projection CNN backbone. Added to s_cond before PiT loop.
        pit_lq_inject: bool = False,
        pit_lq_gate_type: str = "sigma_aware_per_token_per_dim",
    ):
        super().__init__(
            in_channels=in_channels,
            num_groups=num_groups,
            hidden_size=hidden_size,
            pixel_hidden_size=pixel_hidden_size,
            pixel_attn_hidden_size=pixel_attn_hidden_size,
            pixel_num_groups=pixel_num_groups,
            patch_depth=patch_depth,
            pixel_depth=pixel_depth,
            num_text_blocks=num_text_blocks,
            patch_size=patch_size,
            txt_embed_dim=txt_embed_dim,
            txt_max_length=txt_max_length,
            use_text_rope=use_text_rope,
            text_rope_theta=text_rope_theta,
            rope_mode=rope_mode,
            rope_ref_h=rope_ref_h,
            rope_ref_w=rope_ref_w,
            repa_encoder_index=repa_encoder_index,
            enable_ed=enable_ed,
            ed_compress_ratio=ed_compress_ratio,
            ed_depth_per_stage=ed_depth_per_stage,
            ed_window_size=ed_window_size,
            ed_num_heads=ed_num_heads,
            ed_hidden_size=ed_hidden_size,
            ed_use_token_shuffle=ed_use_token_shuffle,
        )

        assert lq_inject_mode == "controlnet", (
            f"Only lq_inject_mode='controlnet' is supported in this inference subset, got '{lq_inject_mode}'"
        )
        self.lq_inject_mode = lq_inject_mode
        self.sr_scale = sr_scale
        self.train_lq_proj_only = train_lq_proj_only

        num_lq_outputs = (patch_depth + lq_interval - 1) // lq_interval

        self.pit_lq_inject = pit_lq_inject

        self.lq_proj = LQProjection2D(
            in_channels=lq_in_channels,
            latent_channels=lq_latent_channels,
            hidden_dim=lq_hidden_dim,
            out_dim=hidden_size,
            patch_size=patch_size,
            sr_scale=sr_scale,
            latent_spatial_down_factor=latent_spatial_down_factor,
            num_res_blocks=lq_num_res_blocks,
            num_outputs=num_lq_outputs,
            gate_type=lq_gate_type,
            interval=lq_interval,
            zero_init=zero_init_lq,
            pit_output=pit_lq_inject,
        )

        # PiT LQ gate (applied to s_cond before pixel blocks)
        if pit_lq_inject:
            from invokeai.backend.pid._src.networks.lq_projection_2d import _build_gate

            self.pit_lq_gate = _build_gate(pit_lq_gate_type, hidden_size, zero_init=zero_init_lq)
        else:
            self.pit_lq_gate = None

        if train_lq_proj_only:
            for p in self.parameters():
                p.requires_grad_(False)
            for p in self.lq_proj.parameters():
                p.requires_grad_(True)
            if self.pit_lq_gate is not None and hasattr(self.pit_lq_gate, "parameters"):
                for p in self.pit_lq_gate.parameters():
                    p.requires_grad_(True)

    def init_weights(self):
        """Initialize LQ projection."""
        self.lq_proj.init_weights()
        log.info("LQ projection init_weights complete")

    def _compute_lq_features(self, lq_video_or_image, lq_latent, lq_mask, Hs, Ws):
        lq_features = self.lq_proj(
            lq_video_or_image=lq_video_or_image,
            lq_latent=lq_latent,
            target_pH=Hs,
            target_pW=Ws,
        )
        if lq_mask is not None:
            lq_features = [f * lq_mask.view(-1, 1, 1) for f in lq_features]
        # Under CP, lq_features are produced at full L (LQ inputs are replicated
        # across CP ranks). Split each along the token axis so they line up with
        # the rank-local image stream the patch blocks consume.
        if self._cp_group is not None:
            lq_features = [split_inputs_cp(f, seq_dim=1, cp_group=self._cp_group) for f in lq_features]
        return lq_features

    def _run_patch_blocks(
        self,
        s_main,
        y_emb,
        condition,
        pos,
        pos_txt,
        attn_mask_joint,
        lq_features,
        degrade_sigma=None,
        feature_indices=None,
    ):
        """Run patch_blocks loop with controlnet-style LQ injection.

        Args:
            feature_indices: Optional set of block indices whose output features should be
                collected and returned (for GAN discriminator). None = no collection.

        Returns:
            (s_main, y_emb, collected_features) where collected_features is a list of
            [B, L, D] tensors (one per index in feature_indices), or None if not requested.
        """
        has_lq = lq_features is not None

        collected_features = [] if feature_indices is not None else None

        for i in range(self.patch_depth):
            if has_lq and self.lq_proj.is_gate_active(i):
                out_idx = self.lq_proj._get_output_index(i)
                if out_idx < len(lq_features):
                    s_main = self.lq_proj.gate(s_main, lq_features[out_idx], sigma=degrade_sigma, out_idx=out_idx)

            s_main, y_emb = self.patch_blocks[i](
                s_main,
                y_emb,
                condition,
                pos,
                pos_txt,
                attn_mask_joint,
            )

            # Collect intermediate features for GAN discriminator
            if feature_indices is not None and i in feature_indices:
                collected_features.append(s_main.clone())

            if 0 < self.repa_encoder_index == (i + 1):
                self.last_repa_tokens = s_main

        return s_main, y_emb, collected_features

    def _unpatchify_features(self, features: list, Hs: int, Ws: int) -> list:
        """Reshape patch token features [B, L, D] → [B, D, Hs, Ws] for discriminator.

        PixDiT tokens are 1-to-1 with spatial patches (no sub-patch splitting in the
        token dimension), so we just reshape to a 2D spatial feature map.
        Compatible with Discriminator_ImageDiT which uses Conv2D heads.

        Under CP, collected features are rank-local [B, L_local, D]. We gather
        them along the token axis here so the discriminator (which has no CP
        plumbing) sees the full feature map.

        Args:
            features: List of [B, L_local_or_full, D] token tensors.
            Hs, Ws: Spatial patch grid dimensions (full).

        Returns:
            List of [B, D, Hs, Ws] tensors.
        """
        result = []
        for feat in features:
            if self._cp_group is not None:
                feat = cat_outputs_cp_with_grad(feat.contiguous(), seq_dim=1, cp_group=self._cp_group)
            B, _L, D = feat.shape
            result.append(feat.view(B, Hs, Ws, D).permute(0, 3, 1, 2))  # [B, D, Hs, Ws]
        return result

    def forward(
        self,
        x,
        t,
        y,
        s=None,
        mask=None,
        lq_video_or_image=None,
        lq_latent=None,
        lq_mask=None,
        degrade_sigma=None,
        # --- Feature extraction for GAN discriminator ---
        feature_indices=None,
        return_features_early: bool = False,
    ):
        B, _, H, W = x.shape
        Hs = H // self.patch_size
        Ws = W // self.patch_size
        L = Hs * Ws

        # Context-parallel local sequence length. When CP is enabled, every rank
        # sees the same full inputs (x, y, t, lq_*) — we patchify on full size,
        # then immediately split tokens along L so the heavy transformer/pixel
        # blocks operate on L_local = L / cp_size each.
        cp_group = self._cp_group
        cp_size = cp_group.size() if cp_group is not None else 1
        if cp_size > 1:
            assert L % cp_size == 0, f"L={L} not divisible by cp_size={cp_size}"
        L_local = L // cp_size

        # Compute LQ features (split along L internally when CP is active).
        has_lq = lq_video_or_image is not None or lq_latent is not None
        lq_features = self._compute_lq_features(lq_video_or_image, lq_latent, lq_mask, Hs, Ws) if has_lq else None

        collected_features = None  # populated by _run_patch_blocks when feature_indices is set

        # Patch tokens — full unfolding on every rank (cheap; identical across ranks).
        pos = self.fetch_pos(Hs, Ws, x.device)  # full pos; the CP-aware attention slices for q internally
        x_patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)

        # Text tokens (replicated across CP ranks; not split).
        if y.dim() != 3:
            raise ValueError("Text embedding y must be [B, L, D]")
        Ltxt = min(y.shape[1], self.txt_max_length)
        y = y[:, :Ltxt, :]
        y_emb = self.y_embedder(y).view(B, Ltxt, self.hidden_size)
        y_emb = y_emb + self.y_pos_embedding[:, :Ltxt, :].to(y_emb.dtype)

        # Condition signal: silu(t_emb), [B, 1, D]
        condition = torch.nn.functional.silu(t_emb)

        # Mask
        pad = None
        pos_txt = self.fetch_pos_text(Ltxt, x.device) if self.use_text_rope else None
        if mask is not None and isinstance(mask, torch.Tensor):
            m = mask
            while m.dim() > 2 and m.size(1) == 1:
                m = m.squeeze(1)
            if m.dim() == 3 and m.size(1) == 1:
                m = m.squeeze(1)
            if m.dim() == 2:
                pad = m == 0

        if s is None:
            s0 = self.s_embedder(x_patches)
            # Split image patch tokens across the CP group along the sequence axis.
            # Everything downstream (lq injection, patch_blocks, pixel pathway)
            # operates on the rank-local slice until the final fold gather.
            if cp_group is not None:
                s0 = split_inputs_cp(s0, seq_dim=1, cp_group=cp_group)
            self.last_repa_tokens = None

            if self.use_ed and self.encoder_ed is not None and self.decoder_ed is not None:
                # Encoder-decoder path (CP not supported here; PixDiT_T2I.enable_context_parallel asserts)
                H_tokens, W_tokens = Hs, Ws
                s_ed = s0 if self.s_ed_proj_in is None else self.s_ed_proj_in(s0)
                if self.s_ed_in_norm is not None:
                    s_ed = self.s_ed_in_norm(s_ed)
                c_ed = condition if self.s_ed_cond_proj is None else self.s_ed_cond_proj(condition)
                bottleneck, skip_tokens, Hb, Wb = self.encoder_ed(s_ed, H_tokens, W_tokens, c_ed)
                pos_b = self.fetch_pos(Hb, Wb, x.device)
                s_main = bottleneck if self.s_ed_proj_out is None else self.s_ed_proj_out(bottleneck)
                if self.s_ed_out_norm is not None:
                    s_main = self.s_ed_out_norm(s_main)
                s_main = torch.nn.functional.silu(t_emb + s_main)

                attn_mask_joint = None
                if pad is not None:
                    L_img_curr = s_main.shape[1]
                    pad_img = torch.zeros((B, L_img_curr), dtype=torch.bool, device=x.device)
                    pad_txt = (
                        pad[:, :Ltxt]
                        if pad.size(1) >= Ltxt
                        else torch.nn.functional.pad(pad, (0, Ltxt - pad.size(1)), value=True)
                    )
                    attn_mask_joint = torch.cat([pad_txt, pad_img], dim=1).view(B, 1, 1, Ltxt + L_img_curr)

                s_main, y_emb, collected_features = self._run_patch_blocks(
                    s_main,
                    y_emb,
                    condition,
                    pos_b,
                    pos_txt,
                    attn_mask_joint,
                    lq_features,
                    degrade_sigma=degrade_sigma,
                    feature_indices=feature_indices,
                )

                s_bottleneck2 = s_main if self.s_ed_proj_in is None else self.s_ed_proj_in(s_main)
                if self.s_ed_in_norm is not None:
                    s_bottleneck2 = self.s_ed_in_norm(s_bottleneck2)
                decoded, _, _ = self.decoder_ed(s_bottleneck2, Hb, Wb, skip_tokens, c_ed)
                s = decoded if self.s_ed_proj_out is None else self.s_ed_proj_out(decoded)
                if self.s_ed_out_norm is not None:
                    s = self.s_ed_out_norm(s)
                s = torch.nn.functional.silu(t_emb + s)
            else:
                # Standard path (no encoder-decoder).
                s_main = s0
                attn_mask_joint = None
                if pad is not None:
                    # SDPA's K dimension is full image length (CP gathers K/V across
                    # CP ranks inside the joint attention). Use full L for the K-side
                    # mask regardless of CP.
                    pad_img = torch.zeros((B, L), dtype=torch.bool, device=x.device)
                    pad_txt = (
                        pad[:, :Ltxt]
                        if pad.size(1) >= Ltxt
                        else torch.nn.functional.pad(pad, (0, Ltxt - pad.size(1)), value=True)
                    )
                    attn_mask_joint = torch.cat([pad_txt, pad_img], dim=1).view(B, 1, 1, Ltxt + L)

                s_main, y_emb, collected_features = self._run_patch_blocks(
                    s_main,
                    y_emb,
                    condition,
                    pos,
                    pos_txt,
                    attn_mask_joint,
                    lq_features,
                    degrade_sigma=degrade_sigma,
                    feature_indices=feature_indices,
                )

                s = torch.nn.functional.silu(t_emb + s_main)

        if not (0 < self.repa_encoder_index <= self.patch_depth):
            self.last_repa_tokens = s

        # Early exit for discriminator feature extraction (skip pixel blocks).
        # `_unpatchify_features` handles the CP all-gather along L internally.
        if return_features_early and feature_indices is not None and collected_features:
            return self._unpatchify_features(collected_features, Hs, Ws)

        # Ensure patch token length matches the rank-local grid (L_local under CP,
        # L otherwise). This guard exists for ED/token-shuffle paths where the
        # block stack may emit a different length than the input.
        batch_size, length, _ = s.shape
        if length != L_local:
            if length > L_local:
                s = s[:, :L_local, :]
            else:
                pad_len = L_local - length
                s = torch.cat([s, s.new_zeros(B, pad_len, s.shape[2])], dim=1)

        # Pixel pathway with optional PiT LQ injection — operates on rank-local
        # patches under CP. lq_features[-1] was already split along L in
        # `_compute_lq_features`, so its B*L_local view lines up with s.
        s_cond = s.reshape(B * L_local, self.hidden_size)
        if self.pit_lq_inject and lq_features is not None:
            pit_lq = lq_features[-1].reshape(B * L_local, self.hidden_size)
            sigma_flat = degrade_sigma.repeat_interleave(L_local) if degrade_sigma is not None else None
            s_cond = self.pit_lq_gate(s_cond, pit_lq, sigma=sigma_flat)

        # Pixel embedder runs on the full image (cheap; identical across CP
        # ranks). Reshape and slice to the rank-local subset of patches so that
        # the per-pixel branch processes exactly L_local patches.
        x_pixels = self.pixel_embedder(x, img_height=H, img_width=W, patch_size=self.patch_size)
        if cp_group is not None:
            P2 = self.patch_size * self.patch_size
            x_pixels = x_pixels.view(B, L, P2, self.pixel_hidden_size)
            x_pixels = split_inputs_cp(x_pixels, seq_dim=1, cp_group=cp_group)
            x_pixels = x_pixels.reshape(B * L_local, P2, self.pixel_hidden_size)
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask)

        x_pixels = self.final_layer(x_pixels)  # [B*L_local, P², C_out]
        C_out = self.out_channels
        P2 = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(B, L_local, P2, C_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(B, C_out * P2, L_local)
        # Gather pixel patches across CP ranks along L so `fold` reconstructs
        # the full image. `cat_outputs_cp_with_grad` keeps gradients on each
        # rank's local slice.
        if cp_group is not None:
            x_pixels = cat_outputs_cp_with_grad(x_pixels.contiguous(), seq_dim=2, cp_group=cp_group)
        output = torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)

        # Return (output, features) when feature extraction is enabled (without early exit)
        if feature_indices is not None and collected_features is not None:
            return output, self._unpatchify_features(collected_features, Hs, Ws)
        return output
