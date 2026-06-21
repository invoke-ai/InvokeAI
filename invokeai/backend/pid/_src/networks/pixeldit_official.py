# PixelDiT T2I — consolidated network architecture.
# Verbatim copy from the original PixelDiT repo, merged into a single file.
# Sources:
#   pixdit_core/modules.py        — building blocks (RMSNorm, RoPE, attention, etc.)
#   pixdit_core/pixeldit_c2i.py   — PatchTokenEmbedder, PixelTokenEmbedder, PiTBlock
#   pixdit_core/pixeldit_t2i.py   — MMDiT joint attention, encoder-decoder, PixDiT_T2I
#
# Only import statements were changed (everything is now local). Logic is unchanged.

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup
from torch.nn.functional import scaled_dot_product_attention

from invokeai.backend.pid._src.utils.context_parallel import cat_outputs_cp_with_grad

# =============================================================================
# From pixdit_core/modules.py
# =============================================================================


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def apply_adaln(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepConditioner(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        mlp_dtype = next(self.mlp.parameters()).dtype
        if t_freq.dtype != mlp_dtype:
            t_freq = t_freq.to(mlp_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return x


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0, scale=16.0):
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    freqs_cis = freqs_cis.reshape(height * width, -1)
    return freqs_cis


def precompute_freqs_cis_2d_ntk(
    dim: int,
    height: int,
    width: int,
    ref_grid_h: int,
    ref_grid_w: int,
    theta: float = 10000.0,
    scale: float = 16.0,
):
    """NTK-aware 2D RoPE.  Identical to precompute_freqs_cis_2d when
    height == ref_grid_h and width == ref_grid_w.  For other resolutions
    the base theta is scaled per-axis following the NTK-aware formula:
        ntk_factor = (current / ref) ** (dim_axis / (dim_axis - 2))
        theta_axis = theta * ntk_factor
    where dim_axis = dim // 2 (half the head dim per spatial axis).
    """
    dim_axis = dim // 2  # each axis gets dim//4 complex pairs → dim//2 real dims
    h_scale = height / ref_grid_h
    w_scale = width / ref_grid_w
    h_ntk = h_scale ** (dim_axis / (dim_axis - 2)) if dim_axis > 2 else 1.0
    w_ntk = w_scale ** (dim_axis / (dim_axis - 2)) if dim_axis > 2 else 1.0
    h_theta = theta * h_ntk
    w_theta = theta * w_ntk

    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)

    freqs_w = 1.0 / (w_theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_h = 1.0 / (h_theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    x_freqs = torch.outer(x_pos, freqs_w).float()
    y_freqs = torch.outer(y_pos, freqs_h).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    freqs_cis = freqs_cis.reshape(height * width, -1)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Context-parallel group; when set, `forward` runs split-Q / gather-K,V.
        self._cp_group: Optional[ProcessGroup] = None

    def set_context_parallel_group(self, cp_group: Optional[ProcessGroup]):
        self._cp_group = cp_group

    def forward(self, x: torch.Tensor, pos, mask) -> torch.Tensor:
        # CP convention: caller passes `pos` of full sequence length (N_full).
        # When `_cp_group` is set, `x` is the rank-local slice [B, N_local, C]
        # with N_local = N_full / cp_size. We gather k/v to full length, apply
        # RoPE with the appropriate slice/full pos, and run SDPA producing
        # local-Q output [B, N_local, C].
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self._cp_group is None:
            q, k = apply_rotary_emb(q, k, freqs_cis=pos)
        else:
            cp_size = self._cp_group.size()
            cp_rank = self._cp_group.rank()
            N_full = pos.shape[0]
            assert N_full % cp_size == 0, f"pos length {N_full} not divisible by cp_size {cp_size}"
            N_local = N_full // cp_size
            assert N == N_local, f"local x length {N} != expected {N_local}"
            pos_local = pos.view(cp_size, N_local, -1)[cp_rank]
            # Apply RoPE to local q with local pos.
            q, _ = apply_rotary_emb(q, q, freqs_cis=pos_local)
            # Gather k, v across CP ranks along the sequence dim, then RoPE with full pos.
            # `all_gather` requires contiguous tensors; the qkv permute leaves k/v as non-contiguous views.
            k = cat_outputs_cp_with_grad(k.contiguous(), seq_dim=1, cp_group=self._cp_group)
            v = cat_outputs_cp_with_grad(v.contiguous(), seq_dim=1, cp_group=self._cp_group)
            _, k = apply_rotary_emb(k, k, freqs_cis=pos)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()

        x = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x


# =============================================================================
# From pixdit_core/pixeldit_c2i.py (PatchTokenEmbedder, PixelTokenEmbedder, PiTBlock)
# =============================================================================


class PatchTokenEmbedder(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        bias: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class PixelTokenEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size_output: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size_output = int(hidden_size_output)
        self.proj = nn.Linear(self.in_channels, self.hidden_size_output, bias=True)
        self._pos_cache = {}

    def _fetch_pixel_pos_patch(self, patch_size: int, device, dtype):
        key = ("patch", patch_size)
        if key in self._pos_cache:
            pe = self._pos_cache[key]
            return pe.to(device=device, dtype=dtype)
        pos_np = get_2d_sincos_pos_embed(self.hidden_size_output, patch_size)
        pos = torch.from_numpy(pos_np).to(device=device, dtype=dtype)  # [P2, D]
        self._pos_cache[key] = pos
        return pos

    def _fetch_pixel_pos_image(self, height: int, width: int, device, dtype):
        if height == width:
            key = ("image", height, width)
            if key in self._pos_cache:
                pe = self._pos_cache[key]
                return pe.to(device=device, dtype=dtype)
            pos_np = get_2d_sincos_pos_embed(self.hidden_size_output, height)
            pos = torch.from_numpy(pos_np).to(device=device, dtype=dtype)  # [H*W, D]
            self._pos_cache[key] = pos
            return pos
        else:
            key = ("image", height, width)
            if key in self._pos_cache:
                pe = self._pos_cache[key]
                return pe.to(device=device, dtype=dtype)
            # Build a non-square grid (H x W) and compute 2D sin/cos embedding
            grid_h = np.arange(height, dtype=np.float32)
            grid_w = np.arange(width, dtype=np.float32)
            grid = np.meshgrid(grid_w, grid_h)  # w first to match existing convention
            grid = np.stack(grid, axis=0).reshape(2, 1, height, width)
            pos_np = get_2d_sincos_pos_embed_from_grid(self.hidden_size_output, grid)
            pos = torch.from_numpy(pos_np).to(device=device, dtype=dtype)  # [H*W, D]
            self._pos_cache[key] = pos
            return pos

    def forward(self, inputs: torch.Tensor, img_height: int = None, img_width: int = None, patch_size: int = None):
        # Two modes:
        # 1) Legacy patch mode: inputs [B*L, P2, C] -> add 2D sincos within patch (P2 = patch_size^2)
        # 2) Image mode: inputs [B, C, H, W] -> patchify inside and add full-image (H*W) pixel-space sincos sampled per patch
        if inputs.dim() == 3:
            # Legacy: [B*L, P2, C]
            batch_tokens, p2, _ = inputs.shape
            patch_sz = int(p2**0.5)
            pos = self._fetch_pixel_pos_patch(patch_sz, inputs.device, inputs.dtype)  # [P2, D]
            x = self.proj(inputs)
            x = x + pos.unsqueeze(0)
            return x
        elif inputs.dim() == 4:
            # Image mode: [B, C, H, W]
            assert img_height is not None and img_width is not None and patch_size is not None, (
                "Need H, W, patch_size for image mode"
            )
            B, C, H, W = inputs.shape
            assert H == img_height and W == img_width, "Input spatial size mismatch"
            assert (H % patch_size == 0) and (W % patch_size == 0), "H and W must be divisible by patch_size"
            Hs, Ws = H // patch_size, W // patch_size
            P2 = patch_size * patch_size
            # linear proj per pixel
            x = inputs.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            x = self.proj(x)  # [B, H, W, D]
            # full-image pixel-space pos
            pos_full = self._fetch_pixel_pos_image(H, W, inputs.device, inputs.dtype)  # [H*W, D]
            pos_full = pos_full.view(H, W, self.hidden_size_output)
            # add pos at image grid then patchify to [B*L, P2, D]
            x = x + pos_full.unsqueeze(0)
            x = x.view(B, Hs, patch_size, Ws, patch_size, self.hidden_size_output)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, Hs, Ws, ps, ps, D]
            x = x.view(B * Hs * Ws, P2, self.hidden_size_output)
            return x
        else:
            raise ValueError("PixelTokenEmbedder expects inputs of shape [B*L,P2,C] or [B,C,H,W]")


class PiTBlock(nn.Module):
    def __init__(
        self,
        pixel_hidden_size: int,
        patch_hidden_size: int,
        patch_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_hidden_size: Optional[int] = None,
        attn_num_heads: Optional[int] = None,
        rope_mode: str = "original",
        rope_ref_grid_h: int = 32,
        rope_ref_grid_w: int = 32,
    ):
        super().__init__()
        self.pixel_dim = int(pixel_hidden_size)
        self.context_dim = int(patch_hidden_size)
        self.patch_size = int(patch_size)
        self.attn_dim = int(attn_hidden_size) if attn_hidden_size is not None else self.context_dim
        self.num_heads = int(attn_num_heads) if attn_num_heads is not None else int(num_heads)
        self.rope_mode = rope_mode
        self.rope_ref_grid_h = rope_ref_grid_h
        self.rope_ref_grid_w = rope_ref_grid_w
        assert self.attn_dim % self.num_heads == 0, "pixel attention hidden size must be divisible by pixel num_heads"
        p2 = self.patch_size * self.patch_size
        self.compress_to_attn = nn.Linear(p2 * self.pixel_dim, self.attn_dim, bias=True)
        self.expand_from_attn = nn.Linear(self.attn_dim, p2 * self.pixel_dim, bias=True)
        self.norm1 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.attn = RotaryAttention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=mlp_ratio, drop=0.0)
        self.adaLN_modulation = nn.Sequential(nn.Linear(self.context_dim, 6 * self.pixel_dim * p2, bias=True))
        self._pos_cache = {}
        # CP group; when set, the attention runs split-Q / gather-K,V across L.
        self._cp_group: Optional[ProcessGroup] = None

    def set_context_parallel_group(self, cp_group: Optional[ProcessGroup]):
        self._cp_group = cp_group
        self.attn.set_context_parallel_group(cp_group)

    def _fetch_pos(self, height: int, width: int, device):
        key = (height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device)
        head_dim = self.attn_dim // self.num_heads
        if self.rope_mode == "ntk_aware":
            pos = precompute_freqs_cis_2d_ntk(head_dim, height, width, self.rope_ref_grid_h, self.rope_ref_grid_w).to(
                device
            )
        else:
            pos = precompute_freqs_cis_2d(head_dim, height, width).to(device)
        self._pos_cache[key] = pos
        return pos

    def forward(
        self, x: torch.Tensor, s_cond: torch.Tensor, image_height: int, image_width: int, patch_size: int, mask=None
    ) -> torch.Tensor:
        # x: [B*L_local, P2, C]; under CP, L_local = (Hs*Ws)/cp_size. Without CP,
        # L_local == L_full. The reshape uses L_local for the (B, L_local, ...)
        # axis; the inner attention all-gathers k/v back to full length.
        BL, P2, C = x.shape
        if C != self.pixel_dim:
            raise ValueError(f"PiTBlock expected pixel_dim={self.pixel_dim}, got {C}")
        assert patch_size == self.patch_size, "PiTBlock expects fixed patch_size"
        assert P2 == patch_size * patch_size, "Token count per patch must equal patch_size^2"
        assert (image_height % patch_size == 0) and (image_width % patch_size == 0), (
            "H and W must be divisible by patch_size"
        )
        Hs, Ws = image_height // patch_size, image_width // patch_size
        L = Hs * Ws
        cp_size = self._cp_group.size() if self._cp_group is not None else 1
        assert L % cp_size == 0, f"L={L} not divisible by cp_size={cp_size}"
        L_local = L // cp_size
        assert s_cond.shape[0] == BL, "s_cond batch must match x batch"
        assert BL % L_local == 0, "Total sequences must be a multiple of local patch count"
        B = BL // L_local
        # adaLN per pixel (within patch): params
        cond_params = self.adaLN_modulation(s_cond)  # [BL, 6*pixel_dim*P2]
        cond_params = cond_params.view(BL, P2, 6 * self.pixel_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond_params, 6, dim=-1)
        x_norm = apply_adaln(self.norm1(x), shift_msa, scale_msa)
        x_flat = x_norm.view(BL, P2 * self.pixel_dim)
        x_comp = self.compress_to_attn(x_flat).view(B, L_local, self.attn_dim)
        # attention across patch tokens (L) — pos is full-length; the CP-aware
        # RotaryAttention gathers k/v across CP ranks internally.
        pos_comp = self._fetch_pos(Hs, Ws, x.device)
        attn_out = self.attn(x_comp, pos_comp, mask)  # [B, L_local, attn_dim]
        attn_flat = self.expand_from_attn(attn_out.view(B * L_local, self.attn_dim))
        attn_exp = attn_flat.view(BL, P2, self.pixel_dim)
        # residual & MLP locally
        x = x + gate_msa * attn_exp
        mlp_out = self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp * mlp_out
        return x


# =============================================================================
# From pixdit_core/pixeldit_t2i.py
# =============================================================================


class MMDiTJointAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Separate QKV projections for image (x) and text (y) streams
        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Per-stream QK normalization (head-wise)
        self.q_norm_x = RMSNorm(self.head_dim)
        self.k_norm_x = RMSNorm(self.head_dim)
        self.q_norm_y = RMSNorm(self.head_dim)
        self.k_norm_y = RMSNorm(self.head_dim)

        # Output projections for each stream
        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop_x = nn.Dropout(proj_drop)
        self.proj_drop_y = nn.Dropout(proj_drop)
        # CP group for the image stream. Text is replicated across CP ranks.
        self._cp_group: Optional[ProcessGroup] = None

    def set_context_parallel_group(self, cp_group: Optional[ProcessGroup]):
        self._cp_group = cp_group

    def forward(
        self,
        x: torch.Tensor,  # [B, Nx, C] image stream (Nx = Nx_local under CP)
        y: torch.Tensor,  # [B, Ny, C] text stream (always full / replicated)
        pos_img: torch.Tensor,  # [Nx_full, head_dim/2] complex RoPE freqs (always full)
        pos_txt: torch.Tensor = None,  # [Ny, head_dim/2] complex RoPE freqs for text (optional)
        attn_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Nx, C = x.shape
        By, Ny, Cy = y.shape
        assert B == By and C == Cy, "x and y must share batch and channel dims"

        # QKV for image
        qkv_x = self.qkv_x(x).reshape(B, Nx, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qx, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]  # [B, Nx, H, Hc]
        qx = self.q_norm_x(qx)
        kx = self.k_norm_x(kx)

        # QKV for text
        qkv_y = self.qkv_y(y).reshape(B, Ny, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qy, ky, vy = qkv_y[0], qkv_y[1], qkv_y[2]  # [B, Ny, H, Hc]
        qy = self.q_norm_y(qy)
        ky = self.k_norm_y(ky)

        # Image RoPE — under CP, q uses the rank-local slice of pos_img, k (after
        # all-gather along the sequence dim) uses the full pos_img.
        if self._cp_group is None:
            qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        else:
            cp_size = self._cp_group.size()
            cp_rank = self._cp_group.rank()
            Nx_full = pos_img.shape[0]
            assert Nx_full % cp_size == 0, f"pos_img length {Nx_full} not divisible by cp_size {cp_size}"
            Nx_local = Nx_full // cp_size
            assert Nx == Nx_local, f"local image stream length {Nx} != expected {Nx_local}"
            pos_img_local = pos_img.view(cp_size, Nx_local, -1)[cp_rank]
            qx, _ = apply_rotary_emb(qx, qx, freqs_cis=pos_img_local)
            # `all_gather` requires contiguous tensors; the qkv permute leaves k/v as non-contiguous views.
            kx = cat_outputs_cp_with_grad(kx.contiguous(), seq_dim=1, cp_group=self._cp_group)
            vx = cat_outputs_cp_with_grad(vx.contiguous(), seq_dim=1, cp_group=self._cp_group)
            _, kx = apply_rotary_emb(kx, kx, freqs_cis=pos_img)
        if pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)

        # SDPA expects [B, H, S, Hc]; build joint sequence [text, image].
        # Under CP: qx is [B, H, Nx_local, Hc]; kx, vx are [B, H, Nx_full, Hc].
        qx = qx.transpose(1, 2)
        kx = kx.transpose(1, 2)
        vx = vx.transpose(1, 2)

        qy = qy.transpose(1, 2)  # [B, H, Ny, Hc]
        ky = ky.transpose(1, 2)
        vy = vy.transpose(1, 2)

        q_joint = torch.cat([qy, qx], dim=2)  # [B, H, Ny + Nx_local, Hc]
        k_joint = torch.cat([ky, kx], dim=2)  # [B, H, Ny + Nx_full,  Hc]
        v_joint = torch.cat([vy, vx], dim=2)

        out_joint = F.scaled_dot_product_attention(q_joint, k_joint, v_joint, dropout_p=0.0, attn_mask=attn_mask)
        # Split back to [text, image]; image output is local under CP.
        out_y = out_joint[:, :, :Ny, :]
        out_x = out_joint[:, :, Ny:, :]

        # Merge heads
        out_y = out_y.transpose(1, 2).reshape(B, Ny, C)
        out_x = out_x.transpose(1, 2).reshape(B, Nx, C)

        # Output projections
        out_x = self.proj_drop_x(self.proj_x(out_x))
        out_y = self.proj_drop_y(self.proj_y(out_y))
        return out_x, out_y


class MMDiTBlockT2I(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4.0, adaLN_modulation_img=None, adaLN_modulation_txt=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.groups = groups
        self.head_dim = hidden_size // groups

        # Per-stream norms
        self.norm_x1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = RMSNorm(hidden_size, eps=1e-6)

        self.attn = MMDiTJointAttention(hidden_size, num_heads=groups, qkv_bias=False)

        self.norm_x2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = RMSNorm(hidden_size, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden_dim)

        # Per-stream AdaLN modulation
        self.adaLN_modulation_img = (
            adaLN_modulation_img
            if adaLN_modulation_img is not None
            else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        )
        self.adaLN_modulation_txt = (
            adaLN_modulation_txt
            if adaLN_modulation_txt is not None
            else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        )

    def set_context_parallel_group(self, cp_group: Optional[ProcessGroup]):
        # The block itself has no CP-affecting state; only the joint attention does.
        self.attn.set_context_parallel_group(cp_group)

    def forward(self, x, y, c, pos_img, pos_txt=None, attn_mask=None):
        # c: [B, 1, C] typically, broadcast across tokens
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(c).chunk(
            6, dim=-1
        )
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(c).chunk(
            6, dim=-1
        )

        # 1) Joint attention with dual-stream
        x_norm = apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
        attn_x, attn_y = self.attn(x_norm, y_norm, pos_img, pos_txt, attn_mask)
        x = x + gate_msa_x * attn_x
        y = y + gate_msa_y * attn_y

        # 2) Per-stream MLP with AdaLN
        x = x + gate_mlp_x * self.mlp_x(apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x))
        y = y + gate_mlp_y * self.mlp_y(apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y))
        return x, y


def _compute_num_stages_from_ratio(compress_ratio: int) -> int:
    if compress_ratio <= 1:
        return 0
    if compress_ratio & (compress_ratio - 1) != 0:
        raise ValueError(f"ed_compress_ratio must be power of 2, got {compress_ratio}")
    return int(math.log2(compress_ratio))


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        use_token_compression: bool = False,
        token_shuffle_window_size: int = 1,
        rope_mode: str = "original",
        rope_ref_grid_h: int = 32,
        rope_ref_grid_w: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.rope_mode = rope_mode
        self.rope_ref_grid_h = rope_ref_grid_h
        self.rope_ref_grid_w = rope_ref_grid_w
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = RotaryAttention(dim, num_heads=num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.adaLN_modulation = nn.Sequential(nn.Linear(dim, 6 * dim, bias=True))
        self.use_token_compression = bool(use_token_compression)
        ts_ws = int(token_shuffle_window_size) if self.use_token_compression else 1

        if self.use_token_compression and ts_ws > 1:

            class _AttnTokenShuffleCompression(nn.Module):
                def __init__(self):
                    super().__init__()
                    s2 = ts_ws * ts_ws
                    adapted_hidden = ((dim + s2 - 1) // s2) * s2
                    needs_adapter_in = adapted_hidden != dim
                    compressed_dim = adapted_hidden // s2
                    self.s = ts_ws
                    self.adapted_hidden = adapted_hidden
                    self.compressed_dim = compressed_dim
                    self.adapter_in = (
                        nn.Sequential(nn.Linear(dim, adapted_hidden, bias=True), nn.GELU())
                        if needs_adapter_in
                        else nn.Identity()
                    )
                    self.proj_down = nn.Linear(adapted_hidden, compressed_dim, bias=True)
                    self.proj_to_attn = (
                        nn.Identity() if adapted_hidden == dim else nn.Linear(adapted_hidden, dim, bias=True)
                    )

                def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
                    B, N, C = x.shape
                    assert N == height * width, f"Token count {N} != {height}*{width}"
                    s = self.s
                    assert height % s == 0 and width % s == 0, (
                        f"Height {height} and Width {width} must be divisible by token shuffle size {s}"
                    )
                    x = x.view(B, height, width, C)
                    x = self.adapter_in(x)
                    x = self.proj_down(x)
                    c_per = self.compressed_dim
                    x = x.view(B, height // s, s, width // s, s, c_per)
                    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
                    x = x.view(B, (height // s) * (width // s), s * s * c_per)
                    x = self.proj_to_attn(x)
                    return x

            class _AttnTokenShuffleExpansion(nn.Module):
                def __init__(self):
                    super().__init__()
                    s2 = ts_ws * ts_ws
                    adapted_hidden = ((dim + s2 - 1) // s2) * s2
                    needs_adapter_out = adapted_hidden != dim
                    compressed_dim = adapted_hidden // s2
                    self.s = ts_ws
                    self.adapted_hidden = adapted_hidden
                    self.compressed_dim = compressed_dim
                    self.proj_from_attn = (
                        nn.Identity() if adapted_hidden == dim else nn.Linear(dim, adapted_hidden, bias=True)
                    )
                    self.proj_up = nn.Sequential(nn.Linear(compressed_dim, adapted_hidden, bias=True), nn.GELU())
                    self.adapter_out = (
                        nn.Sequential(nn.Linear(adapted_hidden, dim, bias=True), nn.GELU())
                        if needs_adapter_out
                        else nn.Identity()
                    )

                def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
                    B, Np, C = x.shape
                    s = self.s
                    Hs, Ws = height // s, width // s
                    assert Np == Hs * Ws, f"Token count {Np} != {Hs}*{Ws}"
                    x = self.proj_from_attn(x)
                    c_per = self.compressed_dim
                    x = x.view(B, Hs, Ws, s, s, c_per)
                    x_flat = x.reshape(B * Hs * Ws * s * s, c_per)
                    x_expanded = self.proj_up(x_flat)
                    x_expanded = x_expanded.view(B, Hs, Ws, s, s, self.adapted_hidden)
                    x_expanded = x_expanded.permute(0, 1, 3, 2, 4, 5).contiguous()
                    x_expanded = x_expanded.view(B, Hs * s, Ws * s, self.adapted_hidden)
                    x_expanded = self.adapter_out(x_expanded)
                    x_expanded = x_expanded.view(B, height * width, dim)
                    return x_expanded

            self._ts_compress = _AttnTokenShuffleCompression()
            self._ts_expand = _AttnTokenShuffleExpansion()
        else:
            self._ts_compress = None
            self._ts_expand = None

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        pos: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        use_ts = (
            self.use_token_compression
            and self._ts_compress is not None
            and self._ts_expand is not None
            and height is not None
            and width is not None
        )
        if use_ts:
            x_norm = apply_adaln(self.norm1(x), shift_msa, scale_msa)
            x_comp = self._ts_compress(x_norm, height, width)
            s = self._ts_compress.s
            Hs, Ws = height // s, width // s
            head_dim = self.dim // self.num_heads
            if self.rope_mode == "ntk_aware":
                pos_comp = precompute_freqs_cis_2d_ntk(head_dim, Hs, Ws, self.rope_ref_grid_h, self.rope_ref_grid_w).to(
                    x.device
                )
            else:
                pos_comp = precompute_freqs_cis_2d(head_dim, Hs, Ws).to(x.device)
            attn_out = self.attn(x_comp, pos_comp, mask)
            attn_out = self._ts_expand(attn_out, height, width)
            x = x + gate_msa * attn_out
        else:
            attn_out = self.attn(apply_adaln(self.norm1(x), shift_msa, scale_msa), pos, mask)
            x = x + gate_msa * attn_out
        x = x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        return x


class _PatchMerging(nn.Module):
    def __init__(self, hidden_size: int, window_size: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = int(window_size)
        s2 = self.window_size * self.window_size
        self.adapted_hidden = ((hidden_size + s2 - 1) // s2) * s2
        self.needs_adapter = self.adapted_hidden != hidden_size
        self.adapter_in = (
            nn.Sequential(nn.Linear(hidden_size, self.adapted_hidden, bias=True), nn.GELU())
            if self.needs_adapter
            else nn.Identity()
        )
        self.compressed_dim = self.adapted_hidden // s2
        self.proj_down = nn.Linear(self.adapted_hidden, self.compressed_dim, bias=True)
        self.proj_to_hidden = (
            nn.Identity()
            if self.adapted_hidden == hidden_size
            else nn.Sequential(nn.Linear(self.adapted_hidden, hidden_size, bias=True), nn.GELU())
        )

    def forward(self, x: torch.Tensor, height: int, width: int):
        B, N, C = x.shape
        assert N == height * width, f"Token count {N} doesn't match H*W={height * width}"
        s = self.window_size
        assert height % s == 0 and width % s == 0, f"H and W must be divisible by {s}"
        x = x.view(B, height, width, C)
        x = self.adapter_in(x)
        x = self.proj_down(x)
        c_per = self.compressed_dim
        x = x.view(B, height // s, s, width // s, s, c_per)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, (height // s) * (width // s), s * s * c_per)
        x = self.proj_to_hidden(x)
        return x, height // s, width // s


class _PatchExpanding(nn.Module):
    def __init__(self, hidden_size: int, window_size: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = int(window_size)
        s2 = self.window_size * self.window_size
        self.adapted_hidden = ((hidden_size + s2 - 1) // s2) * s2
        self.needs_adapter = self.adapted_hidden != hidden_size
        self.proj_from_hidden = (
            nn.Identity()
            if self.adapted_hidden == hidden_size
            else nn.Linear(hidden_size, self.adapted_hidden, bias=True)
        )
        self.compressed_dim = self.adapted_hidden // s2
        self.proj_up = nn.Sequential(nn.Linear(self.compressed_dim, self.adapted_hidden, bias=True), nn.GELU())
        self.adapter_out = (
            nn.Sequential(nn.Linear(self.adapted_hidden, hidden_size, bias=True), nn.GELU())
            if self.needs_adapter
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, height: int, width: int):
        B, Np, C = x.shape
        Hs, Ws = height, width
        s = self.window_size
        x = self.proj_from_hidden(x)
        c_per = self.adapted_hidden // (s * s)
        x = x.view(B, Hs, Ws, s, s, c_per)
        x_flat = x.reshape(B * Hs * Ws * s * s, c_per)
        x_expanded = self.proj_up(x_flat)
        x_expanded = x_expanded.view(B, Hs, Ws, s, s, self.adapted_hidden)
        x_expanded = x_expanded.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_expanded = x_expanded.view(B, Hs * s, Ws * s, self.adapted_hidden)
        x_expanded = self.adapter_out(x_expanded)
        x_expanded = x_expanded.view(B, (Hs * s) * (Ws * s), self.hidden_size)
        return x_expanded, Hs * s, Ws * s


class _EncoderED(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_stages: int,
        depth_per_stage: int = 1,
        num_heads: int = 8,
        window_size: int = 2,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        use_attn_token_shuffle: bool = False,
        rope_mode: str = "original",
        rope_ref_grid_h: int = 32,
        rope_ref_grid_w: int = 32,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_stages = int(num_stages)
        self.window_size = int(window_size)
        self.use_attn_token_shuffle = bool(use_attn_token_shuffle)
        self.rope_mode = rope_mode
        self.rope_ref_grid_h = rope_ref_grid_h
        self.rope_ref_grid_w = rope_ref_grid_w
        self._pos_cache = {}
        stages = []
        for i_stage in range(self.num_stages):
            ts_ws = 2 ** (self.num_stages - i_stage) if self.use_attn_token_shuffle else 1
            blocks = nn.ModuleList(
                [
                    _TransformerBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio,
                        drop,
                        use_token_compression=self.use_attn_token_shuffle,
                        token_shuffle_window_size=ts_ws,
                        rope_mode=rope_mode,
                        rope_ref_grid_h=rope_ref_grid_h,
                        rope_ref_grid_w=rope_ref_grid_w,
                    )
                    for _ in range(int(depth_per_stage))
                ]
            )
            compress = _PatchMerging(hidden_size, window_size=self.window_size)
            stages.append(nn.ModuleDict({"blocks": blocks, "compress": compress}))
        self.stages = nn.ModuleList(stages)

    def _fetch_pos(self, height: int, width: int, device: torch.device):
        key = (height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device)
        head_dim = self.hidden_size // self.num_heads
        if self.rope_mode == "ntk_aware":
            pos = precompute_freqs_cis_2d_ntk(head_dim, height, width, self.rope_ref_grid_h, self.rope_ref_grid_w).to(
                device
            )
        else:
            pos = precompute_freqs_cis_2d(head_dim, height, width).to(device)
        self._pos_cache[key] = pos
        return pos

    def forward(self, x: torch.Tensor, height: int, width: int, c: torch.Tensor):
        H, W = height, width
        skip_tokens = []
        for stage in self.stages:
            for blk in stage["blocks"]:
                pos = self._fetch_pos(H, W, x.device)
                x = blk(x, c, pos, None, H, W) if self.use_attn_token_shuffle else blk(x, c, pos, None)
            skip_tokens.append(x)
            x, H, W = stage["compress"](x, H, W)
        return x, skip_tokens, H, W


class _DecoderED(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_stages: int,
        depth_per_stage: int = 1,
        num_heads: int = 8,
        window_size: int = 2,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        use_attn_token_shuffle: bool = False,
        rope_mode: str = "original",
        rope_ref_grid_h: int = 32,
        rope_ref_grid_w: int = 32,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_stages = int(num_stages)
        self.window_size = int(window_size)
        self.use_attn_token_shuffle = bool(use_attn_token_shuffle)
        self.rope_mode = rope_mode
        self.rope_ref_grid_h = rope_ref_grid_h
        self.rope_ref_grid_w = rope_ref_grid_w
        self._pos_cache = {}
        stages = []
        for i_stage in range(self.num_stages):
            ts_ws = 2**i_stage if self.use_attn_token_shuffle else 1
            blocks = nn.ModuleList(
                [
                    _TransformerBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio,
                        drop,
                        use_token_compression=self.use_attn_token_shuffle,
                        token_shuffle_window_size=ts_ws,
                        rope_mode=rope_mode,
                        rope_ref_grid_h=rope_ref_grid_h,
                        rope_ref_grid_w=rope_ref_grid_w,
                    )
                    for _ in range(int(depth_per_stage))
                ]
            )
            expand = _PatchExpanding(hidden_size, window_size=self.window_size)
            stages.append(nn.ModuleDict({"blocks": blocks, "expand": expand}))
        self.stages = nn.ModuleList(stages)

    def _fetch_pos(self, height: int, width: int, device: torch.device):
        key = (height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device)
        head_dim = self.hidden_size // self.num_heads
        if self.rope_mode == "ntk_aware":
            pos = precompute_freqs_cis_2d_ntk(head_dim, height, width, self.rope_ref_grid_h, self.rope_ref_grid_w).to(
                device
            )
        else:
            pos = precompute_freqs_cis_2d(head_dim, height, width).to(device)
        self._pos_cache[key] = pos
        return pos

    def forward(self, x: torch.Tensor, bottleneck_h: int, bottleneck_w: int, skip_tokens, c: torch.Tensor):
        H, W = bottleneck_h, bottleneck_w
        for i, stage in enumerate(self.stages):
            for blk in stage["blocks"]:
                pos = self._fetch_pos(H, W, x.device)
                x = blk(x, c, pos, None, H, W) if self.use_attn_token_shuffle else blk(x, c, pos, None)
            x, H, W = stage["expand"](x, H, W)
            skip_idx = len(self.stages) - 1 - i
            if 0 <= skip_idx < len(skip_tokens):
                skip = skip_tokens[skip_idx]
                expected_tokens = H * W
                if skip.shape[1] == expected_tokens:
                    x = x + skip
        return x, H, W


# =============================================================================
# Main T2I network: PixDiT_T2I
# =============================================================================


class PixDiT_T2I(nn.Module):
    def __init__(
        self,
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
        # NTK-aware RoPE: set rope_mode="ntk_aware" and provide the reference
        # pixel resolution used during training.  When the actual grid size
        # differs from ref, the base theta is scaled per-axis.
        rope_mode: str = "original",  # "original" | "ntk_aware"
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
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.num_text_blocks = int(num_text_blocks)
        self.patch_size = int(patch_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.txt_embed_dim = int(txt_embed_dim)
        self.txt_max_length = int(txt_max_length)
        self.use_text_rope = bool(use_text_rope)
        self.text_rope_theta = float(text_rope_theta)
        self.rope_mode = rope_mode
        self.rope_ref_grid_h = rope_ref_h // self.patch_size
        self.rope_ref_grid_w = rope_ref_w // self.patch_size
        self.repa_encoder_index = int(repa_encoder_index)
        if self.pixel_depth <= 0:
            raise ValueError("PixDiT_T2I expects pixel_depth > 0 to retain the pixel pathway")

        # Embedders
        self.pixel_embedder = PixelTokenEmbedder(in_channels, self.pixel_hidden_size)
        self.s_embedder = PatchTokenEmbedder(in_channels * patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(hidden_size)
        self.y_embedder = PatchTokenEmbedder(self.txt_embed_dim, hidden_size, bias=True, norm_layer=RMSNorm)
        self.y_pos_embedding = nn.Parameter(torch.randn(1, self.txt_max_length, hidden_size))

        # Blocks
        # Shared AdaLN modulator for conditional blocks (optional)
        self._shared_cond_adaln = None
        self._shared_cond_adaln_img = None
        self._shared_cond_adaln_txt = None
        self.patch_blocks = nn.ModuleList(
            [
                MMDiTBlockT2I(
                    self.hidden_size,
                    self.num_groups,
                    adaLN_modulation_img=self._shared_cond_adaln_img,
                    adaLN_modulation_txt=self._shared_cond_adaln_txt,
                )
                for _ in range(self.patch_depth)
            ]
        )
        # Remove AdaLN-based text refinement; PixDiT keeps cross-attn-only text handling
        self.text_refine_blocks = None
        self.pixel_attn_hidden_size = (
            int(pixel_attn_hidden_size) if pixel_attn_hidden_size is not None else self.hidden_size
        )
        self.pixel_num_groups = int(pixel_num_groups) if pixel_num_groups is not None else self.num_groups
        self.pixel_blocks = nn.ModuleList(
            [
                PiTBlock(
                    self.pixel_hidden_size,
                    self.hidden_size,
                    patch_size=self.patch_size,
                    num_heads=self.num_groups,
                    mlp_ratio=4.0,
                    attn_hidden_size=self.pixel_attn_hidden_size,
                    attn_num_heads=self.pixel_num_groups,
                    rope_mode=self.rope_mode,
                    rope_ref_grid_h=self.rope_ref_grid_h,
                    rope_ref_grid_w=self.rope_ref_grid_w,
                )
                for _ in range(self.pixel_depth)
            ]
        )

        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels)

        self.precompute_pos = {}
        self.precompute_pos_txt = {}  # cache for 1D text RoPE
        self.last_repa_tokens = None

        self.enable_ed = bool(enable_ed)
        self.ed_compress_ratio = int(ed_compress_ratio)
        self.ed_depth_per_stage = int(ed_depth_per_stage)
        self.ed_window_size = int(ed_window_size)
        self.ed_num_heads = int(ed_num_heads) if ed_num_heads is not None else self.num_groups
        self.ed_hidden_size = int(ed_hidden_size) if ed_hidden_size is not None else self.hidden_size
        self.ed_use_token_shuffle = bool(ed_use_token_shuffle)
        self.encoder_ed: Optional[_EncoderED] = None
        self.decoder_ed: Optional[_DecoderED] = None
        self.s_ed_proj_in: Optional[nn.Module] = None
        self.s_ed_proj_out: Optional[nn.Module] = None
        self.s_ed_cond_proj: Optional[nn.Module] = None
        self.s_ed_in_norm: Optional[RMSNorm] = None
        self.s_ed_out_norm: Optional[RMSNorm] = None
        num_stages = _compute_num_stages_from_ratio(self.ed_compress_ratio) if self.enable_ed else 0
        self.use_ed = self.enable_ed and num_stages > 0
        if self.use_ed:
            if self.ed_hidden_size % self.ed_num_heads != 0:
                raise ValueError(
                    f"ed_hidden_size {self.ed_hidden_size} must be divisible by ed_num_heads {self.ed_num_heads}"
                )
            self.s_ed_proj_in = (
                nn.Identity()
                if self.ed_hidden_size == self.hidden_size
                else nn.Linear(self.hidden_size, self.ed_hidden_size, bias=True)
            )
            self.s_ed_proj_out = (
                nn.Identity()
                if self.ed_hidden_size == self.hidden_size
                else nn.Linear(self.ed_hidden_size, self.hidden_size, bias=True)
            )
            self.s_ed_cond_proj = (
                nn.Identity()
                if self.ed_hidden_size == self.hidden_size
                else nn.Linear(self.hidden_size, self.ed_hidden_size, bias=True)
            )
            self.s_ed_in_norm = RMSNorm(self.ed_hidden_size, eps=1e-6)
            self.s_ed_out_norm = RMSNorm(self.hidden_size, eps=1e-6)
            self.encoder_ed = _EncoderED(
                hidden_size=self.ed_hidden_size,
                num_stages=num_stages,
                depth_per_stage=self.ed_depth_per_stage,
                num_heads=self.ed_num_heads,
                window_size=self.ed_window_size,
                use_attn_token_shuffle=self.ed_use_token_shuffle,
                rope_mode=self.rope_mode,
                rope_ref_grid_h=self.rope_ref_grid_h,
                rope_ref_grid_w=self.rope_ref_grid_w,
            )
            self.decoder_ed = _DecoderED(
                hidden_size=self.ed_hidden_size,
                num_stages=num_stages,
                depth_per_stage=self.ed_depth_per_stage,
                num_heads=self.ed_num_heads,
                window_size=self.ed_window_size,
                use_attn_token_shuffle=self.ed_use_token_shuffle,
                rope_mode=self.rope_mode,
                rope_ref_grid_h=self.rope_ref_grid_h,
                rope_ref_grid_w=self.rope_ref_grid_w,
            )

        self.initialize_weights()

        # Context-parallel state — set by `enable_context_parallel`. The base
        # class does not split tokens itself; subclasses (e.g. PidNet)
        # are responsible for splitting along L in `forward` and gathering
        # before the final fold. This attribute is propagated to every patch
        # block (joint MMDiT attention) and pixel block (RotaryAttention).
        self._cp_group: Optional[ProcessGroup] = None
        self._is_context_parallel_enabled: bool = False

    @property
    def is_context_parallel_enabled(self) -> bool:
        return self._is_context_parallel_enabled

    def enable_context_parallel(self, cp_group: ProcessGroup):
        # CP for the ED (encoder-decoder) path is not implemented; refuse to
        # enable CP if the network was built with use_ed=True so we don't
        # silently produce wrong results.
        if self.use_ed:
            raise NotImplementedError(
                "PixDiT_T2I context parallel is not implemented for the encoder-decoder path. "
                "Build with enable_ed=False to use CP."
            )
        for block in self.patch_blocks:
            block.set_context_parallel_group(cp_group)
        for block in self.pixel_blocks:
            block.set_context_parallel_group(cp_group)
        self._cp_group = cp_group
        self._is_context_parallel_enabled = True

    def disable_context_parallel(self):
        for block in self.patch_blocks:
            block.set_context_parallel_group(None)
        for block in self.pixel_blocks:
            block.set_context_parallel_group(None)
        self._cp_group = None
        self._is_context_parallel_enabled = False

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        head_dim = self.hidden_size // self.num_groups
        if self.rope_mode == "ntk_aware":
            pos = precompute_freqs_cis_2d_ntk(head_dim, height, width, self.rope_ref_grid_h, self.rope_ref_grid_w).to(
                device
            )
        else:
            pos = precompute_freqs_cis_2d(head_dim, height, width).to(device)
        self.precompute_pos[(height, width)] = pos
        return pos

    def fetch_pos_text(self, length, device):
        if length in self.precompute_pos_txt:
            return self.precompute_pos_txt[length].to(device)
        # Build 1D RoPE freqs for text stream using the same per-head dim as image
        head_dim = self.hidden_size // self.num_groups
        # Create frequencies for complex rotation: [length, head_dim//2]
        freqs = 1.0 / (self.text_rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        positions = torch.arange(0, length, device=device).float().unsqueeze(1)  # [length,1]
        angles = positions * freqs.unsqueeze(0)  # [length, head_dim//2]
        freqs_cis = torch.polar(torch.ones_like(angles), angles)  # complex64/complex32
        self.precompute_pos_txt[length] = freqs_cis
        return freqs_cis

    def initialize_weights(self):
        # Initialize s_embedder like nn.Linear
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # zero init final layer
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(self, x, t, y, s=None, mask=None):
        B, _, H, W = x.shape
        # Derive grid token count deterministically from spatial size
        Hs = H // self.patch_size
        Ws = W // self.patch_size
        L = Hs * Ws

        # Patch tokens for condition pathway
        pos = self.fetch_pos(Hs, Ws, x.device)
        x_patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)

        # Text tokens -> project to hidden_size and add learned pos
        if y.dim() != 3:
            raise ValueError("Text embedding y must be [B, L, D]")
        Ltxt = min(y.shape[1], self.txt_max_length)
        y = y[:, :Ltxt, :]
        y_emb = self.y_embedder(y).view(B, Ltxt, self.hidden_size)
        y_emb = y_emb + self.y_pos_embedding[:, :Ltxt, :].to(y_emb.dtype)

        # PixDiT design: no AdaLN modulation applied on text stream
        condition = torch.nn.functional.silu(t_emb)

        # Condition blocks on patch tokens with MM-DiT joint attention to text tokens
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
            self.last_repa_tokens = None
            if self.use_ed and self.encoder_ed is not None and self.decoder_ed is not None:
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

                for i in range(self.patch_depth):
                    s_main, y_emb = self.patch_blocks[i](s_main, y_emb, condition, pos_b, pos_txt, attn_mask_joint)
                    if 0 < self.repa_encoder_index == (i + 1):
                        self.last_repa_tokens = s_main
                s_bottleneck2 = s_main if self.s_ed_proj_in is None else self.s_ed_proj_in(s_main)
                if self.s_ed_in_norm is not None:
                    s_bottleneck2 = self.s_ed_in_norm(s_bottleneck2)
                decoded, _, _ = self.decoder_ed(s_bottleneck2, Hb, Wb, skip_tokens, c_ed)
                s = decoded if self.s_ed_proj_out is None else self.s_ed_proj_out(decoded)
                if self.s_ed_out_norm is not None:
                    s = self.s_ed_out_norm(s)
                s = torch.nn.functional.silu(t_emb + s)
            else:
                s_main = s0
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

                for i in range(self.patch_depth):
                    s_main, y_emb = self.patch_blocks[i](s_main, y_emb, condition, pos, pos_txt, attn_mask_joint)
                    if 0 < self.repa_encoder_index == (i + 1):
                        self.last_repa_tokens = s_main
                s = torch.nn.functional.silu(t_emb + s_main)
        # If no valid tap index is specified, expose last conditional output
        if not (0 < self.repa_encoder_index <= self.patch_depth):
            self.last_repa_tokens = s

        # Ensure the patch token length matches the spatial grid L
        batch_size, length, _ = s.shape
        if length != L:
            if length > L:
                s = s[:, :L, :]
            else:
                pad_len = L - length
                s = torch.cat([s, s.new_zeros(B, pad_len, s.shape[2])], dim=1)
            length = L

        # Pixel pathway
        s_cond = s.view(B * L, self.hidden_size)
        x_pixels = self.pixel_embedder(x, img_height=H, img_width=W, patch_size=self.patch_size)
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask)

        # Project back to image and fold
        x_pixels = self.final_layer(x_pixels)  # [B*L, P2, C]
        C_out = self.out_channels
        P2 = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(B, L, P2, C_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(B, C_out * P2, L)
        x_img = torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return x_img
