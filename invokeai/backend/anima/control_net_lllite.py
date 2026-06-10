"""ControlNet-LLLite adapter for Anima (DiT), v2 weight format.

A LLLite adapter is a shared conv trunk (``conditioning1``) that encodes a
conditioning image into per-token embeddings, plus one tiny zero-init MLP per
target Linear in the DiT. Each module perturbs its Linear's *input*:
``y = org_forward(x + up(film_mlp(x, cond + depth_embed)) * multiplier)``, so
multiplier 0 or a missing cond image is an exact passthrough.

On-disk format (v2, named-key): shared trunk under ``lllite_conditioning1.*``,
per-module weights under ``lllite_dit_blocks_{i}_{target}.{down,mid,cond_to_film,up}.*``
plus a per-module ``.depth_embed``; hyperparams in safetensors metadata
(``lllite.*`` keys) with state-dict-shape fallbacks.

Unlike the reference implementation, the model is constructed from
``(state_dict, metadata)`` alone — no transformer instance is needed until
:meth:`AnimaControlNetLLLite.apply_to` binds the modules to the target Linears
by their saved names.

Original source code:
- kohya-ss ControlNet-LLLite for Anima: ComfyUI-Anima-LLLite port
  (``control_net_lllite_anima.py``, ``nodes.py``) of kohya-ss/sd-scripts
  ``networks/control_net_lllite_anima.py``.
  SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

ASPP_DEFAULT_DILATIONS: tuple[int, ...] = (1, 2, 4, 8)

_SAVED_COND_PREFIX = "lllite_conditioning1."
_INTERNAL_COND_PREFIX = "conditioning1."
_INTERNAL_MODULES_PREFIX = "lllite_modules."
_LEGACY_MODULES_PREFIX = "lllite_modules."

MODULE_NAME_PATTERN = re.compile(
    r"^lllite_dit_blocks_(\d+)_(self_attn_q_proj|self_attn_k_proj|self_attn_v_proj|cross_attn_q_proj|mlp_layer1)$"
)

# Saved module name suffix -> attribute path under transformer.blocks[i].
_SUFFIX_TO_ATTR_PATH: dict[str, tuple[str, ...]] = {
    "self_attn_q_proj": ("self_attn", "q_proj"),
    "self_attn_k_proj": ("self_attn", "k_proj"),
    "self_attn_v_proj": ("self_attn", "v_proj"),
    "cross_attn_q_proj": ("cross_attn", "q_proj"),
    "mlp_layer1": ("mlp", "layer1"),
}
_SUFFIX_ORDER = list(_SUFFIX_TO_ATTR_PATH)


# ----------------------------------------------------------------------------
# Conditioning image preprocessing (torch-only, PIL-free)
# ----------------------------------------------------------------------------


def target_cond_hw(latent_h: int, latent_w: int, patch_spatial: int = 2) -> tuple[int, int]:
    """Return the (H, W) the cond image / mask must be resized to.

    The LLLite ``conditioning1`` trunk has total conv stride 16, so the cond
    image must be sized to ``latent_HW * 8`` in input pixel space (= token_HW
    * 16 after DiT patchify with patch_spatial=2). The DiT internally pads the
    latent up to a multiple of ``patch_spatial`` before patchify, so the same
    rounding is mirrored here — otherwise odd latent dims yield a token-count
    mismatch that silently bypasses every LLLite module.
    """
    padded_h = ((latent_h + patch_spatial - 1) // patch_spatial) * patch_spatial
    padded_w = ((latent_w + patch_spatial - 1) // patch_spatial) * patch_spatial
    return padded_h * 8, padded_w * 8


def prepare_cond_image(rgb_bchw_01: torch.Tensor, latent_h: int, latent_w: int, patch_spatial: int = 2) -> torch.Tensor:
    """RGB image (B, 3, H, W) in [0, 1] -> (1, 3, H_t, W_t) in [-1, 1]."""
    if rgb_bchw_01.ndim != 4 or rgb_bchw_01.shape[1] != 3:
        raise ValueError(f"Unexpected cond image shape: {tuple(rgb_bchw_01.shape)} (expected B,3,H,W)")
    img = rgb_bchw_01[:1]
    target_h, target_w = target_cond_hw(latent_h, latent_w, patch_spatial)
    if img.shape[-2] != target_h or img.shape[-1] != target_w:
        img = F.interpolate(img, size=(target_h, target_w), mode="bicubic", align_corners=False)
        img = img.clamp(0.0, 1.0)
    return img * 2.0 - 1.0


def prepare_mask(mask_b1hw_01: torch.Tensor, latent_h: int, latent_w: int, patch_spatial: int = 2) -> torch.Tensor:
    """Mask (B, 1, H, W) or (B, H, W) in [0, 1] -> (1, 1, H_t, W_t) in {0.0, 1.0}.

    1 = inpaint area, 0 = keep. The caller is responsible for the ``*2-1``
    rescale before concat with RGB (see :func:`build_inpaint_cond_image`).
    """
    if mask_b1hw_01.ndim == 3:
        m = mask_b1hw_01.unsqueeze(1)
    elif mask_b1hw_01.ndim == 4 and mask_b1hw_01.shape[1] == 1:
        m = mask_b1hw_01
    else:
        raise ValueError(f"Unexpected mask shape: {tuple(mask_b1hw_01.shape)} (expected B,H,W or B,1,H,W)")
    m = m[:1].float()
    target_h, target_w = target_cond_hw(latent_h, latent_w, patch_spatial)
    if m.shape[-2] != target_h or m.shape[-1] != target_w:
        m = F.interpolate(m, size=(target_h, target_w), mode="nearest")
    return (m >= 0.5).float()


def build_inpaint_cond_image(rgb_pm1: torch.Tensor, mask01: torch.Tensor, masked_input: bool) -> torch.Tensor:
    """rgb_pm1: (1, 3, H, W) in [-1, 1], mask01: (1, 1, H, W) in {0, 1}. Returns (1, 4, H, W).

    The mask channel is rescaled to [-1, +1] (matching the RGB range), and if
    ``masked_input`` is set the RGB is zeroed where ``mask >= 0.5``.
    """
    if masked_input:
        keep = (mask01 < 0.5).to(rgb_pm1.dtype)
        rgb_pm1 = rgb_pm1 * keep
    mask_pm1 = mask01.to(rgb_pm1.dtype) * 2.0 - 1.0
    return torch.cat([rgb_pm1, mask_pm1], dim=1)


# ----------------------------------------------------------------------------
# Conditioning1 trunk (v2)
# ----------------------------------------------------------------------------


def _gn(channels: int) -> nn.GroupNorm:
    g = 8
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = _gn(ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.norm2 = _gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class _ASPP(nn.Module):
    def __init__(self, ch: int, dilations: tuple[int, ...] = ASPP_DEFAULT_DILATIONS):
        super().__init__()
        assert len(dilations) >= 1, "ASPP needs at least one dilation"
        branches = []
        for d in dilations:
            if d == 1:
                conv = nn.Conv2d(ch, ch, kernel_size=1)
            else:
                conv = nn.Conv2d(ch, ch, kernel_size=3, padding=d, dilation=d)
            branches.append(nn.Sequential(conv, _gn(ch), nn.SiLU()))
        self.branches = nn.ModuleList(branches)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=1), _gn(ch), nn.SiLU())

        n_branches = len(dilations) + 1
        self.proj = nn.Sequential(nn.Conv2d(ch * n_branches, ch, kernel_size=1), _gn(ch), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        g = self.global_conv(self.global_pool(x))
        g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
        outs.append(g)
        return self.proj(torch.cat(outs, dim=1))


class _Conditioning1(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        cond_emb_dim: int,
        n_resblocks: int,
        use_aspp: bool = False,
        aspp_dilations: tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
        cond_in_channels: int = 3,
    ):
        super().__init__()
        assert cond_dim % 2 == 0, f"cond_dim must be even, got {cond_dim}"
        assert cond_in_channels >= 1, f"cond_in_channels must be >= 1, got {cond_in_channels}"
        ch_half = cond_dim // 2

        self.cond_in_channels = cond_in_channels
        self.conv1 = nn.Conv2d(cond_in_channels, ch_half, kernel_size=4, stride=4, padding=0)
        self.norm1 = _gn(ch_half)
        self.conv2 = nn.Conv2d(ch_half, ch_half, kernel_size=3, stride=1, padding=1)
        self.norm2 = _gn(ch_half)
        self.conv3 = nn.Conv2d(ch_half, cond_dim, kernel_size=4, stride=4, padding=0)
        self.norm3 = _gn(cond_dim)

        self.resblocks = nn.ModuleList([_ResBlock(cond_dim) for _ in range(n_resblocks)])
        self.aspp = _ASPP(cond_dim, aspp_dilations) if use_aspp else None

        self.proj = nn.Conv2d(cond_dim, cond_emb_dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(cond_emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        h = F.silu(self.norm3(self.conv3(h)))
        for rb in self.resblocks:
            h = rb(h)
        if self.aspp is not None:
            h = self.aspp(h)
        h = self.proj(h)
        b, c, hh, ww = h.shape
        h = h.view(b, c, hh * ww).permute(0, 2, 1).contiguous()
        h = self.out_norm(h)
        return h


# ----------------------------------------------------------------------------
# LLLite module (v2: FiLM + SiLU + 5D path + per-module depth embedding)
# ----------------------------------------------------------------------------


class LLLiteModuleDiT(nn.Module):
    def __init__(
        self,
        name: str,
        in_dim: int,
        cond_emb_dim: int,
        mlp_dim: int,
        dropout: float | None = None,
        multiplier: float = 1.0,
    ):
        super().__init__()
        self.lllite_name = name
        self.in_dim = in_dim
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.multiplier = multiplier

        self.down = nn.Linear(in_dim, mlp_dim)
        self.mid = nn.Linear(mlp_dim + cond_emb_dim, mlp_dim)

        # FiLM: cond_local -> (gamma, beta), zero-init for identity at start.
        self.cond_to_film = nn.Linear(cond_emb_dim, 2 * mlp_dim)
        nn.init.zeros_(self.cond_to_film.weight)
        nn.init.zeros_(self.cond_to_film.bias)

        self.up = nn.Linear(mlp_dim, in_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        self.depth_embed = nn.Parameter(torch.zeros(cond_emb_dim))

        self.cond_emb: torch.Tensor | None = None
        # Wrapped in a list so the original Linear is not registered as a
        # submodule and its weights stay out of state_dict.
        self._org_module: list[nn.Linear] = []
        self._org_forward: Callable[[torch.Tensor], torch.Tensor] | None = None
        self._org_forward_was_instance_attr = False

    def bind(self, org_module: nn.Linear) -> None:
        self.unbind()
        self._org_module = [org_module]
        self._org_forward = org_module.forward
        self._org_forward_was_instance_attr = "forward" in org_module.__dict__
        org_module.forward = self.forward  # type: ignore[method-assign]

    def unbind(self) -> None:
        if self._org_forward is not None:
            org_module = self._org_module[0]
            if self._org_forward_was_instance_attr:
                org_module.forward = self._org_forward  # type: ignore[method-assign]
            else:
                # Restoring by assignment would pin a frozen bound method in the
                # instance __dict__, which silently bypasses later class-level
                # forward swaps that share the module __dict__ (see
                # wrap_custom_layer notes in model_manager/load/load_default.py).
                del org_module.__dict__["forward"]
            self._org_forward = None
            self._org_module = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input layouts:
        #   self/cross attention q/k/v: (B, S, D) — already flattened in the Anima block
        #   mlp.layer1:                 (B, T, H, W, D) — passed un-flattened
        # Flatten the 5D case to 3D for the LLLite path and reshape on exit.
        assert self._org_forward is not None
        if self.multiplier == 0.0 or self.cond_emb is None:
            return self._org_forward(x)

        orig_shape = x.shape
        is_5d = x.dim() == 5
        if is_5d:
            b, t, hh, ww, d = orig_shape
            x = x.reshape(b, t * hh * ww, d)

        cx = self.cond_emb  # (B_c, S, cond_emb_dim)

        # Broadcast cond_emb to the runtime batch (CFG cond+uncond, multi-cond).
        if x.shape[0] != cx.shape[0]:
            if x.shape[0] % cx.shape[0] != 0:
                return self._org_forward(x.reshape(orig_shape) if is_5d else x)
            cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1)

        if x.shape[1] != cx.shape[1]:
            return self._org_forward(x.reshape(orig_shape) if is_5d else x)

        # Run the LLLite mini-MLP in its own parameter dtype, then cast the
        # correction back to ``x``'s dtype before adding. Robust to autocast
        # flows where x and LLLite weights have different dtypes.
        param_dtype = self.down.weight.dtype
        x_proc = x if x.dtype == param_dtype else x.to(param_dtype)
        if cx.dtype != param_dtype or cx.device != x.device:
            cx = cx.to(device=x.device, dtype=param_dtype)

        depth_e = self.depth_embed
        if depth_e.dtype != param_dtype or depth_e.device != x.device:
            depth_e = depth_e.to(device=x.device, dtype=param_dtype)
        cond_local = cx + depth_e

        h = F.silu(self.down(x_proc))

        gb = self.cond_to_film(cond_local)
        gamma, beta = gb.chunk(2, dim=-1)

        m = self.mid(torch.cat([cond_local, h], dim=-1))
        m = m * (1 + gamma) + beta
        m = F.silu(m)

        if self.dropout is not None and self.training:
            m = F.dropout(m, p=self.dropout)

        out = self.up(m) * self.multiplier
        if out.dtype != x.dtype:
            out = out.to(x.dtype)

        y = self._org_forward(x + out)

        if is_5d:
            # org Linear out_features may differ from in_features — recover with -1.
            y = y.reshape(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], -1)
        return y


# ----------------------------------------------------------------------------
# AnimaControlNetLLLite
# ----------------------------------------------------------------------------


def _meta_int(metadata: dict[str, str], key: str, fallback: int) -> int:
    value = metadata.get(key)
    return int(value) if value is not None else fallback


def _meta_bool(metadata: dict[str, str], key: str, fallback: bool) -> bool:
    value = metadata.get(key)
    return str(value).lower() == "true" if value is not None else fallback


class AnimaControlNetLLLite(nn.Module):
    """Self-contained, cacheable LLLite adapter for the Anima transformer.

    Construct via :meth:`from_state_dict`; bind to a transformer with
    :meth:`apply_to` and undo with :meth:`restore`.
    """

    def __init__(
        self,
        module_specs: Sequence[tuple[str, int]],
        cond_emb_dim: int,
        mlp_dim: int,
        cond_dim: int,
        cond_resblocks: int,
        use_aspp: bool = False,
        aspp_dilations: tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
        cond_in_channels: int = 3,
        inpaint_masked_input: bool = False,
        multiplier: float = 1.0,
    ):
        super().__init__()
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.cond_dim = cond_dim
        self.cond_resblocks = cond_resblocks
        self.use_aspp = use_aspp
        self.cond_in_channels = cond_in_channels
        # Training-time RGB-masking policy for cond image preparation; does not
        # alter the forward pass.
        self.inpaint_masked_input = inpaint_masked_input
        self.multiplier = multiplier

        self.conditioning1 = _Conditioning1(
            cond_dim,
            cond_emb_dim,
            cond_resblocks,
            use_aspp=use_aspp,
            aspp_dilations=aspp_dilations,
            cond_in_channels=cond_in_channels,
        )

        modules = []
        for name, in_dim in module_specs:
            if MODULE_NAME_PATTERN.match(name) is None:
                raise ValueError(f"Unrecognized LLLite module name: '{name}'")
            modules.append(LLLiteModuleDiT(name, in_dim, cond_emb_dim, mlp_dim, multiplier=multiplier))
        self.lllite_modules = nn.ModuleList(modules)

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], metadata: dict[str, str] | None
    ) -> AnimaControlNetLLLite:
        """Build the adapter from a saved v2 named-key state dict.

        Hyperparams come from ``lllite.*`` metadata when present, with
        state-dict-shape fallbacks. ``inpaint_masked_input`` is metadata-only
        (not derivable from shapes; defaults to False).
        """
        meta = metadata or {}

        if any(k.startswith(_LEGACY_MODULES_PREFIX) for k in state_dict):
            raise ValueError(
                f"State dict appears to be in a legacy ControlNet-LLLite weight format (keys starting "
                f"with '{_LEGACY_MODULES_PREFIX}'). Only the v2 named-key format is supported."
            )

        module_names: set[str] = set()
        for key in state_dict:
            head, dot, _tail = key.partition(".")
            if dot and MODULE_NAME_PATTERN.match(head):
                module_names.add(head)
        if not module_names:
            raise ValueError("State dict contains no LLLite modules (no 'lllite_dit_blocks_*' keys).")

        def sort_key(name: str) -> tuple[int, int]:
            match = MODULE_NAME_PATTERN.match(name)
            assert match is not None
            return int(match.group(1)), _SUFFIX_ORDER.index(match.group(2))

        sorted_names = sorted(module_names, key=sort_key)
        module_specs: list[tuple[str, int]] = []
        for name in sorted_names:
            down_key = f"{name}.down.weight"
            if down_key not in state_dict:
                raise ValueError(f"LLLite module '{name}' is missing key '{down_key}'")
            module_specs.append((name, state_dict[down_key].shape[1]))

        conv1_weight = state_dict[f"{_SAVED_COND_PREFIX}conv1.weight"]
        conv3_weight = state_dict[f"{_SAVED_COND_PREFIX}conv3.weight"]
        proj_weight = state_dict[f"{_SAVED_COND_PREFIX}proj.weight"]
        resblock_indices = {
            m.group(1) for m in (re.match(rf"^{_SAVED_COND_PREFIX}resblocks\.(\d+)\.", k) for k in state_dict) if m
        }
        has_aspp_keys = any(k.startswith(f"{_SAVED_COND_PREFIX}aspp.") for k in state_dict)

        use_aspp = _meta_bool(meta, "lllite.use_aspp", has_aspp_keys)
        aspp_dilations_meta = meta.get("lllite.aspp_dilations")
        if use_aspp and aspp_dilations_meta:
            aspp_dilations = tuple(int(d) for d in aspp_dilations_meta.split(",") if d.strip())
        else:
            aspp_dilations = ASPP_DEFAULT_DILATIONS

        model = cls(
            module_specs=module_specs,
            cond_emb_dim=_meta_int(meta, "lllite.cond_emb_dim", proj_weight.shape[0]),
            mlp_dim=_meta_int(meta, "lllite.mlp_dim", state_dict[f"{sorted_names[0]}.down.weight"].shape[0]),
            cond_dim=_meta_int(meta, "lllite.cond_dim", conv3_weight.shape[0]),
            cond_resblocks=_meta_int(meta, "lllite.cond_resblocks", len(resblock_indices)),
            use_aspp=use_aspp,
            aspp_dilations=aspp_dilations,
            cond_in_channels=_meta_int(meta, "lllite.cond_in_channels", conv1_weight.shape[1]),
            inpaint_masked_input=_meta_bool(meta, "lllite.inpaint_masked_input", False),
        )

        name_to_idx = {name: i for i, name in enumerate(sorted_names)}
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(_SAVED_COND_PREFIX):
                remapped[_INTERNAL_COND_PREFIX + key[len(_SAVED_COND_PREFIX) :]] = value
                continue
            head, dot, tail = key.partition(".")
            if dot and head in name_to_idx:
                remapped[f"{_INTERNAL_MODULES_PREFIX}{name_to_idx[head]}.{tail}"] = value
            else:
                # Unknown keys are passed through so strict loading reports them.
                remapped[key] = value

        model.load_state_dict(remapped, strict=True)
        model.eval().requires_grad_(False)
        return model

    def set_cond_image(self, cond: torch.Tensor | None) -> None:
        """cond: (B, cond_in_channels, H_t, W_t) in [-1, 1]; ``None`` clears."""
        if cond is None:
            for m in self.lllite_modules:
                m.cond_emb = None
            return
        trunk_weight = self.conditioning1.conv1.weight
        cond = cond.to(device=trunk_weight.device, dtype=trunk_weight.dtype)
        cx = self.conditioning1(cond)  # (B, S, cond_emb_dim)
        for m in self.lllite_modules:
            m.cond_emb = cx

    def clear_cond_image(self) -> None:
        self.set_cond_image(None)

    def set_multiplier(self, multiplier: float) -> None:
        self.multiplier = multiplier
        for m in self.lllite_modules:
            m.multiplier = multiplier

    def apply_to(self, transformer: nn.Module) -> None:
        """Swap the forward of each target Linear in ``transformer``. Idempotent."""
        self.restore()
        for m in self.lllite_modules:
            target = self._resolve_target(transformer, m.lllite_name)
            if not isinstance(target, nn.Linear):
                raise TypeError(f"LLLite target for '{m.lllite_name}' is {type(target).__name__}, expected nn.Linear")
            if target.in_features != m.in_dim:
                raise ValueError(
                    f"LLLite module '{m.lllite_name}' was trained for in_features={m.in_dim}, but the "
                    f"target Linear has in_features={target.in_features}"
                )
            m.bind(target)

    def restore(self) -> None:
        """Undo :meth:`apply_to`. Safe to call when not applied."""
        for m in self.lllite_modules:
            m.unbind()

    @staticmethod
    def _resolve_target(transformer: nn.Module, name: str) -> nn.Module:
        match = MODULE_NAME_PATTERN.match(name)
        if match is None:
            raise ValueError(f"Unrecognized LLLite module name: '{name}'")
        block_idx = int(match.group(1))
        blocks = transformer.blocks
        if block_idx >= len(blocks):
            raise ValueError(
                f"LLLite module '{name}' targets block {block_idx}, but the transformer has only {len(blocks)} blocks"
            )
        target: nn.Module = blocks[block_idx]
        for attr in _SUFFIX_TO_ATTR_PATH[match.group(2)]:
            target = getattr(target, attr)
        return target
