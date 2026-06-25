from __future__ import annotations

import warnings

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F


_BNB_SIBLING_SUFFIXES = (
  ".absmax",
  ".quant_map",
  ".nested_absmax",
  ".nested_quant_map",
)

# Largest magnitude representable by the e4m3 float8 format. Per-row weight
# scales map each row's max abs value onto this so we use the full range.
FP8_E4M3_MAX = 448.0
FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"
# Marker written into the text encoder's config.json so the loader knows to take
# the custom weight-only FP8 path instead of transformers' from_pretrained.
FP8_TEXT_ENCODER_CONFIG_FLAG = "ideogram_fp8_weight_only"


def is_bnb4bit_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  """True if any key looks like a bnb 4-bit quant_state sibling."""
  return any(".quant_state.bitsandbytes__" in k for k in state_dict)


def swap_linears_to_bnb4bit(
  module: nn.Module,
  compute_dtype: torch.dtype,
  *,
  quant_type: str = "nf4",
  compress_statistics: bool = False,
) -> None:
  for name, child in list(module.named_children()):
    if isinstance(child, nn.Linear):
      new_linear = bnb.nn.Linear4bit(
        child.in_features,
        child.out_features,
        bias=child.bias is not None,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
      )
      setattr(module, name, new_linear)
    else:
      swap_linears_to_bnb4bit(
        child,
        compute_dtype,
        quant_type=quant_type,
        compress_statistics=compress_statistics,
      )


def load_bnb4bit_state_dict(
  model: nn.Module,
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
) -> None:
  consumed: set[str] = set()
  for full_name, tensor in state_dict.items():
    if ".quant_state." in full_name or full_name.endswith(_BNB_SIBLING_SUFFIXES):
      continue
    parent_path, _, param_name = full_name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    current = parent._parameters.get(param_name)
    if not isinstance(current, bnb.nn.Params4bit):
      continue
    prefix = full_name + "."
    quantized_stats = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
    # bnb's from_prequantized pops keys it consumes from the dict, so snapshot
    # the names first.
    consumed.add(full_name)
    consumed.update(quantized_stats.keys())
    parent._parameters[param_name] = bnb.nn.Params4bit.from_prequantized(
      data=tensor,
      quantized_stats=quantized_stats,
      requires_grad=False,
      device=device,
    )

  remaining = {k: v for k, v in state_dict.items() if k not in consumed}
  for k in list(remaining):
    if remaining[k].is_floating_point():
      remaining[k] = remaining[k].to(device=device, dtype=dtype)
    else:
      remaining[k] = remaining[k].to(device=device)

  missing, unexpected = model.load_state_dict(remaining, strict=False)
  # Quantized weights are loaded via from_prequantized above, so they appear in
  # `missing` from load_state_dict's perspective — filter those out.
  real_missing = [m for m in missing if m not in consumed]
  if real_missing:
    raise RuntimeError(f"missing keys after quantized load: {real_missing[:10]}")
  if unexpected:
    raise RuntimeError(f"unexpected keys after quantized load: {unexpected[:10]}")

  for p in model.parameters():
    if isinstance(p, bnb.nn.Params4bit):
      continue
    if p.is_floating_point() and p.dtype != dtype:
      p.data = p.data.to(dtype=dtype)
    if p.device != device:
      p.data = p.data.to(device=device)
  for name, b in list(model.named_buffers()):
    if b.is_floating_point() and b.dtype != dtype:
      parent_path, _, leaf = name.rpartition(".")
      parent = model.get_submodule(parent_path) if parent_path else model
      parent.register_buffer(
        leaf,
        b.to(device=device, dtype=dtype),
        persistent=leaf not in parent._non_persistent_buffers_set,
      )
    elif b.device != device:
      parent_path, _, leaf = name.rpartition(".")
      parent = model.get_submodule(parent_path) if parent_path else model
      parent.register_buffer(
        leaf,
        b.to(device=device),
        persistent=leaf not in parent._non_persistent_buffers_set,
      )


# ---------------------------------------------------------------------------
# Weight-only FP8 (e4m3)
#
# Activations stay in the compute dtype (e.g. bfloat16); only Linear weights are
# stored as float8 with a per-output-channel (per-row) float32 scale. At forward
# time the weight is dequantized back to the compute dtype and a normal bf16
# matmul runs, so this needs no FP8 tensor-core hardware and works on any device
# that can store float8 (CPU included). The win is ~2x smaller Linear weights.
# ---------------------------------------------------------------------------


def quantize_weight_to_fp8(
  weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Quantize a 2-D Linear weight to e4m3 float8 with per-row scales.

  Returns ``(weight_fp8, scale)`` where ``weight_fp8`` has shape ``(out, in)``
  in ``float8_e4m3fn`` and ``scale`` has shape ``(out,)`` in float32 such that
  ``weight ≈ weight_fp8.to(dtype) * scale[:, None]``.
  """
  w = weight.detach().to(torch.float32)
  amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
  scale = amax / FP8_E4M3_MAX
  q = (w / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE)
  return q, scale.squeeze(1).to(torch.float32)


def is_fp8_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  """True if the checkpoint carries weight-only FP8 Linear weights."""
  return any(k.endswith(FP8_SCALE_SUFFIX) for k in state_dict) or any(
    v.dtype == FP8_WEIGHT_DTYPE for v in state_dict.values()
  )


class Fp8Linear(nn.Module):
  """Linear layer holding an e4m3 float8 weight + per-row float32 scale.

  The weight and scale are registered as buffers (not parameters) so they load
  via ``load_state_dict`` and are excluded from optimizer/grad machinery. The
  dequantized matmul runs in ``compute_dtype``.
  """

  weight: torch.Tensor
  weight_scale: torch.Tensor
  bias: torch.Tensor | None

  def __init__(
    self,
    in_features: int,
    out_features: int,
    bias: bool,
    compute_dtype: torch.dtype,
  ) -> None:
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.compute_dtype = compute_dtype
    self.register_buffer(
      "weight",
      torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE),
    )
    self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
    if bias:
      self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
    else:
      self.bias = None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
    bias = self.bias.to(x.dtype) if self.bias is not None else None
    return F.linear(x, w, bias)


def swap_linears_to_fp8(
  module: nn.Module,
  state_dict: dict[str, torch.Tensor],
  compute_dtype: torch.dtype,
  *,
  prefix: str = "",
) -> None:
  """Replace each ``nn.Linear`` that has a saved FP8 scale with an ``Fp8Linear``.

  Gating on the presence of ``<name>.weight_scale`` means only layers that were
  actually quantized at save time are swapped; everything else loads normally in
  the compute dtype.
  """
  for name, child in list(module.named_children()):
    child_prefix = f"{prefix}{name}"
    if (
      isinstance(child, nn.Linear) and f"{child_prefix}{FP8_SCALE_SUFFIX}" in state_dict
    ):
      setattr(
        module,
        name,
        Fp8Linear(
          child.in_features,
          child.out_features,
          bias=child.bias is not None,
          compute_dtype=compute_dtype,
        ),
      )
    else:
      swap_linears_to_fp8(child, state_dict, compute_dtype, prefix=f"{child_prefix}.")


def load_fp8_state_dict(
  model: nn.Module,
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
  *,
  assign: bool = False,
  strict: bool = True,
) -> None:
  """Load a weight-only FP8 checkpoint into ``model``.

  ``model`` must already have its FP8 Linear layers swapped in (see
  ``swap_linears_to_fp8``). FP8 weights are kept as float8, scales stay float32,
  and every other floating tensor is cast to ``dtype``.

  ``assign=True`` replaces the module's tensors with the prepared ones rather than
  copying into them. Use it when the model was built with ``from_config`` so the
  non-quantized params take the loaded dtype directly and computed non-persistent
  buffers (e.g. rotary caches) are left untouched. With ``assign=False`` (default),
  the caller must have already put the unquantized params in ``dtype``.

  ``strict=False`` downgrades missing keys to a warning (e.g. tied weights that a
  ``transformers`` model resolves itself); unexpected keys always raise.
  """
  prepared: dict[str, torch.Tensor] = {}
  for k, v in state_dict.items():
    if v.dtype == FP8_WEIGHT_DTYPE:
      prepared[k] = v.to(device=device)
    elif k.endswith(FP8_SCALE_SUFFIX):
      prepared[k] = v.to(device=device, dtype=torch.float32)
    elif v.is_floating_point():
      prepared[k] = v.to(device=device, dtype=dtype)
    else:
      prepared[k] = v.to(device=device)

  missing, unexpected = model.load_state_dict(prepared, strict=False, assign=assign)
  if unexpected:
    raise RuntimeError(f"unexpected keys after fp8 load: {unexpected[:10]}")
  if missing:
    if strict:
      raise RuntimeError(f"missing keys after fp8 load: {missing[:10]}")
    warnings.warn(f"missing keys after fp8 load: {missing[:10]}", stacklevel=2)

  model.to(device)
