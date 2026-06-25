"""Flux2 KL autoencoder."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import torch
from einops import rearrange
from torch import Tensor, nn


@dataclass
class AutoEncoderParams:
  resolution: int = 256
  in_channels: int = 3
  ch: int = 128
  out_ch: int = 3
  ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
  num_res_blocks: int = 2
  z_channels: int = 32


def swish(x: Tensor) -> Tensor:
  return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
  def __init__(self, in_channels: int):
    super().__init__()
    self.in_channels = in_channels

    self.norm = nn.GroupNorm(
      num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )

    self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

  def attention(self, h_: Tensor) -> Tensor:
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    b, c, h, w = q.shape
    q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
    k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
    v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
    h_ = nn.functional.scaled_dot_product_attention(q, k, v)

    return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

  def forward(self, x: Tensor) -> Tensor:
    return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.in_channels = in_channels
    out_channels = in_channels if out_channels is None else out_channels
    self.out_channels = out_channels

    self.norm1 = nn.GroupNorm(
      num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )
    self.conv1 = nn.Conv2d(
      in_channels, out_channels, kernel_size=3, stride=1, padding=1
    )
    self.norm2 = nn.GroupNorm(
      num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
    )
    self.conv2 = nn.Conv2d(
      out_channels, out_channels, kernel_size=3, stride=1, padding=1
    )
    if self.in_channels != self.out_channels:
      self.nin_shortcut = nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=1, padding=0
      )

  def forward(self, x):
    h = x
    h = self.norm1(h)
    h = swish(h)
    h = self.conv1(h)

    h = self.norm2(h)
    h = swish(h)
    h = self.conv2(h)

    if self.in_channels != self.out_channels:
      x = self.nin_shortcut(x)

    return x + h


class Downsample(nn.Module):
  def __init__(self, in_channels: int):
    super().__init__()
    # no asymmetric padding in torch conv, must do it ourselves
    self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

  def forward(self, x: Tensor):
    pad = (0, 1, 0, 1)
    x = nn.functional.pad(x, pad, mode="constant", value=0)
    x = self.conv(x)
    return x


class Upsample(nn.Module):
  def __init__(self, in_channels: int):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x: Tensor):
    x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
    x = self.conv(x)
    return x


class Encoder(nn.Module):
  def __init__(
    self,
    resolution: int,
    in_channels: int,
    ch: int,
    ch_mult: list[int],
    num_res_blocks: int,
    z_channels: int,
  ):
    super().__init__()
    self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
    self.ch = ch
    self.num_resolutions = len(ch_mult)
    self.num_res_blocks = num_res_blocks
    self.resolution = resolution
    self.in_channels = in_channels
    # downsampling
    self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

    curr_res = resolution
    in_ch_mult = (1,) + tuple(ch_mult)
    self.in_ch_mult = in_ch_mult
    self.down = nn.ModuleList()
    block_in = self.ch
    for i_level in range(self.num_resolutions):
      block = nn.ModuleList()
      attn = nn.ModuleList()
      block_in = ch * in_ch_mult[i_level]
      block_out = ch * ch_mult[i_level]
      for _ in range(self.num_res_blocks):
        block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
        block_in = block_out
      down = nn.Module()
      down.block = block
      down.attn = attn
      if i_level != self.num_resolutions - 1:
        down.downsample = Downsample(block_in)
        curr_res = curr_res // 2
      self.down.append(down)

    # middle
    self.mid = nn.Module()
    self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
    self.mid.attn_1 = AttnBlock(block_in)
    self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

    # end
    self.norm_out = nn.GroupNorm(
      num_groups=32, num_channels=block_in, eps=1e-6, affine=True
    )
    self.conv_out = nn.Conv2d(
      block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1
    )

  def forward(self, x: Tensor) -> Tensor:
    # downsampling
    hs = [self.conv_in(x)]
    for i_level in range(self.num_resolutions):
      for i_block in range(self.num_res_blocks):
        h = self.down[i_level].block[i_block](hs[-1])  # type: ignore[index, operator]
        if len(self.down[i_level].attn) > 0:  # type: ignore[arg-type]
          h = self.down[i_level].attn[i_block](h)  # type: ignore[index, operator]
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(self.down[i_level].downsample(hs[-1]))  # type: ignore[operator]

    # middle
    h = hs[-1]
    h = self.mid.block_1(h)  # type: ignore[operator]
    h = self.mid.attn_1(h)  # type: ignore[operator]
    h = self.mid.block_2(h)  # type: ignore[operator]
    # end
    h = self.norm_out(h)
    h = swish(h)
    h = self.conv_out(h)
    h = self.quant_conv(h)
    return h


class Decoder(nn.Module):
  def __init__(
    self,
    ch: int,
    out_ch: int,
    ch_mult: list[int],
    num_res_blocks: int,
    in_channels: int,
    resolution: int,
    z_channels: int,
  ):
    super().__init__()
    self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
    self.ch = ch
    self.num_resolutions = len(ch_mult)
    self.num_res_blocks = num_res_blocks
    self.resolution = resolution
    self.in_channels = in_channels
    self.ffactor = 2 ** (self.num_resolutions - 1)

    # compute in_ch_mult, block_in and curr_res at lowest res
    block_in = ch * ch_mult[self.num_resolutions - 1]
    curr_res = resolution // 2 ** (self.num_resolutions - 1)
    self.z_shape = (1, z_channels, curr_res, curr_res)

    # z to block_in
    self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

    # middle
    self.mid = nn.Module()
    self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
    self.mid.attn_1 = AttnBlock(block_in)
    self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

    # upsampling
    self.up = nn.ModuleList()
    for i_level in reversed(range(self.num_resolutions)):
      block = nn.ModuleList()
      attn = nn.ModuleList()
      block_out = ch * ch_mult[i_level]
      for _ in range(self.num_res_blocks + 1):
        block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
        block_in = block_out
      up = nn.Module()
      up.block = block
      up.attn = attn
      if i_level != 0:
        up.upsample = Upsample(block_in)
        curr_res = curr_res * 2
      self.up.insert(0, up)  # prepend to get consistent order

    # end
    self.norm_out = nn.GroupNorm(
      num_groups=32, num_channels=block_in, eps=1e-6, affine=True
    )
    self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

  def forward(self, z: Tensor) -> Tensor:
    z = self.post_quant_conv(z)

    # get dtype for proper tracing
    upscale_dtype = next(self.up.parameters()).dtype

    # z to block_in
    h = self.conv_in(z)

    # middle
    h = self.mid.block_1(h)  # type: ignore[operator]
    h = self.mid.attn_1(h)  # type: ignore[operator]
    h = self.mid.block_2(h)  # type: ignore[operator]

    # cast to proper dtype
    h = h.to(upscale_dtype)
    # upsampling
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = self.up[i_level].block[i_block](h)  # type: ignore[index, operator]
        if len(self.up[i_level].attn) > 0:  # type: ignore[arg-type]
          h = self.up[i_level].attn[i_block](h)  # type: ignore[index, operator]
      if i_level != 0:
        h = self.up[i_level].upsample(h)  # type: ignore[operator]

    # end
    h = self.norm_out(h)
    h = swish(h)
    h = self.conv_out(h)
    return h


class AutoEncoder(nn.Module):
  def __init__(self, params: AutoEncoderParams):
    super().__init__()
    self.params = params
    self.encoder = Encoder(
      resolution=params.resolution,
      in_channels=params.in_channels,
      ch=params.ch,
      ch_mult=params.ch_mult,
      num_res_blocks=params.num_res_blocks,
      z_channels=params.z_channels,
    )
    self.decoder = Decoder(
      resolution=params.resolution,
      in_channels=params.in_channels,
      ch=params.ch,
      out_ch=params.out_ch,
      ch_mult=params.ch_mult,
      num_res_blocks=params.num_res_blocks,
      z_channels=params.z_channels,
    )

    self.bn_eps = 1e-4
    self.bn_momentum = 0.1
    self.ps = [2, 2]
    self.bn = torch.nn.BatchNorm2d(
      math.prod(self.ps) * params.z_channels,
      eps=self.bn_eps,
      momentum=self.bn_momentum,
      affine=False,
      track_running_stats=True,
    )


_NUM_RESOLUTIONS = 4


def convert_diffusers_state_dict(src: dict[str, Tensor]) -> dict[str, Tensor]:
  out: dict[str, Tensor] = {}
  attn_substrings = (".mid.attn_1.",)
  for src_key, tensor in src.items():
    dst_key = _rewrite_diffusers_key(src_key)
    if dst_key is None:
      raise KeyError(f"Unrecognized diffusers VAE state-dict key: {src_key}")
    if (
      any(s in dst_key for s in attn_substrings)
      and dst_key.endswith(".weight")
      and tensor.ndim == 2
    ):
      tensor = tensor.unsqueeze(-1).unsqueeze(-1)
    out[dst_key] = tensor
  return out


def _rewrite_diffusers_key(key: str) -> str | None:
  if key.startswith("bn."):
    return key

  if key.startswith("quant_conv."):
    return key.replace("quant_conv.", "encoder.quant_conv.", 1)
  if key.startswith("post_quant_conv."):
    return key.replace("post_quant_conv.", "decoder.post_quant_conv.", 1)

  if key == "encoder.conv_norm_out.weight":
    return "encoder.norm_out.weight"
  if key == "encoder.conv_norm_out.bias":
    return "encoder.norm_out.bias"
  if key == "decoder.conv_norm_out.weight":
    return "decoder.norm_out.weight"
  if key == "decoder.conv_norm_out.bias":
    return "decoder.norm_out.bias"

  m = re.match(r"^(encoder|decoder)\.mid_block\.resnets\.(\d+)\.(.+)$", key)
  if m:
    side, idx, rest = m.group(1), int(m.group(2)), m.group(3)
    rest = rest.replace("conv_shortcut", "nin_shortcut")
    return f"{side}.mid.block_{idx + 1}.{rest}"
  m = re.match(r"^(encoder|decoder)\.mid_block\.attentions\.0\.(.+)$", key)
  if m:
    side, rest = m.group(1), m.group(2)
    rest = (
      rest.replace("group_norm.", "norm.")
      .replace("to_q.", "q.")
      .replace("to_k.", "k.")
      .replace("to_v.", "v.")
      .replace("to_out.0.", "proj_out.")
    )
    return f"{side}.mid.attn_1.{rest}"

  m = re.match(r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
  if m:
    level, res_idx, rest = m.group(1), m.group(2), m.group(3)
    rest = rest.replace("conv_shortcut", "nin_shortcut")
    return f"encoder.down.{level}.block.{res_idx}.{rest}"
  m = re.match(r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.(.+)$", key)
  if m:
    return f"encoder.down.{m.group(1)}.downsample.conv.{m.group(2)}"

  m = re.match(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
  if m:
    diffusers_idx = int(m.group(1))
    res_idx = m.group(2)
    rest = m.group(3).replace("conv_shortcut", "nin_shortcut")
    return f"decoder.up.{_NUM_RESOLUTIONS - 1 - diffusers_idx}.block.{res_idx}.{rest}"
  m = re.match(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.(.+)$", key)
  if m:
    diffusers_idx = int(m.group(1))
    return (
      f"decoder.up.{_NUM_RESOLUTIONS - 1 - diffusers_idx}.upsample.conv.{m.group(2)}"
    )

  if key.startswith(
    ("encoder.conv_in.", "encoder.conv_out.", "decoder.conv_in.", "decoder.conv_out.")
  ):
    return key

  return None
