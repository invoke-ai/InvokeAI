import torch


if torch.backends.mps.is_available():
    torch.empty = torch.zeros


_torch_layer_norm = torch.nn.functional.layer_norm
def new_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    if input.device.type == "mps" and input.dtype == torch.float16:
        input = input.float()
        if weight is not None:
            weight = weight.float()
        if bias is not None:
            bias = bias.float()
        return _torch_layer_norm(input, normalized_shape, weight, bias, eps).half()
    else:
        return _torch_layer_norm(input, normalized_shape, weight, bias, eps)

torch.nn.functional.layer_norm = new_layer_norm


_torch_tensor_permute = torch.Tensor.permute
def new_torch_tensor_permute(input, *dims):
    result = _torch_tensor_permute(input, *dims)
    if input.device == "mps" and input.dtype == torch.float16:
        result = result.contiguous()
    return result

torch.Tensor.permute = new_torch_tensor_permute


_torch_lerp = torch.lerp
def new_torch_lerp(input, end, weight, *, out=None):
    if input.device.type == "mps" and input.dtype == torch.float16:
        input = input.float()
        end = end.float()
        if isinstance(weight, torch.Tensor):
            weight = weight.float()
        if out is not None:
            out_fp32 = torch.zeros_like(out, dtype=torch.float32)
        else:
            out_fp32 = None
        result = _torch_lerp(input, end, weight, out=out_fp32)
        if out is not None:
            out.copy_(out_fp32.half())
            del out_fp32
        return result.half()

    else:
        return _torch_lerp(input, end, weight, out=out)

torch.lerp = new_torch_lerp


_torch_interpolate = torch.nn.functional.interpolate
def new_torch_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if input.device.type == "mps" and input.dtype == torch.float16:
        return _torch_interpolate(input.float(), size, scale_factor, mode, align_corners, recompute_scale_factor, antialias).half()
    else:
        return _torch_interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)

torch.nn.functional.interpolate = new_torch_interpolate
