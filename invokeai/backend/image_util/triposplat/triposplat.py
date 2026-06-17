import numpy as np
import torch
import torch.nn.functional as F
import safetensors.torch
from PIL import Image, ImageFilter
from torchvision import transforms
from tqdm.auto import tqdm

from .model import (
    DinoV3ViT, Flux2VAEEncoder, BiRefNet,
    OctreeProbabilityFixedlenDecoder, ElasticGaussianFixedlenDecoder,
    LatentSeqMMFlowModel, OctreeGaussianDecoder,
)


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------

class Gaussian:
    def __init__(self, aabb: list, sh_degree: int = 0, mininum_kernel_size: float = 0.0,
                 scaling_bias: float = 0.01, opacity_bias: float = 0.1,
                 scaling_activation: str = "exp", device='cuda'):
        self.sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)

        if scaling_activation == "exp":
            self._scaling_activation = torch.exp
            self._inverse_scaling_activation = torch.log
        elif scaling_activation == "softplus":
            self._scaling_activation = F.softplus
            self._inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))

        self._opacity_activation = torch.sigmoid
        self._inverse_opacity_activation = lambda x: torch.log(x / (1 - x))

        self.scale_bias = self._inverse_scaling_activation(torch.tensor(self.scaling_bias)).to(self.device)
        self.rots_bias = torch.zeros(4, device=self.device)
        self.rots_bias[0] = 1
        self.opacity_bias_val = self._inverse_opacity_activation(torch.tensor(self.opacity_bias)).to(self.device)

        self._storage = {}

    def _get_store(self, name):
        return self._storage.get(name)

    def _set_store(self, name, value):
        self._storage[name] = value

    @property
    def _xyz(self):
        return self._get_store("_xyz")
    @_xyz.setter
    def _xyz(self, value):
        if value is None:
            self._set_store("_xyz", None); self._set_store("xyz", None); return
        self._set_store("_xyz", value)
        self._set_store("xyz", value * self.aabb[None, 3:] + self.aabb[None, :3])

    @property
    def get_xyz(self):
        return self._get_store("xyz")

    @property
    def _features_dc(self):
        return self._get_store("_features_dc")
    @_features_dc.setter
    def _features_dc(self, value):
        self._set_store("_features_dc", value)

    @property
    def _opacity(self):
        return self._get_store("_opacity")
    @_opacity.setter
    def _opacity(self, value):
        if value is None:
            self._set_store("_opacity", None); self._set_store("opacity", None); return
        self._set_store("_opacity", value)
        self._set_store("opacity", self._opacity_activation(value + self.opacity_bias_val))

    @property
    def get_opacity(self):
        return self._get_store("opacity")

    @property
    def _scaling(self):
        return self._get_store("_scaling")
    @_scaling.setter
    def _scaling(self, value):
        if value is None:
            self._set_store("_scaling", None); self._set_store("scaling", None); return
        self._set_store("_scaling", value)
        s = self._scaling_activation(value + self.scale_bias)
        s = torch.square(s) + self.mininum_kernel_size ** 2
        self._set_store("scaling", torch.sqrt(s))

    @property
    def get_scaling(self):
        return self._get_store("scaling")

    @property
    def _rotation(self):
        return self._get_store("_rotation")
    @_rotation.setter
    def _rotation(self, value):
        self._set_store("_rotation", value)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        dc = self._features_dc
        for i in range(dc.shape[1] * dc.shape[2]):
            l.append(f'f_dc_{i}')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rot_{i}')
        return l

    _DEFAULT_TRANSFORM = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]

    def _get_ply_data(self, transform=None):
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._inverse_opacity_activation(self.get_opacity).detach().cpu().numpy()
        scale = torch.log(self.get_scaling).detach().cpu().numpy()
        rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform.T)
            R_mat = _quat_to_matrix(rotation)
            R_mat = np.matmul(transform, R_mat)
            rotation = _matrix_to_quat(R_mat)
        return xyz, normals, f_dc, opacities, scale, rotation

    def _transformed_xyz_rot(self, transform=None):
        if transform is None:
            transform = self._DEFAULT_TRANSFORM
        transform = np.array(transform, dtype=np.float32)
        xyz = self.get_xyz.detach().cpu().numpy().astype(np.float32)
        rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()
        xyz = np.matmul(xyz, transform.T)
        R_mat = _quat_to_matrix(rotation)
        R_mat = np.matmul(transform, R_mat)
        rotation = _matrix_to_quat(R_mat)
        return xyz, rotation

    def to_ply_bytes(self, transform=None) -> bytes:
        if transform is None:
            transform = self._DEFAULT_TRANSFORM
        xyz, normals, f_dc, opacities, scale, rotation = self._get_ply_data(transform=transform)
        dtype_full = [(attr, 'f4') for attr in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)))
        return _binary_ply_bytes(elements, dtype_full)

    def to_splat_bytes(self, transform=None) -> bytes:
        if transform is None:
            transform = self._DEFAULT_TRANSFORM
        xyz, rotation = self._transformed_xyz_rot(transform=transform)
        scale = self.get_scaling.detach().cpu().numpy().astype(np.float32)
        opacity = self.get_opacity.detach().cpu().numpy()
        f_dc = self._features_dc.detach().cpu().numpy()
        C0 = 0.28209479177387814
        # .splat packs color as 4 bytes RGBA: RGB from the SH DC term, A from opacity.
        rgb = np.clip((f_dc[:, 0, :] * C0 + 0.5) * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(opacity[:, 0:1] * 255, 0, 255).astype(np.uint8)
        rgba = np.concatenate([rgb, alpha], axis=1)
        rot = rotation / np.linalg.norm(rotation, axis=-1, keepdims=True)
        rot_u8 = np.clip(rot * 128 + 128, 0, 255).astype(np.uint8)
        order = np.argsort(-opacity[:, 0] * np.prod(scale, axis=-1))
        xyz, scale, rgba, rot_u8 = xyz[order], scale[order], rgba[order], rot_u8[order]
        # Per-splat record is exactly 32 bytes: xyz(12) + scale(12) + rgba(4) + rot(4).
        data = np.concatenate([
            xyz.astype(np.float32).view(np.uint8).reshape(-1, 12),
            scale.astype(np.float32).view(np.uint8).reshape(-1, 12),
            rgba.reshape(-1, 4),
            rot_u8.reshape(-1, 4),
        ], axis=1).reshape(-1)
        return data.tobytes()

    def save_ply(self, path, transform=None):
        with open(path, 'wb') as f:
            f.write(self.to_ply_bytes(transform=transform))

    def save_splat(self, path, transform=None):
        with open(path, 'wb') as f:
            f.write(self.to_splat_bytes(transform=transform))


def _binary_ply_bytes(elements, dtype_full) -> bytes:
    num_vertices = len(elements)
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {num_vertices}\n"
    type_map = {'f4': 'float', 'u1': 'uchar', 'i4': 'int'}
    for name, t in dtype_full:
        header += f"property {type_map.get(t, t)} {name}\n"
    header += "end_header\n"
    return header.encode('ascii') + elements.tobytes()


def _quat_to_matrix(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], axis=-1).reshape(-1, 3, 3)
    return R


def _matrix_to_quat(R):
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = np.zeros((R.shape[0], 4), dtype=R.dtype)
    s = np.sqrt(np.maximum(trace + 1, 0)) * 2
    q[:, 0] = 0.25 * s
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / np.where(s != 0, s, 1)
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / np.where(s != 0, s, 1)
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / np.where(s != 0, s, 1)
    m01 = (R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2]) & (s == 0)
    s1 = np.sqrt(np.maximum(1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], 0)) * 2
    q[m01, 0] = (R[m01, 2, 1] - R[m01, 1, 2]) / s1[m01]
    q[m01, 1] = 0.25 * s1[m01]
    q[m01, 2] = (R[m01, 0, 1] + R[m01, 1, 0]) / s1[m01]
    q[m01, 3] = (R[m01, 0, 2] + R[m01, 2, 0]) / s1[m01]
    m11 = (R[:, 1, 1] > R[:, 0, 0]) & (R[:, 1, 1] >= R[:, 2, 2]) & (s == 0)
    s2 = np.sqrt(np.maximum(1 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2], 0)) * 2
    q[m11, 0] = (R[m11, 0, 2] - R[m11, 2, 0]) / s2[m11]
    q[m11, 1] = (R[m11, 0, 1] + R[m11, 1, 0]) / s2[m11]
    q[m11, 2] = 0.25 * s2[m11]
    q[m11, 3] = (R[m11, 1, 2] + R[m11, 2, 1]) / s2[m11]
    m21 = (R[:, 2, 2] > R[:, 0, 0]) & (R[:, 2, 2] > R[:, 1, 1]) & (s == 0)
    s3 = np.sqrt(np.maximum(1 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1], 0)) * 2
    q[m21, 0] = (R[m21, 1, 0] - R[m21, 0, 1]) / s3[m21]
    q[m21, 1] = (R[m21, 0, 2] + R[m21, 2, 0]) / s3[m21]
    q[m21, 2] = (R[m21, 1, 2] + R[m21, 2, 1]) / s3[m21]
    q[m21, 3] = 0.25 * s3[m21]
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _build_gaussians(decoder: ElasticGaussianFixedlenDecoder, points_pred: dict, pred: dict):
    x = points_pred
    offset = decoder._get_offset(pred['features'])
    h = pred["features"]
    ret = []
    for i in range(h.shape[0]):
        g = Gaussian(
            sh_degree=0,
            aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
            mininum_kernel_size=decoder.rep_config['filter_kernel_size_3d'],
            scaling_bias=decoder.rep_config['scaling_bias'],
            opacity_bias=decoder.rep_config['opacity_bias'],
            scaling_activation=decoder.rep_config['scaling_activation'],
            device=x['points'].device,
        )
        _x = x["points"][i, :, None, :]
        for k, v in decoder.layout.items():
            if k == '_xyz':
                setattr(g, k, (offset[i] + _x).flatten(0, 1))
            elif k in ('_xyz_center', '_offset_scale'):
                continue
            else:
                feats = h[i][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                setattr(g, k, feats * decoder.rep_config['lr'][k])
        ret.append(g)
    return ret


# ---------------------------------------------------------------------------
# Euler flow sampler
# ---------------------------------------------------------------------------

class FlowEulerCfgSampler:
    def __init__(self, sigma_min: float = 1e-5):
        self.sigma_min = sigma_min

    def _get_batch_size(self, x_t):
        return next(iter(x_t.values())).shape[0] if isinstance(x_t, dict) else x_t.shape[0]

    def _get_device(self, x_t):
        return next(iter(x_t.values())).device if isinstance(x_t, dict) else x_t.device

    def _inference_model(self, model, x_t, t, cond=None):
        batch = self._get_batch_size(x_t)
        device = self._get_device(x_t)
        t_scaled = torch.tensor([1000 * t] * batch, device=device, dtype=torch.float32)
        if isinstance(cond, dict):
            for k, v in cond.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == 1 and batch > 1:
                    cond[k] = v.repeat(batch, *([1] * (len(v.shape) - 1)))
        elif cond is not None and cond.shape[0] == 1 and batch > 1:
            cond = cond.repeat(batch, *([1] * (len(cond.shape) - 1)))
        return model(x_t, t_scaled, cond)

    def _cfg_prediction(self, model, x_t, t, cond, neg_cond, guidance_scale):
        # Diffusers-style convention: guidance_scale == 1 (or <= 1, or None) means no CFG —
        # only the conditional pass runs, halving the per-step cost. > 1 enables CFG and
        # blends as `pred = s * cond + (1 - s) * uncond = s * cond - (s - 1) * uncond`.
        pred_v = self._inference_model(model, x_t, t, cond)
        if isinstance(guidance_scale, dict):
            if not any(s > 1 for s in guidance_scale.values()):
                return pred_v
            neg_pred_v = self._inference_model(model, x_t, t, neg_cond)
            for key in pred_v:
                s = guidance_scale.get(key, 1.0)
                if s > 1:
                    pred_v[key] = s * pred_v[key] - (s - 1) * neg_pred_v[key]
            return pred_v
        if guidance_scale is None or guidance_scale <= 1:
            return pred_v
        neg_pred_v = self._inference_model(model, x_t, t, neg_cond)
        for key in pred_v:
            pred_v[key] = guidance_scale * pred_v[key] - (guidance_scale - 1) * neg_pred_v[key]
        return pred_v

    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps=50, shift=1.0,
               guidance_scale=None, show_progress=False, callback=None):
        sample = noise
        t_seq = shift * np.linspace(1, 0, steps + 1) / (1 + (shift - 1) * np.linspace(1, 0, steps + 1))
        t_pairs = list(zip(t_seq[:-1], t_seq[1:]))
        iterator = tqdm(t_pairs, desc="Sampling", total=steps) if show_progress else t_pairs
        for i, (t, t_prev) in enumerate(iterator):
            x_t = {k: v.clone() for k, v in sample.items()} if isinstance(sample, dict) else sample.clone()
            pred_v = self._cfg_prediction(model, x_t, t, cond, neg_cond, guidance_scale)
            dt = t - t_prev
            if isinstance(sample, dict):
                for key in sample:
                    sample[key] = sample[key] - pred_v[key] * dt
            else:
                sample = sample - pred_v * dt
            if callback is not None:
                callback(i + 1, steps)
        return sample


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------

def _place(m, device, dtype):
    if device is not None or dtype is not None:
        m = m.to(device=device, dtype=dtype)
    return m.eval()


def load_dinov3(path: str, device=None, dtype=None) -> DinoV3ViT:
    m = DinoV3ViT()
    m.load_safetensors(path)
    return _place(m, device, dtype)


def load_vae_encoder(path: str, device=None, dtype=None) -> Flux2VAEEncoder:
    m = Flux2VAEEncoder()
    m.load_safetensors(path)
    return _place(m, device, dtype)


def load_rmbg(path: str, device=None, dtype=None) -> BiRefNet:
    m = BiRefNet()
    m.load_safetensors(path)
    return _place(m, device, dtype)


FLOW_MODEL_ARGS = dict(
    q_token_length=8192, in_channels=16, cam_channels=5, out_channels=16,
    model_channels=1024, cond_channels=1280, cond2_channels=128,
    num_refiner_blocks=2, num_blocks=24, num_heads=16, mlp_ratio=4,
    qk_rms_norm=True, share_mod=True, use_shift_table=True,
)


def load_flow_model(path: str, device=None, dtype=None) -> LatentSeqMMFlowModel:
    m = LatentSeqMMFlowModel(**FLOW_MODEL_ARGS)
    m.load_safetensors(path)
    return _place(m, device, dtype)


OCTREE_DECODER_ARGS = dict(
    model_channels=1024, cond_channels=16,
    num_blocks=4, num_heads=16, mlp_ratio=4, share_mod=True,
)

GS_DECODER_ARGS = dict(
    in_channels=3, model_channels=1024, cond_channels=16,
    attn_mode="full", num_blocks=16, num_heads=16, mlp_ratio=4,
    use_learned_offset_scale=True, use_per_offset=True,
    representation_config=dict(
        lr=dict(_xyz=1.0, _features_dc=1.0, _opacity=1.0, _scaling=1.0, _rotation=0.1),
        perturb_offset=True, perturbe_size=1.5, offset_scale=0.05, num_gaussians=32,
        filter_kernel_size_3d=0.0009, scaling_bias=0.004, opacity_bias=0.1,
        scaling_activation="softplus",
    ),
)


def load_decoder(path: str, device=None, dtype=None) -> OctreeGaussianDecoder:
    m = OctreeGaussianDecoder(OCTREE_DECODER_ARGS, GS_DECODER_ARGS)
    m.load_safetensors(path)
    return _place(m, device, dtype)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

_CANVAS_SIZE = 1024


def _image_to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        return Image.open(image)
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.ndim == 4:
            assert t.shape[0] == 1, (
                f"batched image input is not supported (got B={t.shape[0]}); "
                "pass one image at a time"
            )
            t = t[0]
        arr = (t.clamp(0, 1) * 255).to(torch.uint8).numpy()
        mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
        return Image.fromarray(arr, mode=mode)
    raise TypeError(f"unsupported image type: {type(image)}")


def preprocess_image(image, rmbg: BiRefNet, erode_radius: int = 1) -> Image.Image:
    image = _image_to_pil(image)
    size = _CANVAS_SIZE
    w, h = image.size
    s = size / min(w, h)
    image = image.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.LANCZOS)
    has_real_alpha = (image.mode == "RGBA"
                      and np.array(image.getchannel(3), dtype=np.int32).min() < 255)
    if not has_real_alpha:
        image = rmbg.remove_background(image.convert("RGB"))
    if erode_radius > 0:
        image.putalpha(image.getchannel(3).filter(ImageFilter.MinFilter(2 * erode_radius + 1)))
    alpha = np.array(image.getchannel(3))
    ys, xs = np.nonzero(alpha)
    bbox = [xs.min(), ys.min(), xs.max(), ys.max()]
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    half = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2 * 1.2
    image = image.crop([int(cx - half), int(cy - half), int(cx + half), int(cy + half)])
    image = image.resize((size, size), Image.LANCZOS)
    bg = Image.new("RGB", (size, size), (0, 0, 0))
    bg.paste(image, mask=image.split()[3])
    return bg


_DINOV3_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@torch.no_grad()
def encode_image(image: Image.Image, dinov3: DinoV3ViT, vae_encoder: Flux2VAEEncoder,
                 generator: torch.Generator = None) -> dict:
    device = next(dinov3.parameters()).device
    img_tensor   = transforms.ToTensor()(image).unsqueeze(0).to(device=device, dtype=torch.float32)
    img_normed   = _DINOV3_NORMALIZE(img_tensor)
    dinov3_dtype = next(dinov3.parameters()).dtype
    vae_dtype    = next(vae_encoder.parameters()).dtype
    dinov3_feat = dinov3(pixel_values=img_normed.to(dinov3_dtype))
    dinov3_feat = F.layer_norm(dinov3_feat.float(), dinov3_feat.shape[-1:])
    vae_feat = vae_encoder.encode(img_tensor.to(vae_dtype) * 2 - 1,
                                  deterministic=False, generator=generator)
    # pad 5 zero tokens so feature2's token length matches feature1's (cls + 4 registers + patches)
    zero_reg = torch.zeros(vae_feat.shape[0], 5, vae_feat.shape[2],
                           dtype=vae_feat.dtype, device=vae_feat.device)
    vae_feat = torch.cat([zero_reg, vae_feat], dim=1)
    return {'feature1': dinov3_feat, 'feature2': vae_feat}


@torch.no_grad()
def sample_latent(flow_model: LatentSeqMMFlowModel, cond: dict,
                  steps: int = 50, guidance_scale: float = 7.0, shift: float = 3.0,
                  generator: torch.Generator = None,
                  show_progress: bool = False, callback=None) -> dict:
    device = flow_model.device
    neg_cond = {k: torch.zeros_like(v) for k, v in cond.items()}
    noise = {'latent': torch.randn(1, flow_model.q_token_length, flow_model.in_channels,
                                   device=device, generator=generator)}
    if flow_model.cam_channels is not None:
        noise['camera'] = torch.randn(1, 1, flow_model.cam_channels,
                                      device=device, generator=generator)
    sampler = FlowEulerCfgSampler()
    return sampler.sample(flow_model, noise, cond=cond, neg_cond=neg_cond,
                          steps=steps, guidance_scale=guidance_scale, shift=shift,
                          show_progress=show_progress, callback=callback)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TripoSplatPipeline:
    def __init__(self, ckpt_path: str, decoder_path: str, dinov3_path: str,
                 flux2_vae_encoder_path: str, rmbg_path: str, device: str = "cuda"):
        self._device = torch.device(device)
        self.dinov3      = load_dinov3      (dinov3_path,             device=self._device, dtype=torch.bfloat16)
        self.vae_encoder = load_vae_encoder (flux2_vae_encoder_path,  device=self._device, dtype=torch.bfloat16)
        self.rmbg        = load_rmbg        (rmbg_path,               device=self._device, dtype=torch.float16)
        self.flow_model  = load_flow_model  (ckpt_path,               device=self._device, dtype=torch.float16)
        self.decoder     = load_decoder     (decoder_path,            device=self._device, dtype=torch.float16)

    def preprocess_image(self, image, erode_radius: int = 1) -> Image.Image:
        return preprocess_image(image, self.rmbg, erode_radius=erode_radius)

    def encode_image(self, image: Image.Image, generator: torch.Generator = None) -> dict:
        return encode_image(image, self.dinov3, self.vae_encoder, generator=generator)

    def sample_latent(self, cond: dict, steps: int = 50, guidance_scale: float = 7.0,
                      shift: float = 3.0, generator: torch.Generator = None,
                      show_progress: bool = False, callback=None) -> dict:
        return sample_latent(self.flow_model, cond, steps=steps, guidance_scale=guidance_scale,
                             shift=shift, generator=generator,
                             show_progress=show_progress, callback=callback)

    def decode_latent(self, latent: torch.Tensor, num_gaussians: int = 262144):
        return self.decoder.decode(latent, num_gaussians=num_gaussians)

    _NUM_GAUSSIANS_MIN = 32768
    _NUM_GAUSSIANS_MAX = 262144

    def _validate_num_gaussians(self, n: int) -> int:
        assert self._NUM_GAUSSIANS_MIN <= n <= self._NUM_GAUSSIANS_MAX, (
            f"num_gaussians must be in [{self._NUM_GAUSSIANS_MIN}, {self._NUM_GAUSSIANS_MAX}], got {n}"
        )
        gpp = self.decoder.gaussians_per_point
        if n % gpp == 0:
            return n
        rounded = round(n / gpp) * gpp
        print(f"[TripoSplatPipeline] num_gaussians={n} is not a multiple of {gpp}; rounding to {rounded}")
        return rounded

    @torch.no_grad()
    def run(self, image, seed: int = 42, steps: int = 20, guidance_scale: float = 3.0,
            shift: float = 3.0, num_gaussians=262144, erode_radius: int = 1,
            show_progress: bool = False, callback=None):
        """
        Args:
            image: Input image. Accepts a file path / PIL.Image / torch.Tensor
                (`[1,H,W,C]` or `[H,W,C]`, float in `[0, 1]`, optional alpha
                channel as the 4th channel).
            seed: RNG seed for the VAE encoder's stochastic latent sampling and
                the initial flow-matching noise. Same seed → same output.
            steps: Number of Euler integrator steps in the flow-matching sampler.
                More steps → better fidelity, linear runtime cost.
                Recommend: 10~20.
            guidance_scale: Classifier-free-guidance strength (diffusers
                convention). `≤ 1.0` disables CFG. Higher → more detail,
                stronger adherence to the input image; too high can cause color
                oversaturation.
                Recommend: 3.0.
            shift: Flow-matching timestep schedule shift. `1.0` gives a uniform
                schedule; `>1.0` allocates more steps to the early/high-noise end.
                Recommend: 3.0.
            num_gaussians: Target Gaussian-splat count. An `int` returns a
                single `Gaussian`. A `list` / `tuple` of ints returns a
                `list[Gaussian]`. Each count is rounded to the nearest multiple
                of 32. More gaussians → more detail but higher rendering and
                storage cost.
                Recommend: 32768~262144.
            erode_radius: Pixel radius used to erode the alpha matte after
                background removal, to avoid segmentation-border bleed before
                compositing on black. `0` disables; `1` is a 3×3 minimum filter.
                Recommend: 1.
            show_progress: Print a `tqdm` progress bar over sampler steps.
            callback: Optional `fn(step, total)` invoked after each sampler step.
                Useful for external progress UIs (e.g. ComfyUI's
                `ProgressBar.update`).

        Returns:
            `(gaussian, prepared_image)` for an `int` `num_gaussians`, or
            `(list_of_gaussians, prepared_image)` for a `list` / `tuple`. The
            second element is the RGB composite the encoders actually saw —
            useful for display / debugging.
        """
        if isinstance(num_gaussians, (list, tuple)):
            counts = [self._validate_num_gaussians(n) for n in num_gaussians]
        else:
            counts = [self._validate_num_gaussians(num_gaussians)]

        gen = torch.Generator(device=self._device).manual_seed(seed)
        prepared = self.preprocess_image(image, erode_radius=erode_radius)
        cond = self.encode_image(prepared, generator=gen)
        out = self.sample_latent(cond, steps=steps, guidance_scale=guidance_scale, shift=shift,
                                 generator=gen, show_progress=show_progress, callback=callback)
        gaussians = [self.decode_latent(out['latent'], num_gaussians=n) for n in counts]
        if isinstance(num_gaussians, (list, tuple)):
            return gaussians, prepared
        return gaussians[0], prepared
