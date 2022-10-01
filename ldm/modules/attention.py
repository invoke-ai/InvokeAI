from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint

import psutil
import os
import numpy as np

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.mem_total_gb = psutil.virtual_memory().total // (1 << 30)

    def einsum_op_compvis(self, q, k, v):
        s = einsum('b i d, b j d -> b i j', q, k)
        s = s.softmax(dim=-1, dtype=s.dtype)
        return einsum('b i j, b j d -> b i d', s, v)

    def einsum_op_slice_0(self, q, k, v, slice_size):
        r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
        for i in range(0, q.shape[0], slice_size):
            end = i + slice_size
            r[i:end] = self.einsum_op_compvis(q[i:end], k[i:end], v[i:end])
        return r

    def einsum_op_slice_1(self, q, k, v, slice_size):
        r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            r[:, i:end] = self.einsum_op_compvis(q[:, i:end], k, v)
        return r

    def einsum_op_mps_v1(self, q, k, v):
        if q.shape[1] <= 4096: # (512x512) max q.shape[1]: 4096
            return self.einsum_op_compvis(q, k, v)
        else:
            slice_size = math.floor(2**30 / (q.shape[0] * q.shape[1]))
            return self.einsum_op_slice_1(q, k, v, slice_size)

    def einsum_op_mps_v2(self, q, k, v):
        if self.mem_total_gb > 8 and q.shape[1] <= 4096:
            return self.einsum_op_compvis(q, k, v)
        else:
            return self.einsum_op_slice_0(q, k, v, 1)

    def einsum_op_tensor_mem(self, q, k, v, max_tensor_mb):
        size_mb = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size() // (1 << 20)
        if size_mb <= max_tensor_mb:
            return self.einsum_op_compvis(q, k, v)
        div = 1 << int((size_mb - 1) / max_tensor_mb).bit_length()
        if div <= q.shape[0]:
            return self.einsum_op_slice_0(q, k, v, q.shape[0] // div)
        return self.einsum_op_slice_1(q, k, v, max(q.shape[1] // div, 1))

    def einsum_op_cuda(self, q, k, v):
        stats = torch.cuda.memory_stats(q.device)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(q.device)
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch
        # Divide factor of safety as there's copying and fragmentation
        return self.einsum_op_tensor_mem(q, k, v, mem_free_total / 3.3 / (1 << 20))

    def einsum_op(self, q, k, v):
        if q.device.type == 'cuda':
            return self.einsum_op_cuda(q, k, v)

        if q.device.type == 'mps':
            if self.mem_total_gb >= 32:
                return self.einsum_op_mps_v1(q, k, v)
            return self.einsum_op_mps_v2(q, k, v)

        # Smaller slices are faster due to L2/L3/SLC caches.
        # Tested on i7 with 8MB L3 cache.
        return self.einsum_op_tensor_mem(q, k, v, 32)

    def forward(
        self, x, context=None, scontext=None, pmask=None, time=None, mask=None
    ):
        """
        x.shape: (6,4096,320)
        context.shape(6,77,768)
        q, k, v shape: (6, hw, 320), (6, 77, 320), (6, 77, 320)
        -> q,k,v shape: (32, hw, 40=320/8=self.head), (32, 77, 40=320/8=self.head), (32, 77, 40=320/8=self.head)

        - visualization.
        1. aggregate all attention map across the "timesteps" and "heads"
        2. Normalization divided by "max" with respecto to "each token"
        """

        h = self.heads
        if scontext == 'selfattn':
            sim, attn, v = self.get_attmap(
                x=x, h=self.heads, context=context, mask=None
            )
            sattn = None
        else:
            if scontext is None:
                sim, attn, v = self.get_attmap(
                    x=x, h=self.heads, context=context, mask=None
                )
                sattn = None

                """ cross attention control: only reweighting is possible. """
                """
                ex) A photo of a house on a snowy mountain
                : for controlling "snowy":
                the token index=8.
                the weights for sample1~3 are -2, 1, 5 in this example.
                """
                attn = self.cross_attention_control(
                    tattmap=attn,
                    t=time,
                    token_idx=[2],
                    weights=[[-2.0, 1.0, 5.0]],
                )
            else:
                x, sx = x.chunk(2)
                sim, attn, v = self.get_attmap(
                    x=x, h=self.heads, context=context, mask=None
                )
                ssim, sattn, sv = self.get_attmap(
                    x=sx, h=self.heads, context=scontext, mask=None
                )

                """ cross attention control """
                bh, hw, tleng = attn.shape
                attn = self.cross_attention_control(
                    tattmap=attn,
                    sattmap=sattn,
                    pmask=pmask,
                    t=time,
                    token_idx=[0],
                    weights=[[1.0, 1.0, 1.0]],
                )

        """ target prompt """
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)

        if scontext != 'selfattn':
            if scontext is not None:
                """source prompt"""
                sout = einsum('b i j, b j d -> b i d', sattn, sv)
                sout = rearrange(sout, '(b h) n d -> b n (h d)', h=h)
                sout = self.to_out(sout)

                """ aggregate again befroe return """
                out = torch.cat((out, sout))
                sim = torch.cat((sim, ssim))

                return out, sim

        return out, sim

    def get_attmap(self, x, h, context, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        query, key, val = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v)
        )
        sim = einsum('b i d, b j d -> b i j', query, key) * self.scale
        # attention, what we cannot get enough of
        rear_sim = rearrange(sim, '(b h) n d -> b h n d', h=h)

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        return rear_sim, attn, val

    def cross_attention_control(
        self,
        tattmap,
        sattmap=None,
        pmask=None,
        t=0,
        tthres=0,
        token_idx=[0],
        weights=[[1.0, 1.0, 1.0]],
    ):
        attn = tattmap
        sattn = sattmap

        h = 8
        bh, n, d = attn.shape

        if t >= tthres:
            """1. swap & ading new phrase"""
            if sattmap is not None:
                bh, n, d = attn.shape
                pmask, sindices, indices = pmask
                pmask = pmask.view(1, 1, -1).repeat(bh, n, 1)
                attn = (1 - pmask) * attn[:, :, indices] + (pmask) * sattn[
                    :, :, sindices
                ]

            """ 2. reweighting """
            attn = rearrange(
                attn, '(b h) n d -> b h n d', h=h
            )
            num_iter = bh // (h * 2)
            for k in range(len(token_idx)):
                for i in range(num_iter):
                    attn[num_iter + i, :, :, token_idx[k]] *= weights[k][i]
            attn = rearrange(
                attn, 'b h n d -> (b h) n d', h=h
            )

        return attn


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, scontext=None, pmask=None, time=None):
        return checkpoint(self._forward, (x, context, scontext, pmask, time), self.parameters(), self.checkpoint, )

    def _forward(self, x, context=None, scontext=None, pmask=None, time=None):
        x_att, self_attmap = self.attn1(
            self.norm1(x), scontext='selfattn', time=time
        )
        x += x_att
        x_att, cross_attmap = self.attn2(
            self.norm2(x),
            context=context,
            scontext=scontext,
            pmask=pmask,
            time=time,
        ) 
        x += x_att
        x = self.ff(self.norm3(x)) + x
        return x, self_attmap, cross_attmap


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 is_get_attn=False, attn_save_dir='./attenion_map_savedir'):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

        self.is_get_attn = is_get_attn
        if self.is_get_attn:
            self.attmap_save_dir = attn_save_dir
            os.makedirs(
                os.path.join(self.attmap_save_dir, 'selfatt'), exist_ok=True
            )
            os.makedirs(
                os.path.join(self.attmap_save_dir, 'crossatt'), exist_ok=True
            )

    def forward(self, x, context=None,
                scontext=None, pmask=None, timestep_str=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            time = int(timestep_str.split('_')[1].split('time')[1])
            x, self_attmap, cross_attmap = block(x, context=context,
                                                 scontext=scontext,
                                                 pmask=pmask,
                                                 time=time)
            if self.is_get_attn:
                if scontext is not None:
                    """save attention map"""
                    cross_attmap, scross_attmap = cross_attmap.chunk(2)
                    np.save(
                        os.path.join(
                            self.attmap_save_dir, 'crossatt', timestep_str
                        ),
                        self.avg_attmap(cross_attmap).detach().cpu().numpy(),
                    )
                else:
                    """save attention map"""
                    np.save(
                        os.path.join(
                            self.attmap_save_dir, 'crossatt', timestep_str
                        ),
                        self.avg_attmap(cross_attmap).detach().cpu().numpy(),
                    )
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in

    def avg_attmap(self, attmap, token_idx=0):
        """
        num_sample(=batch_size) = 3
        uc,c = 2 #(unconditional, condiitonal)
        -> 3*2=6

        attmap.shape: similarity matrix.
        token_idx: index of token for visualizing, 77: [SOS, ...text..., EOS]
        """
        nsample2, head, hw, context_dim = attmap.shape
        attmap_sm = F.softmax(
            attmap.float(), dim=-1
        )
        att_map_sm = attmap_sm[
            nsample2 // 2 :, :, :, :
        ]
        att_map_mean = torch.mean(att_map_sm, dim=1)

        b, hw, context_dim = att_map_mean.shape
        h = int(math.sqrt(hw))
        w = h

        return att_map_mean.view(
            b, h, w, context_dim
        )
