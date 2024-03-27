# copied from https://github.com/tencent-ailab/IP-Adapter (Apache License 2.0)

# tencent ailab comment: modified from
# https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
import math

import torch
import torch.nn as nn


# FFN
def FeedForward(dim: int, mult: int = 4):
    inner_dim = dim * mult
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x: torch.Tensor, heads: int):
    bs, length, _ = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, L, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, L, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        depth: int = 8,
        dim_head: int = 64,
        heads: int = 16,
        num_queries: int = 8,
        embedding_dim: int = 768,
        output_dim: int = 1024,
        ff_mult: int = 4,
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        depth: int = 8,
        dim_head: int = 64,
        heads: int = 16,
        num_queries: int = 8,
        ff_mult: int = 4,
    ):
        """A convenience function that initializes a Resampler from a state_dict.

        Some of the shape parameters are inferred from the state_dict (e.g. dim, embedding_dim, etc.). At the time of
        writing, we did not have a need for inferring ALL of the shape parameters from the state_dict, but this would be
        possible if needed in the future.

        Args:
            state_dict (dict[torch.Tensor]): The state_dict to load.
            depth (int, optional):
            dim_head (int, optional):
            heads (int, optional):
            ff_mult (int, optional):

        Returns:
            Resampler
        """
        dim = state_dict["latents"].shape[2]
        num_queries = state_dict["latents"].shape[1]
        embedding_dim = state_dict["proj_in.weight"].shape[-1]
        output_dim = state_dict["norm_out.weight"].shape[0]

        model = cls(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_queries=num_queries,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            ff_mult=ff_mult,
        )
        model.load_state_dict(state_dict)
        return model

    def forward(self, x: torch.Tensor):
        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)
