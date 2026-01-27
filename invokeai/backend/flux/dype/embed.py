"""DyPE-enhanced position embedding module."""

import torch
from torch import Tensor, nn

from invokeai.backend.flux.dype.base import DyPEConfig
from invokeai.backend.flux.dype.rope import rope_dype


class DyPEEmbedND(nn.Module):
    """N-dimensional position embedding with DyPE support.

    This class replaces the standard EmbedND from FLUX with a DyPE-aware version
    that dynamically scales position embeddings based on resolution and timestep.

    The key difference from EmbedND:
    - Maintains step state (current_sigma, target dimensions)
    - Uses rope_dype() instead of rope() for frequency computation
    - Applies timestep-dependent scaling for better high-resolution generation
    """

    def __init__(
        self,
        dim: int,
        theta: int,
        axes_dim: list[int],
        dype_config: DyPEConfig,
    ):
        """Initialize DyPE position embedder.

        Args:
            dim: Total embedding dimension (sum of axes_dim)
            theta: RoPE base frequency
            axes_dim: Dimension allocation per axis (e.g., [16, 56, 56] for FLUX)
            dype_config: DyPE configuration
        """
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.dype_config = dype_config

        # Step state - updated before each denoising step
        self._current_sigma: float = 1.0
        self._target_height: int = 1024
        self._target_width: int = 1024

    def set_step_state(self, sigma: float, height: int, width: int) -> None:
        """Update the step state before each denoising step.

        This method should be called by the DyPE extension before each step
        to update the current noise level and target dimensions.

        Args:
            sigma: Current noise level (timestep value, 1.0 = full noise)
            height: Target image height in pixels
            width: Target image width in pixels
        """
        self._current_sigma = sigma
        self._target_height = height
        self._target_width = width

    def forward(self, ids: Tensor) -> Tensor:
        """Compute position embeddings with DyPE scaling.

        Args:
            ids: Position indices tensor with shape (batch, seq_len, n_axes)
                 For FLUX: n_axes=3 (time/channel, height, width)

        Returns:
            Position embedding tensor with shape (batch, 1, seq_len, dim)
        """
        n_axes = ids.shape[-1]

        # Compute RoPE for each axis with DyPE scaling
        embeddings = []
        for i in range(n_axes):
            axis_emb = rope_dype(
                pos=ids[..., i],
                dim=self.axes_dim[i],
                theta=self.theta,
                current_sigma=self._current_sigma,
                target_height=self._target_height,
                target_width=self._target_width,
                dype_config=self.dype_config,
            )
            embeddings.append(axis_emb)

        # Concatenate embeddings from all axes
        emb = torch.cat(embeddings, dim=-3)

        return emb.unsqueeze(1)

    @classmethod
    def from_embednd(
        cls,
        embed_nd: nn.Module,
        dype_config: DyPEConfig,
    ) -> "DyPEEmbedND":
        """Create a DyPEEmbedND from an existing EmbedND.

        This is a convenience method for patching an existing FLUX model.

        Args:
            embed_nd: Original EmbedND module from FLUX
            dype_config: DyPE configuration

        Returns:
            New DyPEEmbedND with same parameters
        """
        return cls(
            dim=embed_nd.dim,
            theta=embed_nd.theta,
            axes_dim=embed_nd.axes_dim,
            dype_config=dype_config,
        )
