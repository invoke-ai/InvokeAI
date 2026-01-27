"""DyPE extension for FLUX denoising pipeline."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from invokeai.backend.flux.dype.base import DyPEConfig
from invokeai.backend.flux.dype.embed import DyPEEmbedND

if TYPE_CHECKING:
    from invokeai.backend.flux.model import Flux


@dataclass
class DyPEExtension:
    """Extension for Dynamic Position Extrapolation in FLUX models.

    This extension manages the patching of the FLUX model's position embedder
    and updates the step state during denoising.

    Usage:
        1. Create extension with config and target dimensions
        2. Call patch_model() to replace pe_embedder with DyPE version
        3. Call update_step_state() before each denoising step
        4. Call restore_model() after denoising to restore original embedder
    """

    config: DyPEConfig
    target_height: int
    target_width: int

    def patch_model(self, model: "Flux") -> tuple[DyPEEmbedND, object]:
        """Patch the model's position embedder with DyPE version.

        Args:
            model: The FLUX model to patch

        Returns:
            Tuple of (new DyPE embedder, original embedder for restoration)
        """
        original_embedder = model.pe_embedder

        dype_embedder = DyPEEmbedND.from_embednd(
            embed_nd=original_embedder,
            dype_config=self.config,
        )

        # Set initial state
        dype_embedder.set_step_state(
            sigma=1.0,
            height=self.target_height,
            width=self.target_width,
        )

        # Replace the embedder
        model.pe_embedder = dype_embedder

        return dype_embedder, original_embedder

    def update_step_state(
        self,
        embedder: DyPEEmbedND,
        timestep: float,
        timestep_index: int,
        total_steps: int,
    ) -> None:
        """Update the step state in the DyPE embedder.

        This should be called before each denoising step to update the
        current noise level for timestep-dependent scaling.

        Args:
            embedder: The DyPE embedder to update
            timestep: Current timestep value (sigma/noise level)
            timestep_index: Current step index (0-based)
            total_steps: Total number of denoising steps
        """
        embedder.set_step_state(
            sigma=timestep,
            height=self.target_height,
            width=self.target_width,
        )

    @staticmethod
    def restore_model(model: "Flux", original_embedder: object) -> None:
        """Restore the original position embedder.

        Args:
            model: The FLUX model to restore
            original_embedder: The original embedder saved from patch_model()
        """
        model.pe_embedder = original_embedder
