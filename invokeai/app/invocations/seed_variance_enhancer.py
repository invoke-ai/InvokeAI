"""Seed Variance Enhancer for Z-Image conditioning.

Adds controlled random noise to conditioning embeddings to increase output diversity,
particularly useful for Z-Image models with low seed variance.

Based on the ComfyUI SeedVarianceEnhancer node by ChangeTheConstants.
Released under MIT No Attribution License.
"""

from enum import Enum

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import Input, InputField, ZImageConditioningField
from invokeai.app.invocations.primitives import ZImageConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    ZImageConditioningInfo,
)

# Seed offset for noise mask generation (v2.2 behavior for consistency with prompt variations)
NOISE_MASK_SEED_OFFSET = 1

# Factors for suggested strength calculation based on embedding standard deviation
MIN_STRENGTH_FACTOR = 0.1
MAX_STRENGTH_FACTOR = 10.0


class MaskStartPosition(str, Enum):
    """Which end of the prompt will be protected from noise."""

    BEGINNING = "beginning"
    END = "end"


@invocation(
    "seed_variance_enhancer",
    title="Seed Variance Enhancer - Z-Image",
    tags=["conditioning", "z-image", "variance", "seed", "noise"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class SeedVarianceEnhancerInvocation(BaseInvocation):
    """Adds random noise to Z-Image conditioning embeddings to increase output diversity.

    This node compensates for low seed variance by adding controlled noise to the conditioning
    embeddings. Works specifically with Z-Image models.

    Typical settings for Z-Image Turbo:
    - randomize_percent: 50%
    - strength: 15-40
    - Experiment with different values for your specific prompts

    Masking features allow protecting portions of the prompt from noise exposure.
    """

    conditioning: ZImageConditioningField = InputField(
        description="The Z-Image conditioning to enhance with variance.",
        input=Input.Connection,
    )
    randomize_percent: float = InputField(
        default=50.0,
        ge=1.0,
        le=100.0,
        description="Percentage of embedding values to which random noise is added.",
    )
    strength: float = InputField(
        default=20.0,
        description="Scale of the random noise. Typical range: 15-40 for Z-Image Turbo.",
    )
    seed: int = InputField(
        default=0,
        ge=0,
        description="Random seed for noise generation and value selection.",
    )
    mask_starts_at: MaskStartPosition = InputField(
        default=MaskStartPosition.BEGINNING,
        description="Which end of prompt will be protected from noise.",
    )
    mask_percent: float = InputField(
        default=0.0,
        ge=0.0,
        le=99.0,
        description="Percentage of prompt protected from noise. 0 = no masking.",
    )
    log_statistics: bool = InputField(
        default=False,
        description="Log embedding statistics to console for debugging.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ZImageConditioningOutput:
        # Load the conditioning data
        conditioning_data = context.conditioning.load(self.conditioning.conditioning_name)

        # Early return if strength is zero
        if self.strength == 0:
            if self.log_statistics:
                context.logger.info(
                    "Seed Variance Enhancer strength is zero. Passing conditioning through unchanged."
                )
                self._log_statistics(context, conditioning_data)
            return ZImageConditioningOutput(conditioning=self.conditioning)

        # Validate we have Z-Image conditioning
        if not conditioning_data.conditionings or len(conditioning_data.conditionings) == 0:
            context.logger.warning("Seed Variance Enhancer received empty conditioning.")
            return ZImageConditioningOutput(conditioning=self.conditioning)

        conditioning_info = conditioning_data.conditionings[0]
        if not isinstance(conditioning_info, ZImageConditioningInfo):
            context.logger.warning(
                f"Seed Variance Enhancer expected Z-Image conditioning, got {type(conditioning_info).__name__}"
            )
            return ZImageConditioningOutput(conditioning=self.conditioning)

        # Get the prompt embeddings tensor
        prompt_embeds = conditioning_info.prompt_embeds

        if self.log_statistics:
            self._log_statistics(context, conditioning_data)

        # Apply noise to the embeddings
        noisy_prompt_embeds = self._apply_noise(context, prompt_embeds)

        # Create new conditioning with noisy embeddings
        new_conditioning_info = ZImageConditioningInfo(prompt_embeds=noisy_prompt_embeds)
        new_conditioning_data = ConditioningFieldData(conditionings=[new_conditioning_info])

        # Save and return
        conditioning_name = context.conditioning.save(new_conditioning_data)
        return ZImageConditioningOutput(
            conditioning=ZImageConditioningField(
                conditioning_name=conditioning_name,
                mask=self.conditioning.mask,
            )
        )

    def _apply_noise(self, context: InvocationContext, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """Apply random noise to prompt embeddings.
        
        Z-Image uses 2D tensors: [seq_len, hidden_size]
        """
        # Normalize parameters
        randomize_percent = max(1, min(100, self.randomize_percent)) / 100.0
        mask_percent = max(0, min(99, self.mask_percent)) / 100.0

        # Generate noise
        torch.manual_seed(self.seed)
        noise = torch.rand_like(prompt_embeds) * 2 * self.strength - self.strength

        # Reset seed for value selection to ensure consistency with prompt variations
        torch.manual_seed(self.seed + NOISE_MASK_SEED_OFFSET)
        noise_mask = torch.bernoulli(torch.full_like(prompt_embeds, randomize_percent)).bool()

        # Check for null sequences (padding)
        first_null, last_nonnull, is_null_list = self._find_null_sequences(prompt_embeds)

        # Apply masking if needed
        # For 2D tensor: dimension 0 = seq_len, dimension 1 = hidden_size
        seq_dim = 0
        if mask_percent > 0 or last_nonnull < prompt_embeds.size(seq_dim) - 1:
            seq_len = (
                last_nonnull + 1
                if last_nonnull >= 0 and last_nonnull < prompt_embeds.size(seq_dim) - 1
                else prompt_embeds.size(seq_dim)
            )

            # Determine mask range
            if self.mask_starts_at == MaskStartPosition.END:
                mask_start = seq_len - int(seq_len * mask_percent)
                mask_end = prompt_embeds.size(seq_dim)
            else:  # BEGINNING
                mask_start = 0
                mask_end = int(seq_len * mask_percent)

            # Create position-based mask for 2D tensor [seq_len, hidden_size]
            prompt_mask = (
                torch.arange(prompt_embeds.size(seq_dim), device=prompt_embeds.device)
                .unsqueeze(1)  # [seq_len, 1]
                .expand(prompt_embeds.size(seq_dim), prompt_embeds.size(1))  # [seq_len, hidden_size]
            )
            prompt_mask = (prompt_mask >= mask_start) & (prompt_mask < mask_end)

            # Include null sequences in protected region
            if first_null > -1:
                if self.log_statistics:
                    context.logger.info("Seed Variance Enhancer is masking null sequences from noise")

                # Convert is_null_list to tensor: True where sequences should be protected (null sequences)
                null_mask_tensor = torch.tensor(is_null_list, device=prompt_embeds.device, dtype=torch.bool)
                null_mask_tensor = null_mask_tensor.unsqueeze(1).expand(
                    prompt_embeds.size(seq_dim), prompt_embeds.size(1)
                )
                prompt_mask = prompt_mask | null_mask_tensor

            # Combine with noise mask
            noise_mask = noise_mask & (~prompt_mask)

        # Apply masked noise
        modified_noise = noise * noise_mask
        noisy_embeds = prompt_embeds + modified_noise

        return noisy_embeds

    def _find_null_sequences(self, tensor: torch.Tensor) -> tuple[int, int, list[int]]:
        """Find sequences in tensor that contain all zeros (padding).
        
        Z-Image uses 2D tensors: [seq_len, hidden_size]

        Returns:
            Tuple of (first_null_index, last_nonnull_index, is_null_list)
            where is_null_list contains 1 for null sequences and 0 for non-null sequences
        """
        first_null = -1
        last_nonnull = -1
        
        # For 2D tensor: dimension 0 = seq_len, dimension 1 = hidden_size
        is_null_list = [0] * tensor.size(0)

        if tensor.dim() == 2:
            for i in range(tensor.size(0)):
                sequence = tensor[i, :]  # Get the i-th sequence (all hidden dimensions)
                is_all_zero = torch.all(sequence == 0)

                is_null_list[i] = 1 if is_all_zero else 0

                if not is_all_zero:
                    last_nonnull = i

                if is_all_zero and first_null == -1:
                    first_null = i

        return first_null, last_nonnull, is_null_list

    def _log_statistics(self, context: InvocationContext, conditioning_data: ConditioningFieldData) -> None:
        """Log statistics about the conditioning tensor.
        
        Z-Image uses 2D tensors: [seq_len, hidden_size]
        """
        if not conditioning_data.conditionings:
            context.logger.warning("Conditioning data has no conditionings")
            return

        conditioning_info = conditioning_data.conditionings[0]
        if not isinstance(conditioning_info, ZImageConditioningInfo):
            context.logger.warning(f"Expected ZImageConditioningInfo, got {type(conditioning_info).__name__}")
            return

        tensor = conditioning_info.prompt_embeds
        if not isinstance(tensor, torch.Tensor):
            context.logger.warning("Conditioning does not contain a tensor")
            return

        # Find null sequences
        first_null, last_nonnull, is_null_list = self._find_null_sequences(tensor)

        # Calculate statistics on non-null portion
        # For 2D tensor: dimension 0 = seq_len, dimension 1 = hidden_size
        if last_nonnull < tensor.size(0) - 1 and last_nonnull >= 0:
            sliced_tensor = tensor[: last_nonnull + 1, :]
            mean = torch.mean(sliced_tensor).item()
            std = torch.std(sliced_tensor).item()
            min_val = torch.min(sliced_tensor).item()
            max_val = torch.max(sliced_tensor).item()
        else:
            mean = torch.mean(tensor).item()
            std = torch.std(tensor).item()
            min_val = torch.min(tensor).item()
            max_val = torch.max(tensor).item()

        context.logger.info("=== Seed Variance Enhancer - Embedding Statistics ===")
        context.logger.info(f"Dimensions: {list(tensor.shape)}")
        context.logger.info(f"Min: {min_val:.6f}, Max: {max_val:.6f}")
        context.logger.info(f"Mean: {mean:.6f}, Std Dev: {std:.6f}")
        context.logger.info(
            f"Suggested strength range: {std * MIN_STRENGTH_FACTOR:.6f} - {std * MAX_STRENGTH_FACTOR:.6f}"
        )

        if first_null != -1:
            num_null = sum(1 for x in is_null_list if x == 1)
            context.logger.info(
                f"Null sequences: First at {first_null}, Last non-null at {last_nonnull}, Total null: {num_null}"
            )
