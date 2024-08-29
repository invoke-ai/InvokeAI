import torch


def merge_intermediate_latents_with_init_latents(
    init_latents: torch.Tensor,
    intermediate_latents: torch.Tensor,
    timestep: float,
    noise: torch.Tensor,
    inpaint_mask: torch.Tensor,
) -> torch.Tensor:
    # Noise the init_latents for the current timestep.
    noised_init_latents = noise * timestep + (1.0 - timestep) * init_latents

    # Merge the intermediate_latents with the noised_init_latents using the inpaint_mask.
    return intermediate_latents * inpaint_mask + noised_init_latents * (1.0 - inpaint_mask)
