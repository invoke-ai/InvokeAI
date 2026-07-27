"""Named sampler configurations for Ideogram 4 inference."""

from __future__ import annotations

from invokeai.backend.ideogram4.scheduler import SamplerParameters

# guidance_schedule is in loop-INDEX order: index 0 is the LAST (polish) step.
# Each preset does the first N_main sampling steps at gw=7, then N_cleanup
# polish steps at gw=3.
PRESETS: dict[str, SamplerParameters] = {
    "V4_QUALITY_48": SamplerParameters(
        num_steps=48,
        guidance_schedule=(3.0,) * 3 + (7.0,) * 45,
        mu=0.0,
        std=1.5,
    ),
    "V4_DEFAULT_20": SamplerParameters(
        num_steps=20,
        guidance_schedule=(3.0,) * 2 + (7.0,) * 18,
        mu=0.0,
        std=1.75,
    ),
    "V4_TURBO_12": SamplerParameters(
        num_steps=12,
        guidance_schedule=(3.0,) * 1 + (7.0,) * 11,
        mu=0.5,
        std=1.75,
    ),
}
