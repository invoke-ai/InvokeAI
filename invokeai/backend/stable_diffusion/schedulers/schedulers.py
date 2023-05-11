from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, KDPM2DiscreteScheduler, \
    KDPM2AncestralDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, \
    HeunDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler, UniPCMultistepScheduler, \
    DPMSolverSinglestepScheduler

SCHEDULER_MAP = dict(
    ddim=(DDIMScheduler, dict()),
    k_lms=(LMSDiscreteScheduler, dict()),
    plms=(PNDMScheduler, dict()),
    k_euler=(EulerDiscreteScheduler, dict(use_karras_sigmas=False)),
    euler_karras=(EulerDiscreteScheduler, dict(use_karras_sigmas=True)),
    k_euler_a=(EulerAncestralDiscreteScheduler, dict()),
    k_dpm_2=(KDPM2DiscreteScheduler, dict()),
    k_dpm_2_a=(KDPM2AncestralDiscreteScheduler, dict()),
    dpmpp_2s=(DPMSolverSinglestepScheduler, dict()),
    k_dpmpp_2=(DPMSolverMultistepScheduler, dict(use_karras_sigmas=False)),
    k_dpmpp_2_karras=(DPMSolverMultistepScheduler, dict(use_karras_sigmas=True)),
    k_heun=(HeunDiscreteScheduler, dict()),
    unipc=(UniPCMultistepScheduler, dict(cpu_only=True))
)
