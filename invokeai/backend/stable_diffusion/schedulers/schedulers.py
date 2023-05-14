from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, KDPM2DiscreteScheduler, \
    KDPM2AncestralDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, \
    HeunDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler, UniPCMultistepScheduler, \
    DPMSolverSinglestepScheduler, DEISMultistepScheduler, DDPMScheduler

SCHEDULER_MAP = dict(
    ddim=(DDIMScheduler, dict()),
    ddpm=(DDPMScheduler, dict()),
    deis=(DEISMultistepScheduler, dict()),
    lms=(LMSDiscreteScheduler, dict()),
    pndm=(PNDMScheduler, dict()),
    heun=(HeunDiscreteScheduler, dict(use_karras_sigmas=False)),
    heun_k=(HeunDiscreteScheduler, dict(use_karras_sigmas=True)),
    euler=(EulerDiscreteScheduler, dict(use_karras_sigmas=False)),
    euler_k=(EulerDiscreteScheduler, dict(use_karras_sigmas=True)),
    euler_a=(EulerAncestralDiscreteScheduler, dict()),
    kdpm_2=(KDPM2DiscreteScheduler, dict()),
    kdpm_2_a=(KDPM2AncestralDiscreteScheduler, dict()),
    dpmpp_2s=(DPMSolverSinglestepScheduler, dict()),
    dpmpp_2m=(DPMSolverMultistepScheduler, dict(use_karras_sigmas=False)),
    dpmpp_2m_k=(DPMSolverMultistepScheduler, dict(use_karras_sigmas=True)),
    unipc=(UniPCMultistepScheduler, dict(cpu_only=True))
)
