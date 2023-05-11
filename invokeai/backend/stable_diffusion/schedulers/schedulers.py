from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, KDPM2DiscreteScheduler, \
    KDPM2AncestralDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, \
    HeunDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler, UniPCMultistepScheduler

SCHEDULER_MAP = dict(
    ddim=(DDIMScheduler, dict(cpu_only=False)),
    dpmpp_2=(DPMSolverMultistepScheduler, dict(cpu_only=False)),
    k_dpm_2=(KDPM2DiscreteScheduler, dict(cpu_only=False)),
    k_dpm_2_a=(KDPM2AncestralDiscreteScheduler, dict(cpu_only=False)),
    k_dpmpp_2=(DPMSolverMultistepScheduler, dict(cpu_only=False)),
    k_euler=(EulerDiscreteScheduler, dict(cpu_only=False)),
    k_euler_a=(EulerAncestralDiscreteScheduler, dict(cpu_only=False)),
    k_heun=(HeunDiscreteScheduler, dict(cpu_only=False)),
    k_lms=(LMSDiscreteScheduler, dict(cpu_only=False)),
    plms=(PNDMScheduler, dict(cpu_only=False)),
    unipc=(UniPCMultistepScheduler, dict(cpu_only=True))
)
