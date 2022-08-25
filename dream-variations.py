from ldm.simplet2i import T2I
import transformers
import numpy as np
import torch
from pytorch_lightning import seed_everything
import random

t2i = T2I(
    latent_diffusion_weights=False,
    config  = "configs/stable-diffusion/v1-inference.yaml"
)

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays v1 v2 """

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


transformers.logging.set_verbosity_error()
seed = 3516972428
seed_everything(seed)
init_code = torch.randn([t2i.batch_size,
                          t2i.latent_channels,
                          t2i.height // t2i.downsampling_factor,
                          t2i.width  // t2i.downsampling_factor],
                          device=t2i.device)


print("loading model...")
t2i.load_model()

##### interpolate
# noise = torch.randn([t2i.batch_size,
#                           t2i.latent_channels,
#                           t2i.height // t2i.downsampling_factor,
#                           t2i.width  // t2i.downsampling_factor],
#                           device=t2i.device)
# for i in range(20):
#   print("running generation " + str(i))
#   code = slerp(i / 20., init_code, noise)
#   outputs = t2i.txt2img("elf queen with rainbow hair, golden hour. colored pencil drawing by rossdraws andrei riabovitchev trending on artstation", start_code=code, seed=seed)

# generate variants
strength = 0.05
prompt = "elf queen with rainbow hair, golden hour. colored pencil drawing by rossdraws andrei riabovitchev trending on artstation"

print("generating base image")
t2i.txt2img(prompt, start_code=init_code, seed=seed)
for i in range(20):
  random.seed() # reset RNG to an actually random state, so we can get a random seed
  iter_seed = random.randrange(0,np.iinfo(np.uint32).max)
  print("iteration " + str(i) + " running, seed = " + str(iter_seed))
  seed_everything(iter_seed)
  noise = torch.randn([t2i.batch_size,
                            t2i.latent_channels,
                            t2i.height // t2i.downsampling_factor,
                            t2i.width  // t2i.downsampling_factor],
                            device=t2i.device)
  code = slerp(strength, init_code, noise)
  seed_everything(iter_seed)
  t2i.txt2img(prompt, start_code=code, seed=iter_seed)
