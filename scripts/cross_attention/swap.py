"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from difflib import SequenceMatcher
import copy


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def get_indice(model, prompts, sprompts, device='cuda'):
    """from cross attention control(https://github.com/bloc97/CrossAttentionControl)"""
    # input_ids: 49406, 1125, 539, 320, 2368, 6765, 525, 320, 11652, 49407]
    tokenizer = model.cond_stage_model.tokenizer
    tokens_length = tokenizer.model_max_length

    tokens = tokenizer(
        prompts[0],
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt',
        return_overflowing_tokens=True,
    )
    stokens = tokenizer(
        sprompts[0],
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt',
        return_overflowing_tokens=True,
    )

    p_ids = tokens.input_ids.numpy()[0]
    sp_ids = stokens.input_ids.numpy()[0]

    mask = torch.zeros(tokens_length)
    indices_target = torch.arange(tokens_length, dtype=torch.long)
    indices = torch.zeros(tokens_length, dtype=torch.long)

    for name, a0, a1, b0, b1 in SequenceMatcher(None, sp_ids, p_ids).get_opcodes():
        if b0 < tokens_length:
            if name == 'equal' or (name == 'replace' and a1 - a0 == b1 - b0):
                mask[b0:b1] = 1
                indices[b0:b1] = indices_target[a0:a1]

    mask = mask.to(device)
    indices = indices.to(device)
    indices_target = indices_target.to(device)

    return [mask, indices, indices_target]


def load_model_from_config(config, ckpt, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert('RGB')
    w, h = image.size
    print(f'loaded input image of size ({w}, {h}) from {path}')
    w, h = (
        512,
        512,
    )  # map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


""" sw 추가 """


def load_mask(path):
    image = Image.open(path).convert('RGB')
    w, h = image.size
    print(f'loaded input mask of size ({w}, {h}) from {path}')
    w, h = (
        512 // 8,
        512 // 8,
    )  # map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.NEAREST)

    image = ((np.array(image) > 0.5) * 255).astype(np.uint8)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image   # don't do normalization


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--prompt',
        type=str,
        nargs='?',
        default='a painting of a virus monster playing guitar',
        help='the prompt to render',
    )

    parser.add_argument(
        '--init-img', type=str, default=None, nargs='?', help='path to the input image'
    )

    parser.add_argument(
        '--outdir',
        type=str,
        nargs='?',
        help='dir to write results to',
        default='outputs/swap-samples',
    )

    parser.add_argument(
        '--skip_grid',
        action='store_true',
        help='do not save a grid, only individual samples. Helpful when evaluating lots of samples',
    )

    parser.add_argument(
        '--skip_save',
        action='store_true',
        help='do not save indiviual samples. For speed measurements.',
    )

    parser.add_argument(
        '--ddim_steps',
        type=int,
        default=50,
        help='number of ddim sampling steps',
    )

    parser.add_argument(
        '--plms',
        action='store_true',
        help='use plms sampling',
    )
    parser.add_argument(
        '--fixed_code',
        action='store_true',
        help='if enabled, uses the same starting code across all samples ',
    )

    parser.add_argument(
        '--ddim_eta',
        type=float,
        default=0.0,
        help='ddim eta (eta=0.0 corresponds to deterministic sampling',
    )
    parser.add_argument(
        '--n_iter',
        type=int,
        default=1,
        help='sample this often',
    )
    parser.add_argument(
        '--C',
        type=int,
        default=4,
        help='latent channels',
    )
    parser.add_argument(
        '--f',
        type=int,
        default=8,
        help='downsampling factor, most often 8 or 16',
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=2,
        help='how many samples to produce for each given prompt. A.k.a batch size',
    )
    parser.add_argument(
        '--n_rows',
        type=int,
        default=0,
        help='rows in the grid (default: n_samples)',
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=7.5,
        help='unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))',
    )

    parser.add_argument(
        '--strength',
        type=float,
        default=0.75,
        help='strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image',
    )
    parser.add_argument(
        '--from-file',
        type=str,
        help='if specified, load prompts from this file',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/stable-diffusion/v1-inference.yaml',
        help='path to config which constructs model',
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='models/ldm/stable-diffusion-v1/model.ckpt',
        help='path to checkpoint of model',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='the seed (for reproducible sampling)',
    )
    parser.add_argument(
        '--precision',
        type=str,
        help='evaluate at this precision',
        choices=['full', 'autocast'],
        default='autocast',
    )

    """ sw 추가 argument """
    parser.add_argument(
        '--init-mask', type=str, default=None, nargs='?', help='path to the input mask'
    )

    parser.add_argument(
        '--is_get_attn',
        action='store_true',
        help='get attention map?',
    )

    parser.add_argument('--save_attn_dir', type=str, default='./save_attention_map_dir')

    parser.add_argument(
        '--use_mask',
        action='store_true',
        help='use mask?',
    )

    parser.add_argument(
        '--sprompt',
        type=str,
        default=None,
        help='prompt of source image',
    )

    parser.add_argument(
        '--is-swap',
        action='store_true',
        help='use attention map swapping?',
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f'{opt.config}')
    """ sw 추가한 부분, attention 저장여부, 경로를 argument command로 다루기 위해서. 
        configs/stable-diffusion.
    """
    config['model']['params']['unet_config']['params']['is_get_attn'] = opt.is_get_attn
    config['model']['params']['unet_config']['params'][
        'save_attn_dir'
    ] = opt.save_attn_dir
    # config["model"]["params"]["unet_config"]["params"]["is_swap"]=opt.is_swap

    model = load_model_from_config(config, f'{opt.ckpt}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    # for name, module in model.model.diffusion_model.named_modules():
    #     module_name = type(module).__name__
    #     if module_name == "CrossAttention" and "attn2" in name:
    #         import pdb; pdb.set_trace()
    #         module.use_last_attn_slice = use
    # #import pdb; pdb.set_trace()

    if opt.plms:
        raise NotImplementedError('PLMS sampler not (yet) supported')
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f'reading prompts from {opt.from_file}')
        with open(opt.from_file, 'r') as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # assert os.path.isfile(opt.init_img)

    """ 1. (optional) load image and mask """
    if opt.init_img is not None:
        init_image = load_img(opt.init_img).to(device)  # torch.Size([1, 3, 512, 512])
        init_image = repeat(
            init_image, '1 ... -> b ...', b=batch_size
        )   # torch.Size([2, 3, 512, 512])

    if opt.use_mask and opt.init_mask is not None:
        init_mask = load_mask(opt.init_mask).to(device)   # torch.Size([1, 3, 512, 512])
        init_mask = repeat(
            init_mask, '1 ... -> b ...', b=batch_size
        )   # torch.Size([2, 3, 512, 512])

    """
    model.encode_first_stage(init_image).__class__: <class 'ldm.modules.distributions.distributions.DiagonalGaussianDistribution'>
    model.encode_first_stage(init_image).logvar.shape: torch.Size([2, 4, 64, 64])
    init_latent.shape = torch.Size([2, 4, 64, 64])
    """

    """ mazke z_init """
    if opt.init_img is not None:
        init_latent = model.get_first_stage_encoding(
            model.encode_first_stage(init_image)
        )  # move to latent space
        init_latent_bg = init_latent
    else:
        init_latent = None
        init_latent_bg = None

    sampler.make_schedule(
        ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False
    )

    # assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = opt.ddim_steps  # int(opt.strength * opt.ddim_steps) # t_enc=40, 0.8*50
    print(f'target t_enc is {t_enc} steps')

    precision_scope = (
        autocast if opt.precision == 'autocast' else nullcontext
    )   # default = "autocast"
    with torch.no_grad():
        with precision_scope('cuda'):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                sall_samples = list()
                for n in trange(opt.n_iter, desc='Sampling'):
                    for prompts in tqdm(data, desc='data'):
                        uc = None
                        if opt.scale != 1.0:   # default  = 5.0
                            uc = model.get_learned_conditioning(
                                batch_size * ['']
                            )   # uc.shape (2, 77, 768)
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        c = model.get_learned_conditioning(prompts)

                        if opt.is_swap:
                            sprompts = batch_size * [opt.sprompt]
                            sc = model.get_learned_conditioning(sprompts)
                        else:
                            sc = None

                        # encode (scaled latent)
                        # z_enc.shape=init_latent.shape: torch.Size([2, 4, 64, 64]) , imgsize=512 기준.
                        """ 1. z_t """
                        if init_latent is not None:
                            z_enc = sampler.stochastic_encode(
                                init_latent, torch.tensor([t_enc] * batch_size).to(device)
                            )
                        else:
                            if opt.fixed_code:
                                start_code = (
                                    torch.Tensor(np.load('./z_fixed.npy'))
                                    .float()
                                    .to(device=device)
                                )
                                z_enc = start_code[0:1].repeat(3, 1, 1, 1)
                                sz_enc = start_code[0:1].repeat(3, 1, 1, 1)
                            else:
                                z_enc = torch.randn((batch_size, 4, 64, 64)).to(device)
                                sz_enc = (
                                    torch.empty_like(z_enc).copy_(z_enc).to(device)
                                )   # torch.Tensor(z_enc).to(device)
                        # decode it
                        # sample.shape: (2,4,64,64) , imgsize=512 기준.
                        """
                        decode -> p_sample_ddim 
                        sample -> ddim_sampling -> p_sample_ddim
                        """

                        if not opt.is_swap:
                            if not opt.use_mask:
                                samples = sampler.decode(
                                    z_enc,
                                    c,
                                    t_enc,
                                    unconditional_guidance_scale=opt.scale,
                                    unconditional_conditioning=uc,
                                )
                            else:
                                samples = sampler.decode(
                                    z_enc,
                                    c,
                                    t_enc,
                                    x0=init_latent_bg,
                                    mask=init_mask,
                                    unconditional_guidance_scale=opt.scale,
                                    unconditional_conditioning=uc,
                                )
                        else:
                            pmask = get_indice(model, prompts, sprompts)   # .to(device)
                            if not opt.use_mask:
                                samples, s_samples = sampler.decode(
                                    z_enc,
                                    c,
                                    t_enc,
                                    sx_latent=sz_enc,
                                    scond=sc,
                                    pmask=pmask,
                                    unconditional_guidance_scale=opt.scale,
                                    unconditional_conditioning=uc,
                                )
                            else:
                                samples, s_samples = sampler.decode(
                                    z_enc,
                                    c,
                                    t_enc,
                                    sx_latent=sz_enc,
                                    scond=sc,
                                    pmask=pmask,
                                    x0=init_latent_bg,
                                    mask=init_mask,
                                    unconditional_guidance_scale=opt.scale,
                                    unconditional_conditioning=uc,
                                )

                        x_samples = model.decode_first_stage(
                            samples
                        )   # (2,3,512,512), imgsize=256 기준., # swap된 sample.
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if opt.is_swap:
                            sx_samples = model.decode_first_stage(
                                s_samples
                            )   # (2,3,512,512), imgsize=256 기준.
                            sx_samples = torch.clamp(
                                (sx_samples + 1.0) / 2.0, min=0.0, max=1.0
                            )

                        if not opt.skip_save:
                            """save samples of target prompt"""
                            for x_sample in x_samples:
                                x_sample = 255.0 * rearrange(
                                    x_sample.cpu().numpy(), 'c h w -> h w c'
                                )
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f'{base_count:05}.png')
                                )
                                base_count += 1

                            if opt.is_swap:
                                """save samples of source prompt"""
                                for sx_sample in sx_samples:
                                    sx_sample = 255.0 * rearrange(
                                        sx_sample.cpu().numpy(), 'c h w -> h w c'
                                    )
                                    Image.fromarray(sx_sample.astype(np.uint8)).save(
                                        os.path.join(sample_path, f'{base_count:05}.png')
                                    )
                        all_samples.append(x_samples)
                        if opt.is_swap:
                            sall_samples.append(sx_samples)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255.0 * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    Image.fromarray(grid.astype(np.uint8)).save(
                        os.path.join(outpath, prompts[0].replace(' ', '_') + '_tar.png')
                    )

                    if opt.is_swap:
                        sgrid = torch.stack(sall_samples, 0)
                        sgrid = rearrange(sgrid, 'n b c h w -> (n b) c h w')
                        sgrid = make_grid(sgrid, nrow=n_rows)

                        # to image
                        sgrid = 255.0 * rearrange(sgrid, 'c h w -> h w c').cpu().numpy()
                        # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                        Image.fromarray(sgrid.astype(np.uint8)).save(
                            os.path.join(
                                outpath, sprompts[0].replace(' ', '_') + '_src.png'
                            )
                        )

                    grid_count += 1

                toc = time.time()

    print(f'Your samples are ready and waiting for you here: \n{outpath} \n' f' \nEnjoy.')


if __name__ == '__main__':
    main()
