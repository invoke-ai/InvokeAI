# Copyright 2023 Lincoln D. Stein and the InvokeAI Team

import argparse
import shlex
from argparse import ArgumentParser

# note that this includes both old sampler names and new scheduler names
# in order to be able to parse both 2.0 and 3.0-pre-nodes versions of invokeai.init
SAMPLER_CHOICES = [
    "ddim",
    "ddpm",
    "deis",
    "lms",
    "lms_k",
    "pndm",
    "heun",
    "heun_k",
    "euler",
    "euler_k",
    "euler_a",
    "kdpm_2",
    "kdpm_2_a",
    "dpmpp_2s",
    "dpmpp_2s_k",
    "dpmpp_2m",
    "dpmpp_2m_k",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_k",
    "dpmpp_sde",
    "dpmpp_sde_k",
    "unipc",
    "k_dpm_2_a",
    "k_dpm_2",
    "k_dpmpp_2_a",
    "k_dpmpp_2",
    "k_euler_a",
    "k_euler",
    "k_heun",
    "k_lms",
    "plms",
]

PRECISION_CHOICES = [
    "auto",
    "float32",
    "autocast",
    "float16",
]


class FileArgumentParser(ArgumentParser):
    """
    Supports reading defaults from an init file.
    """

    def convert_arg_line_to_args(self, arg_line):
        return shlex.split(arg_line, comments=True)


legacy_parser = FileArgumentParser(
    description="""
Generate images using Stable Diffusion.
    Use --web to launch the web interface.
    Use --from_file to load prompts from a file path or standard input ("-").
    Otherwise you will be dropped into an interactive command prompt (type -h for help.)
    Other command-line arguments are defaults that can usually be overridden
    prompt the command prompt.
    """,
    fromfile_prefix_chars="@",
)
general_group = legacy_parser.add_argument_group("General")
model_group = legacy_parser.add_argument_group("Model selection")
file_group = legacy_parser.add_argument_group("Input/output")
web_server_group = legacy_parser.add_argument_group("Web server")
render_group = legacy_parser.add_argument_group("Rendering")
postprocessing_group = legacy_parser.add_argument_group("Postprocessing")
deprecated_group = legacy_parser.add_argument_group("Deprecated options")

deprecated_group.add_argument("--laion400m")
deprecated_group.add_argument("--weights")  # deprecated
general_group.add_argument("--version", "-V", action="store_true", help="Print InvokeAI version number")
model_group.add_argument(
    "--root_dir",
    default=None,
    help='Path to directory containing "models", "outputs" and "configs". If not present will read from environment variable INVOKEAI_ROOT. Defaults to ~/invokeai.',
)
model_group.add_argument(
    "--config",
    "-c",
    "-config",
    dest="conf",
    default="./configs/models.yaml",
    help="Path to configuration file for alternate models.",
)
model_group.add_argument(
    "--model",
    help='Indicates which diffusion model to load (defaults to "default" stanza in configs/models.yaml)',
)
model_group.add_argument(
    "--weight_dirs",
    nargs="+",
    type=str,
    help="List of one or more directories that will be auto-scanned for new model weights to import",
)
model_group.add_argument(
    "--png_compression",
    "-z",
    type=int,
    default=6,
    choices=range(0, 9),
    dest="png_compression",
    help="level of PNG compression, from 0 (none) to 9 (maximum). Default is 6.",
)
model_group.add_argument(
    "-F",
    "--full_precision",
    dest="full_precision",
    action="store_true",
    help="Deprecated way to set --precision=float32",
)
model_group.add_argument(
    "--max_loaded_models",
    dest="max_loaded_models",
    type=int,
    default=2,
    help="Maximum number of models to keep in memory for fast switching, including the one in GPU",
)
model_group.add_argument(
    "--free_gpu_mem",
    dest="free_gpu_mem",
    action="store_true",
    help="Force free gpu memory before final decoding",
)
model_group.add_argument(
    "--sequential_guidance",
    dest="sequential_guidance",
    action="store_true",
    help="Calculate guidance in serial instead of in parallel, lowering memory requirement " "at the expense of speed",
)
model_group.add_argument(
    "--xformers",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable/disable xformers support (default enabled if installed)",
)
model_group.add_argument(
    "--always_use_cpu", dest="always_use_cpu", action="store_true", help="Force use of CPU even if GPU is available"
)
model_group.add_argument(
    "--precision",
    dest="precision",
    type=str,
    choices=PRECISION_CHOICES,
    metavar="PRECISION",
    help=f'Set model precision. Defaults to auto selected based on device. Options: {", ".join(PRECISION_CHOICES)}',
    default="auto",
)
model_group.add_argument(
    "--ckpt_convert",
    action=argparse.BooleanOptionalAction,
    dest="ckpt_convert",
    default=True,
    help="Deprecated option. Legacy ckpt files are now always converted to diffusers when loaded.",
)
model_group.add_argument(
    "--internet",
    action=argparse.BooleanOptionalAction,
    dest="internet_available",
    default=True,
    help="Indicate whether internet is available for just-in-time model downloading (default: probe automatically).",
)
model_group.add_argument(
    "--nsfw_checker",
    "--safety_checker",
    action=argparse.BooleanOptionalAction,
    dest="safety_checker",
    default=False,
    help="Check for and blur potentially NSFW images. Use --no-nsfw_checker to disable.",
)
model_group.add_argument(
    "--autoimport",
    default=None,
    type=str,
    help="Check the indicated directory for .ckpt/.safetensors weights files at startup and import directly",
)
model_group.add_argument(
    "--autoconvert",
    default=None,
    type=str,
    help="Check the indicated directory for .ckpt/.safetensors weights files at startup and import as optimized diffuser models",
)
model_group.add_argument(
    "--patchmatch",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Load the patchmatch extension for outpainting. Use --no-patchmatch to disable.",
)
file_group.add_argument(
    "--from_file",
    dest="infile",
    type=str,
    help="If specified, load prompts from this file",
)
file_group.add_argument(
    "--outdir",
    "-o",
    type=str,
    help="Directory to save generated images and a log of prompts and seeds. Default: ROOTDIR/outputs",
    default="outputs",
)
file_group.add_argument(
    "--prompt_as_dir",
    "-p",
    action="store_true",
    help="Place images in subdirectories named after the prompt.",
)
render_group.add_argument(
    "--fnformat",
    default="{prefix}.{seed}.png",
    type=str,
    help="Overwrite the filename format. You can use any argument as wildcard enclosed in curly braces. Default is {prefix}.{seed}.png",
)
render_group.add_argument("-s", "--steps", type=int, default=50, help="Number of steps")
render_group.add_argument(
    "-W",
    "--width",
    type=int,
    help="Image width, multiple of 64",
)
render_group.add_argument(
    "-H",
    "--height",
    type=int,
    help="Image height, multiple of 64",
)
render_group.add_argument(
    "-C",
    "--cfg_scale",
    default=7.5,
    type=float,
    help='Classifier free guidance (CFG) scale - higher numbers cause generator to "try" harder.',
)
render_group.add_argument(
    "--sampler",
    "-A",
    "-m",
    dest="sampler_name",
    type=str,
    choices=SAMPLER_CHOICES,
    metavar="SAMPLER_NAME",
    help=f'Set the default sampler. Supported samplers: {", ".join(SAMPLER_CHOICES)}',
    default="k_lms",
)
render_group.add_argument(
    "--log_tokenization", "-t", action="store_true", help="shows how the prompt is split into tokens"
)
render_group.add_argument(
    "-f",
    "--strength",
    type=float,
    help="img2img strength for noising/unnoising. 0.0 preserves image exactly, 1.0 replaces it completely",
)
render_group.add_argument(
    "-T",
    "-fit",
    "--fit",
    action=argparse.BooleanOptionalAction,
    help="If specified, will resize the input image to fit within the dimensions of width x height (512x512 default)",
)

render_group.add_argument("--grid", "-g", action=argparse.BooleanOptionalAction, help="generate a grid")
render_group.add_argument(
    "--embedding_directory",
    "--embedding_path",
    dest="embedding_path",
    default="embeddings",
    type=str,
    help="Path to a directory containing .bin and/or .pt files, or a single .bin/.pt file. You may use subdirectories. (default is ROOTDIR/embeddings)",
)
render_group.add_argument(
    "--lora_directory",
    dest="lora_path",
    default="loras",
    type=str,
    help="Path to a directory containing LoRA files; subdirectories are not supported. (default is ROOTDIR/loras)",
)
render_group.add_argument(
    "--embeddings",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable embedding directory (default). Use --no-embeddings to disable.",
)
render_group.add_argument("--enable_image_debugging", action="store_true", help="Generates debugging image to display")
render_group.add_argument(
    "--karras_max",
    type=int,
    default=None,
    help="control the point at which the K* samplers will shift from using the Karras noise schedule (good for low step counts) to the LatentDiffusion noise schedule (good for high step counts). Set to 0 to use LatentDiffusion for all step values, and to a high value (e.g. 1000) to use Karras for all step values. [29].",
)
# Restoration related args
postprocessing_group.add_argument(
    "--no_restore",
    dest="restore",
    action="store_false",
    help="Disable face restoration with GFPGAN or codeformer",
)
postprocessing_group.add_argument(
    "--no_upscale",
    dest="esrgan",
    action="store_false",
    help="Disable upscaling with ESRGAN",
)
postprocessing_group.add_argument(
    "--esrgan_bg_tile",
    type=int,
    default=400,
    help="Tile size for background sampler, 0 for no tile during testing. Default: 400.",
)
postprocessing_group.add_argument(
    "--esrgan_denoise_str",
    type=float,
    default=0.75,
    help="esrgan denoise str. 0 is no denoise, 1 is max denoise.  Default: 0.75",
)
postprocessing_group.add_argument(
    "--gfpgan_model_path",
    type=str,
    default="./models/gfpgan/GFPGANv1.4.pth",
    help="Indicates the path to the GFPGAN model",
)
web_server_group.add_argument(
    "--web",
    dest="web",
    action="store_true",
    help="Start in web server mode.",
)
web_server_group.add_argument(
    "--web_develop",
    dest="web_develop",
    action="store_true",
    help="Start in web server development mode.",
)
web_server_group.add_argument(
    "--web_verbose",
    action="store_true",
    help="Enables verbose logging",
)
web_server_group.add_argument(
    "--cors",
    nargs="*",
    type=str,
    help="Additional allowed origins, comma-separated",
)
web_server_group.add_argument(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Web server: Host or IP to listen on. Set to 0.0.0.0 to accept traffic from other devices on your network.",
)
web_server_group.add_argument("--port", type=int, default="9090", help="Web server: Port to listen on")
web_server_group.add_argument(
    "--certfile",
    type=str,
    default=None,
    help="Web server: Path to certificate file to use for SSL. Use together with --keyfile",
)
web_server_group.add_argument(
    "--keyfile",
    type=str,
    default=None,
    help="Web server: Path to private key file to use for SSL. Use together with --certfile",
)
web_server_group.add_argument(
    "--gui",
    dest="gui",
    action="store_true",
    help="Start InvokeAI GUI",
)
