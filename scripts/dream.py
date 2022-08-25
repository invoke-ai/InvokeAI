#!/usr/bin/env python3
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

import argparse
import shlex
import atexit
import os
import sys
import copy
from PIL import Image, PngImagePlugin

skip_load_model = False
t2i = None
log_path = ""

# check if readline is available
try:
    import readline
    readline_available = True
except ModuleNotFoundError:
    readline_available = False


def init() -> None:
    print("Setup...")

    # command line history will be stored in "~/.dream_history"
    if readline_available:
        init_readline()

    #sys.path.append('.')

    from pytorch_lightning import logging
    from ldm.simplet2i import T2I

    # prevent warning message on frozen clip tokenizer
    import transformers
    transformers.logging.set_verbosity_error()

    argv_opts = parse_argv()

    if argv_opts.laion400m:
        # defaults for older latent diffusion weights
        width = 256
        height = 256
        config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        weights = "models/ldm/text2img-large/model.ckpt"
    else:
        # defaults for stable diffusion
        width = 512
        height = 512
        config = "configs/stable-diffusion/v1-inference.yaml"
        weights = "models/ldm/stable-diffusion-v1/model.ckpt"

    # create text2image object with default parameters
    # overridden in the user input loop
    global t2i
    t2i = T2I(
        height=height,
        batch_size=argv_opts.batch_size,
        outdir=argv_opts.outdir,
        sampler_name=argv_opts.sampler_name,
        weights=weights,
        full_precision=argv_opts.full_precision,
        config=config,
        latent_diffusion_weights=argv_opts.laion400m,
        embedding_path=argv_opts.embedding_path,
        device=argv_opts.device,
    )

    # set up logging
    global log_path
    log_path = os.path.join(argv_opts.outdir, "dream_log.txt")

    # ensure output directory
    if not os.path.exists(argv_opts.outdir):
        os.makedirs(argv_opts.outdir)

    # supress random seed message
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # load infile
    infile_lines: list = None
    if argv_opts.infile:
        with open(argv_opts.infile, "r") as file:
            infile = file.read()
        infile = infile.split('\n')

    # preload model
    try:
        if skip_load_model:
            print("######################")
            print("# Model not loaded!! #")
            print("######################")
        else:
            t2i.load_model()
    except FileNotFoundError:
        print(f"Cannot find weights at {weights}")
        sys.exit(1)

    print("Initialisation complete.")
    print("('help' for help, 'q' to quit, 'cd' to change output dir, 'pwd' to print output dir)")

    cmd_parser = parse_cmd()

    #with open(log_path, 'a') as log:
    # start user loop
    done = False
    while not done:
        try:
            done = user_loop(cmd_parser, infile_lines)
        except KeyboardInterrupt:
            print("Task cancelled. Enter q if you want to quit.")
        except EOFError:
            return True


def user_loop(
    cmd_parser: argparse.ArgumentParser(),
    infile_lines: list,
) -> bool:
    # read command from interactive cli
    if infile_lines is None:
        command = input("dream> ")

    # read command from infile
    else:
        try:
            command = infile_lines.pop(0)
        # infile_lines is empty; terminate
        except IndexError:
            return True

        # skip empty lines
        if not command.strip():
            return False

    # skip if command is a comment
    if command.startswith(("#", "//")):
        return False

    # escape single quotes
    command = command.replace("'", "\\'")

    try:
        elements = shlex.split(command)
    except ValueError as e:
        print(e)
        return False

    # skip if elements are empty
    if not elements:
        return False

    # quit if first element is 'q'
    if elements[0] == "q":
        return True

    # change output directory
    if elements[0] == "cd":
        new_dir = change_dir(elements)

        # only set dir if not empty
        if new_dir:
            t2i.outdir = new_dir

        del new_dir
        return False

    # print output directory
    if elements[0] == "pwd":
        print(f"Current output directory: {t2i.outdir}")
        return False

    # show help
    if elements[0] == "help":
        cmd_parser.print_help()
        return False

    # remove '!dream' command
    if elements[0] == "!dream":
        elements.pop(0)

    # seperate prompt from dash arguments
    args = []
    args_set = False

    for idx, element in enumerate(elements):
        if element[0] == "-":
            # element is a dash argument
            args.append(" ".join(elements[:idx]))
            args += elements[idx:]
            args_set = True
            break

    if not args_set:
        args.append(" ".join(elements))

    try:
        cmd_opts = cmd_parser.parse_args(args)
    except SystemExit:
        return False

    if not cmd_opts.prompt:
        print("Prompt required.")
        return False

    generate(cmd_opts)

    return False


def generate(cmd_opts: argparse.Namespace) -> None:
    results = []

    # cmd_opts that are not to be given to t2i.txt2img and t2i.img2img
    invalid_keys = ("repeats", "feedback")

    for r in range(cmd_opts.repeats + 1):
        t2i_args = eval_params(copy.deepcopy(vars(cmd_opts)), r)

        # in feedback mode: replaces the init_img with the first image from the last result
        if cmd_opts.feedback and results:
            t2i_args = {**t2i_args, "init_img": results[-1][0][0]}

        # applies invalid keys
        t2i_args = {k: v for k, v in t2i_args.items() if k not in invalid_keys}

        try:
            if not cmd_opts.init_img:
                results.append([img + [t2i_args] for img in t2i.txt2img(**t2i_args)])
            else:
                assert os.path.exists(cmd_opts.init_img), f"No file found at {cmd_opts.init_img}. On Linux systems, pressing <tab> after -I will autocomplete a list of possible image files."

                if cmd_opts.width or cmd_opts.height:
                    print("Warning: width and height options are ignored when modifying an init image")

                results.append([img + [t2i_args] for img in t2i.img2img(**t2i_args)])

        except AssertionError as e:
            print(e)
            return

    write_log(cmd_opts, results)


def write_log(cmd_opts: argparse.Namespace, results: list) -> None:
    log_message = []

    last_seed = None
    img_num = 1
    batch_size = cmd_opts.batch_size or t2i.batch_size
    seen = []

    seeds = [img[1] for result in results for img in result]
    if batch_size > 1:
        seeds = f"(seeds for each batch row: {seeds})"
    else:
        seeds = f"(seeds for individual images: {seeds})"

    for repeat in results:
        for result in repeat:
            seed = result[1]
            prompt_str = normalise_args(result[2])
            log_message.append(f"# {result[0]}: {prompt_str} -S {seed}")

            if batch_size > 1:
                if seed != lasts_seed:
                    img_num = 1
                else:
                    img_num += 1

                log_message[-1] += f" # (batch image {img_num} of {batch_size})"
                last_seed = seed

            print(log_message[-1])
            log_message[-1] += "\n"

            if result[0] not in seen:
                seen.append(result[0])

                try:
                    if cmd_opts.grid:
                        write_prompt_to_png(result[0], f"{prompt_str} -g -S {seed} {seeds}")
                    else:
                        write_prompt_to_png(result[0], f"{prompt_str} -S {seed}")
                except FileNotFoundError:
                    print(f"Could not open file '{result[0]}' for reading.")

    log_message = [normalise_args(vars(cmd_opts)) + "\n"] + log_message
    print("Prompt:", log_message[0], end="")

    with open(log_path, "a") as file:
        file.writelines(log_message)


def normalise_args(opts: dict) -> str:
    args = []

    args.append(f"\"{opts.get('prompt')}\"")
    args.append(f"-s {opts.get('steps') or t2i.steps}")
    args.append(f"-b {opts.get('batch_size') or t2i.batch_size}")
    args.append(f"-W {opts.get('width') or t2i.width}")
    args.append(f"-H {opts.get('height') or t2i.height}")
    args.append(f"-C {opts.get('cfg_scale') or t2i.cfg_scale}")
    #args.append(f"-m {t2i.sampler_name}")
    opts.get("repeats") and args.append(f"-r {opts.get('repeats')}")
    opts.get("feedback") and args.append("-B")
    opts.get("init_img") and args.append(f"-I {opts.get('init_img')}")
    opts.get("strength") and opts.get("init_img") and args.append(f"-f {opts.get('strength')}")
    #t2i.full_precision and args.append("-F")

    return " ".join(args)


def write_prompt_to_png(path, prompt) -> None:
    info = PngImagePlugin.PngInfo()
    info.add_text("Dream", prompt)

    with Image.open(path) as img:
        img.save(path, "PNG", pnginfo=info)


def eval_params(t2i_args: vars, r: int) -> vars:
    params = ("steps", "seed", "width", "height", "cfg_scale", "strength")
    floats = ("cfg_scale", "strength")

    for p in params:
        if not t2i_args[p]:
            continue

        convert = float if p in floats else int
        split = t2i_args[p].split(":")
        if len(split) > 1:
            t2i_args[p] = convert(split[0]) + r * convert(split[1])
        else:
            t2i_args[p] = convert(split[0])

    return t2i_args


def parse_argv() -> argparse.Namespace():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg("-l", "--laion400m", "--latent_diffusion",
            dest="laion400m",
            action="store_true",
            help="fallback to the latent diffusion (laion400m) weights and config")
    add_arg("--from_file",
            dest="infile",
            type=str,
            help="if specified, load prompts from this file")
    add_arg("-n", "--iterations",
            type=int,
            default=1,
            help="number of images to generate")
    add_arg("-F", "--full_precision",
            dest="full_precision",
            action="store_true",
            help="use slower full precision math for calculations")
    add_arg("-b", "--batch_size",
            type=int,
            default=1,
            help="number of images to produce per iteration (faster, but doesn't generate individual seeds")
    add_arg("--sampler", "-m",
            dest="sampler_name",
            choices=["ddim", "k_dpm_2_a", "k_dpm_2", "k_euler_a", "k_euler", "k_heun", "k_lms", "plms"],
            default="k_lms",
            help="which sampler to use (k_lms) - can only be set on command line")
    add_arg("-o", "--outdir",
            type=str,
            default="outputs/img-samples",
            help="directory in which to place generated images and a log of prompts and seeds")
    add_arg("--embedding_path",
            type=str,
            help="Path to a pre-trained embedding manager checkpoint - can only be set on command line")
    add_arg("-d", "--device",
           type=str,
           default="cuda",
           help="device to run stable diffusion on. defaults to cuda `torch.cuda.current_device()` if avalible")

    return parser.parse_args()


def parse_cmd() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg("prompt")
    add_arg("-s", "--steps", type=str,
            help="number of steps")
    add_arg("-S", "--seed", type=str,
            help="image seed")
    add_arg("-n", "--iterations", type=int, default=1,
            help="number of samplings to perform (slower than batches, but will provide seeds for individual images)")
    add_arg("-b", "--batch_size", type=int, default=1,
            help="number of images to produce per sampling (will not provide seeds for individual images!)")
    add_arg("-W", "--width", type=str,
            help="image width, must be a multiple of 64")
    add_arg("-H", "--height", type=str,
            help="image height, must be a multiple of 64")
    add_arg("-C", "--cfg_scale", type=str, default="7",
            help="prompt configuration scale")
    add_arg("-g", "--grid", action="store_true",
            help="generate a grid")
    add_arg("-i", "--individual", action="store_true",
            help="generate individual files (default)")
    add_arg("-I", "--init_img", type=str,
            help="path to input image for img2img mode (supersedes width and height)")
    add_arg("-f", "--strength", type=str, default="0.75",
            help="strength for noising/unnoising. 0.0 preserves image exactly, 1.0 replaces it completely")
    add_arg("-r", "--repeats", type=int, default=0,
            help="number of times values are incremented")
    add_arg("-B", "--feedback", action="store_true",
            help="feeds the first generated image back into the next one as an init_img")
    add_arg("-x", "--skip_normalize", action="store_true",
            help="skip subprompt weight normalization")

    return parser


def change_dir(elements) -> str:
    if len(elements) == 2:
        d = elements[1]
        if os.path.exists(d):
            return d
        print(f"Directory '{d} does not exist.'")
    else:
        print("Invalid number of arguments. Usage: cd <path>")

if readline_available:
    def init_readline() -> None:
        readline.set_completer(Completer(
            ["cd", "pwd", "help", "q"]
        ).complete)
        readline.set_completer_delims(" ")
        readline.parse_and_bind("tab: complete")
        load_history()

    def load_history() -> None:
        histfile = os.path.join(os.path.expanduser("~"), ".dream_history")

        try:
            readline.read_history_file(histfile)
            readline.set_history_length(1000)
        except FileNotFoundError:
            pass

        atexit.register(readline.write_history_file, histfile)

    class Completer():
        def __init__(self, options: list):
            self.options = options + list(parse_cmd()._option_string_actions.keys())

        def complete(self, text, state):
            buffer = readline.get_line_buffer()

            if text.startswith(("-I", "--init_img")):
                return self._path_completions(text, state, (".png"))

            if buffer.strip().endswith("cd") or text.startswith((".", "/")):
                return self._path_completions(text, state, ())

            response = None

            if state == 0:
                # This is the first time for this text, so build a match list.
                if text:
                    self.matches = [
                        s for s in self.options if s and s.startswith(text)
                    ]
                else:
                    self.matches = self.options[:]

            # Return state'th item from the match list, if we have that many.
            try:
                response = self.matches[state]
            except IndexError:
                response = None

            return response

        def _path_completions(self, text, state, extensions):
            # get the path so far
            if text.startswith("-I"):
                path = text.replace("-I", "", 1).lstrip()
            elif text.startswith("--init_img="):
                path = text.replace("--init_img=", "", 1).lstrip()
            else:
                path = text

            matches = []
            path = os.path.expanduser(path)

            if not len(path):
                matches.append(text + "./")
            else:
                directory = os.path.dirname(path)
                dir_list = os.listdri(directory)

                for n in dir_list:
                    if n.startswith(".") and len(n):
                        continue

                    full_path = os.path.join(directory, n)

                    if full_path.startswith(path):
                        if os.path.isdir(full_path):
                            matches.append(os.path.join(os.path.dirname(text), n) + "/")
                        elif n.endswith(extensions):
                            matches.append(os.path.join(os.path.dirname(text), n))

            try:
                response = matches[state]
            except IndexError:
                response = None

            return response


if __name__ == "__main__":
    init()

