import os
import re
import shlex
import sys
import traceback
from argparse import Namespace
from packaging import version
from pathlib import Path
from typing import Union

import click

from compel import PromptParser

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pyparsing  # type: ignore

import ldm.invoke

from ..generate import Generate
from .args import (Args, dream_cmd_from_png, metadata_dumps,
                             metadata_from_png)
from .generator.diffusers_pipeline import PipelineIntermediateState
from .globals import Globals, global_config_dir
from .image_util import make_grid
from .log import write_log
from .model_manager import ModelManager
from .pngwriter import PngWriter, retrieve_metadata, write_metadata
from .readline import Completer, get_completer
from ..util import url_attachment_name

# global used in multiple functions (fix)
infile = None

def main():
    """Initialize command-line parsers and the diffusion model"""
    global infile

    opt = Args()
    args = opt.parse_args()
    if not args:
        sys.exit(-1)

    if args.laion400m:
        print(
            "--laion400m flag has been deprecated. Please use --model laion400m instead."
        )
        sys.exit(-1)
    if args.weights:
        print(
            "--weights argument has been deprecated. Please edit ./configs/models.yaml, and select the weights using --model instead."
        )
        sys.exit(-1)
    if args.max_loaded_models is not None:
        if args.max_loaded_models <= 0:
            print("--max_loaded_models must be >= 1; using 1")
            args.max_loaded_models = 1

    # alert - setting a few globals here
    Globals.try_patchmatch = args.patchmatch
    Globals.always_use_cpu = args.always_use_cpu
    Globals.internet_available = args.internet_available and check_internet()
    Globals.disable_xformers = not args.xformers
    Globals.sequential_guidance = args.sequential_guidance
    Globals.ckpt_convert = args.ckpt_convert

    # run any post-install patches needed
    run_patches()

    print(f">> Internet connectivity is {Globals.internet_available}")

    if not args.conf:
        config_file = os.path.join(Globals.root, "configs", "models.yaml")
        if not os.path.exists(config_file):
            report_model_error(
                opt, FileNotFoundError(f"The file {config_file} could not be found.")
            )

    print(f">> {ldm.invoke.__app_name__}, version {ldm.invoke.__version__}")
    print(f'>> InvokeAI runtime directory is "{Globals.root}"')

    # loading here to avoid long delays on startup
    # these two lines prevent a horrible warning message from appearing
    # when the frozen CLIP tokenizer is imported
    import transformers  # type: ignore

    from ldm.generate import Generate

    transformers.logging.set_verbosity_error()
    import diffusers

    diffusers.logging.set_verbosity_error()

    # Loading Face Restoration and ESRGAN Modules
    gfpgan, codeformer, esrgan = load_face_restoration(opt)

    # normalize the config directory relative to root
    if not os.path.isabs(opt.conf):
        opt.conf = os.path.normpath(os.path.join(Globals.root, opt.conf))

    if opt.embeddings:
        if not os.path.isabs(opt.embedding_path):
            embedding_path = os.path.normpath(
                os.path.join(Globals.root, opt.embedding_path)
            )
        else:
            embedding_path = opt.embedding_path
    else:
        embedding_path = None

    # migrate legacy models
    ModelManager.migrate_models()

    # load the infile as a list of lines
    if opt.infile:
        try:
            if os.path.isfile(opt.infile):
                infile = open(opt.infile, "r", encoding="utf-8")
            elif opt.infile == "-":  # stdin
                infile = sys.stdin
            else:
                raise FileNotFoundError(f"{opt.infile} not found.")
        except (FileNotFoundError, IOError) as e:
            print(f"{e}. Aborting.")
            sys.exit(-1)

    model = opt.model or retrieve_last_used_model()

    # creating a Generate object:
    try:
        gen = Generate(
            conf=opt.conf,
            model=model,
            sampler_name=opt.sampler_name,
            embedding_path=embedding_path,
            full_precision=opt.full_precision,
            precision=opt.precision,
            gfpgan=gfpgan,
            codeformer=codeformer,
            esrgan=esrgan,
            free_gpu_mem=opt.free_gpu_mem,
            safety_checker=opt.safety_checker,
            max_loaded_models=opt.max_loaded_models,
        )
    except (FileNotFoundError, TypeError, AssertionError) as e:
        report_model_error(opt, e)
    except (IOError, KeyError) as e:
        print(f"{e}. Aborting.")
        sys.exit(-1)

    if opt.seamless:
        print(">> changed to seamless tiling mode")

    # preload the model
    try:
        gen.load_model()
    except KeyError:
        pass
    except Exception as e:
        report_model_error(opt, e)

    # completer is the readline object
    completer = get_completer(opt, models=gen.model_manager.list_models())
            
    # try to autoconvert new models
    if path := opt.autoimport:
        gen.model_manager.heuristic_import(
            str(path),
            convert=False,
            commit_to_conf=opt.conf,
            config_file_callback=lambda x: _pick_configuration_file(completer,x),
        )

    if path := opt.autoconvert:
        gen.model_manager.heuristic_import(
            str(path), convert=True, commit_to_conf=opt.conf
        )

    # web server loops forever
    if opt.web or opt.gui:
        invoke_ai_web_server_loop(gen, gfpgan, codeformer, esrgan)
        save_last_used_model(gen.model_name)
        sys.exit(0)

    if not infile:
        print(
            "\n* Initialization done! Awaiting your command (-h for help, 'q' to quit)"
        )

    try:
        main_loop(gen, opt, completer)
    except KeyboardInterrupt:
        print(
            f'\nGoodbye!\nYou can start InvokeAI again by running the "invoke.bat" (or "invoke.sh") script from {Globals.root}'
        )
    except Exception:
        print(">> An error occurred:")
        traceback.print_exc()


# TODO: main_loop() has gotten busy. Needs to be refactored.
def main_loop(gen, opt, completer):
    """prompt/read/execute loop"""
    global infile
    done = False
    doneAfterInFile = infile is not None
    path_filter = re.compile(r'[<>:"/\\|?*]')
    last_results = list()

    # The readline completer reads history from the .dream_history file located in the
    # output directory specified at the time of script launch. We do not currently support
    # changing the history file midstream when the output directory is changed.
    set_default_output_dir(opt, completer)
    if gen.model:
        add_embedding_terms(gen, completer)
    output_cntr = completer.get_current_history_length() + 1

    # os.pathconf is not available on Windows
    if hasattr(os, "pathconf"):
        path_max = os.pathconf(opt.outdir, "PC_PATH_MAX")
        name_max = os.pathconf(opt.outdir, "PC_NAME_MAX")
    else:
        path_max = 260
        name_max = 255

    while not done:
        operation = "generate"

        try:
            command = get_next_command(infile, gen.model_name)
        except EOFError:
            done = infile is None or doneAfterInFile
            infile = None
            continue

        # skip empty lines
        if not command.strip():
            continue

        if command.startswith(("#", "//")):
            continue

        if len(command.strip()) == 1 and command.startswith("q"):
            done = True
            break

        if not command.startswith("!history"):
            completer.add_history(command)

        if command.startswith("!"):
            command, operation = do_command(command, gen, opt, completer)

        if operation is None:
            continue

        if opt.parse_cmd(command) is None:
            continue

        if opt.init_img:
            try:
                if not opt.prompt:
                    oldargs = metadata_from_png(opt.init_img)
                    opt.prompt = oldargs.prompt
                    print(f'>> Retrieved old prompt "{opt.prompt}" from {opt.init_img}')
            except (OSError, AttributeError, KeyError):
                pass

        if len(opt.prompt) == 0:
            opt.prompt = ""

        # width and height are set by model if not specified
        if not opt.width:
            opt.width = gen.width
        if not opt.height:
            opt.height = gen.height

        # retrieve previous value of init image if requested
        if opt.init_img is not None and re.match("^-\\d+$", opt.init_img):
            try:
                opt.init_img = last_results[int(opt.init_img)][0]
                print(f">> Reusing previous image {opt.init_img}")
            except IndexError:
                print(f">> No previous initial image at position {opt.init_img} found")
                opt.init_img = None
                continue

        # the outdir can change with each command, so we adjust it here
        set_default_output_dir(opt, completer)

        # try to relativize pathnames
        for attr in ("init_img", "init_mask", "init_color"):
            if getattr(opt, attr) and not os.path.exists(getattr(opt, attr)):
                basename = getattr(opt, attr)
                path = os.path.join(opt.outdir, basename)
                setattr(opt, attr, path)

        # retrieve previous value of seed if requested
        # Exception: for postprocess operations negative seed values
        # mean "discard the original seed and generate a new one"
        # (this is a non-obvious hack and needs to be reworked)
        if opt.seed is not None and opt.seed < 0 and operation != "postprocess":
            try:
                opt.seed = last_results[opt.seed][1]
                print(f">> Reusing previous seed {opt.seed}")
            except IndexError:
                print(f">> No previous seed at position {opt.seed} found")
                opt.seed = None
                continue

        if opt.strength is None:
            opt.strength = 0.75 if opt.out_direction is None else 0.83

        if opt.with_variations is not None:
            opt.with_variations = split_variations(opt.with_variations)

        if opt.prompt_as_dir and operation == "generate":
            # sanitize the prompt to a valid folder name
            subdir = path_filter.sub("_", opt.prompt)[:name_max].rstrip(" .")

            # truncate path to maximum allowed length
            # 39 is the length of '######.##########.##########-##.png', plus two separators and a NUL
            subdir = subdir[: (path_max - 39 - len(os.path.abspath(opt.outdir)))]
            current_outdir = os.path.join(opt.outdir, subdir)

            print('Writing files to directory: "' + current_outdir + '"')

            # make sure the output directory exists
            if not os.path.exists(current_outdir):
                os.makedirs(current_outdir)
        else:
            if not os.path.exists(opt.outdir):
                os.makedirs(opt.outdir)
            current_outdir = opt.outdir

        # Here is where the images are actually generated!
        last_results = []
        try:
            file_writer = PngWriter(current_outdir)
            results = []  # list of filename, prompt pairs
            grid_images = dict()  # seed -> Image, only used if `opt.grid`
            prior_variations = opt.with_variations or []
            prefix = file_writer.unique_prefix()
            step_callback = (
                make_step_callback(gen, opt, prefix)
                if opt.save_intermediates > 0
                else None
            )

            def image_writer(
                image,
                seed,
                upscaled=False,
                first_seed=None,
                use_prefix=None,
                prompt_in=None,
                attention_maps_image=None,
            ):
                # note the seed is the seed of the current image
                # the first_seed is the original seed that noise is added to
                # when the -v switch is used to generate variations
                nonlocal prior_variations
                nonlocal prefix

                path = None
                if opt.grid:
                    grid_images[seed] = image

                elif operation == "mask":
                    filename = f"{prefix}.{use_prefix}.{seed}.png"
                    tm = opt.text_mask[0]
                    th = opt.text_mask[1] if len(opt.text_mask) > 1 else 0.5
                    formatted_dream_prompt = (
                        f"!mask {opt.input_file_path} -tm {tm} {th}"
                    )
                    path = file_writer.save_image_and_prompt_to_png(
                        image=image,
                        dream_prompt=formatted_dream_prompt,
                        metadata={},
                        name=filename,
                        compress_level=opt.png_compression,
                    )
                    results.append([path, formatted_dream_prompt])

                else:
                    if use_prefix is not None:
                        prefix = use_prefix
                    postprocessed = upscaled if upscaled else operation == "postprocess"
                    opt.prompt = (
                        gen.huggingface_concepts_library.replace_triggers_with_concepts(
                            opt.prompt or prompt_in
                        )
                    )  # to avoid the problem of non-unique concept triggers
                    filename, formatted_dream_prompt = prepare_image_metadata(
                        opt,
                        prefix,
                        seed,
                        operation,
                        prior_variations,
                        postprocessed,
                        first_seed,
                        gen.model_name,
                    )
                    path = file_writer.save_image_and_prompt_to_png(
                        image=image,
                        dream_prompt=formatted_dream_prompt,
                        metadata=metadata_dumps(
                            opt,
                            seeds=[
                                seed
                                if opt.variation_amount == 0
                                and len(prior_variations) == 0
                                else first_seed
                            ],
                            model_hash=gen.model_hash,
                            model_id=gen.model_name,
                        ),
                        name=filename,
                        compress_level=opt.png_compression,
                    )

                    # update rfc metadata
                    if operation == "postprocess":
                        tool = re.match(
                            "postprocess:(\w+)", opt.last_operation
                        ).groups()[0]
                        add_postprocessing_to_metadata(
                            opt,
                            opt.input_file_path,
                            filename,
                            tool,
                            formatted_dream_prompt,
                        )

                    if (not postprocessed) or opt.save_original:
                        # only append to results if we didn't overwrite an earlier output
                        results.append([path, formatted_dream_prompt])

                # so that the seed autocompletes (on linux|mac when -S or --seed specified
                if completer and operation == "generate":
                    completer.add_seed(seed)
                    completer.add_seed(first_seed)
                last_results.append([path, seed])

            if operation == "generate":
                catch_ctrl_c = (
                    infile is None
                )  # if running interactively, we catch keyboard interrupts
                opt.last_operation = "generate"
                try:
                    gen.prompt2image(
                        image_callback=image_writer,
                        step_callback=step_callback,
                        catch_interrupts=catch_ctrl_c,
                        **vars(opt),
                    )
                except (PromptParser.ParsingException, pyparsing.ParseException) as e:
                    print("** An error occurred while processing your prompt **")
                    print(f"** {str(e)} **")
            elif operation == "postprocess":
                print(f">> fixing {opt.prompt}")
                opt.last_operation = do_postprocess(gen, opt, image_writer)

            elif operation == "mask":
                print(f">> generating masks from {opt.prompt}")
                do_textmask(gen, opt, image_writer)

            if opt.grid and len(grid_images) > 0:
                grid_img = make_grid(list(grid_images.values()))
                grid_seeds = list(grid_images.keys())
                first_seed = last_results[0][1]
                filename = f"{prefix}.{first_seed}.png"
                formatted_dream_prompt = opt.dream_prompt_str(
                    seed=first_seed, grid=True, iterations=len(grid_images)
                )
                formatted_dream_prompt += f" # {grid_seeds}"
                metadata = metadata_dumps(
                    opt, seeds=grid_seeds, model_hash=gen.model_hash
                )
                path = file_writer.save_image_and_prompt_to_png(
                    image=grid_img,
                    dream_prompt=formatted_dream_prompt,
                    metadata=metadata,
                    name=filename,
                )
                results = [[path, formatted_dream_prompt]]

        except AssertionError as e:
            print(e)
            continue

        except OSError as e:
            print(e)
            continue

        print("Outputs:")
        log_path = os.path.join(current_outdir, "invoke_log")
        output_cntr = write_log(results, log_path, ("txt", "md"), output_cntr)
        print()

    print(
        f'\nGoodbye!\nYou can start InvokeAI again by running the "invoke.bat" (or "invoke.sh") script from {Globals.root}'
    )
    save_last_used_model(gen.model_name)


# TO DO: remove repetitive code and the awkward command.replace() trope
# Just do a simple parse of the command!
def do_command(command: str, gen, opt: Args, completer) -> tuple:
    global infile
    operation = "generate"  # default operation, alternative is 'postprocess'
    command = command.replace('\\','/') # windows

    if command.startswith(
        "!dream"
    ):  # in case a stored prompt still contains the !dream command
        command = command.replace("!dream ", "", 1)

    elif command.startswith("!fix"):
        command = command.replace("!fix ", "", 1)
        operation = "postprocess"

    elif command.startswith("!mask"):
        command = command.replace("!mask ", "", 1)
        operation = "mask"

    elif command.startswith("!switch"):
        model_name = command.replace("!switch ", "", 1)
        try:
            gen.set_model(model_name)
            add_embedding_terms(gen, completer)
        except KeyError as e:
            print(str(e))
        except Exception as e:
            report_model_error(opt, e)
        completer.add_history(command)
        operation = None

    elif command.startswith("!models"):
        gen.model_manager.print_models()
        completer.add_history(command)
        operation = None

    elif command.startswith("!import"):
        path = shlex.split(command)
        if len(path) < 2:
            print(
                "** please provide (1) a URL to a .ckpt file to import; (2) a local path to a .ckpt file; or (3) a diffusers repository id in the form stabilityai/stable-diffusion-2-1"
            )
        else:
            try:
                import_model(path[1], gen, opt, completer)
                completer.add_history(command)
            except KeyboardInterrupt:
                print('\n')
        operation = None

    elif command.startswith(("!convert","!optimize")):
        path = shlex.split(command)
        if len(path) < 2:
            print("** please provide the path to a .ckpt or .safetensors model")
        else:
            try:
                convert_model(path[1], gen, opt, completer)
                completer.add_history(command)
            except KeyboardInterrupt:
                print('\n')
        operation = None
 
    elif command.startswith("!edit"):
        path = shlex.split(command)
        if len(path) < 2:
            print("** please provide the name of a model")
        else:
            edit_model(path[1], gen, opt, completer)
        completer.add_history(command)
        operation = None

    elif command.startswith("!del"):
        path = shlex.split(command)
        if len(path) < 2:
            print("** please provide the name of a model")
        else:
            del_config(path[1], gen, opt, completer)
        completer.add_history(command)
        operation = None

    elif command.startswith("!fetch"):
        file_path = command.replace("!fetch", "", 1).strip()
        retrieve_dream_command(opt, file_path, completer)
        completer.add_history(command)
        operation = None

    elif command.startswith("!replay"):
        file_path = command.replace("!replay", "", 1).strip()
        if infile is None and os.path.isfile(file_path):
            infile = open(file_path, "r", encoding="utf-8")
        completer.add_history(command)
        operation = None

    elif command.startswith("!trigger"):
        print("Embedding trigger strings: ", ", ".join(gen.embedding_trigger_strings))
        operation = None

    elif command.startswith("!history"):
        completer.show_history()
        operation = None

    elif command.startswith("!search"):
        search_str = command.replace("!search", "", 1).strip()
        completer.show_history(search_str)
        operation = None

    elif command.startswith("!clear"):
        completer.clear_history()
        operation = None

    elif re.match("^!(\d+)", command):
        command_no = re.match("^!(\d+)", command).groups()[0]
        command = completer.get_line(int(command_no))
        completer.set_line(command)
        operation = None

    else:  # not a recognized command, so give the --help text
        command = "-h"
    return command, operation


def set_default_output_dir(opt: Args, completer: Completer):
    """
    If opt.outdir is relative, we add the root directory to it
    normalize the outdir relative to root and make sure it exists.
    """
    if not os.path.isabs(opt.outdir):
        opt.outdir = os.path.normpath(os.path.join(Globals.root, opt.outdir))
    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)
    completer.set_default_dir(opt.outdir)


def import_model(model_path: str, gen, opt, completer, convert=False):
    """
    model_path can be (1) a URL to a .ckpt file; (2) a local .ckpt file path;
    (3) a huggingface repository id; or (4) a local directory containing a
    diffusers model.
    """
    default_name = Path(model_path).stem
    model_name = None
    model_desc = None

    if (
        Path(model_path).is_dir()
        and not (Path(model_path) / "model_index.json").exists()
    ):
        pass
    else:
        if model_path.startswith(('http:','https:')):
            try:
                default_name = url_attachment_name(model_path)
                default_name = Path(default_name).stem
            except Exception as e:
                print(f'** URL: {str(e)}')
            model_name, model_desc = _get_model_name_and_desc(
                gen.model_manager,
                completer,
                model_name=default_name,
            )
    imported_name = gen.model_manager.heuristic_import(
        model_path,
        model_name=model_name,
        description=model_desc,
        convert=convert,
        config_file_callback=lambda x: _pick_configuration_file(completer,x),
    )
    if not imported_name:
        print("** Aborting import.")
        return

    if not _verify_load(imported_name, gen):
        print("** model failed to load. Discarding configuration entry")
        gen.model_manager.del_model(imported_name)
        return
    if click.confirm("Make this the default model?", default=False):
        gen.model_manager.set_default_model(imported_name)

    gen.model_manager.commit(opt.conf)
    completer.update_models(gen.model_manager.list_models())
    print(f">> {imported_name} successfully installed")

def _pick_configuration_file(completer, checkpoint_path: Path)->Path:
    print(
f"""
Please select the type of the model at checkpoint {checkpoint_path}:
[1] A Stable Diffusion v1.x ckpt/safetensors model
[2] A Stable Diffusion v1.x inpainting ckpt/safetensors model
[3] A Stable Diffusion v2.x base model (512 pixels; there should be no 'parameterization:' line in its yaml file)
[4] A Stable Diffusion v2.x v-predictive model (768 pixels; look for a 'parameterization: "v"' line in its yaml file)
[5] Other (you will be prompted to enter the config file path)
[Q] I have no idea! Skip the import.
""")
    choices = [
        global_config_dir() / 'stable-diffusion' / x
        for x in [
                'v1-inference.yaml',
                'v1-inpainting-inference.yaml',
                'v2-inference.yaml',
                'v2-inference-v.yaml',
        ]
    ]

    ok = False
    while not ok:
        try:
            choice = input('select 0-5, Q > ').strip()
            if choice.startswith(('q','Q')):
                return
            if choice == '5':
                completer.complete_extensions(('.yaml'))
                choice = Path(input('Select config file for this model> ').strip()).absolute()
                completer.complete_extensions(None)
                ok = choice.exists()
            else:
                choice = choices[int(choice)-1]
                ok = True
        except (ValueError, IndexError):
            print(f'{choice} is not a valid choice')
        except EOFError:
            return
    return choice


def _verify_load(model_name: str, gen) -> bool:
    print(">> Verifying that new model loads...")
    current_model = gen.model_name
    try:
        if not gen.set_model(model_name):
            return
    except Exception as e:
        print(f"** model failed to load: {str(e)}")
        print(
            "** note that importing 2.X checkpoints is not supported. Please use !convert_model instead."
        )
        return False
    if click.confirm("Keep model loaded?", default=True):
        gen.set_model(model_name)
    else:
        print(">> Restoring previous model")
        gen.set_model(current_model)
    return True


def _get_model_name_and_desc(
    model_manager, completer, model_name: str = "", model_description: str = ""
):
    model_name = _get_model_name(model_manager.list_models(), completer, model_name)
    model_description = model_description or f"Imported model {model_name}"
    completer.set_line(model_description)
    model_description = (
        input(f"Description for this model [{model_description}]: ").strip()
        or model_description
    )
    return model_name, model_description

def convert_model(model_name_or_path: Union[Path, str], gen, opt, completer):
    model_name_or_path = model_name_or_path.replace("\\", "/")  # windows
    manager = gen.model_manager
    ckpt_path = None
    original_config_file = None
    if model_name_or_path == gen.model_name:
        print("** Can't convert the active model. !switch to another model first. **")
        return
    elif model_info := manager.model_info(model_name_or_path):
        if "weights" in model_info:
            ckpt_path = Path(model_info["weights"])
            original_config_file = Path(model_info["config"])
            model_name = model_name_or_path
            model_description = model_info["description"]
            vae_path = model_info.get("vae")
        else:
            print(f"** {model_name_or_path} is not a legacy .ckpt weights file")
            return
        model_name = manager.convert_and_import(
            ckpt_path,
            diffusers_path=Path(
                Globals.root, "models", Globals.converted_ckpts_dir, model_name_or_path
            ),
            model_name=model_name,
            model_description=model_description,
            original_config_file=original_config_file,
            vae_path=vae_path,
        )
    else:
        try:
            import_model(model_name_or_path, gen, opt, completer, convert=True)
        except KeyboardInterrupt:
            return

    manager.commit(opt.conf)
    if ckpt_path and click.confirm(f"Delete the original .ckpt file at {ckpt_path}?", default=False):
        ckpt_path.unlink(missing_ok=True)
        print(f"{ckpt_path} deleted")


def del_config(model_name: str, gen, opt, completer):
    current_model = gen.model_name
    if model_name == current_model:
        print("** Can't delete active model. !switch to another model first. **")
        return
    if model_name not in gen.model_manager.config:
        print(f"** Unknown model {model_name}")
        return

    if not click.confirm(
        f"Remove {model_name} from the list of models known to InvokeAI?", default=True
    ):
        return

    delete_completely = click.confirm(
        "Completely remove the model file or directory from disk?", default=False
    )
    gen.model_manager.del_model(model_name, delete_files=delete_completely)
    gen.model_manager.commit(opt.conf)
    print(f"** {model_name} deleted")
    completer.update_models(gen.model_manager.list_models())


def edit_model(model_name: str, gen, opt, completer):
    manager = gen.model_manager
    if not (info := manager.model_info(model_name)):
        print(f"** Unknown model {model_name}")
        return

    print(f"\n>> Editing model {model_name} from configuration file {opt.conf}")
    new_name = _get_model_name(manager.list_models(), completer, model_name)

    completer.complete_extensions(('.yaml','.ckpt','.safetensors','.pt'))
    for attribute in info.keys():
        if type(info[attribute]) != str:
            continue
        if attribute == "format":
            continue
        completer.set_line(info[attribute])
        info[attribute] = input(f"{attribute}: ") or info[attribute]
    completer.complete_extensions(None)

    if info["format"] == "diffusers":
        vae = info.get("vae", dict(repo_id=None, path=None, subfolder=None))
        completer.set_line(vae.get("repo_id") or "stabilityai/sd-vae-ft-mse")
        vae["repo_id"] = input("External VAE repo_id: ").strip() or None
        if not vae["repo_id"]:
            completer.set_line(vae.get("path") or "")
            vae["path"] = (
                input("Path to a local diffusers VAE model (usually none): ").strip()
                or None
            )
        completer.set_line(vae.get("subfolder") or "")
        vae["subfolder"] = (
            input("Name of subfolder containing the VAE model (usually none): ").strip()
            or None
        )
        info["vae"] = vae

    if new_name != model_name:
        manager.del_model(model_name)

    # this does the update
    manager.add_model(new_name, info, True)

    if click.confirm("Make this the default model?", default=False):
        manager.set_default_model(new_name)
    manager.commit(opt.conf)
    completer.update_models(manager.list_models())
    print(">> Model successfully updated")


def _get_model_name(existing_names, completer, default_name: str = "") -> str:
    done = False
    completer.set_line(default_name)
    while not done:
        model_name = input(f"Short name for this model [{default_name}]: ").strip()
        if len(model_name) == 0:
            model_name = default_name
        if not re.match("^[\w._+:/-]+$", model_name):
            print(
                '** model name must contain only words, digits and the characters "._+:/-" **'
            )
        elif model_name != default_name and model_name in existing_names:
            print(f"** the name {model_name} is already in use. Pick another.")
        else:
            done = True
    return model_name


def do_textmask(gen, opt, callback):
    image_path = opt.prompt
    if not os.path.exists(image_path):
        image_path = os.path.join(opt.outdir, image_path)
    assert os.path.exists(
        image_path
    ), '** "{opt.prompt}" not found. Please enter the name of an existing image file to mask **'
    assert (
        opt.text_mask is not None and len(opt.text_mask) >= 1
    ), "** Please provide a text mask with -tm **"
    opt.input_file_path = image_path
    tm = opt.text_mask[0]
    threshold = float(opt.text_mask[1]) if len(opt.text_mask) > 1 else 0.5
    gen.apply_textmask(
        image_path=image_path,
        prompt=tm,
        threshold=threshold,
        callback=callback,
    )


def do_postprocess(gen, opt, callback):
    file_path = opt.prompt  # treat the prompt as the file pathname
    if opt.new_prompt is not None:
        opt.prompt = opt.new_prompt
    else:
        opt.prompt = None

    if os.path.dirname(file_path) == "":  # basename given
        file_path = os.path.join(opt.outdir, file_path)

    opt.input_file_path = file_path

    tool = None
    if opt.facetool_strength > 0:
        tool = opt.facetool
    elif opt.embiggen:
        tool = "embiggen"
    elif opt.upscale:
        tool = "upscale"
    elif opt.out_direction:
        tool = "outpaint"
    elif opt.outcrop:
        tool = "outcrop"
    opt.save_original = True  # do not overwrite old image!
    opt.last_operation = f"postprocess:{tool}"
    try:
        gen.apply_postprocessor(
            image_path=file_path,
            tool=tool,
            facetool_strength=opt.facetool_strength,
            codeformer_fidelity=opt.codeformer_fidelity,
            save_original=opt.save_original,
            upscale=opt.upscale,
            upscale_denoise_str=opt.esrgan_denoise_str,
            out_direction=opt.out_direction,
            outcrop=opt.outcrop,
            callback=callback,
            opt=opt,
        )
    except OSError:
        print(traceback.format_exc(), file=sys.stderr)
        print(f"** {file_path}: file could not be read")
        return
    except (KeyError, AttributeError):
        print(traceback.format_exc(), file=sys.stderr)
        return
    return opt.last_operation


def add_postprocessing_to_metadata(opt, original_file, new_file, tool, command):
    original_file = (
        original_file
        if os.path.exists(original_file)
        else os.path.join(opt.outdir, original_file)
    )
    new_file = (
        new_file if os.path.exists(new_file) else os.path.join(opt.outdir, new_file)
    )
    try:
        meta = retrieve_metadata(original_file)["sd-metadata"]
    except AttributeError:
        try:
            meta = retrieve_metadata(new_file)["sd-metadata"]
        except AttributeError:
            meta = {}

    if "image" not in meta:
        meta = metadata_dumps(opt, seeds=[opt.seed])["image"]
        meta["image"] = {}
    img_data = meta.get("image")
    pp = img_data.get("postprocessing", []) or []
    pp.append(
        {
            "tool": tool,
            "dream_command": command,
        }
    )
    meta["image"]["postprocessing"] = pp
    write_metadata(new_file, meta)


def prepare_image_metadata(
        opt,
        prefix,
        seed,
        operation="generate",
        prior_variations=[],
        postprocessed=False,
        first_seed=None,
        model_id='unknown',
):
    if postprocessed and opt.save_original:
        filename = choose_postprocess_name(opt, prefix, seed)
    else:
        wildcards = dict(opt.__dict__)
        wildcards["prefix"] = prefix
        wildcards["seed"] = seed
        wildcards["model_id"] = model_id
        try:
            filename = opt.fnformat.format(**wildcards)
        except KeyError as e:
            print(
                f"** The filename format contains an unknown key '{e.args[0]}'. Will use {{prefix}}.{{seed}}.png' instead"
            )
            filename = f"{prefix}.{seed}.png"
        except IndexError:
            print(
                "** The filename format is broken or complete. Will use '{prefix}.{seed}.png' instead"
            )
            filename = f"{prefix}.{seed}.png"

    if opt.variation_amount > 0:
        first_seed = first_seed or seed
        this_variation = [[seed, opt.variation_amount]]
        opt.with_variations = prior_variations + this_variation
        formatted_dream_prompt = opt.dream_prompt_str(seed=first_seed,model_id=model_id)
    elif len(prior_variations) > 0:
        formatted_dream_prompt = opt.dream_prompt_str(seed=first_seed,model_id=model_id)
    elif operation == "postprocess":
        formatted_dream_prompt = "!fix " + opt.dream_prompt_str(
            seed=seed, prompt=opt.input_file_path, model_id=model_id,
        )
    else:
        formatted_dream_prompt = opt.dream_prompt_str(seed=seed,model_id=model_id)
    return filename, formatted_dream_prompt

def choose_postprocess_name(opt, prefix, seed) -> str:
    match = re.search("postprocess:(\w+)", opt.last_operation)
    if match:
        modifier = match.group(
            1
        )  # will look like "gfpgan", "upscale", "outpaint" or "embiggen"
    else:
        modifier = "postprocessed"

    counter = 0
    filename = None
    available = False
    while not available:
        if counter == 0:
            filename = f"{prefix}.{seed}.{modifier}.png"
        else:
            filename = f"{prefix}.{seed}.{modifier}-{counter:02d}.png"
        available = not os.path.exists(os.path.join(opt.outdir, filename))
        counter += 1
    return filename


def get_next_command(infile=None, model_name="no model") -> str:  # command string
    if infile is None:
        command = input(f"({model_name}) invoke> ").strip()
    else:
        command = infile.readline()
        if not command:
            raise EOFError
        else:
            command = command.strip()
        if len(command) > 0:
            print(f"#{command}")
    return command


def invoke_ai_web_server_loop(gen: Generate, gfpgan, codeformer, esrgan):
    print("\n* --web was specified, starting web server...")
    from invokeai.backend import InvokeAIWebServer

    # Change working directory to the stable-diffusion directory
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    invoke_ai_web_server = InvokeAIWebServer(
        generate=gen, gfpgan=gfpgan, codeformer=codeformer, esrgan=esrgan
    )

    try:
        invoke_ai_web_server.run()
    except KeyboardInterrupt:
        pass


def add_embedding_terms(gen, completer):
    """
    Called after setting the model, updates the autocompleter with
    any terms loaded by the embedding manager.
    """
    trigger_strings = gen.model.textual_inversion_manager.get_all_trigger_strings()
    completer.add_embedding_terms(trigger_strings)


def split_variations(variations_string) -> list:
    # shotgun parsing, woo
    parts = []
    broken = False  # python doesn't have labeled loops...
    for part in variations_string.split(","):
        seed_and_weight = part.split(":")
        if len(seed_and_weight) != 2:
            print(f'** Could not parse with_variation part "{part}"')
            broken = True
            break
        try:
            seed = int(seed_and_weight[0])
            weight = float(seed_and_weight[1])
        except ValueError:
            print(f'** Could not parse with_variation part "{part}"')
            broken = True
            break
        parts.append([seed, weight])
    if broken:
        return None
    elif len(parts) == 0:
        return None
    else:
        return parts


def load_face_restoration(opt):
    try:
        gfpgan, codeformer, esrgan = None, None, None
        if opt.restore or opt.esrgan:
            from ldm.invoke.restoration import Restoration

            restoration = Restoration()
            if opt.restore:
                gfpgan, codeformer = restoration.load_face_restore_models(
                    opt.gfpgan_model_path
                )
            else:
                print(">> Face restoration disabled")
            if opt.esrgan:
                esrgan = restoration.load_esrgan(opt.esrgan_bg_tile)
            else:
                print(">> Upscaling disabled")
        else:
            print(">> Face restoration and upscaling disabled")
    except (ModuleNotFoundError, ImportError):
        print(traceback.format_exc(), file=sys.stderr)
        print(">> You may need to install the ESRGAN and/or GFPGAN modules")
    return gfpgan, codeformer, esrgan


def make_step_callback(gen, opt, prefix):
    destination = os.path.join(opt.outdir, "intermediates", prefix)
    os.makedirs(destination, exist_ok=True)
    print(f">> Intermediate images will be written into {destination}")

    def callback(state: PipelineIntermediateState):
        latents = state.latents
        step = state.step
        if step % opt.save_intermediates == 0 or step == opt.steps - 1:
            filename = os.path.join(destination, f"{step:04}.png")
            image = gen.sample_to_lowres_estimated_image(latents)
            image = image.resize((image.size[0]*8,image.size[1]*8))
            image.save(filename, "PNG")

    return callback


def retrieve_dream_command(opt, command, completer):
    """
    Given a full or partial path to a previously-generated image file,
    will retrieve and format the dream command used to generate the image,
    and pop it into the readline buffer (linux, Mac), or print out a comment
    for cut-and-paste (windows)

    Given a wildcard path to a folder with image png files,
    will retrieve and format the dream command used to generate the images,
    and save them to a file commands.txt for further processing
    """
    if len(command) == 0:
        return

    tokens = command.split()
    dir, basename = os.path.split(tokens[0])
    if len(dir) == 0:
        path = os.path.join(opt.outdir, basename)
    else:
        path = tokens[0]

    if len(tokens) > 1:
        return write_commands(opt, path, tokens[1])

    cmd = ""
    try:
        cmd = dream_cmd_from_png(path)
    except OSError:
        print(f"## {tokens[0]}: file could not be read")
    except (KeyError, AttributeError, IndexError):
        print(f"## {tokens[0]}: file has no metadata")
    except:
        print(f"## {tokens[0]}: file could not be processed")
    if len(cmd) > 0:
        completer.set_line(cmd)


def write_commands(opt, file_path: str, outfilepath: str):
    dir, basename = os.path.split(file_path)
    try:
        paths = sorted(list(Path(dir).glob(basename)))
    except ValueError:
        print(f'## "{basename}": unacceptable pattern')
        return

    commands = []
    cmd = None
    for path in paths:
        try:
            cmd = dream_cmd_from_png(path)
        except (KeyError, AttributeError, IndexError):
            print(f"## {path}: file has no metadata")
        except:
            print(f"## {path}: file could not be processed")
        if cmd:
            commands.append(f"# {path}")
            commands.append(cmd)
    if len(commands) > 0:
        dir, basename = os.path.split(outfilepath)
        if len(dir) == 0:
            outfilepath = os.path.join(opt.outdir, basename)
        with open(outfilepath, "w", encoding="utf-8") as f:
            f.write("\n".join(commands))
        print(f">> File {outfilepath} with commands created")


def report_model_error(opt: Namespace, e: Exception):
    print(f'** An error occurred while attempting to initialize the model: "{str(e)}"')
    print(
        "** This can be caused by a missing or corrupted models file, and can sometimes be fixed by (re)installing the models."
    )
    yes_to_all = os.environ.get("INVOKE_MODEL_RECONFIGURE")
    if yes_to_all:
        print(
            "** Reconfiguration is being forced by environment variable INVOKE_MODEL_RECONFIGURE"
        )
    else:
        if not click.confirm(
                'Do you want to run invokeai-configure script to select and/or reinstall models?',
                default=False
        ):
            return

    print("invokeai-configure is launching....\n")

    # Match arguments that were set on the CLI
    # only the arguments accepted by the configuration script are parsed
    root_dir = ["--root", opt.root_dir] if opt.root_dir is not None else []
    config = ["--config", opt.conf] if opt.conf is not None else []
    previous_args = sys.argv
    sys.argv = ["invokeai-configure"]
    sys.argv.extend(root_dir)
    sys.argv.extend(config)
    if yes_to_all is not None:
        for arg in yes_to_all.split():
            sys.argv.append(arg)

    from ldm.invoke.config import invokeai_configure

    invokeai_configure.main()
    print("** InvokeAI will now restart")
    sys.argv = previous_args
    main()  # would rather do a os.exec(), but doesn't exist?
    sys.exit(0)


def check_internet() -> bool:
    """
    Return true if the internet is reachable.
    It does this by pinging huggingface.co.
    """
    import urllib.request

    host = "http://huggingface.co"
    try:
        urllib.request.urlopen(host, timeout=1)
        return True
    except:
        return False


def retrieve_last_used_model()->str:
    """
    Return name of the last model used.
    """
    model_file_path = Path(Globals.root,'.last_model')
    if not model_file_path.exists():
        return None
    with open(model_file_path,'r') as f:
        return f.readline()

def save_last_used_model(model_name:str):
    """
    Save name of the last model used.
    """
    model_file_path = Path(Globals.root,'.last_model')
    with open(model_file_path,'w') as f:
        f.write(model_name)

# This routine performs any patch-ups needed after installation
def run_patches():
    install_missing_config_files()
    version_file = Path(Globals.root,'.version')
    if version_file.exists():
        with open(version_file,'r') as f:
            root_version = version.parse(f.readline() or 'v2.3.2')
    else:
        root_version = version.parse('v2.3.2')
    app_version = version.parse(ldm.invoke.__version__)
    if root_version < app_version:
        try:
            do_version_update(root_version, ldm.invoke.__version__)
            with open(version_file,'w') as f:
                f.write(ldm.invoke.__version__)
        except:
            print("** Update failed. Will try again on next launch")

def install_missing_config_files():
    """
    install ckpt configuration files that may have been added to the
    distro after original root directory configuration
    """
    import invokeai.configs as conf
    from shutil import copyfile
    
    root_configs = Path(global_config_dir(), 'stable-diffusion')
    repo_configs = Path(conf.__path__[0], 'stable-diffusion')
    for src in repo_configs.iterdir():
        dest = root_configs / src.name
        if not dest.exists():
            copyfile(src,dest)
    
def do_version_update(root_version: version.Version, app_version: Union[str, version.Version]):
    """
    Make any updates to the launcher .sh and .bat scripts that may be needed
    from release to release. This is not an elegant solution. Instead, the 
    launcher should be moved into the source tree and installed using pip.
    """
    if root_version < version.Version('v2.3.3'):
        if sys.platform == "linux":
            print('>> Downloading new version of launcher script and its config file')
            from ldm.util import download_with_progress_bar
            url_base = f'https://raw.githubusercontent.com/invoke-ai/InvokeAI/v{str(app_version)}/installer/templates/'

            dest = Path(Globals.root,'invoke.sh.in')
            assert download_with_progress_bar(url_base+'invoke.sh.in',dest)
            dest.replace(Path(Globals.root,'invoke.sh'))
            os.chmod(Path(Globals.root,'invoke.sh'), 0o0755)

            dest = Path(Globals.root,'dialogrc')
            assert download_with_progress_bar(url_base+'dialogrc',dest)
            dest.replace(Path(Globals.root,'.dialogrc'))

if __name__ == '__main__':
    main()
    
