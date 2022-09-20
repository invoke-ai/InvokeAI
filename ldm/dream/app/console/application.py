import json
import os
import re
import sys
import ldm.dream.readline
from PIL import Image
from dependency_injector.wiring import inject, Provide
from omegaconf import OmegaConf
from ldm.dream.args import Args, metadata_dumps
from ldm.dream.pngwriter import PngWriter
from ldm.dream.image_util import make_grid
from ldm.generate import Generate
from ldm.dream.app.services.generation.services import GeneratorService
from ldm.dream.app.services.models import DreamResult, JobRequest
from ldm.dream.app.services.storage.services import ImageStorageService, JobQueueService
from ldm.dream.app.console.containers import Container, SignalServiceOverride


def run_console_app(opt, parser: Args):
    # Change working directory to the stable-diffusion directory
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

    print("* Initializing, be patient...\n")
    sys.path.append(".")
    from ldm.generate import Generate

    # these two lines prevent a horrible warning message from appearing
    # when the frozen CLIP tokenizer is imported
    import transformers

    transformers.logging.set_verbosity_error()

    # load the infile as a list of lines
    infile = None
    if "infile" in opt and opt.infile:
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

    if "seamless" in opt and opt.seamless:
        print(">> changed to seamless tiling mode")

    # Set up dependency injection container
    container = Container()
    container.config.from_dict(opt.__dict__)
    container.generator_package.config.from_dict(opt.__dict__)
    container.wire(modules=[__name__])

    main_loop(parser, infile)


# TODO: main_loop() has gotten busy. Needs to be refactored.
@inject
def main_loop(
    opt,
    infile,
    job_queue_service: JobQueueService = Provide[
        Container.generator_package.generation_queue_service
    ],
    image_storage_service: ImageStorageService = Provide[
        Container.storage_package.image_storage_service
    ],
    generator_service: GeneratorService = Provide[
        Container.generator_package.generator_service
    ],
    signal_service: SignalServiceOverride = Provide[
        Container.signal_service
    ],  # TODO: use eventing
    generator: Generate = Provide[
        Container.generator_package.model_singleton
    ],  # TODO: remove the need for this
):
    """prompt/read/execute loop"""
    done = False
    path_filter = re.compile(r'[<>:"/\\|?*]')
    last_results = list()
    model_config = OmegaConf.load(opt.conf)[opt.model]

    # Wait for generator to be warmed
    generator_service.wait_for_ready()

    # os.pathconf is not available on Windows
    if hasattr(os, "pathconf"):
        path_max = os.pathconf(opt.outdir, "PC_PATH_MAX")
        name_max = os.pathconf(opt.outdir, "PC_NAME_MAX")
    else:
        path_max = 260
        name_max = 255

    while not done:
        try:
            command = get_next_command(infile)
        except EOFError:
            done = True
            continue

        # skip empty lines
        if not command.strip():
            continue

        if command.startswith(("#", "//")):
            continue

        if len(command.strip()) == 1 and command.startswith("q"):
            done = True
            break

        if command.startswith(
            "!dream"
        ):  # in case a stored prompt still contains the !dream command
            command.replace("!dream", "", 1)

        if opt.parse_cmd(command) is None:
            continue
        if len(opt.prompt) == 0:
            print("\nTry again with a prompt!")
            continue

        # width and height are set by model if not specified
        if not opt.width:
            opt.width = model_config.width
        if not opt.height:
            opt.height = model_config.height

        # retrieve previous value!
        if opt.init_img is not None and re.match("^-\\d+$", opt.init_img):
            try:
                opt.init_img = last_results[int(opt.init_img)][0]
                print(f">> Reusing previous image {opt.init_img}")
            except IndexError:
                print(f">> No previous initial image at position {opt.init_img} found")
                opt.init_img = None
                continue

        if opt.seed is not None and opt.seed < 0:  # retrieve previous value!
            try:
                opt.seed = last_results[opt.seed][1]
                print(f">> Reusing previous seed {opt.seed}")
            except IndexError:
                print(f">> No previous seed at position {opt.seed} found")
                opt.seed = None
                continue

        # TODO - move this into a module
        if opt.with_variations is not None:
            # shotgun parsing, woo
            parts = []
            broken = False  # python doesn't have labeled loops...
            for part in opt.with_variations.split(","):
                seed_and_weight = part.split(":")
                if len(seed_and_weight) != 2:
                    print(f'could not parse with_variation part "{part}"')
                    broken = True
                    break
                try:
                    seed = int(seed_and_weight[0])
                    weight = float(seed_and_weight[1])
                except ValueError:
                    print(f'could not parse with_variation part "{part}"')
                    broken = True
                    break
                parts.append([seed, weight])
            if broken:
                continue
            if len(parts) > 0:
                opt.with_variations = parts
            else:
                opt.with_variations = None

        if opt.outdir:
            if not os.path.exists(opt.outdir):
                os.makedirs(opt.outdir)
            current_outdir = opt.outdir
        elif opt.prompt_as_dir:
            # sanitize the prompt to a valid folder name
            subdir = path_filter.sub("_", opt.prompt)[:name_max].rstrip(" .")

            # truncate path to maximum allowed length
            # 27 is the length of '######.##########.##.png', plus two separators and a NUL
            subdir = subdir[: (path_max - 27 - len(os.path.abspath(opt.outdir)))]
            current_outdir = os.path.join(opt.outdir, subdir)

            print('Writing files to directory: "' + current_outdir + '"')

            # make sure the output directory exists
            if not os.path.exists(current_outdir):
                os.makedirs(current_outdir)
        else:
            current_outdir = opt.outdir

        # Here is where the images are actually generated!
        last_results = []
        try:
            file_writer = PngWriter(current_outdir)  # TODO: write usage image storage
            results = []  # list of filename, prompt pairs

            # Generate enabled flags
            # TODO: generate JobRequest from Args
            overrides = {
                "enable_generate": True if opt.prompt else False,
                "enable_upscale": True if opt.upscale else False,
                "enable_gfpgan": True if opt.gfpgan_strength != 0 else False,
                "enable_init_image": True if opt.init_img else False,
                "enable_img2img": True if opt.init_img else False,
            }

            # Enqueue job
            job = JobRequest.from_json({**opt.__dict__, **overrides})
            job_queue_service.push(job)

            # Wait for completion
            while True:
                signal = signal_service.get_signal()
                if signal.event == "job_done":
                    break
                elif signal.event == "job_canceled":
                    break
                elif signal.event == "dream_result":
                    pass  # Job already captures all results

            def get_image(dreamResult: DreamResult) -> Image:
                path = image_storage_service.path(dreamResult.id)
                return Image.open(path)

            # TODO: probably need to have a file writing utility
            # TODO: Grids should be generated as part of the generator (or future pipelining)
            if opt.grid and len(job.results) > 0:
                grid_img = make_grid(list(map(get_image, job.results)))
                grid_seeds = list(map(lambda dreamResult: dreamResult.id, job.results))
                first_seed = job.results[0].seed
                filename = f"{job.id}.{first_seed}.grid.png"
                formatted_dream_prompt = opt.dream_prompt_str(
                    seed=first_seed, grid=True, iterations=len(job.results)
                )
                formatted_dream_prompt += f" # {grid_seeds}"
                metadata = (
                    metadata_dumps(  # TODO: standardize metadata with model class
                        opt, seeds=grid_seeds, model_hash=generator.model_hash
                    )
                )
                path = file_writer.save_image_and_prompt_to_png(
                    image=grid_img,
                    dream_prompt=formatted_dream_prompt,  # NOTE: metadata already has all this information
                    metadata=json.dumps(metadata),
                    name=filename,
                )

                # TODO: Make sure opt.save_original cleans up non-final results here
                results = [[path, formatted_dream_prompt]]
            else:
                # TODO: Make sure opt.save_original cleans up non-final results here
                results = list(
                    map(
                        lambda dreamResult: [
                            image_storage_service.path(dreamResult.id),
                            opt.dream_prompt_str(seed=dreamResult.seed),
                        ],
                        job.results,
                    )
                )

        except AssertionError as e:
            print(e)
            continue

        except OSError as e:
            print(e)
            continue

        print("Outputs:")
        log_path = os.path.join(current_outdir, "dream_log.txt")
        write_log_message(results, log_path)
        print()

    print("goodbye!")


def get_next_command(infile=None) -> str:  # command string
    if infile is None:
        command = input("dream> ")
    else:
        command = infile.readline()
        if not command:
            raise EOFError
        else:
            command = command.strip()
        if len(command) > 0:
            print(f"#{command}")
    return command


output_cntr = 0


def write_log_message(results, log_path):
    """logs the name of the output image, prompt, and prompt args to the terminal and log file"""
    global output_cntr
    log_lines = [f"{path}: {prompt}\n" for path, prompt in results]
    for l in log_lines:
        output_cntr += 1
        print(f"[{output_cntr}] {l}", end="")

    with open(log_path, "a", encoding="utf-8") as file:
        file.writelines(log_lines)


if __name__ == "__main__":
    """Load configuration and run application"""
    arg_parser = Args()
    args = arg_parser.parse_args()

    # Start server
    try:
        run_console_app(args, arg_parser)
    except KeyboardInterrupt:
        pass
