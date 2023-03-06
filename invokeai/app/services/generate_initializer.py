import os
import sys
import traceback
from argparse import Namespace

import invokeai.version
from invokeai.backend import Generate, ModelManager

from ...backend import Globals


# TODO: most of this code should be split into individual services as the Generate.py code is deprecated
def get_generate(args, config) -> Generate:
    if not args.conf:
        config_file = os.path.join(Globals.root, "configs", "models.yaml")
        if not os.path.exists(config_file):
            report_model_error(
                args, FileNotFoundError(f"The file {config_file} could not be found.")
            )

    print(f">> {invokeai.version.__app_name__}, version {invokeai.version.__version__}")
    print(f'>> InvokeAI runtime directory is "{Globals.root}"')

    # these two lines prevent a horrible warning message from appearing
    # when the frozen CLIP tokenizer is imported
    import transformers  # type: ignore

    transformers.logging.set_verbosity_error()
    import diffusers

    diffusers.logging.set_verbosity_error()

    # Loading Face Restoration and ESRGAN Modules
    gfpgan, codeformer, esrgan = load_face_restoration(args)

    # normalize the config directory relative to root
    if not os.path.isabs(args.conf):
        args.conf = os.path.normpath(os.path.join(Globals.root, args.conf))

    if args.embeddings:
        if not os.path.isabs(args.embedding_path):
            embedding_path = os.path.normpath(
                os.path.join(Globals.root, args.embedding_path)
            )
        else:
            embedding_path = args.embedding_path
    else:
        embedding_path = None

    # migrate legacy models
    ModelManager.migrate_models()

    # load the infile as a list of lines
    if args.infile:
        try:
            if os.path.isfile(args.infile):
                infile = open(args.infile, "r", encoding="utf-8")
            elif args.infile == "-":  # stdin
                infile = sys.stdin
            else:
                raise FileNotFoundError(f"{args.infile} not found.")
        except (FileNotFoundError, IOError) as e:
            print(f"{e}. Aborting.")
            sys.exit(-1)

    # creating a Generate object:
    try:
        gen = Generate(
            conf=args.conf,
            model=args.model,
            sampler_name=args.sampler_name,
            embedding_path=embedding_path,
            full_precision=args.full_precision,
            precision=args.precision,
            gfpgan=gfpgan,
            codeformer=codeformer,
            esrgan=esrgan,
            free_gpu_mem=args.free_gpu_mem,
            safety_checker=args.safety_checker,
            max_loaded_models=args.max_loaded_models,
        )
    except (FileNotFoundError, TypeError, AssertionError) as e:
        report_model_error(opt, e)
    except (IOError, KeyError) as e:
        print(f"{e}. Aborting.")
        sys.exit(-1)

    if args.seamless:
        print(">> changed to seamless tiling mode")

    # preload the model
    try:
        gen.load_model()
    except KeyError:
        pass
    except Exception as e:
        report_model_error(args, e)

    # try to autoconvert new models
    # autoimport new .ckpt files
    if path := args.autoconvert:
        gen.model_manager.autoconvert_weights(
            conf_path=args.conf,
            weights_directory=path,
        )

    return gen


def load_face_restoration(opt):
    try:
        gfpgan, codeformer, esrgan = None, None, None
        if opt.restore or opt.esrgan:
            from invokeai.backend.restoration import Restoration

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
        response = input(
            "Do you want to run invokeai-configure script to select and/or reinstall models? [y] "
        )
        if response.startswith(("n", "N")):
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

    from invokeai.frontend.install import invokeai_configure

    invokeai_configure()
    # TODO: Figure out how to restart
    # print('** InvokeAI will now restart')
    # sys.argv = previous_args
    # main() # would rather do a os.exec(), but doesn't exist?
    # sys.exit(0)


# Temporary initializer for Generate until we migrate off of it
def old_get_generate(args, config) -> Generate:
    # TODO: Remove the need for globals
    from invokeai.backend.globals import Globals

    # alert - setting globals here
    Globals.root = os.path.expanduser(
        args.root_dir or os.environ.get("INVOKEAI_ROOT") or os.path.abspath(".")
    )
    Globals.try_patchmatch = args.patchmatch

    print(f'>> InvokeAI runtime directory is "{Globals.root}"')

    # these two lines prevent a horrible warning message from appearing
    # when the frozen CLIP tokenizer is imported
    import transformers

    transformers.logging.set_verbosity_error()

    # Loading Face Restoration and ESRGAN Modules
    gfpgan, codeformer, esrgan = None, None, None
    try:
        if config.restore or config.esrgan:
            from ldm.invoke.restoration import Restoration

            restoration = Restoration()
            if config.restore:
                gfpgan, codeformer = restoration.load_face_restore_models(
                    config.gfpgan_model_path
                )
            else:
                print(">> Face restoration disabled")
            if config.esrgan:
                esrgan = restoration.load_esrgan(config.esrgan_bg_tile)
            else:
                print(">> Upscaling disabled")
        else:
            print(">> Face restoration and upscaling disabled")
    except (ModuleNotFoundError, ImportError):
        print(traceback.format_exc(), file=sys.stderr)
        print(">> You may need to install the ESRGAN and/or GFPGAN modules")

    # normalize the config directory relative to root
    if not os.path.isabs(config.conf):
        config.conf = os.path.normpath(os.path.join(Globals.root, config.conf))

    if config.embeddings:
        if not os.path.isabs(config.embedding_path):
            embedding_path = os.path.normpath(
                os.path.join(Globals.root, config.embedding_path)
            )
    else:
        embedding_path = None

    # TODO: lazy-initialize this by wrapping it
    try:
        generate = Generate(
            conf=config.conf,
            model=config.model,
            sampler_name=config.sampler_name,
            embedding_path=embedding_path,
            full_precision=config.full_precision,
            precision=config.precision,
            gfpgan=gfpgan,
            codeformer=codeformer,
            esrgan=esrgan,
            free_gpu_mem=config.free_gpu_mem,
            safety_checker=config.safety_checker,
            max_loaded_models=config.max_loaded_models,
        )
    except (FileNotFoundError, TypeError, AssertionError):
        # emergency_model_reconfigure() # TODO?
        sys.exit(-1)
    except (IOError, KeyError) as e:
        print(f"{e}. Aborting.")
        sys.exit(-1)

    generate.free_gpu_mem = config.free_gpu_mem

    return generate
