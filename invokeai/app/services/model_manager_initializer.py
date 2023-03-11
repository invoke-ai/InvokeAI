import os
import sys
import torch
from argparse import Namespace
from omegaconf import OmegaConf
from pathlib import Path

import invokeai.version
from ...backend import ModelManager
from ...backend.util import choose_precision, choose_torch_device
from ...backend import Globals

# TODO: most of this code should be split into individual services as the Generate.py code is deprecated
def get_model_manager(args, config) -> ModelManager:
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

    # creating the model manager
    try:
        device = torch.device(choose_torch_device())
        precision = 'float16' if args.precision=='float16' \
        else 'float32' if args.precision=='float32' \
        else choose_precision(device)
        
        model_manager = ModelManager(
            OmegaConf.load(args.conf),
            precision=precision,
            device_type=device,
            max_loaded_models=args.max_loaded_models,
            embedding_path = Path(embedding_path),
        )
    except (FileNotFoundError, TypeError, AssertionError) as e:
        report_model_error(args, e)
    except (IOError, KeyError) as e:
        print(f"{e}. Aborting.")
        sys.exit(-1)

    if args.seamless:
        #TODO: do something here ?
        print(">> changed to seamless tiling mode")

    # try to autoconvert new models
    # autoimport new .ckpt files
    if path := args.autoconvert:
        model_manager.autoconvert_weights(
            conf_path=args.conf,
            weights_directory=path,
        )

    return model_manager

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
