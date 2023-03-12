import os
import sys
import torch
from argparse import Namespace
from invokeai.backend import Args
from omegaconf import OmegaConf
from pathlib import Path

import invokeai.version
from ...backend import ModelManager
from ...backend.util import choose_precision, choose_torch_device
from ...backend import Globals

# TODO: Replace with an abstract class base ModelManagerBase
def get_model_manager(config: Args) -> ModelManager:
    if not config.conf:
        config_file = os.path.join(Globals.root, "configs", "models.yaml")
        if not os.path.exists(config_file):
            report_model_error(
                config, FileNotFoundError(f"The file {config_file} could not be found.")
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
    if not os.path.isabs(config.conf):
        config.conf = os.path.normpath(os.path.join(Globals.root, config.conf))

    if config.embeddings:
        if not os.path.isabs(config.embedding_path):
            embedding_path = os.path.normpath(
                os.path.join(Globals.root, config.embedding_path)
            )
        else:
            embedding_path = config.embedding_path
    else:
        embedding_path = None

    # migrate legacy models
    ModelManager.migrate_models()

    # creating the model manager
    try:
        device = torch.device(choose_torch_device())
        precision = 'float16' if config.precision=='float16' \
        else 'float32' if config.precision=='float32' \
        else choose_precision(device)
        
        model_manager = ModelManager(
            OmegaConf.load(config.conf),
            precision=precision,
            device_type=device,
            max_loaded_models=config.max_loaded_models,
            embedding_path = Path(embedding_path),
        )
    except (FileNotFoundError, TypeError, AssertionError) as e:
        report_model_error(config, e)
    except (IOError, KeyError) as e:
        print(f"{e}. Aborting.")
        sys.exit(-1)

    # try to autoconvert new models
    # autoimport new .ckpt files
    if path := config.autoconvert:
        model_manager.autoconvert_weights(
            conf_path=config.conf,
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
    previous_config = sys.argv
    sys.argv = ["invokeai-configure"]
    sys.argv.extend(root_dir)
    sys.argv.extend(config.to_dict())
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
