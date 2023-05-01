import os
import sys
import torch
from argparse import Namespace
from invokeai.backend import Args
from omegaconf import OmegaConf
from pathlib import Path

import invokeai.version
from .config import InvokeAISettings
from ...backend import ModelManager
from ...backend.util import choose_precision, choose_torch_device

# TODO: Replace with an abstract class base ModelManagerBase
def get_model_manager(config: InvokeAISettings) -> ModelManager:
    model_config = config.model_conf_path
    if not model_config.exists():
        report_model_error(
            config, FileNotFoundError(f"The file {model_config} could not be found.")
        )

    print(f">> {invokeai.version.__app_name__}, version {invokeai.version.__version__}")
    print(f'>> InvokeAI runtime directory is "{config.root_dir}"')

    # these two lines prevent a horrible warning message from appearing
    # when the frozen CLIP tokenizer is imported
    import transformers  # type: ignore

    transformers.logging.set_verbosity_error()
    import diffusers

    diffusers.logging.set_verbosity_error()
    embedding_path = config.embedding_path

    # migrate legacy models
    ModelManager.migrate_models()

    # creating the model manager
    try:
        device = torch.device(choose_torch_device())
        precision = 'float16' if config.precision=='float16' \
        else 'float32' if config.precision=='float32' \
        else choose_precision(device)

        model_manager = ModelManager(
            OmegaConf.load(model_config),
            precision=precision,
            device_type=device,
            max_loaded_models=config.max_loaded_models,
            embedding_path = embedding_path,
        )
    except (FileNotFoundError, TypeError, AssertionError) as e:
        report_model_error(config, e)
    except (IOError, KeyError) as e:
        print(f"{e}. Aborting.")
        sys.exit(-1)

    # try to autoconvert new models
    # autoimport new .ckpt files
    if config.autoconvert_path:
        model_manager.heuristic_import(
            config.autoconvert_path,
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
