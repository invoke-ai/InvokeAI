import uvicorn

from invokeai.app.invocations.load_custom_nodes import load_custom_nodes
from invokeai.app.services.config.config_default import get_config
from invokeai.app.util.torch_cuda_allocator import configure_torch_cuda_allocator
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.frontend.cli.arg_parser import InvokeAIArgs


def get_app():
    """Import the app and event loop. We wrap this in a function to more explicitly control when it happens, because
    importing from api_app does a bunch of stuff - it's more like calling a function than importing a module.
    """
    from invokeai.app.api_app import app, loop

    return app, loop


def run_app() -> None:
    """The main entrypoint for the app."""
    # Parse the CLI arguments.
    InvokeAIArgs.parse_args()

    # Load config.
    app_config = get_config()

    logger = InvokeAILogger.get_logger(config=app_config)

    # Configure the torch CUDA memory allocator.
    # NOTE: It is important that this happens before torch is imported.
    if app_config.pytorch_cuda_alloc_conf:
        configure_torch_cuda_allocator(app_config.pytorch_cuda_alloc_conf, logger)

    # Import from startup_utils here to avoid importing torch before configure_torch_cuda_allocator() is called.
    from invokeai.app.util.startup_utils import (
        apply_monkeypatches,
        check_cudnn,
        enable_dev_reload,
        find_open_port,
        register_mime_types,
    )

    # Find an open port, and modify the config accordingly.
    orig_config_port = app_config.port
    app_config.port = find_open_port(app_config.port)
    if orig_config_port != app_config.port:
        logger.warning(f"Port {orig_config_port} is already in use. Using port {app_config.port}.")

    # Miscellaneous startup tasks.
    apply_monkeypatches()
    register_mime_types()
    if app_config.dev_reload:
        enable_dev_reload()
    check_cudnn(logger)

    # Initialize the app and event loop.
    app, loop = get_app()

    # Load custom nodes. This must be done after importing the Graph class, which itself imports all modules from the
    # invocations module. The ordering here is implicit, but important - we want to load custom nodes after all the
    # core nodes have been imported so that we can catch when a custom node clobbers a core node.
    load_custom_nodes(custom_nodes_path=app_config.custom_nodes_path, logger=logger)

    # Start the server.
    config = uvicorn.Config(
        app=app,
        host=app_config.host,
        port=app_config.port,
        loop="asyncio",
        log_level=app_config.log_level_network,
        ssl_certfile=app_config.ssl_certfile,
        ssl_keyfile=app_config.ssl_keyfile,
    )
    server = uvicorn.Server(config)

    # replace uvicorn's loggers with InvokeAI's for consistent appearance
    uvicorn_logger = InvokeAILogger.get_logger("uvicorn")
    uvicorn_logger.handlers.clear()
    for hdlr in logger.handlers:
        uvicorn_logger.addHandler(hdlr)

    loop.run_until_complete(server.serve())
