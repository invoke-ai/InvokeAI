def get_app():
    """Import the app and event loop. We wrap this in a function to more explicitly control when it happens, because
    importing from api_app does a bunch of stuff - it's more like calling a function than importing a module.
    """
    from invokeai.app.api_app import app, loop

    return app, loop


def run_app() -> None:
    """The main entrypoint for the app."""
    from invokeai.frontend.cli.arg_parser import InvokeAIArgs

    # Parse the CLI arguments before doing anything else, which ensures CLI args correctly override settings from other
    # sources like `invokeai.yaml` or env vars.
    InvokeAIArgs.parse_args()

    import uvicorn

    from invokeai.app.services.config.config_default import get_config
    from invokeai.app.util.torch_cuda_allocator import configure_torch_cuda_allocator
    from invokeai.backend.util.logging import InvokeAILogger

    # Load config.
    app_config = get_config()

    logger = InvokeAILogger.get_logger(config=app_config)

    # Configure the torch CUDA memory allocator.
    # NOTE: It is important that this happens before torch is imported.
    if app_config.pytorch_cuda_alloc_conf:
        configure_torch_cuda_allocator(app_config.pytorch_cuda_alloc_conf, logger)

    # This import must happen after configure_torch_cuda_allocator() is called, because the module imports torch.
    from invokeai.app.invocations.baseinvocation import InvocationRegistry
    from invokeai.app.invocations.load_custom_nodes import load_custom_nodes
    from invokeai.backend.util.devices import TorchDevice

    torch_device_name = TorchDevice.get_torch_device_name()
    logger.info(f"Using torch device: {torch_device_name}")

    # Import from startup_utils here to avoid importing torch before configure_torch_cuda_allocator() is called.
    from invokeai.app.util.startup_utils import (
        apply_monkeypatches,
        check_cudnn,
        enable_dev_reload,
        find_open_port,
        register_mime_types,
    )

    # Find an open port, and modify the config accordingly.
    first_open_port = find_open_port(app_config.port)
    if app_config.port != first_open_port:
        orig_config_port = app_config.port
        app_config.port = first_open_port
        logger.warning(f"Port {orig_config_port} is already in use. Using port {app_config.port}.")

    # Miscellaneous startup tasks.
    apply_monkeypatches()
    register_mime_types()
    check_cudnn(logger)

    # Initialize the app and event loop.
    app, loop = get_app()

    # Load custom nodes. This must be done after importing the Graph class, which itself imports all modules from the
    # invocations module. The ordering here is implicit, but important - we want to load custom nodes after all the
    # core nodes have been imported so that we can catch when a custom node clobbers a core node.
    load_custom_nodes(custom_nodes_path=app_config.custom_nodes_path, logger=logger)

    # Check all invocations and ensure their outputs are registered.
    for invocation in InvocationRegistry.get_invocation_classes():
        invocation_type = invocation.get_type()
        output_annotation = invocation.get_output_annotation()
        if output_annotation not in InvocationRegistry.get_output_classes():
            logger.warning(
                f'Invocation "{invocation_type}" has unregistered output class "{output_annotation.__name__}"'
            )

    if app_config.dev_reload:
        # load_custom_nodes seems to bypass jurrigged's import sniffer, so be sure to call it *after* they're already
        # imported.
        enable_dev_reload(custom_nodes_path=app_config.custom_nodes_path)

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
