"""This is a wrapper around the main app entrypoint, to allow for CLI args to be parsed before running the app."""


def run_app() -> None:
    # Before doing _anything_, parse CLI args!
    from invokeai.frontend.cli.arg_parser import InvokeAIArgs

    InvokeAIArgs.parse_args()

    from invokeai.app.api_app import invoke_api

    invoke_api()
