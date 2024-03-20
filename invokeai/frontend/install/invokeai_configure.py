"""
Wrapper for invokeai.backend.configure.invokeai_configure
"""

import argparse


def run_configure() -> None:
    # Before doing _anything_, parse CLI args!
    from invokeai.frontend.cli.arg_parser import InvokeAIArgs

    parser = argparse.ArgumentParser(description="InvokeAI model downloader")
    parser.add_argument(
        "--skip-sd-weights",
        dest="skip_sd_weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="skip downloading the large Stable Diffusion weight files",
    )
    parser.add_argument(
        "--skip-support-models",
        dest="skip_support_models",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="skip downloading the support models",
    )
    parser.add_argument(
        "--full-precision",
        dest="full_precision",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="use 32-bit weights instead of faster 16-bit weights",
    )
    parser.add_argument(
        "--yes",
        "-y",
        dest="yes_to_all",
        action="store_true",
        help='answer "yes" to all prompts',
    )
    parser.add_argument(
        "--default_only",
        action="store_true",
        help="when --yes specified, only install the default model",
    )
    parser.add_argument(
        "--root_dir",
        dest="root",
        type=str,
        default=None,
        help="path to root of install directory",
    )

    opt = parser.parse_args()
    InvokeAIArgs.args = opt

    from invokeai.backend.install.invokeai_configure import main as invokeai_configure

    invokeai_configure(opt)
