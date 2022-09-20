#!/usr/bin/env python3
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

import sys
from ldm.dream.args import Args

# Placeholder to be replaced with proper class that tracks the
# outputs and associates with the prompt that generated them.
# Just want to get the formatting look right for now.
output_cntr = 0


def main():
    """Initialize command-line parsers and the diffusion model"""
    arg_parser = Args()
    args = arg_parser.parse_args()
    if not args:
        sys.exit(-1)

    if args.web:
        # Start server
        try:
            from ldm.dream.app.api.application import run_web_app

            run_web_app(args.__dict__)
            sys.exit(0)
        except KeyboardInterrupt:
            sys.exit(0)

    else:
        from ldm.dream.app.console.application import run_console_app

        run_console_app(args, arg_parser)


if __name__ == "__main__":
    main()
