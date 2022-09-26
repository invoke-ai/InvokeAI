# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import argparse
import shlex
from typing import Literal, Union, get_args, get_origin, get_type_hints
from pydantic import BaseModel
from pydantic.fields import Field
from .invocations.baseinvocation import BaseInvocation
from .services.invocation_context import InvocationFieldLink
from .services.invocation_services import InvocationServices
from .services.invoker import Invoker
from .invocations import *
from ..args import Args
from ...generate import Generate
from .service_bases import EventServiceBase


class Command(BaseModel):
    invocation: Union[BaseInvocation.get_invocations()] = Field(discriminator="type")


def invoke_cli():
    args = Args()
    config = args.parse_args()

    generate = Generate(
        model=config.model,
        sampler_name=config.sampler_name,
        embedding_path=config.embedding_path,
        full_precision=config.full_precision,
    )

    # NOTE: load model on first use
    #generate.load_model()

    events = EventServiceBase()

    services = InvocationServices(
        generate = generate,
        events = events
    )
    invoker = Invoker(services)
    context = invoker.create_context()
    
    parser = argparse.ArgumentParser()

    class InvalidArgs(Exception):
        pass
    def exit(*args, **kwargs):
        raise InvalidArgs
    parser.exit = exit

    subparsers = parser.add_subparsers(dest='type')
    invocation_parsers = dict()

    invocations = BaseInvocation.get_all_subclasses()
    for invocation in invocations:
        hints = get_type_hints(invocation)
        cmd_name = get_args(hints['type'])[0]
        command_parser = subparsers.add_parser(cmd_name, help=invocation.__doc__)
        invocation_parsers[cmd_name] = command_parser


        fields = invocation.__fields__
        for name, field in fields.items():
            if name in ['id', 'type']:
                continue
            
            if get_origin(field.type_) == Literal:
                allowed_values = get_args(field.type_)
                allowed_types = set()
                for val in allowed_values:
                    allowed_types.add(type(val))
                allowed_types_list = list(allowed_types)
                field_type = allowed_types_list[0] if len(allowed_types) == 1 else Union[allowed_types_list]

                command_parser.add_argument(
                    f"--{name}",
                    dest=name,
                    type=field_type,
                    default=field.default,
                    choices = allowed_values,                    
                    help=field.field_info.description
                )
            else:
                command_parser.add_argument(
                    f"--{name}",
                    dest=name,
                    type=field.type_,
                    default=field.default,   
                    help=field.field_info.description
                )

    while (True):
        try:
            cmd = input("> ")
        except KeyboardInterrupt:
            # Ctrl-c exits
            break

        if cmd in ['exit','q']:
            break;

        if cmd in ['--help','help','h','?']:
            parser.print_help()
            continue

        try:
            args = vars(parser.parse_args(shlex.split(cmd)))
            args['id'] = str(len(context.history))
            invocation = Command(invocation = args)
            links = []
            if len(context.history) > 0:
                links.append(InvocationFieldLink(-1, "*", "*"))

            invoker.invoke(context, invocation.invocation, links)

        except InvalidArgs:
            print('Invalid command, use "help" to list commands')
            continue

        except SystemExit:
            continue

if __name__ == "__main__":
    invoke_cli()
