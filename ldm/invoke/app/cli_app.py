# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import argparse
import shlex
import os
from typing import Any, Dict, Literal, Union, get_args, get_origin, get_type_hints
from pydantic import BaseModel
from pydantic.fields import Field

from ldm.invoke.app.invocations.image import ImageField
from .services.image_storage import DiskImageStorage
from .services.session_manager import DiskSessionManager
from .services.invocation_queue import MemoryInvocationQueue
from .invocations.baseinvocation import BaseInvocation
from .services.invocation_session import InvocationFieldLink
from .services.invocation_services import InvocationServices
from .services.invoker import Invoker, InvokerServices
from .invocations import *
from ..args import Args
from ...generate import Generate
from .services.events import EventServiceBase


class Command(BaseModel):
    invocation: Union[BaseInvocation.get_invocations()] = Field(discriminator="type")


class InvalidArgs(Exception):
    pass


def get_invocation_parser() -> argparse.ArgumentParser:

    # Create invocation parser
    parser = argparse.ArgumentParser()
    def exit(*args, **kwargs):
        raise InvalidArgs
    parser.exit = exit

    subparsers = parser.add_subparsers(dest='type')
    invocation_parsers = dict()

    # Add history parser
    history_parser = subparsers.add_parser('history', help="Shows the invocation history")
    history_parser.add_argument('count', nargs='?', default=5, type=int, help="The number of history entries to show")

    # Add default parser
    default_parser = subparsers.add_parser('default', help="Define a default value for all inputs with a specified name")
    default_parser.add_argument('input', type=str, help="The input field")
    default_parser.add_argument('value', help="The default value")
    
    default_parser = subparsers.add_parser('reset_default', help="Resets a default value")
    default_parser.add_argument('input', type=str, help="The input field")

    # Create subparsers for each invocation
    invocations = BaseInvocation.get_all_subclasses()
    for invocation in invocations:
        hints = get_type_hints(invocation)
        cmd_name = get_args(hints['type'])[0]
        command_parser = subparsers.add_parser(cmd_name, help=invocation.__doc__)
        invocation_parsers[cmd_name] = command_parser

        # Add linking capability
        command_parser.add_argument('--link', '-l', action='append', nargs=3,
            help="A link in the format 'dest_field source_node source_field'. source_node can be relative to history (e.g. -1)")

        command_parser.add_argument('--link_node', '-ln', action='append',
            help="A link from all fields in the specified node. Node can be relative to history (e.g. -1)")

        # Convert all fields to arguments
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

    return parser


def get_invocation_command(invocation) -> str:
    fields = invocation.__fields__.items()
    type_hints = get_type_hints(type(invocation))
    command = [invocation.type]
    for name,field in fields:
        if name in ['id', 'type']:
            continue

        # TODO: add links

        # Skip image fields when serializing command
        type_hint = type_hints.get(name) or None
        if type_hint is ImageField or ImageField in get_args(type_hint):
            continue

        field_value = getattr(invocation, name)
        field_default = field.default
        if field_value != field_default:
            if type_hint is str or str in get_args(type_hint):
                command.append(f'--{name} "{field_value}"')
            else:
                command.append(f'--{name} {field_value}')
        
    return ' '.join(command)


def invoke_cli():
    args = Args()
    config = args.parse_args()

    # TODO: this should move either inside ApiDependencies, or to their own invocations (or...their own services?)
    # Loading Face Restoration and ESRGAN Modules
    try:
        gfpgan, codeformer, esrgan = None, None, None
        if config.restore or config.esrgan:
            from ldm.invoke.restoration import Restoration
            restoration = Restoration()
            if config.restore:
                gfpgan, codeformer = restoration.load_face_restore_models(config.gfpgan_dir, config.gfpgan_model_path)
            else:
                print('>> Face restoration disabled')
            if config.esrgan:
                esrgan = restoration.load_esrgan(config.esrgan_bg_tile)
            else:
                print('>> Upscaling disabled')
        else:
            print('>> Face restoration and upscaling disabled')
    except (ModuleNotFoundError, ImportError):
        print('>> You may need to install the ESRGAN and/or GFPGAN modules')

    # Create Generate
    generate = Generate(
        model          = config.model,
        sampler_name   = config.sampler_name,
        embedding_path = config.embedding_path,
        full_precision = config.full_precision,
        gfpgan         = gfpgan,
        codeformer     = codeformer,
        esrgan         = esrgan
    )
    generate.free_gpu_mem = config.free_gpu_mem

    # NOTE: load model on first use, uncomment to load at startup
    # TODO: Make this a config option?
    #generate.load_model()

    events = EventServiceBase()

    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../outputs'))

    services = InvocationServices(
        generate = generate,
        events = events,
        images          = DiskImageStorage(output_folder)
    )

    invoker_services = InvokerServices(
        queue           = MemoryInvocationQueue(),
        session_manager = DiskSessionManager(output_folder)
    )

    invoker = Invoker(services, invoker_services)
    session = invoker.create_session()
    
    parser = get_invocation_parser()

    # Uncomment to print out previous sessions at startup
    # print(invoker_services.session_manager.list())

    # Defaults storage
    defaults: Dict[str, Any] = dict()

    while True:
        try:
            cmd_input = input("> ")
        except KeyboardInterrupt:
            # Ctrl-c exits
            break

        if cmd_input in ['exit','q']:
            break;

        if cmd_input in ['--help','help','h','?']:
            parser.print_help()
            continue

        try:
            # Split the command for piping
            cmds = cmd_input.split('|')
            start_id = len(session.history)
            current_id = start_id
            new_invocations = list()
            for cmd in cmds:
                # Parse args to create invocation
                args = vars(parser.parse_args(shlex.split(cmd.strip())))

                # Check for special commands
                if args['type'] == 'history':
                    history_count = args['count'] or 5
                    for i in range(min(history_count, len(session.history))):
                        entry_id = session.history[-1 - i]
                        entry = session.invocation_results[entry_id]
                        output_names = set(entry.outputs.__fields__.keys()).difference(set(['type']))
                        outputs = ', '.join(output_names)
                        print(f'{entry_id}: {get_invocation_command(entry.invocation)} => {outputs}')
                    continue

                if args['type'] == 'reset_default':
                    if args['input'] in defaults:
                        del defaults[args['input']]
                    continue

                if args['type'] == 'default':
                    field = args['input']
                    field_value = args['value']
                    defaults[field] = field_value
                    continue

                # Override defaults
                for field_name,field_default in defaults.items():
                    if field_name in args:
                        args[field_name] = field_default

                # Parse invocation
                args['id'] = current_id
                command = Command(invocation = args)

                # Pipe previous command output (if there was a previous command)
                links = []
                if len(session.history) > 0 or current_id != start_id:
                    from_id = -1 if current_id == start_id else str(current_id - 1)
                    links.append(InvocationFieldLink(
                        from_node_id=from_id,
                        from_field = "*",
                        to_field = "*"))
                
                # Parse provided links
                if 'link_node' in args and args['link_node']:
                    for link in args['link_node']:
                        links.append(InvocationFieldLink(
                            from_node_id = link,
                            from_field = "*",
                            to_field = "*"
                        ))
                
                if 'link' in args and args['link']:
                    for link in args['link']:
                        links.append(InvocationFieldLink(
                            from_field = link[0],
                            from_node_id = link[1],
                            to_field = link[2]
                        ))

                new_invocations.append((command.invocation, links))

                current_id = current_id + 1

            # Command line was parsed successfully
            # Add the invocations to the session
            for invocation in new_invocations:
                session.add_invocation(invocation[0], invocation[1])

            # Execute all available invocations
            invoker.invoke(session, invoke_all = True)
            session.wait_for_all()

        except InvalidArgs:
            print('Invalid command, use "help" to list commands')
            continue

        except SystemExit:
            continue
    
    invoker.stop()


if __name__ == "__main__":
    invoke_cli()
