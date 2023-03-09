# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import argparse
import os
import shlex
import time
from typing import (
    Union,
    get_type_hints,
)

from pydantic import BaseModel
from pydantic.fields import Field

from ..backend import Args
from .cli.commands import BaseCommand, CliContext, ExitCli, add_parsers, get_graph_execution_history
from .invocations import *
from .invocations.baseinvocation import BaseInvocation
from .services.events import EventServiceBase
from .services.generate_initializer import get_generate
from .services.graph import EdgeConnection, GraphExecutionState
from .services.image_storage import DiskImageStorage
from .services.invocation_queue import MemoryInvocationQueue
from .services.invocation_services import InvocationServices
from .services.invoker import Invoker
from .services.processor import DefaultInvocationProcessor
from .services.sqlite import SqliteItemStorage


class CliCommand(BaseModel):
    command: Union[BaseCommand.get_commands() + BaseInvocation.get_invocations()] = Field(discriminator="type")  # type: ignore


class InvalidArgs(Exception):
    pass


def add_invocation_args(command_parser):
    # Add linking capability
    command_parser.add_argument(
        "--link",
        "-l",
        action="append",
        nargs=3,
        help="A link in the format 'dest_field source_node source_field'. source_node can be relative to history (e.g. -1)",
    )

    command_parser.add_argument(
        "--link_node",
        "-ln",
        action="append",
        help="A link from all fields in the specified node. Node can be relative to history (e.g. -1)",
    )


def get_command_parser() -> argparse.ArgumentParser:
    # Create invocation parser
    parser = argparse.ArgumentParser()

    def exit(*args, **kwargs):
        raise InvalidArgs

    parser.exit = exit
    subparsers = parser.add_subparsers(dest="type")

    # Create subparsers for each invocation
    invocations = BaseInvocation.get_all_subclasses()
    add_parsers(subparsers, invocations, add_arguments=add_invocation_args)

    # Create subparsers for each command
    commands = BaseCommand.get_all_subclasses()
    add_parsers(subparsers, commands, exclude_fields=["type"])

    return parser


def generate_matching_edges(
    a: BaseInvocation, b: BaseInvocation
) -> list[tuple[EdgeConnection, EdgeConnection]]:
    """Generates all possible edges between two invocations"""
    atype = type(a)
    btype = type(b)

    aoutputtype = atype.get_output_type()

    afields = get_type_hints(aoutputtype)
    bfields = get_type_hints(btype)

    matching_fields = set(afields.keys()).intersection(bfields.keys())

    # Remove invalid fields
    invalid_fields = set(["type", "id"])
    matching_fields = matching_fields.difference(invalid_fields)

    edges = [
        (
            EdgeConnection(node_id=a.id, field=field),
            EdgeConnection(node_id=b.id, field=field),
        )
        for field in matching_fields
    ]
    return edges


class SessionError(Exception):
    """Raised when a session error has occurred"""
    pass


def invoke_all(context: CliContext):
    """Runs all invocations in the specified session"""
    context.invoker.invoke(context.session, invoke_all=True)
    while not context.session.is_complete():
        # Wait some time
        session = context.get_session()
        time.sleep(0.1)

    # Print any errors
    if context.session.has_error():
        for n in context.session.errors:
            print(
                f"Error in node {n} (source node {context.session.prepared_source_mapping[n]}): {session.errors[n]}"
            )
        
        raise SessionError()


def invoke_cli():
    args = Args()
    config = args.parse_args()

    generate = get_generate(args, config)

    # NOTE: load model on first use, uncomment to load at startup
    # TODO: Make this a config option?
    # generate.load_model()

    events = EventServiceBase()

    output_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../outputs")
    )

    # TODO: build a file/path manager?
    db_location = os.path.join(output_folder, "invokeai.db")

    services = InvocationServices(
        generate=generate,
        events=events,
        images=DiskImageStorage(output_folder),
        queue=MemoryInvocationQueue(),
        graph_execution_manager=SqliteItemStorage[GraphExecutionState](
            filename=db_location, table_name="graph_executions"
        ),
        processor=DefaultInvocationProcessor(),
    )

    invoker = Invoker(services)
    session: GraphExecutionState = invoker.create_execution_state()
    parser = get_command_parser()

    # Uncomment to print out previous sessions at startup
    # print(services.session_manager.list())

    context = CliContext(invoker, session, parser)

    while True:
        try:
            cmd_input = input("> ")
        except KeyboardInterrupt:
            # Ctrl-c exits
            break

        try:
            # Refresh the state of the session
            history = list(get_graph_execution_history(context.session))

            # Split the command for piping
            cmds = cmd_input.split("|")
            start_id = len(history)
            current_id = start_id
            new_invocations = list()
            for cmd in cmds:
                if cmd is None or cmd.strip() == "":
                    raise InvalidArgs("Empty command")

                # Parse args to create invocation
                args = vars(context.parser.parse_args(shlex.split(cmd.strip())))

                # Override defaults
                for field_name, field_default in context.defaults.items():
                    if field_name in args:
                        args[field_name] = field_default

                # Parse invocation
                args["id"] = current_id
                command = CliCommand(command=args)

                # Run any CLI commands immediately
                if isinstance(command.command, BaseCommand):
                    # Invoke all current nodes to preserve operation order
                    invoke_all(context)

                    # Run the command
                    command.command.run(context)
                    continue

                # Pipe previous command output (if there was a previous command)
                edges = []
                if len(history) > 0 or current_id != start_id:
                    from_id = (
                        history[0] if current_id == start_id else str(current_id - 1)
                    )
                    from_node = (
                        next(filter(lambda n: n[0].id == from_id, new_invocations))[0]
                        if current_id != start_id
                        else context.session.graph.get_node(from_id)
                    )
                    matching_edges = generate_matching_edges(
                        from_node, command.command
                    )
                    edges.extend(matching_edges)

                # Parse provided links
                if "link_node" in args and args["link_node"]:
                    for link in args["link_node"]:
                        link_node = context.session.graph.get_node(link)
                        matching_edges = generate_matching_edges(
                            link_node, command.command
                        )
                        matching_destinations = [e[1] for e in matching_edges]
                        edges = [e for e in edges if e[1] not in matching_destinations]
                        edges.extend(matching_edges)

                if "link" in args and args["link"]:
                    for link in args["link"]:
                        edges = [e for e in edges if e[1].node_id != command.command.id and e[1].field != link[2]]
                        edges.append(
                            (
                                EdgeConnection(node_id=link[1], field=link[0]),
                                EdgeConnection(
                                    node_id=command.command.id, field=link[2]
                                ),
                            )
                        )

                new_invocations.append((command.command, edges))

                current_id = current_id + 1

                # Add the node to the session
                context.session.add_node(command.command)
                for edge in edges:
                    print(edge)
                    context.session.add_edge(edge)

            # Execute all remaining nodes
            invoke_all(context)

        except InvalidArgs:
            print('Invalid command, use "help" to list commands')
            continue

        except SessionError:
            # Start a new session
            print("Session error: creating a new session")
            context.session = context.invoker.create_execution_state()

        except ExitCli:
            break

        except SystemExit:
            continue

    invoker.stop()


if __name__ == "__main__":
    invoke_cli()
