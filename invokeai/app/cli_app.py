# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import argparse
import os
import shlex
import time
from typing import (
    Any,
    Dict,
    Iterable,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel
from pydantic.fields import Field

from ..args import Args
from .invocations import *
from .invocations.baseinvocation import BaseInvocation
from .invocations.image import ImageField
from .services.events import EventServiceBase
from .services.generate_initializer import get_generate
from .services.graph import EdgeConnection, GraphExecutionState
from .services.image_storage import DiskImageStorage
from .services.invocation_queue import MemoryInvocationQueue
from .services.invocation_services import InvocationServices
from .services.invoker import Invoker
from .services.processor import DefaultInvocationProcessor
from .services.sqlite import SqliteItemStorage


class InvocationCommand(BaseModel):
    invocation: Union[BaseInvocation.get_invocations()] = Field(discriminator="type")  # type: ignore


class InvalidArgs(Exception):
    pass


def get_invocation_parser() -> argparse.ArgumentParser:
    # Create invocation parser
    parser = argparse.ArgumentParser()

    def exit(*args, **kwargs):
        raise InvalidArgs

    parser.exit = exit

    subparsers = parser.add_subparsers(dest="type")
    invocation_parsers = dict()

    # Add history parser
    history_parser = subparsers.add_parser(
        "history", help="Shows the invocation history"
    )
    history_parser.add_argument(
        "count",
        nargs="?",
        default=5,
        type=int,
        help="The number of history entries to show",
    )

    # Add default parser
    default_parser = subparsers.add_parser(
        "default", help="Define a default value for all inputs with a specified name"
    )
    default_parser.add_argument("input", type=str, help="The input field")
    default_parser.add_argument("value", help="The default value")

    default_parser = subparsers.add_parser(
        "reset_default", help="Resets a default value"
    )
    default_parser.add_argument("input", type=str, help="The input field")

    # Create subparsers for each invocation
    invocations = BaseInvocation.get_all_subclasses()
    for invocation in invocations:
        hints = get_type_hints(invocation)
        cmd_name = get_args(hints["type"])[0]
        command_parser = subparsers.add_parser(cmd_name, help=invocation.__doc__)
        invocation_parsers[cmd_name] = command_parser

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

        # Convert all fields to arguments
        fields = invocation.__fields__
        for name, field in fields.items():
            if name in ["id", "type"]:
                continue

            if get_origin(field.type_) == Literal:
                allowed_values = get_args(field.type_)
                allowed_types = set()
                for val in allowed_values:
                    allowed_types.add(type(val))
                allowed_types_list = list(allowed_types)
                field_type = allowed_types_list[0] if len(allowed_types) == 1 else Union[allowed_types_list]  # type: ignore

                command_parser.add_argument(
                    f"--{name}",
                    dest=name,
                    type=field_type,
                    default=field.default,
                    choices=allowed_values,
                    help=field.field_info.description,
                )
            else:
                command_parser.add_argument(
                    f"--{name}",
                    dest=name,
                    type=field.type_,
                    default=field.default,
                    help=field.field_info.description,
                )

    return parser


def get_invocation_command(invocation) -> str:
    fields = invocation.__fields__.items()
    type_hints = get_type_hints(type(invocation))
    command = [invocation.type]
    for name, field in fields:
        if name in ["id", "type"]:
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
                command.append(f"--{name} {field_value}")

    return " ".join(command)


def get_graph_execution_history(
    graph_execution_state: GraphExecutionState,
) -> Iterable[str]:
    """Gets the history of fully-executed invocations for a graph execution"""
    return (
        n
        for n in reversed(graph_execution_state.executed_history)
        if n in graph_execution_state.graph.nodes
    )


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

    parser = get_invocation_parser()

    # Uncomment to print out previous sessions at startup
    # print(services.session_manager.list())

    # Defaults storage
    defaults: Dict[str, Any] = dict()

    while True:
        try:
            cmd_input = input("> ")
        except KeyboardInterrupt:
            # Ctrl-c exits
            break

        if cmd_input in ["exit", "q"]:
            break

        if cmd_input in ["--help", "help", "h", "?"]:
            parser.print_help()
            continue

        try:
            # Refresh the state of the session
            session = invoker.services.graph_execution_manager.get(session.id)
            history = list(get_graph_execution_history(session))

            # Split the command for piping
            cmds = cmd_input.split("|")
            start_id = len(history)
            current_id = start_id
            new_invocations = list()
            for cmd in cmds:
                if cmd is None or cmd.strip() == "":
                    raise InvalidArgs("Empty command")

                # Parse args to create invocation
                args = vars(parser.parse_args(shlex.split(cmd.strip())))

                # Check for special commands
                # TODO: These might be better as Pydantic models, similar to the invocations
                if args["type"] == "history":
                    history_count = args["count"] or 5
                    for i in range(min(history_count, len(history))):
                        entry_id = history[-1 - i]
                        entry = session.graph.get_node(entry_id)
                        print(f"{entry_id}: {get_invocation_command(entry.invocation)}")
                    continue

                if args["type"] == "reset_default":
                    if args["input"] in defaults:
                        del defaults[args["input"]]
                    continue

                if args["type"] == "default":
                    field = args["input"]
                    field_value = args["value"]
                    defaults[field] = field_value
                    continue

                # Override defaults
                for field_name, field_default in defaults.items():
                    if field_name in args:
                        args[field_name] = field_default

                # Parse invocation
                args["id"] = current_id
                command = InvocationCommand(invocation=args)

                # Pipe previous command output (if there was a previous command)
                edges = []
                if len(history) > 0 or current_id != start_id:
                    from_id = (
                        history[0] if current_id == start_id else str(current_id - 1)
                    )
                    from_node = (
                        next(filter(lambda n: n[0].id == from_id, new_invocations))[0]
                        if current_id != start_id
                        else session.graph.get_node(from_id)
                    )
                    matching_edges = generate_matching_edges(
                        from_node, command.invocation
                    )
                    edges.extend(matching_edges)

                # Parse provided links
                if "link_node" in args and args["link_node"]:
                    for link in args["link_node"]:
                        link_node = session.graph.get_node(link)
                        matching_edges = generate_matching_edges(
                            link_node, command.invocation
                        )
                        edges.extend(matching_edges)

                if "link" in args and args["link"]:
                    for link in args["link"]:
                        edges.append(
                            (
                                EdgeConnection(node_id=link[1], field=link[0]),
                                EdgeConnection(
                                    node_id=command.invocation.id, field=link[2]
                                ),
                            )
                        )

                new_invocations.append((command.invocation, edges))

                current_id = current_id + 1

            # Command line was parsed successfully
            # Add the invocations to the session
            for invocation in new_invocations:
                session.add_node(invocation[0])
                for edge in invocation[1]:
                    session.add_edge(edge)

            # Execute all available invocations
            invoker.invoke(session, invoke_all=True)
            while not session.is_complete():
                # Wait some time
                session = invoker.services.graph_execution_manager.get(session.id)
                time.sleep(0.1)

            # Print any errors
            if session.has_error():
                for n in session.errors:
                    print(
                        f"Error in node {n} (source node {session.prepared_source_mapping[n]}): {session.errors[n]}"
                    )

                # Start a new session
                print("Creating a new session")
                session = invoker.create_execution_state()

        except InvalidArgs:
            print('Invalid command, use "help" to list commands')
            continue

        except SystemExit:
            continue

    invoker.stop()


if __name__ == "__main__":
    invoke_cli()
