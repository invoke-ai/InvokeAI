# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import argparse
import re
import shlex
import sys
import time
from typing import Union, get_type_hints, Optional

from pydantic import BaseModel, ValidationError
from pydantic.fields import Field

# This should come early so that the logger can pick up its configuration options
from .services.config import InvokeAIAppConfig
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.version.invokeai_version import __version__


from invokeai.app.services.board_image_record_storage import (
    SqliteBoardImageRecordStorage,
)
from invokeai.app.services.board_images import (
    BoardImagesService,
    BoardImagesServiceDependencies,
)
from invokeai.app.services.board_record_storage import SqliteBoardRecordStorage
from invokeai.app.services.boards import BoardService, BoardServiceDependencies
from invokeai.app.services.image_record_storage import SqliteImageRecordStorage
from invokeai.app.services.images import ImageService, ImageServiceDependencies
from invokeai.app.services.resource_name import SimpleNameService
from invokeai.app.services.urls import LocalUrlService
from invokeai.app.services.invocation_stats import InvocationStatsService
from .services.default_graphs import default_text_to_image_graph_id, create_system_graphs
from .services.latent_storage import DiskLatentsStorage, ForwardCacheLatentsStorage

from .cli.commands import BaseCommand, CliContext, ExitCli, SortedHelpFormatter, add_graph_parsers, add_parsers
from .cli.completer import set_autocompleter
from .invocations.baseinvocation import BaseInvocation
from .services.events import EventServiceBase
from .services.graph import (
    Edge,
    EdgeConnection,
    GraphExecutionState,
    GraphInvocation,
    LibraryGraph,
    are_connection_types_compatible,
)
from .services.image_file_storage import DiskImageFileStorage
from .services.invocation_queue import MemoryInvocationQueue
from .services.invocation_services import InvocationServices
from .services.invoker import Invoker
from .services.model_manager_service import ModelManagerService
from .services.processor import DefaultInvocationProcessor
from .services.sqlite import SqliteItemStorage

import torch
import invokeai.backend.util.hotfixes  # noqa: F401 (monkeypatching on import)

if torch.backends.mps.is_available():
    import invokeai.backend.util.mps_fixes  # noqa: F401 (monkeypatching on import)


config = InvokeAIAppConfig.get_config()
config.parse_args()
logger = InvokeAILogger().getLogger(config=config)


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
        help="A link in the format 'source_node source_field dest_field'. source_node can be relative to history (e.g. -1)",
    )

    command_parser.add_argument(
        "--link_node",
        "-ln",
        action="append",
        help="A link from all fields in the specified node. Node can be relative to history (e.g. -1)",
    )


def get_command_parser(services: InvocationServices) -> argparse.ArgumentParser:
    # Create invocation parser
    parser = argparse.ArgumentParser(formatter_class=SortedHelpFormatter)

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

    # Create subparsers for exposed CLI graphs
    # TODO: add a way to identify these graphs
    text_to_image = services.graph_library.get(default_text_to_image_graph_id)
    add_graph_parsers(subparsers, [text_to_image], add_arguments=add_invocation_args)

    return parser


class NodeField:
    alias: str
    node_path: str
    field: str
    field_type: type

    def __init__(self, alias: str, node_path: str, field: str, field_type: type):
        self.alias = alias
        self.node_path = node_path
        self.field = field
        self.field_type = field_type


def fields_from_type_hints(hints: dict[str, type], node_path: str) -> dict[str, NodeField]:
    return {k: NodeField(alias=k, node_path=node_path, field=k, field_type=v) for k, v in hints.items()}


def get_node_input_field(graph: LibraryGraph, field_alias: str, node_id: str) -> NodeField:
    """Gets the node field for the specified field alias"""
    exposed_input = next(e for e in graph.exposed_inputs if e.alias == field_alias)
    node_type = type(graph.graph.get_node(exposed_input.node_path))
    return NodeField(
        alias=exposed_input.alias,
        node_path=f"{node_id}.{exposed_input.node_path}",
        field=exposed_input.field,
        field_type=get_type_hints(node_type)[exposed_input.field],
    )


def get_node_output_field(graph: LibraryGraph, field_alias: str, node_id: str) -> NodeField:
    """Gets the node field for the specified field alias"""
    exposed_output = next(e for e in graph.exposed_outputs if e.alias == field_alias)
    node_type = type(graph.graph.get_node(exposed_output.node_path))
    node_output_type = node_type.get_output_type()
    return NodeField(
        alias=exposed_output.alias,
        node_path=f"{node_id}.{exposed_output.node_path}",
        field=exposed_output.field,
        field_type=get_type_hints(node_output_type)[exposed_output.field],
    )


def get_node_inputs(invocation: BaseInvocation, context: CliContext) -> dict[str, NodeField]:
    """Gets the inputs for the specified invocation from the context"""
    node_type = type(invocation)
    if node_type is not GraphInvocation:
        return fields_from_type_hints(get_type_hints(node_type), invocation.id)
    else:
        graph: LibraryGraph = context.invoker.services.graph_library.get(context.graph_nodes[invocation.id])
        return {e.alias: get_node_input_field(graph, e.alias, invocation.id) for e in graph.exposed_inputs}


def get_node_outputs(invocation: BaseInvocation, context: CliContext) -> dict[str, NodeField]:
    """Gets the outputs for the specified invocation from the context"""
    node_type = type(invocation)
    if node_type is not GraphInvocation:
        return fields_from_type_hints(get_type_hints(node_type.get_output_type()), invocation.id)
    else:
        graph: LibraryGraph = context.invoker.services.graph_library.get(context.graph_nodes[invocation.id])
        return {e.alias: get_node_output_field(graph, e.alias, invocation.id) for e in graph.exposed_outputs}


def generate_matching_edges(a: BaseInvocation, b: BaseInvocation, context: CliContext) -> list[Edge]:
    """Generates all possible edges between two invocations"""
    afields = get_node_outputs(a, context)
    bfields = get_node_inputs(b, context)

    matching_fields = set(afields.keys()).intersection(bfields.keys())

    # Remove invalid fields
    invalid_fields = set(["type", "id"])
    matching_fields = matching_fields.difference(invalid_fields)

    # Validate types
    matching_fields = [
        f for f in matching_fields if are_connection_types_compatible(afields[f].field_type, bfields[f].field_type)
    ]

    edges = [
        Edge(
            source=EdgeConnection(node_id=afields[alias].node_path, field=afields[alias].field),
            destination=EdgeConnection(node_id=bfields[alias].node_path, field=bfields[alias].field),
        )
        for alias in matching_fields
    ]
    return edges


class SessionError(Exception):
    """Raised when a session error has occurred"""

    pass


def invoke_all(context: CliContext):
    """Runs all invocations in the specified session"""
    context.invoker.invoke(context.session, invoke_all=True)
    while not context.get_session().is_complete():
        # Wait some time
        time.sleep(0.1)

    # Print any errors
    if context.session.has_error():
        for n in context.session.errors:
            context.invoker.services.logger.error(
                f"Error in node {n} (source node {context.session.prepared_source_mapping[n]}): {context.session.errors[n]}"
            )

        raise SessionError()


def invoke_cli():
    logger.info(f"InvokeAI version {__version__}")
    # get the optional list of invocations to execute on the command line
    parser = config.get_parser()
    parser.add_argument("commands", nargs="*")
    invocation_commands = parser.parse_args().commands

    # get the optional file to read commands from.
    # Simplest is to use it for STDIN
    if infile := config.from_file:
        sys.stdin = open(infile, "r")

    model_manager = ModelManagerService(config, logger)

    events = EventServiceBase()
    output_folder = config.output_path

    # TODO: build a file/path manager?
    if config.use_memory_db:
        db_location = ":memory:"
    else:
        db_location = config.db_path
        db_location.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f'InvokeAI database location is "{db_location}"')

    graph_execution_manager = SqliteItemStorage[GraphExecutionState](
        filename=db_location, table_name="graph_executions"
    )

    urls = LocalUrlService()
    image_record_storage = SqliteImageRecordStorage(db_location)
    image_file_storage = DiskImageFileStorage(f"{output_folder}/images")
    names = SimpleNameService()

    board_record_storage = SqliteBoardRecordStorage(db_location)
    board_image_record_storage = SqliteBoardImageRecordStorage(db_location)

    boards = BoardService(
        services=BoardServiceDependencies(
            board_image_record_storage=board_image_record_storage,
            board_record_storage=board_record_storage,
            image_record_storage=image_record_storage,
            url=urls,
            logger=logger,
        )
    )

    board_images = BoardImagesService(
        services=BoardImagesServiceDependencies(
            board_image_record_storage=board_image_record_storage,
            board_record_storage=board_record_storage,
            image_record_storage=image_record_storage,
            url=urls,
            logger=logger,
        )
    )

    images = ImageService(
        services=ImageServiceDependencies(
            board_image_record_storage=board_image_record_storage,
            image_record_storage=image_record_storage,
            image_file_storage=image_file_storage,
            url=urls,
            logger=logger,
            names=names,
            graph_execution_manager=graph_execution_manager,
        )
    )

    services = InvocationServices(
        model_manager=model_manager,
        events=events,
        latents=ForwardCacheLatentsStorage(DiskLatentsStorage(f"{output_folder}/latents")),
        images=images,
        boards=boards,
        board_images=board_images,
        queue=MemoryInvocationQueue(),
        graph_library=SqliteItemStorage[LibraryGraph](filename=db_location, table_name="graphs"),
        graph_execution_manager=graph_execution_manager,
        processor=DefaultInvocationProcessor(),
        performance_statistics=InvocationStatsService(graph_execution_manager),
        logger=logger,
        configuration=config,
    )

    system_graphs = create_system_graphs(services.graph_library)
    system_graph_names = set([g.name for g in system_graphs])
    set_autocompleter(services)

    invoker = Invoker(services)
    session: GraphExecutionState = invoker.create_execution_state()
    parser = get_command_parser(services)

    re_negid = re.compile("^-[0-9]+$")

    # Uncomment to print out previous sessions at startup
    # print(services.session_manager.list())

    context = CliContext(invoker, session, parser)
    set_autocompleter(services)

    command_line_args_exist = len(invocation_commands) > 0
    done = False

    while not done:
        try:
            if command_line_args_exist:
                cmd_input = invocation_commands.pop(0)
                done = len(invocation_commands) == 0
            else:
                cmd_input = input("invoke> ")
        except (KeyboardInterrupt, EOFError):
            # Ctrl-c exits
            break

        try:
            # Refresh the state of the session
            # history = list(get_graph_execution_history(context.session))
            history = list(reversed(context.nodes_added))

            # Split the command for piping
            cmds = cmd_input.split("|")
            start_id = len(context.nodes_added)
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
                command: CliCommand = None  # type:ignore
                system_graph: Optional[LibraryGraph] = None
                if args["type"] in system_graph_names:
                    system_graph = next(filter(lambda g: g.name == args["type"], system_graphs))
                    invocation = GraphInvocation(graph=system_graph.graph, id=str(current_id))
                    for exposed_input in system_graph.exposed_inputs:
                        if exposed_input.alias in args:
                            node = invocation.graph.get_node(exposed_input.node_path)
                            field = exposed_input.field
                            setattr(node, field, args[exposed_input.alias])
                    command = CliCommand(command=invocation)
                    context.graph_nodes[invocation.id] = system_graph.id
                else:
                    args["id"] = current_id
                    command = CliCommand(command=args)

                if command is None:
                    continue

                # Run any CLI commands immediately
                if isinstance(command.command, BaseCommand):
                    # Invoke all current nodes to preserve operation order
                    invoke_all(context)

                    # Run the command
                    command.command.run(context)
                    continue

                # TODO: handle linking with library graphs
                # Pipe previous command output (if there was a previous command)
                edges: list[Edge] = list()
                if len(history) > 0 or current_id != start_id:
                    from_id = history[0] if current_id == start_id else str(current_id - 1)
                    from_node = (
                        next(filter(lambda n: n[0].id == from_id, new_invocations))[0]
                        if current_id != start_id
                        else context.session.graph.get_node(from_id)
                    )
                    matching_edges = generate_matching_edges(from_node, command.command, context)
                    edges.extend(matching_edges)

                # Parse provided links
                if "link_node" in args and args["link_node"]:
                    for link in args["link_node"]:
                        node_id = link
                        if re_negid.match(node_id):
                            node_id = str(current_id + int(node_id))

                        link_node = context.session.graph.get_node(node_id)
                        matching_edges = generate_matching_edges(link_node, command.command, context)
                        matching_destinations = [e.destination for e in matching_edges]
                        edges = [e for e in edges if e.destination not in matching_destinations]
                        edges.extend(matching_edges)

                if "link" in args and args["link"]:
                    for link in args["link"]:
                        edges = [
                            e
                            for e in edges
                            if e.destination.node_id != command.command.id or e.destination.field != link[2]
                        ]

                        node_id = link[0]
                        if re_negid.match(node_id):
                            node_id = str(current_id + int(node_id))

                        # TODO: handle missing input/output
                        node_output = get_node_outputs(context.session.graph.get_node(node_id), context)[link[1]]
                        node_input = get_node_inputs(command.command, context)[link[2]]

                        edges.append(
                            Edge(
                                source=EdgeConnection(node_id=node_output.node_path, field=node_output.field),
                                destination=EdgeConnection(node_id=node_input.node_path, field=node_input.field),
                            )
                        )

                new_invocations.append((command.command, edges))

                current_id = current_id + 1

                # Add the node to the session
                context.add_node(command.command)
                for edge in edges:
                    print(edge)
                    context.add_edge(edge)

            # Execute all remaining nodes
            invoke_all(context)

        except InvalidArgs:
            invoker.services.logger.warning('Invalid command, use "help" to list commands')
            continue

        except ValidationError:
            invoker.services.logger.warning('Invalid command arguments, run "<command> --help" for summary')

        except SessionError:
            # Start a new session
            invoker.services.logger.warning("Session error: creating a new session")
            context.reset()

        except ExitCli:
            break

        except SystemExit:
            continue

    invoker.stop()


if __name__ == "__main__":
    if config.version:
        print(f"InvokeAI version {__version__}")
    else:
        invoke_cli()
