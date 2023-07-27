# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from abc import ABC, abstractmethod
import argparse
from typing import Any, Callable, Iterable, Literal, Union, get_args, get_origin, get_type_hints
from pydantic import BaseModel, Field
import networkx as nx
import matplotlib.pyplot as plt

import invokeai.backend.util.logging as logger
from ..invocations.baseinvocation import BaseInvocation
from ..invocations.image import ImageField
from ..services.graph import GraphExecutionState, LibraryGraph, Edge
from ..services.invoker import Invoker


def add_field_argument(command_parser, name: str, field, default_override=None):
    default = (
        default_override
        if default_override is not None
        else field.default
        if field.default_factory is None
        else field.default_factory()
    )
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
            default=default,
            choices=allowed_values,
            help=field.field_info.description,
        )
    else:
        command_parser.add_argument(
            f"--{name}",
            dest=name,
            type=field.type_,
            default=default,
            help=field.field_info.description,
        )


def add_parsers(
    subparsers,
    commands: list[type],
    command_field: str = "type",
    exclude_fields: list[str] = ["id", "type"],
    add_arguments: Union[Callable[[argparse.ArgumentParser], None], None] = None,
):
    """Adds parsers for each command to the subparsers"""

    # Create subparsers for each command
    for command in commands:
        hints = get_type_hints(command)
        cmd_name = get_args(hints[command_field])[0]
        command_parser = subparsers.add_parser(cmd_name, help=command.__doc__)

        if add_arguments is not None:
            add_arguments(command_parser)

        # Convert all fields to arguments
        fields = command.__fields__  # type: ignore
        for name, field in fields.items():
            if name in exclude_fields:
                continue

            add_field_argument(command_parser, name, field)


def add_graph_parsers(
    subparsers, graphs: list[LibraryGraph], add_arguments: Union[Callable[[argparse.ArgumentParser], None], None] = None
):
    for graph in graphs:
        command_parser = subparsers.add_parser(graph.name, help=graph.description)

        if add_arguments is not None:
            add_arguments(command_parser)

        # Add arguments for inputs
        for exposed_input in graph.exposed_inputs:
            node = graph.graph.get_node(exposed_input.node_path)
            field = node.__fields__[exposed_input.field]
            default_override = getattr(node, exposed_input.field)
            add_field_argument(command_parser, exposed_input.alias, field, default_override)


class CliContext:
    invoker: Invoker
    session: GraphExecutionState
    parser: argparse.ArgumentParser
    defaults: dict[str, Any]
    graph_nodes: dict[str, str]
    nodes_added: list[str]

    def __init__(self, invoker: Invoker, session: GraphExecutionState, parser: argparse.ArgumentParser):
        self.invoker = invoker
        self.session = session
        self.parser = parser
        self.defaults = dict()
        self.graph_nodes = dict()
        self.nodes_added = list()

    def get_session(self):
        self.session = self.invoker.services.graph_execution_manager.get(self.session.id)
        return self.session

    def reset(self):
        self.session = self.invoker.create_execution_state()
        self.graph_nodes = dict()
        self.nodes_added = list()
        # Leave defaults unchanged

    def add_node(self, node: BaseInvocation):
        self.get_session()
        self.session.graph.add_node(node)
        self.nodes_added.append(node.id)
        self.invoker.services.graph_execution_manager.set(self.session)

    def add_edge(self, edge: Edge):
        self.get_session()
        self.session.add_edge(edge)
        self.invoker.services.graph_execution_manager.set(self.session)


class ExitCli(Exception):
    """Exception to exit the CLI"""

    pass


class BaseCommand(ABC, BaseModel):
    """A CLI command"""

    # All commands must include a type name like this:
    # type: Literal['your_command_name'] = 'your_command_name'

    @classmethod
    def get_all_subclasses(cls):
        subclasses = []
        toprocess = [cls]
        while len(toprocess) > 0:
            next = toprocess.pop(0)
            next_subclasses = next.__subclasses__()
            subclasses.extend(next_subclasses)
            toprocess.extend(next_subclasses)
        return subclasses

    @classmethod
    def get_commands(cls):
        return tuple(BaseCommand.get_all_subclasses())

    @classmethod
    def get_commands_map(cls):
        # Get the type strings out of the literals and into a dictionary
        return dict(map(lambda t: (get_args(get_type_hints(t)["type"])[0], t), BaseCommand.get_all_subclasses()))

    @abstractmethod
    def run(self, context: CliContext) -> None:
        """Run the command. Raise ExitCli to exit."""
        pass


class ExitCommand(BaseCommand):
    """Exits the CLI"""

    type: Literal["exit"] = "exit"

    def run(self, context: CliContext) -> None:
        raise ExitCli()


class HelpCommand(BaseCommand):
    """Shows help"""

    type: Literal["help"] = "help"

    def run(self, context: CliContext) -> None:
        context.parser.print_help()


def get_graph_execution_history(
    graph_execution_state: GraphExecutionState,
) -> Iterable[str]:
    """Gets the history of fully-executed invocations for a graph execution"""
    return (n for n in reversed(graph_execution_state.executed_history) if n in graph_execution_state.graph.nodes)


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


class HistoryCommand(BaseCommand):
    """Shows the invocation history"""

    type: Literal["history"] = "history"

    # Inputs
    # fmt: off
    count: int = Field(default=5, gt=0, description="The number of history entries to show")
    # fmt: on

    def run(self, context: CliContext) -> None:
        history = list(get_graph_execution_history(context.get_session()))
        for i in range(min(self.count, len(history))):
            entry_id = history[-1 - i]
            entry = context.get_session().graph.get_node(entry_id)
            logger.info(f"{entry_id}: {get_invocation_command(entry)}")


class SetDefaultCommand(BaseCommand):
    """Sets a default value for a field"""

    type: Literal["default"] = "default"

    # Inputs
    # fmt: off
    field: str = Field(description="The field to set the default for")
    value: str = Field(description="The value to set the default to, or None to clear the default")
    # fmt: on

    def run(self, context: CliContext) -> None:
        if self.value is None:
            if self.field in context.defaults:
                del context.defaults[self.field]
        else:
            context.defaults[self.field] = self.value


class DrawGraphCommand(BaseCommand):
    """Debugs a graph"""

    type: Literal["draw_graph"] = "draw_graph"

    def run(self, context: CliContext) -> None:
        session: GraphExecutionState = context.invoker.services.graph_execution_manager.get(context.session.id)
        nxgraph = session.graph.nx_graph_flat()

        # Draw the networkx graph
        plt.figure(figsize=(20, 20))
        pos = nx.spectral_layout(nxgraph)
        nx.draw_networkx_nodes(nxgraph, pos, node_size=1000)
        nx.draw_networkx_edges(nxgraph, pos, width=2)
        nx.draw_networkx_labels(nxgraph, pos, font_size=20, font_family="sans-serif")
        plt.axis("off")
        plt.show()


class DrawExecutionGraphCommand(BaseCommand):
    """Debugs an execution graph"""

    type: Literal["draw_xgraph"] = "draw_xgraph"

    def run(self, context: CliContext) -> None:
        session: GraphExecutionState = context.invoker.services.graph_execution_manager.get(context.session.id)
        nxgraph = session.execution_graph.nx_graph_flat()

        # Draw the networkx graph
        plt.figure(figsize=(20, 20))
        pos = nx.spectral_layout(nxgraph)
        nx.draw_networkx_nodes(nxgraph, pos, node_size=1000)
        nx.draw_networkx_edges(nxgraph, pos, width=2)
        nx.draw_networkx_labels(nxgraph, pos, font_size=20, font_family="sans-serif")
        plt.axis("off")
        plt.show()


class SortedHelpFormatter(argparse.HelpFormatter):
    def _iter_indented_subactions(self, action):
        try:
            get_subactions = action._get_subactions
        except AttributeError:
            pass
        else:
            self._indent()
            if isinstance(action, argparse._SubParsersAction):
                for subaction in sorted(get_subactions(), key=lambda x: x.dest):
                    yield subaction
            else:
                for subaction in get_subactions():
                    yield subaction
                self._dedent()
