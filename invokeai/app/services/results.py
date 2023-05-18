from enum import Enum

from abc import ABC, abstractmethod
import json
import sqlite3
from threading import Lock
from typing import Any, Union

import networkx as nx

from pydantic import BaseModel, Field, parse_obj_as, parse_raw_as
from invokeai.app.invocations.image import ImageOutput
from invokeai.app.services.graph import Edge, GraphExecutionState
from invokeai.app.invocations.latent import LatentsOutput
from invokeai.app.services.item_storage import PaginatedResults
from invokeai.app.util.misc import get_timestamp


class ResultType(str, Enum):
    image_output = "image_output"
    latents_output = "latents_output"


class Result(BaseModel):
    """A session result"""

    id: str = Field(description="Result ID")
    session_id: str = Field(description="Session ID")
    node_id: str = Field(description="Node ID")
    data: Union[LatentsOutput, ImageOutput] = Field(description="The result data")


class ResultWithSession(BaseModel):
    """A result with its session"""

    result: Result = Field(description="The result")
    session: GraphExecutionState = Field(description="The session")


# Create a directed graph
from typing import Any, TypedDict, Union
from networkx import DiGraph
import networkx as nx
import json


# We need to use a loose class for nodes to allow for graceful parsing - we cannot use the stricter
# model used by the system, because we may be a graph in an old format. We can, however, use the 
# Edge model, because the edge format does not change.
class LooseGraph(BaseModel):
    id: str
    nodes: dict[str, dict[str, Any]]
    edges: list[Edge]


# An intermediate type used during parsing
class NearestAncestor(TypedDict):
    node_id: str
    metadata: dict[str, Any]


# The ancestor types that contain the core metadata
ANCESTOR_TYPES = ['t2l', 'l2l']

# The core metadata parameters in the ancestor types
ANCESTOR_PARAMS = ['steps', 'model', 'cfg_scale', 'scheduler', 'strength']

# The core metadata parameters in the noise node
NOISE_FIELDS = ['seed', 'width', 'height']

# Find nearest t2l or l2l ancestor from a given l2i node
def find_nearest_ancestor(G: DiGraph, node_id: str) -> Union[NearestAncestor, None]:
    """Returns metadata for the nearest ancestor of a given node.

    Parameters:
    G (DiGraph): A directed graph.
    node_id (str): The ID of the starting node.

    Returns:
    NearestAncestor | None: An object with the ID and metadata of the nearest ancestor.
    """

    # Retrieve the node from the graph
    node = G.nodes[node_id]

    # If the node type is one of the core metadata node types, gather necessary metadata and return
    if node.get('type') in ANCESTOR_TYPES:
        parsed_metadata = {param: val for param, val in node.items() if param in ANCESTOR_PARAMS}
        return NearestAncestor(node_id=node_id, metadata=parsed_metadata)
        

    # Else, look for the ancestor in the predecessor nodes
    for predecessor in G.predecessors(node_id):
        result = find_nearest_ancestor(G, predecessor)
        if result:
            return result
        
    # If there are no valid ancestors, return None
    return None


def get_additional_metadata(graph: LooseGraph, node_id: str) -> Union[dict[str, Any], None]:
    """Collects additional metadata from nodes connected to a given node.

    Parameters:
    graph (LooseGraph): The graph.
    node_id (str): The ID of the node.

    Returns:
    dict | None: A dictionary containing additional metadata.
    """

    metadata = {}

    # Iterate over all edges in the graph
    for edge in graph.edges:
        dest_node_id = edge.destination.node_id
        dest_field =  edge.destination.field
        source_node = graph.nodes[edge.source.node_id]
        
        # If the destination node ID matches the given node ID, gather necessary metadata
        if dest_node_id == node_id:
            # If the destination field is 'positive_conditioning', add the 'prompt' from the source node
            if dest_field == 'positive_conditioning':
                metadata['positive_conditioning'] = source_node.get('prompt')
            # If the destination field is 'negative_conditioning', add the 'prompt' from the source node
            if dest_field == 'negative_conditioning':
                metadata['negative_conditioning'] = source_node.get('prompt')
            # If the destination field is 'noise', add the core noise fields from the source node
            if dest_field == 'noise':
                for field in NOISE_FIELDS:
                    metadata[field] = source_node.get(field)
    return metadata

def build_core_metadata(graph_raw: str, node_id: str) -> Union[dict, None]:
    """Builds the core metadata for a given node.

    Parameters:
    graph_raw (str): The graph structure as a raw string.
    node_id (str): The ID of the node.

    Returns:
    dict | None: A dictionary containing core metadata.
    """

    # Create a directed graph to facilitate traversal
    G = nx.DiGraph()

    # Convert the raw graph string into a JSON object
    graph = parse_obj_as(LooseGraph, graph_raw)

    # Add nodes and edges to the graph
    for node_id, node_data in graph.nodes.items():
        G.add_node(node_id, **node_data)
    for edge in graph.edges:
        G.add_edge(edge.source.node_id, edge.destination.node_id)

    # Find the nearest ancestor of the given node
    ancestor = find_nearest_ancestor(G, node_id)

    # If no ancestor was found, return None
    if ancestor is None:
        return None
    
    metadata = ancestor['metadata']
    ancestor_id = ancestor['node_id']

    # Get additional metadata related to the ancestor
    addl_metadata = get_additional_metadata(graph, ancestor_id)

    # If additional metadata was found, add it to the main metadata
    if addl_metadata is not None:
        metadata.update(addl_metadata)

    return metadata



class ResultsServiceABC(ABC):
    """The Results service is responsible for retrieving results."""

    @abstractmethod
    def get(
        self, result_id: str, result_type: ResultType
    ) -> Union[ResultWithSession, None]:
        pass

    @abstractmethod
    def get_many(
        self, result_type: ResultType, page: int = 0, per_page: int = 10
    ) -> PaginatedResults[ResultWithSession]:
        pass

    @abstractmethod
    def search(
        self, query: str, page: int = 0, per_page: int = 10
    ) -> PaginatedResults[ResultWithSession]:
        pass

    @abstractmethod
    def handle_graph_execution_state_change(self, session: GraphExecutionState) -> None:
        pass


class SqliteResultsService(ResultsServiceABC):
    """SQLite implementation of the Results service."""

    _filename: str
    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _lock: Lock

    def __init__(self, filename: str):
        super().__init__()

        self._filename = filename
        self._lock = Lock()

        self._conn = sqlite3.connect(
            self._filename, check_same_thread=False
        )  # TODO: figure out a better threading solution
        self._cursor = self._conn.cursor()

        self._create_table()

    def _create_table(self):
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                CREATE TABLE IF NOT EXISTS results (
                  id TEXT PRIMARY KEY, -- the result's name
                  result_type TEXT, -- `image_output` | `latents_output`
                  node_id TEXT, -- the node that produced this result
                  session_id TEXT, -- the session that produced this result
                  created_at INTEGER, -- the time at which this result was created
                  data TEXT -- the result itself
                );
                """
            )
            self._cursor.execute(
                """--sql
                CREATE UNIQUE INDEX IF NOT EXISTS idx_result_id ON results(id);
                """
            )
        finally:
            self._lock.release()

    def _parse_joined_result(self, result_row: Any, column_names: list[str]):
        result_raw = {}
        session_raw = {}

        for idx, name in enumerate(column_names):
            if name == "session":
                session_raw = json.loads(result_row[idx])
            elif name == "data":
                result_raw[name] = json.loads(result_row[idx])
            else:
                result_raw[name] = result_row[idx]

        graph_raw = session_raw['execution_graph']

        result = parse_obj_as(Result, result_raw)
        session = parse_obj_as(GraphExecutionState, session_raw)

        m = build_core_metadata(graph_raw, result.node_id)
        print(m)

        # g = session.execution_graph.nx_graph()
        # ancestors = nx.dag.ancestors(g, result.node_id)

        # nodes = [session.execution_graph.get_node(result.node_id)]
        # for ancestor in ancestors:
        #     nodes.append(session.execution_graph.get_node(ancestor))

        # filtered_nodes = filter(lambda n: n.type in NODE_TYPE_ALLOWLIST, nodes)
        # print(list(map(lambda n: n.dict(), filtered_nodes)))
        # metadata = {}
        # for node in nodes:
        #     if (node.type in ['txt2img', 'img2img',])
        #     for field, value in node.dict().items():
        #         if field not in ['type', 'id']:
        #             if field not in metadata:
        #                 metadata[field] = value

        # print(ancestors)
        # print(nodes)
        # print(metadata)

        # for node in nodes:
        #     print(node.dict())

        # print(nodes)

        return ResultWithSession(
            result=result,
            session=session,
        )

    def get(
        self, result_id: str, result_type: ResultType
    ) -> Union[ResultWithSession, None]:
        """Retrieves a result by ID and type."""
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT
                    results.id AS id,
                    results.result_type AS result_type,
                    results.node_id AS node_id,
                    results.session_id AS session_id,
                    results.data AS data,
                    graph_executions.item AS session
                FROM results
                JOIN graph_executions ON results.session_id = graph_executions.id
                WHERE results.id = ? AND results.result_type = ?
                """,
                (result_id, result_type),
            )

            result_row = self._cursor.fetchone()

            if result_row is None:
                return None

            column_names = list(map(lambda x: x[0], self._cursor.description))
            result_parsed = self._parse_joined_result(result_row, column_names)
        finally:
            self._lock.release()

        if not result_parsed:
            return None

        return result_parsed

    def get_many(
        self,
        result_type: ResultType,
        page: int = 0,
        per_page: int = 10,
    ) -> PaginatedResults[ResultWithSession]:
        """Lists results of a given type."""
        try:
            self._lock.acquire()

            self._cursor.execute(
                f"""--sql
                SELECT
                    results.id AS id,
                    results.result_type AS result_type,
                    results.node_id AS node_id,
                    results.session_id AS session_id,
                    results.data AS data,
                    graph_executions.item AS session
                FROM results
                JOIN graph_executions ON results.session_id = graph_executions.id
                WHERE results.result_type = ?
                LIMIT ? OFFSET ?;
                """,
                (result_type.value, per_page, page * per_page),
            )

            result_rows = self._cursor.fetchall()
            column_names = list(map(lambda c: c[0], self._cursor.description))

            result_parsed = []

            for result_row in result_rows:
                result_parsed.append(
                    self._parse_joined_result(result_row, column_names)
                )

            self._cursor.execute("""SELECT count(*) FROM results;""")
            count = self._cursor.fetchone()[0]
        finally:
            self._lock.release()

        pageCount = int(count / per_page) + 1

        return PaginatedResults[ResultWithSession](
            items=result_parsed,
            page=page,
            pages=pageCount,
            per_page=per_page,
            total=count,
        )

    def search(
        self,
        query: str,
        page: int = 0,
        per_page: int = 10,
    ) -> PaginatedResults[ResultWithSession]:
        """Finds results by query."""
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT results.data, graph_executions.item
                FROM results
                JOIN graph_executions ON results.session_id = graph_executions.id
                WHERE item LIKE ?
                LIMIT ? OFFSET ?;
                """,
                (f"%{query}%", per_page, page * per_page),
            )

            result_rows = self._cursor.fetchall()

            items = list(
                map(
                    lambda r: ResultWithSession(
                        result=parse_raw_as(Result, r[0]),
                        session=parse_raw_as(GraphExecutionState, r[1]),
                    ),
                    result_rows,
                )
            )
            self._cursor.execute(
                """--sql
                SELECT count(*) FROM results WHERE item LIKE ?;
                """,
                (f"%{query}%",),
            )
            count = self._cursor.fetchone()[0]
        finally:
            self._lock.release()

        pageCount = int(count / per_page) + 1

        return PaginatedResults[ResultWithSession](
            items=items, page=page, pages=pageCount, per_page=per_page, total=count
        )

    def handle_graph_execution_state_change(self, session: GraphExecutionState) -> None:
        """Updates the results table with the results from the session."""
        with self._conn as conn:
            for node_id, result in session.results.items():
                # We'll only process 'image_output' or 'latents_output'
                if result.type not in ["image_output", "latents_output"]:
                    continue

                # The id depends on the result type
                if result.type == "image_output":
                    id = result.image.image_name
                    result_type = "image_output"
                else:
                    id = result.latents.latents_name
                    result_type = "latents_output"

                # Insert the result into the results table, ignoring if it already exists
                conn.execute(
                    """--sql
                    INSERT OR IGNORE INTO results (id, result_type, node_id, session_id, created_at, data)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        id,
                        result_type,
                        node_id,
                        session.id,
                        get_timestamp(),
                        result.json(),
                    ),
                )
