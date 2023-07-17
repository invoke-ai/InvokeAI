import json
from typing import Optional

from pydantic import ValidationError

from invokeai.app.services.graph import Edge


def get_metadata_graph_from_raw_session(session_raw: str) -> Optional[dict]:
    """
    Parses raw session string, returning a dict of the graph.

    Only the general graph shape is validated; none of the fields are validated.

    Any `metadata_accumulator` nodes and edges are removed.

    Any validation failure will return None.
    """

    graph = json.loads(session_raw).get("graph", None)

    # sanity check make sure the graph is at least reasonably shaped
    if (
        type(graph) is not dict
        or "nodes" not in graph
        or type(graph["nodes"]) is not dict
        or "edges" not in graph
        or type(graph["edges"]) is not list
    ):
        # something has gone terribly awry, return an empty dict
        return None

    try:
        # delete the `metadata_accumulator` node
        del graph["nodes"]["metadata_accumulator"]
    except KeyError:
        # no accumulator node, all good
        pass

    # delete any edges to or from it
    for i, edge in enumerate(graph["edges"]):
        try:
            # try to parse the edge
            Edge(**edge)
        except ValidationError:
            # something has gone terribly awry, return an empty dict
            return None

        if (
            edge["source"]["node_id"] == "metadata_accumulator"
            or edge["destination"]["node_id"] == "metadata_accumulator"
        ):
            del graph["edges"][i]

    return graph
