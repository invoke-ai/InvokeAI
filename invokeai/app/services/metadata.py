from abc import ABC, abstractmethod
from typing import Any, Union
import networkx as nx

from invokeai.app.models.metadata import ImageMetadata
from invokeai.app.services.graph import Graph, GraphExecutionState


class MetadataServiceBase(ABC):
    """Handles building metadata for nodes, images, and outputs."""

    @abstractmethod
    def create_image_metadata(
        self, session: GraphExecutionState, node_id: str
    ) -> ImageMetadata:
        """Builds an ImageMetadata object for a node."""
        pass


class CoreMetadataService(MetadataServiceBase):
    _ANCESTOR_TYPES = ["t2l", "l2l"]
    """The ancestor types that contain the core metadata"""

    _ANCESTOR_PARAMS = ["type", "steps", "model", "cfg_scale", "scheduler", "strength"]
    """The core metadata parameters in the ancestor types"""

    _NOISE_FIELDS = ["seed", "width", "height"]
    """The core metadata parameters in the noise node"""

    def create_image_metadata(
        self, session: GraphExecutionState, node_id: str
    ) -> ImageMetadata:
        metadata = self._build_metadata_from_graph(session, node_id)

        return metadata

    def _find_nearest_ancestor(self, G: nx.DiGraph, node_id: str) -> Union[str, None]:
        """
        Finds the id of the nearest ancestor (of a valid type) of a given node.

        Parameters:
        G (nx.DiGraph): The execution graph, converted in to a networkx DiGraph. Its nodes must
        have the same data as the execution graph.
        node_id (str): The ID of the node.

        Returns:
        str | None: The ID of the nearest ancestor, or None if there are no valid ancestors.
        """

        # Retrieve the node from the graph
        node = G.nodes[node_id]

        # If the node type is one of the core metadata node types, return its id
        if node.get("type") in self._ANCESTOR_TYPES:
            return node.get("id")

        # Else, look for the ancestor in the predecessor nodes
        for predecessor in G.predecessors(node_id):
            result = self._find_nearest_ancestor(G, predecessor)
            if result:
                return result

        # If there are no valid ancestors, return None
        return None

    def _get_additional_metadata(
        self, graph: Graph, node_id: str
    ) -> Union[dict[str, Any], None]:
        """
        Returns additional metadata for a given node.

        Parameters:
        graph (Graph): The execution graph.
        node_id (str): The ID of the node.

        Returns:
        dict[str, Any] | None: A dictionary of additional metadata.
        """

        metadata = {}

        # Iterate over all edges in the graph
        for edge in graph.edges:
            dest_node_id = edge.destination.node_id
            dest_field = edge.destination.field
            source_node_dict = graph.nodes[edge.source.node_id].dict()

            # If the destination node ID matches the given node ID, gather necessary metadata
            if dest_node_id == node_id:
                # Prompt
                if dest_field == "positive_conditioning":
                    metadata["positive_conditioning"] = source_node_dict.get("prompt")
                # Negative prompt
                if dest_field == "negative_conditioning":
                    metadata["negative_conditioning"] = source_node_dict.get("prompt")
                # Seed, width and height
                if dest_field == "noise":
                    for field in self._NOISE_FIELDS:
                        metadata[field] = source_node_dict.get(field)
        return metadata

    def _build_metadata_from_graph(
        self, session: GraphExecutionState, node_id: str
    ) -> ImageMetadata:
        """
        Builds an ImageMetadata object for a node.

        Parameters:
        session (GraphExecutionState): The session.
        node_id (str): The ID of the node.

        Returns:
        ImageMetadata: The metadata for the node.
        """

        # We need to do all the traversal on the execution graph
        graph = session.execution_graph

        # Find the nearest `t2l`/`l2l` ancestor of the given node
        ancestor_id = self._find_nearest_ancestor(graph.nx_graph_with_data(), node_id)

        # If no ancestor was found, return an empty ImageMetadata object
        if ancestor_id is None:
            return ImageMetadata()

        ancestor_node = graph.get_node(ancestor_id)

        # Grab all the core metadata from the ancestor node
        ancestor_metadata = {
            param: val
            for param, val in ancestor_node.dict().items()
            if param in self._ANCESTOR_PARAMS
        }

        # Get this image's prompts and noise parameters
        addl_metadata = self._get_additional_metadata(graph, ancestor_id)

        # If additional metadata was found, add it to the main metadata
        if addl_metadata is not None:
            ancestor_metadata.update(addl_metadata)

        return ImageMetadata(**ancestor_metadata)
