import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypedDict
from PIL import Image, PngImagePlugin
from pydantic import BaseModel

from invokeai.app.models.image import ImageType, is_image_type


class MetadataImageField(TypedDict):
    """Pydantic-less ImageField, used for metadata parsing."""

    image_type: ImageType
    image_name: str


class MetadataLatentsField(TypedDict):
    """Pydantic-less LatentsField, used for metadata parsing."""

    latents_name: str


class MetadataColorField(TypedDict):
    """Pydantic-less ColorField, used for metadata parsing"""

    r: int
    g: int
    b: int
    a: int


# TODO: This is a placeholder for `InvocationsUnion` pending resolution of circular imports
NodeMetadata = Dict[
    str,
    None
    | str
    | int
    | float
    | bool
    | MetadataImageField
    | MetadataLatentsField
    | MetadataColorField,
]


class InvokeAIMetadata(TypedDict, total=False):
    """InvokeAI-specific metadata format."""

    session_id: Optional[str]
    node: Optional[NodeMetadata]


def build_invokeai_metadata_pnginfo(
    metadata: InvokeAIMetadata | None,
) -> PngImagePlugin.PngInfo:
    """Builds a PngInfo object with key `"invokeai"` and value `metadata`"""
    pnginfo = PngImagePlugin.PngInfo()

    if metadata is not None:
        pnginfo.add_text("invokeai", json.dumps(metadata))

    return pnginfo


class MetadataServiceBase(ABC):
    @abstractmethod
    def get_metadata(self, image: Image.Image) -> InvokeAIMetadata | None:
        """Gets the InvokeAI metadata from a PIL Image, skipping invalid values"""
        pass

    @abstractmethod
    def build_metadata(
        self, session_id: str, node: BaseModel
    ) -> InvokeAIMetadata | None:
        """Builds an InvokeAIMetadata object"""
        pass

    # @abstractmethod
    # def create_metadata(self, session_id: str, node_id: str) -> dict:
    #     """Creates metadata for a result"""
    #     pass


class PngMetadataService(MetadataServiceBase):
    """Handles loading and building metadata for images."""

    # TODO: Use `InvocationsUnion` to **validate** metadata as representing a fully-functioning node
    def _load_metadata(self, image: Image.Image) -> dict | None:
        """Loads a specific info entry from a PIL Image."""

        try:
            info = image.info.get("invokeai")

            if type(info) is not str:
                return None

            loaded_metadata = json.loads(info)

            if type(loaded_metadata) is not dict:
                return None

            if len(loaded_metadata.items()) == 0:
                return None

            return loaded_metadata
        except:
            return None

    def get_metadata(self, image: Image.Image) -> dict | None:
        """Retrieves an image's metadata as a dict"""
        loaded_metadata = self._load_metadata(image)

        return loaded_metadata

    def build_metadata(self, session_id: str, node: BaseModel) -> InvokeAIMetadata:
        metadata = InvokeAIMetadata(session_id=session_id, node=node.dict())

        return metadata


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


# # Create a directed graph
# from typing import Any, TypedDict, Union
# from networkx import DiGraph
# import networkx as nx
# import json


# # We need to use a loose class for nodes to allow for graceful parsing - we cannot use the stricter
# # model used by the system, because we may be a graph in an old format. We can, however, use the
# # Edge model, because the edge format does not change.
# class LooseGraph(BaseModel):
#     id: str
#     nodes: dict[str, dict[str, Any]]
#     edges: list[Edge]


# # An intermediate type used during parsing
# class NearestAncestor(TypedDict):
#     node_id: str
#     metadata: dict[str, Any]


# # The ancestor types that contain the core metadata
# ANCESTOR_TYPES = ['t2l', 'l2l']

# # The core metadata parameters in the ancestor types
# ANCESTOR_PARAMS = ['steps', 'model', 'cfg_scale', 'scheduler', 'strength']

# # The core metadata parameters in the noise node
# NOISE_FIELDS = ['seed', 'width', 'height']

# # Find nearest t2l or l2l ancestor from a given l2i node
# def find_nearest_ancestor(G: DiGraph, node_id: str) -> Union[NearestAncestor, None]:
#     """Returns metadata for the nearest ancestor of a given node.

#     Parameters:
#     G (DiGraph): A directed graph.
#     node_id (str): The ID of the starting node.

#     Returns:
#     NearestAncestor | None: An object with the ID and metadata of the nearest ancestor.
#     """

#     # Retrieve the node from the graph
#     node = G.nodes[node_id]

#     # If the node type is one of the core metadata node types, gather necessary metadata and return
#     if node.get('type') in ANCESTOR_TYPES:
#         parsed_metadata = {param: val for param, val in node.items() if param in ANCESTOR_PARAMS}
#         return NearestAncestor(node_id=node_id, metadata=parsed_metadata)


#     # Else, look for the ancestor in the predecessor nodes
#     for predecessor in G.predecessors(node_id):
#         result = find_nearest_ancestor(G, predecessor)
#         if result:
#             return result

#     # If there are no valid ancestors, return None
#     return None


# def get_additional_metadata(graph: LooseGraph, node_id: str) -> Union[dict[str, Any], None]:
#     """Collects additional metadata from nodes connected to a given node.

#     Parameters:
#     graph (LooseGraph): The graph.
#     node_id (str): The ID of the node.

#     Returns:
#     dict | None: A dictionary containing additional metadata.
#     """

#     metadata = {}

#     # Iterate over all edges in the graph
#     for edge in graph.edges:
#         dest_node_id = edge.destination.node_id
#         dest_field =  edge.destination.field
#         source_node = graph.nodes[edge.source.node_id]

#         # If the destination node ID matches the given node ID, gather necessary metadata
#         if dest_node_id == node_id:
#             # If the destination field is 'positive_conditioning', add the 'prompt' from the source node
#             if dest_field == 'positive_conditioning':
#                 metadata['positive_conditioning'] = source_node.get('prompt')
#             # If the destination field is 'negative_conditioning', add the 'prompt' from the source node
#             if dest_field == 'negative_conditioning':
#                 metadata['negative_conditioning'] = source_node.get('prompt')
#             # If the destination field is 'noise', add the core noise fields from the source node
#             if dest_field == 'noise':
#                 for field in NOISE_FIELDS:
#                     metadata[field] = source_node.get(field)
#     return metadata

# def build_core_metadata(graph_raw: str, node_id: str) -> Union[dict, None]:
#     """Builds the core metadata for a given node.

#     Parameters:
#     graph_raw (str): The graph structure as a raw string.
#     node_id (str): The ID of the node.

#     Returns:
#     dict | None: A dictionary containing core metadata.
#     """

#     # Create a directed graph to facilitate traversal
#     G = nx.DiGraph()

#     # Convert the raw graph string into a JSON object
#     graph = parse_obj_as(LooseGraph, graph_raw)

#     # Add nodes and edges to the graph
#     for node_id, node_data in graph.nodes.items():
#         G.add_node(node_id, **node_data)
#     for edge in graph.edges:
#         G.add_edge(edge.source.node_id, edge.destination.node_id)

#     # Find the nearest ancestor of the given node
#     ancestor = find_nearest_ancestor(G, node_id)

#     # If no ancestor was found, return None
#     if ancestor is None:
#         return None

#     metadata = ancestor['metadata']
#     ancestor_id = ancestor['node_id']

#     # Get additional metadata related to the ancestor
#     addl_metadata = get_additional_metadata(graph, ancestor_id)

#     # If additional metadata was found, add it to the main metadata
#     if addl_metadata is not None:
#         metadata.update(addl_metadata)

#     return metadata
