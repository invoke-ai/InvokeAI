from ..invocations.latent import LatentsToImageInvocation, TextToLatentsInvocation
from ..invocations.image import ImageNSFWBlurInvocation
from ..invocations.noise import NoiseInvocation
from ..invocations.compel import CompelInvocation
from ..invocations.params import ParamIntInvocation
from .graph import Edge, EdgeConnection, ExposedNodeInput, ExposedNodeOutput, Graph, LibraryGraph
from .item_storage import ItemStorageABC


default_text_to_image_graph_id = "539b2af5-2b4d-4d8c-8071-e54a3255fc74"


def create_text_to_image() -> LibraryGraph:
    return LibraryGraph(
        id=default_text_to_image_graph_id,
        name="t2i",
        description="Converts text to an image",
        graph=Graph(
            nodes={
                "width": ParamIntInvocation(id="width", a=512),
                "height": ParamIntInvocation(id="height", a=512),
                "seed": ParamIntInvocation(id="seed", a=-1),
                "3": NoiseInvocation(id="3"),
                "4": CompelInvocation(id="4"),
                "5": CompelInvocation(id="5"),
                "6": TextToLatentsInvocation(id="6"),
                "7": LatentsToImageInvocation(id="7"),
                "8": ImageNSFWBlurInvocation(id="8"),
            },
            edges=[
                Edge(
                    source=EdgeConnection(node_id="width", field="a"),
                    destination=EdgeConnection(node_id="3", field="width"),
                ),
                Edge(
                    source=EdgeConnection(node_id="height", field="a"),
                    destination=EdgeConnection(node_id="3", field="height"),
                ),
                Edge(
                    source=EdgeConnection(node_id="seed", field="a"),
                    destination=EdgeConnection(node_id="3", field="seed"),
                ),
                Edge(
                    source=EdgeConnection(node_id="3", field="noise"),
                    destination=EdgeConnection(node_id="6", field="noise"),
                ),
                Edge(
                    source=EdgeConnection(node_id="6", field="latents"),
                    destination=EdgeConnection(node_id="7", field="latents"),
                ),
                Edge(
                    source=EdgeConnection(node_id="4", field="conditioning"),
                    destination=EdgeConnection(node_id="6", field="positive_conditioning"),
                ),
                Edge(
                    source=EdgeConnection(node_id="5", field="conditioning"),
                    destination=EdgeConnection(node_id="6", field="negative_conditioning"),
                ),
                Edge(
                    source=EdgeConnection(node_id="7", field="image"),
                    destination=EdgeConnection(node_id="8", field="image"),
                ),
            ],
        ),
        exposed_inputs=[
            ExposedNodeInput(node_path="4", field="prompt", alias="positive_prompt"),
            ExposedNodeInput(node_path="5", field="prompt", alias="negative_prompt"),
            ExposedNodeInput(node_path="width", field="a", alias="width"),
            ExposedNodeInput(node_path="height", field="a", alias="height"),
            ExposedNodeInput(node_path="seed", field="a", alias="seed"),
        ],
        exposed_outputs=[ExposedNodeOutput(node_path="8", field="image", alias="image")],
    )


def create_system_graphs(graph_library: ItemStorageABC[LibraryGraph]) -> list[LibraryGraph]:
    """Creates the default system graphs, or adds new versions if the old ones don't match"""

    # TODO: Uncomment this when we are ready to fix this up to prevent breaking changes
    graphs: list[LibraryGraph] = list()

    # text_to_image = graph_library.get(default_text_to_image_graph_id)

    # # TODO: Check if the graph is the same as the default one, and if not, update it
    # #if text_to_image is None:
    text_to_image = create_text_to_image()
    graph_library.set(text_to_image)

    graphs.append(text_to_image)

    return graphs
