from ..invocations.latent import LatentsToImageInvocation, NoiseInvocation, TextToLatentsInvocation
from ..invocations.params import ParamIntInvocation
from .graph import Edge, EdgeConnection, ExposedNodeInput, ExposedNodeOutput, Graph, LibraryGraph
from .item_storage import ItemStorageABC


default_text_to_image_graph_id = '539b2af5-2b4d-4d8c-8071-e54a3255fc74'


def create_text_to_image() -> LibraryGraph:
    return LibraryGraph(
        id=default_text_to_image_graph_id,
        name='t2i',
        description='Converts text to an image',
        graph=Graph(
            nodes={
                'width': ParamIntInvocation(id='width', a=512),
                'height': ParamIntInvocation(id='height', a=512),
                '3': NoiseInvocation(id='3'),
                '4': TextToLatentsInvocation(id='4'),
                '5': LatentsToImageInvocation(id='5')
            },
            edges=[
                Edge(source=EdgeConnection(node_id='width', field='a'), destination=EdgeConnection(node_id='3', field='width')),
                Edge(source=EdgeConnection(node_id='height', field='a'), destination=EdgeConnection(node_id='3', field='height')),
                Edge(source=EdgeConnection(node_id='3', field='noise'), destination=EdgeConnection(node_id='4', field='noise')),
                Edge(source=EdgeConnection(node_id='4', field='latents'), destination=EdgeConnection(node_id='5', field='latents')),
            ]
        ),
        exposed_inputs=[
            ExposedNodeInput(node_path='4', field='prompt', alias='prompt'),
            ExposedNodeInput(node_path='width', field='a', alias='width'),
            ExposedNodeInput(node_path='height', field='a', alias='height')
        ],
        exposed_outputs=[
            ExposedNodeOutput(node_path='5', field='image', alias='image')
        ])


def create_system_graphs(graph_library: ItemStorageABC[LibraryGraph]) -> list[LibraryGraph]:
    """Creates the default system graphs, or adds new versions if the old ones don't match"""

    graphs: list[LibraryGraph] = list()

    text_to_image = graph_library.get(default_text_to_image_graph_id)
    
    # TODO: Check if the graph is the same as the default one, and if not, update it
    #if text_to_image is None:
    text_to_image = create_text_to_image()
    graph_library.set(text_to_image)

    graphs.append(text_to_image)

    return graphs
