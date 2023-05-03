import { RootState } from 'app/store/store';
import { Graph } from 'services/api';
import { buildImg2ImgNode } from './linearGraphBuilder/buildImageToImageNode';
import { buildTxt2ImgNode } from './linearGraphBuilder/buildTextToImageNode';
import { buildRangeNode } from './linearGraphBuilder/buildRangeNode';
import { buildIterateNode } from './linearGraphBuilder/buildIterateNode';
import { buildEdges } from './linearGraphBuilder/buildEdges';

/**
 * Builds the Linear workflow graph.
 */
export const buildLinearGraph = (state: RootState): Graph => {
  // The base node is either a txt2img or img2img node
  const baseNode = state.generation.isImageToImageEnabled
    ? buildImg2ImgNode(state)
    : buildTxt2ImgNode(state);

  // We always range and iterate nodes, no matter the iteration count
  // This is required to provide the correct seeds to the backend engine
  const rangeNode = buildRangeNode(state);
  const iterateNode = buildIterateNode();

  // Build the edges for the nodes selected.
  const edges = buildEdges(baseNode, rangeNode, iterateNode);

  // Assemble!
  const graph = {
    nodes: {
      [rangeNode.id]: rangeNode,
      [iterateNode.id]: iterateNode,
      [baseNode.id]: baseNode,
    },
    edges,
  };

  // TODO: hires fix requires latent space upscaling; we don't have nodes for this yet

  return graph;
};
