import { RootState } from 'app/store/store';
import { Graph } from 'services/api';
import { buildImg2ImgNode } from '../nodeBuilders/buildImageToImageNode';
import { buildRangeNode } from '../nodeBuilders/buildRangeNode';
import { buildIterateNode } from '../nodeBuilders/buildIterateNode';
import { buildEdges } from '../edgeBuilders/buildEdges';

/**
 * Builds the Linear workflow graph.
 */
export const buildImageToImageGraph = (state: RootState): Graph => {
  const baseNode = buildImg2ImgNode(state);

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
