import { RootState } from 'app/store/store';
import {
  CompelInvocation,
  Graph,
  LatentsToImageInvocation,
  TextToLatentsInvocation,
} from 'services/api';
import { NonNullableGraph } from 'features/nodes/types/types';
import { addNoiseNodes } from '../nodeBuilders/addNoiseNodes';

const POSITIVE_CONDITIONING = 'positive_conditioning';
const NEGATIVE_CONDITIONING = 'negative_conditioning';
const TEXT_TO_LATENTS = 'text_to_latents';
const LATENTS_TO_IMAGE = 'latnets_to_image';

/**
 * Builds the Text to Image tab graph.
 */
export const buildTextToImageGraph = (state: RootState): Graph => {
  const {
    positivePrompt,
    negativePrompt,
    model,
    cfgScale: cfg_scale,
    scheduler,
    steps,
  } = state.generation;

  let graph: NonNullableGraph = {
    nodes: {},
    edges: [],
  };

  // Create the conditioning, t2l and l2i nodes
  const positiveConditioningNode: CompelInvocation = {
    id: POSITIVE_CONDITIONING,
    type: 'compel',
    prompt: positivePrompt,
    model,
  };

  const negativeConditioningNode: CompelInvocation = {
    id: NEGATIVE_CONDITIONING,
    type: 'compel',
    prompt: negativePrompt,
    model,
  };

  const textToLatentsNode: TextToLatentsInvocation = {
    id: TEXT_TO_LATENTS,
    type: 't2l',
    cfg_scale,
    model,
    scheduler,
    steps,
  };

  const latentsToImageNode: LatentsToImageInvocation = {
    id: LATENTS_TO_IMAGE,
    type: 'l2i',
    model,
  };

  // Add to the graph
  graph.nodes[POSITIVE_CONDITIONING] = positiveConditioningNode;
  graph.nodes[NEGATIVE_CONDITIONING] = negativeConditioningNode;
  graph.nodes[TEXT_TO_LATENTS] = textToLatentsNode;
  graph.nodes[LATENTS_TO_IMAGE] = latentsToImageNode;

  // Connect them
  graph.edges.push({
    source: { node_id: POSITIVE_CONDITIONING, field: 'conditioning' },
    destination: {
      node_id: TEXT_TO_LATENTS,
      field: 'positive_conditioning',
    },
  });

  graph.edges.push({
    source: { node_id: NEGATIVE_CONDITIONING, field: 'conditioning' },
    destination: {
      node_id: TEXT_TO_LATENTS,
      field: 'negative_conditioning',
    },
  });

  graph.edges.push({
    source: { node_id: TEXT_TO_LATENTS, field: 'latents' },
    destination: {
      node_id: LATENTS_TO_IMAGE,
      field: 'latents',
    },
  });

  // Create and add the noise nodes
  graph = addNoiseNodes(graph, TEXT_TO_LATENTS, state);

  return graph;
};
