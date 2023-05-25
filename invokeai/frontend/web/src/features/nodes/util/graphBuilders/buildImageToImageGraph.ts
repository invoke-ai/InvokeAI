import { RootState } from 'app/store/store';
import {
  CompelInvocation,
  Graph,
  ImageToLatentsInvocation,
  LatentsToImageInvocation,
  LatentsToLatentsInvocation,
} from 'services/api';
import { NonNullableGraph } from 'features/nodes/types/types';
import { addNoiseNodes } from '../nodeBuilders/addNoiseNodes';
import { log } from 'app/logging/useLogger';

const moduleLog = log.child({ namespace: 'buildImageToImageGraph' });

const POSITIVE_CONDITIONING = 'positive_conditioning';
const NEGATIVE_CONDITIONING = 'negative_conditioning';
const IMAGE_TO_LATENTS = 'image_to_latents';
const LATENTS_TO_LATENTS = 'latents_to_latents';
const LATENTS_TO_IMAGE = 'latents_to_image';

/**
 * Builds the Image to Image tab graph.
 */
export const buildImageToImageGraph = (state: RootState): Graph => {
  const {
    positivePrompt,
    negativePrompt,
    model,
    cfgScale: cfg_scale,
    scheduler,
    steps,
    initialImage,
    img2imgStrength: strength,
  } = state.generation;

  if (!initialImage) {
    moduleLog.error('No initial image found in state');
    throw new Error('No initial image found in state');
  }

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

  const imageToLatentsNode: ImageToLatentsInvocation = {
    id: IMAGE_TO_LATENTS,
    type: 'i2l',
    model,
    image: {
      image_name: initialImage?.image_name,
      image_type: initialImage?.image_type,
    },
  };

  const latentsToLatentsNode: LatentsToLatentsInvocation = {
    id: LATENTS_TO_LATENTS,
    type: 'l2l',
    cfg_scale,
    model,
    scheduler,
    steps,
    strength,
  };

  const latentsToImageNode: LatentsToImageInvocation = {
    id: LATENTS_TO_IMAGE,
    type: 'l2i',
    model,
  };

  // Add to the graph
  graph.nodes[POSITIVE_CONDITIONING] = positiveConditioningNode;
  graph.nodes[NEGATIVE_CONDITIONING] = negativeConditioningNode;
  graph.nodes[IMAGE_TO_LATENTS] = imageToLatentsNode;
  graph.nodes[LATENTS_TO_LATENTS] = latentsToLatentsNode;
  graph.nodes[LATENTS_TO_IMAGE] = latentsToImageNode;

  // Connect them
  graph.edges.push({
    source: { node_id: POSITIVE_CONDITIONING, field: 'conditioning' },
    destination: {
      node_id: LATENTS_TO_LATENTS,
      field: 'positive_conditioning',
    },
  });

  graph.edges.push({
    source: { node_id: NEGATIVE_CONDITIONING, field: 'conditioning' },
    destination: {
      node_id: LATENTS_TO_LATENTS,
      field: 'negative_conditioning',
    },
  });

  graph.edges.push({
    source: { node_id: IMAGE_TO_LATENTS, field: 'latents' },
    destination: {
      node_id: LATENTS_TO_LATENTS,
      field: 'latents',
    },
  });

  graph.edges.push({
    source: { node_id: LATENTS_TO_LATENTS, field: 'latents' },
    destination: {
      node_id: LATENTS_TO_IMAGE,
      field: 'latents',
    },
  });

  // Create and add the noise nodes
  graph = addNoiseNodes(graph, latentsToLatentsNode.id, state);

  return graph;
};
