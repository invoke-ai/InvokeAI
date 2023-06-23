import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { RandomIntInvocation, RangeOfSizeInvocation } from 'services/api';
import {
  ITERATE,
  LATENTS_TO_IMAGE,
  MODEL_LOADER,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  RANDOM_INT,
  RANGE_OF_SIZE,
  TEXT_TO_IMAGE_GRAPH,
  TEXT_TO_LATENTS,
} from './constants';
import { addControlNetToLinearGraph } from '../addControlNetToLinearGraph';
import { modelIdToPipelineModelField } from '../modelIdToPipelineModelField';

/**
 * Builds the Canvas tab's Text to Image graph.
 */
export const buildCanvasTextToImageGraph = (
  state: RootState
): NonNullableGraph => {
  const {
    positivePrompt,
    negativePrompt,
    model: modelId,
    cfgScale: cfg_scale,
    scheduler,
    steps,
    iterations,
    seed,
    shouldRandomizeSeed,
  } = state.generation;

  // The bounding box determines width and height, not the width and height params
  const { width, height } = state.canvas.boundingBoxDimensions;

  const model = modelIdToPipelineModelField(modelId);

  /**
   * The easiest way to build linear graphs is to do it in the node editor, then copy and paste the
   * full graph here as a template. Then use the parameters from app state and set friendlier node
   * ids.
   *
   * The only thing we need extra logic for is handling randomized seed, control net, and for img2img,
   * the `fit` param. These are added to the graph at the end.
   */

  // copy-pasted graph from node editor, filled in with state values & friendly node ids
  const graph: NonNullableGraph = {
    id: TEXT_TO_IMAGE_GRAPH,
    nodes: {
      [POSITIVE_CONDITIONING]: {
        type: 'compel',
        id: POSITIVE_CONDITIONING,
        prompt: positivePrompt,
      },
      [NEGATIVE_CONDITIONING]: {
        type: 'compel',
        id: NEGATIVE_CONDITIONING,
        prompt: negativePrompt,
      },
      [RANGE_OF_SIZE]: {
        type: 'range_of_size',
        id: RANGE_OF_SIZE,
        // start: 0, // seed - must be connected manually
        size: iterations,
        step: 1,
      },
      [NOISE]: {
        type: 'noise',
        id: NOISE,
        width,
        height,
      },
      [TEXT_TO_LATENTS]: {
        type: 't2l',
        id: TEXT_TO_LATENTS,
        cfg_scale,
        scheduler,
        steps,
      },
      [MODEL_LOADER]: {
        type: 'pipeline_model_loader',
        id: MODEL_LOADER,
        model,
      },
      [LATENTS_TO_IMAGE]: {
        type: 'l2i',
        id: LATENTS_TO_IMAGE,
      },
      [ITERATE]: {
        type: 'iterate',
        id: ITERATE,
      },
    },
    edges: [
      {
        source: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: TEXT_TO_LATENTS,
          field: 'negative_conditioning',
        },
      },
      {
        source: {
          node_id: POSITIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: TEXT_TO_LATENTS,
          field: 'positive_conditioning',
        },
      },
      {
        source: {
          node_id: MODEL_LOADER,
          field: 'clip',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: MODEL_LOADER,
          field: 'clip',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: MODEL_LOADER,
          field: 'unet',
        },
        destination: {
          node_id: TEXT_TO_LATENTS,
          field: 'unet',
        },
      },
      {
        source: {
          node_id: TEXT_TO_LATENTS,
          field: 'latents',
        },
        destination: {
          node_id: LATENTS_TO_IMAGE,
          field: 'latents',
        },
      },
      {
        source: {
          node_id: MODEL_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: LATENTS_TO_IMAGE,
          field: 'vae',
        },
      },
      {
        source: {
          node_id: RANGE_OF_SIZE,
          field: 'collection',
        },
        destination: {
          node_id: ITERATE,
          field: 'collection',
        },
      },
      {
        source: {
          node_id: ITERATE,
          field: 'item',
        },
        destination: {
          node_id: NOISE,
          field: 'seed',
        },
      },
      {
        source: {
          node_id: NOISE,
          field: 'noise',
        },
        destination: {
          node_id: TEXT_TO_LATENTS,
          field: 'noise',
        },
      },
    ],
  };

  // handle seed
  if (shouldRandomizeSeed) {
    // Random int node to generate the starting seed
    const randomIntNode: RandomIntInvocation = {
      id: RANDOM_INT,
      type: 'rand_int',
    };

    graph.nodes[RANDOM_INT] = randomIntNode;

    // Connect random int to the start of the range of size so the range starts on the random first seed
    graph.edges.push({
      source: { node_id: RANDOM_INT, field: 'a' },
      destination: { node_id: RANGE_OF_SIZE, field: 'start' },
    });
  } else {
    // User specified seed, so set the start of the range of size to the seed
    (graph.nodes[RANGE_OF_SIZE] as RangeOfSizeInvocation).start = seed;
  }

  // add controlnet
  addControlNetToLinearGraph(graph, TEXT_TO_LATENTS, state);

  return graph;
};
