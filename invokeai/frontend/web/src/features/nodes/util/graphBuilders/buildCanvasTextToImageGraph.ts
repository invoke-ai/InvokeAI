import { log } from 'app/logging/useLogger';
import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { initialGenerationState } from 'features/parameters/store/generationSlice';
import { addControlNetToLinearGraph } from './addControlNetToLinearGraph';
import { addDynamicPromptsToGraph } from './addDynamicPromptsToGraph';
import { addLoRAsToGraph } from './addLoRAsToGraph';
import { addVAEToGraph } from './addVAEToGraph';
import {
  CLIP_SKIP,
  LATENTS_TO_IMAGE,
  MAIN_MODEL_LOADER,
  METADATA_ACCUMULATOR,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  TEXT_TO_IMAGE_GRAPH,
  TEXT_TO_LATENTS,
} from './constants';

const moduleLog = log.child({ namespace: 'nodes' });

/**
 * Builds the Canvas tab's Text to Image graph.
 */
export const buildCanvasTextToImageGraph = (
  state: RootState
): NonNullableGraph => {
  const {
    positivePrompt,
    negativePrompt,
    model,
    cfgScale: cfg_scale,
    scheduler,
    steps,
    clipSkip,
    shouldUseCpuNoise,
    shouldUseNoiseSettings,
  } = state.generation;

  // The bounding box determines width and height, not the width and height params
  const { width, height } = state.canvas.boundingBoxDimensions;

  const { shouldAutoSave } = state.canvas;

  if (!model) {
    moduleLog.error('No model found in state');
    throw new Error('No model found in state');
  }

  const use_cpu = shouldUseNoiseSettings
    ? shouldUseCpuNoise
    : initialGenerationState.shouldUseCpuNoise;

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
        is_intermediate: true,
        prompt: positivePrompt,
      },
      [NEGATIVE_CONDITIONING]: {
        type: 'compel',
        id: NEGATIVE_CONDITIONING,
        is_intermediate: true,
        prompt: negativePrompt,
      },
      [NOISE]: {
        type: 'noise',
        id: NOISE,
        is_intermediate: true,
        width,
        height,
        use_cpu,
      },
      [TEXT_TO_LATENTS]: {
        type: 't2l',
        id: TEXT_TO_LATENTS,
        is_intermediate: true,
        cfg_scale,
        scheduler,
        steps,
      },
      [MAIN_MODEL_LOADER]: {
        type: 'main_model_loader',
        id: MAIN_MODEL_LOADER,
        is_intermediate: true,
        model,
      },
      [CLIP_SKIP]: {
        type: 'clip_skip',
        id: CLIP_SKIP,
        is_intermediate: true,
        skipped_layers: clipSkip,
      },
      [LATENTS_TO_IMAGE]: {
        type: 'l2i',
        id: LATENTS_TO_IMAGE,
        is_intermediate: !shouldAutoSave,
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
          node_id: MAIN_MODEL_LOADER,
          field: 'clip',
        },
        destination: {
          node_id: CLIP_SKIP,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: CLIP_SKIP,
          field: 'clip',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: CLIP_SKIP,
          field: 'clip',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: MAIN_MODEL_LOADER,
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

  // add metadata accumulator, which is only mostly populated - some fields are added later
  graph.nodes[METADATA_ACCUMULATOR] = {
    id: METADATA_ACCUMULATOR,
    type: 'metadata_accumulator',
    generation_mode: 'txt2img',
    cfg_scale,
    height,
    width,
    positive_prompt: '', // set in addDynamicPromptsToGraph
    negative_prompt: negativePrompt,
    model,
    seed: 0, // set in addDynamicPromptsToGraph
    steps,
    rand_device: use_cpu ? 'cpu' : 'cuda',
    scheduler,
    vae: undefined, // option; set in addVAEToGraph
    controlnets: [], // populated in addControlNetToLinearGraph
    loras: [], // populated in addLoRAsToGraph
    clip_skip: clipSkip,
  };

  graph.edges.push({
    source: {
      node_id: METADATA_ACCUMULATOR,
      field: 'metadata',
    },
    destination: {
      node_id: LATENTS_TO_IMAGE,
      field: 'metadata',
    },
  });

  // add LoRA support
  addLoRAsToGraph(state, graph, TEXT_TO_LATENTS);

  // optionally add custom VAE
  addVAEToGraph(state, graph);

  // add dynamic prompts - also sets up core iteration and seed
  addDynamicPromptsToGraph(state, graph);

  // add controlnet, mutating `graph`
  addControlNetToLinearGraph(state, graph, TEXT_TO_LATENTS);

  return graph;
};
