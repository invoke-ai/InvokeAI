import { logger } from 'app/logging/logger';
import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { initialGenerationState } from 'features/parameters/store/generationSlice';
import { addDynamicPromptsToGraph } from './addDynamicPromptsToGraph';
import { addSDXLRefinerToGraph } from './addSDXLRefinerToGraph';
import {
  LATENTS_TO_IMAGE,
  METADATA_ACCUMULATOR,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  SDXL_MODEL_LOADER,
  SDXL_TEXT_TO_IMAGE_GRAPH,
  SDXL_TEXT_TO_LATENTS,
} from './constants';
import { addNSFWCheckerToGraph } from './addNSFWCheckerToGraph';
import { addWatermarkerToGraph } from './addWatermarkerToGraph';

export const buildLinearSDXLTextToImageGraph = (
  state: RootState
): NonNullableGraph => {
  const log = logger('nodes');
  const {
    positivePrompt,
    negativePrompt,
    model,
    cfgScale: cfg_scale,
    scheduler,
    steps,
    width,
    height,
    clipSkip,
    shouldUseCpuNoise,
    shouldUseNoiseSettings,
    vaePrecision,
  } = state.generation;

  const {
    positiveStylePrompt,
    negativeStylePrompt,
    shouldUseSDXLRefiner,
    refinerStart,
  } = state.sdxl;

  const use_cpu = shouldUseNoiseSettings
    ? shouldUseCpuNoise
    : initialGenerationState.shouldUseCpuNoise;

  if (!model) {
    log.error('No model found in state');
    throw new Error('No model found in state');
  }

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
    id: SDXL_TEXT_TO_IMAGE_GRAPH,
    nodes: {
      [SDXL_MODEL_LOADER]: {
        type: 'sdxl_model_loader',
        id: SDXL_MODEL_LOADER,
        model,
      },
      [POSITIVE_CONDITIONING]: {
        type: 'sdxl_compel_prompt',
        id: POSITIVE_CONDITIONING,
        prompt: positivePrompt,
        style: positiveStylePrompt,
      },
      [NEGATIVE_CONDITIONING]: {
        type: 'sdxl_compel_prompt',
        id: NEGATIVE_CONDITIONING,
        prompt: negativePrompt,
        style: negativeStylePrompt,
      },
      [NOISE]: {
        type: 'noise',
        id: NOISE,
        width,
        height,
        use_cpu,
      },
      [SDXL_TEXT_TO_LATENTS]: {
        type: 't2l_sdxl',
        id: SDXL_TEXT_TO_LATENTS,
        cfg_scale,
        scheduler,
        steps,
        denoising_end: shouldUseSDXLRefiner ? refinerStart : 1,
      },
      [LATENTS_TO_IMAGE]: {
        type: 'l2i',
        id: LATENTS_TO_IMAGE,
        fp32: vaePrecision === 'fp32' ? true : false,
      },
    },
    edges: [
      {
        source: {
          node_id: SDXL_MODEL_LOADER,
          field: 'unet',
        },
        destination: {
          node_id: SDXL_TEXT_TO_LATENTS,
          field: 'unet',
        },
      },
      {
        source: {
          node_id: SDXL_MODEL_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: LATENTS_TO_IMAGE,
          field: 'vae',
        },
      },
      {
        source: {
          node_id: SDXL_MODEL_LOADER,
          field: 'clip',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: SDXL_MODEL_LOADER,
          field: 'clip2',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip2',
        },
      },
      {
        source: {
          node_id: SDXL_MODEL_LOADER,
          field: 'clip',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: SDXL_MODEL_LOADER,
          field: 'clip2',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip2',
        },
      },
      {
        source: {
          node_id: POSITIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: SDXL_TEXT_TO_LATENTS,
          field: 'positive_conditioning',
        },
      },
      {
        source: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: SDXL_TEXT_TO_LATENTS,
          field: 'negative_conditioning',
        },
      },
      {
        source: {
          node_id: NOISE,
          field: 'noise',
        },
        destination: {
          node_id: SDXL_TEXT_TO_LATENTS,
          field: 'noise',
        },
      },
      {
        source: {
          node_id: SDXL_TEXT_TO_LATENTS,
          field: 'latents',
        },
        destination: {
          node_id: LATENTS_TO_IMAGE,
          field: 'latents',
        },
      },
    ],
  };

  // add metadata accumulator, which is only mostly populated - some fields are added later
  graph.nodes[METADATA_ACCUMULATOR] = {
    id: METADATA_ACCUMULATOR,
    type: 'metadata_accumulator',
    generation_mode: 'sdxl_txt2img',
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
    vae: undefined,
    controlnets: [],
    loras: [],
    clip_skip: clipSkip,
    positive_style_prompt: positiveStylePrompt,
    negative_style_prompt: negativeStylePrompt,
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

  // Add Refiner if enabled
  if (shouldUseSDXLRefiner) {
    addSDXLRefinerToGraph(state, graph, SDXL_TEXT_TO_LATENTS);
  }

  // add dynamic prompts - also sets up core iteration and seed
  addDynamicPromptsToGraph(state, graph);

  // NSFW & watermark - must be last thing added to graph
  if (state.system.shouldUseNSFWChecker) {
    // must add before watermarker!
    addNSFWCheckerToGraph(state, graph);
  }

  if (state.system.shouldUseWatermarker) {
    // must add after nsfw checker!
    addWatermarkerToGraph(state, graph);
  }

  return graph;
};
