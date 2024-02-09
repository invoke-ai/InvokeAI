import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type { NonNullableGraph } from 'services/api/types';

import { addControlNetToLinearGraph } from './addControlNetToLinearGraph';
import { addIPAdapterToLinearGraph } from './addIPAdapterToLinearGraph';
import { addLinearUIOutputNode } from './addLinearUIOutputNode';
import { addNSFWCheckerToGraph } from './addNSFWCheckerToGraph';
import { addSDXLLoRAsToGraph } from './addSDXLLoRAstoGraph';
import { addSDXLRefinerToGraph } from './addSDXLRefinerToGraph';
import { addSeamlessToLinearGraph } from './addSeamlessToLinearGraph';
import { addT2IAdaptersToLinearGraph } from './addT2IAdapterToLinearGraph';
import { addVAEToGraph } from './addVAEToGraph';
import { addWatermarkerToGraph } from './addWatermarkerToGraph';
import {
  CANVAS_OUTPUT,
  LATENTS_TO_IMAGE,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  SDXL_CANVAS_TEXT_TO_IMAGE_GRAPH,
  SDXL_DENOISE_LATENTS,
  SDXL_MODEL_LOADER,
  SDXL_REFINER_SEAMLESS,
  SEAMLESS,
} from './constants';
import { getSDXLStylePrompts } from './getSDXLStylePrompt';
import { addCoreMetadataNode } from './metadata';

/**
 * Builds the Canvas tab's Text to Image graph.
 */
export const buildCanvasSDXLTextToImageGraph = (state: RootState): NonNullableGraph => {
  const log = logger('nodes');
  const {
    positivePrompt,
    negativePrompt,
    model,
    cfgScale: cfg_scale,
    cfgRescaleMultiplier: cfg_rescale_multiplier,
    scheduler,
    seed,
    steps,
    vaePrecision,
    shouldUseCpuNoise,
    seamlessXAxis,
    seamlessYAxis,
  } = state.generation;

  // The bounding box determines width and height, not the width and height params
  const { width, height } = state.canvas.boundingBoxDimensions;

  const { scaledBoundingBoxDimensions, boundingBoxScaleMethod } = state.canvas;

  const fp32 = vaePrecision === 'fp32';
  const is_intermediate = true;
  const isUsingScaledDimensions = ['auto', 'manual'].includes(boundingBoxScaleMethod);

  const { refinerModel, refinerStart } = state.sdxl;

  if (!model) {
    log.error('No model found in state');
    throw new Error('No model found in state');
  }

  const use_cpu = shouldUseCpuNoise;

  let modelLoaderNodeId = SDXL_MODEL_LOADER;

  // Construct Style Prompt
  const { positiveStylePrompt, negativeStylePrompt } = getSDXLStylePrompts(state);

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
    id: SDXL_CANVAS_TEXT_TO_IMAGE_GRAPH,
    nodes: {
      [modelLoaderNodeId]: {
        type: 'sdxl_model_loader',
        id: modelLoaderNodeId,
        is_intermediate,
        model,
      },
      [POSITIVE_CONDITIONING]: {
        type: 'sdxl_compel_prompt',
        id: POSITIVE_CONDITIONING,
        is_intermediate,
        prompt: positivePrompt,
        style: positiveStylePrompt,
      },
      [NEGATIVE_CONDITIONING]: {
        type: 'sdxl_compel_prompt',
        id: NEGATIVE_CONDITIONING,
        is_intermediate,
        prompt: negativePrompt,
        style: negativeStylePrompt,
      },
      [NOISE]: {
        type: 'noise',
        id: NOISE,
        is_intermediate,
        seed,
        width: !isUsingScaledDimensions ? width : scaledBoundingBoxDimensions.width,
        height: !isUsingScaledDimensions ? height : scaledBoundingBoxDimensions.height,
        use_cpu,
      },
      [SDXL_DENOISE_LATENTS]: {
        type: 'denoise_latents',
        id: SDXL_DENOISE_LATENTS,
        is_intermediate,
        cfg_scale,
        scheduler,
        steps,
        denoising_start: 0,
        denoising_end: refinerModel ? refinerStart : 1,
      },
    },
    edges: [
      // Connect Model Loader to UNet and CLIP
      {
        source: {
          node_id: modelLoaderNodeId,
          field: 'unet',
        },
        destination: {
          node_id: SDXL_DENOISE_LATENTS,
          field: 'unet',
        },
      },
      {
        source: {
          node_id: modelLoaderNodeId,
          field: 'clip',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: modelLoaderNodeId,
          field: 'clip2',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip2',
        },
      },
      {
        source: {
          node_id: modelLoaderNodeId,
          field: 'clip',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: modelLoaderNodeId,
          field: 'clip2',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip2',
        },
      },
      // Connect everything to Denoise Latents
      {
        source: {
          node_id: POSITIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: SDXL_DENOISE_LATENTS,
          field: 'positive_conditioning',
        },
      },
      {
        source: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: SDXL_DENOISE_LATENTS,
          field: 'negative_conditioning',
        },
      },
      {
        source: {
          node_id: NOISE,
          field: 'noise',
        },
        destination: {
          node_id: SDXL_DENOISE_LATENTS,
          field: 'noise',
        },
      },
    ],
  };

  // Decode Latents To Image & Handle Scaled Before Processing
  if (isUsingScaledDimensions) {
    graph.nodes[LATENTS_TO_IMAGE] = {
      id: LATENTS_TO_IMAGE,
      type: 'l2i',
      is_intermediate,
      fp32,
    };

    graph.nodes[CANVAS_OUTPUT] = {
      id: CANVAS_OUTPUT,
      type: 'img_resize',
      is_intermediate,
      width: width,
      height: height,
      use_cache: false,
    };

    graph.edges.push(
      {
        source: {
          node_id: SDXL_DENOISE_LATENTS,
          field: 'latents',
        },
        destination: {
          node_id: LATENTS_TO_IMAGE,
          field: 'latents',
        },
      },
      {
        source: {
          node_id: LATENTS_TO_IMAGE,
          field: 'image',
        },
        destination: {
          node_id: CANVAS_OUTPUT,
          field: 'image',
        },
      }
    );
  } else {
    graph.nodes[CANVAS_OUTPUT] = {
      type: 'l2i',
      id: CANVAS_OUTPUT,
      is_intermediate,
      fp32,
      use_cache: false,
    };

    graph.edges.push({
      source: {
        node_id: SDXL_DENOISE_LATENTS,
        field: 'latents',
      },
      destination: {
        node_id: CANVAS_OUTPUT,
        field: 'latents',
      },
    });
  }

  addCoreMetadataNode(
    graph,
    {
      generation_mode: 'txt2img',
      cfg_scale,
      cfg_rescale_multiplier,
      width: !isUsingScaledDimensions ? width : scaledBoundingBoxDimensions.width,
      height: !isUsingScaledDimensions ? height : scaledBoundingBoxDimensions.height,
      positive_prompt: positivePrompt,
      negative_prompt: negativePrompt,
      positive_style_prompt: positiveStylePrompt,
      negative_style_prompt: negativeStylePrompt,
      model,
      seed,
      steps,
      rand_device: use_cpu ? 'cpu' : 'cuda',
      scheduler,
    },
    CANVAS_OUTPUT
  );

  // Add Seamless To Graph
  if (seamlessXAxis || seamlessYAxis) {
    addSeamlessToLinearGraph(state, graph, modelLoaderNodeId);
    modelLoaderNodeId = SEAMLESS;
  }

  // Add Refiner if enabled
  if (refinerModel) {
    addSDXLRefinerToGraph(state, graph, SDXL_DENOISE_LATENTS, modelLoaderNodeId);
    if (seamlessXAxis || seamlessYAxis) {
      modelLoaderNodeId = SDXL_REFINER_SEAMLESS;
    }
  }

  // add LoRA support
  addSDXLLoRAsToGraph(state, graph, SDXL_DENOISE_LATENTS, modelLoaderNodeId);

  // optionally add custom VAE
  addVAEToGraph(state, graph, modelLoaderNodeId);

  // add controlnet, mutating `graph`
  addControlNetToLinearGraph(state, graph, SDXL_DENOISE_LATENTS);

  // Add IP Adapter
  addIPAdapterToLinearGraph(state, graph, SDXL_DENOISE_LATENTS);
  addT2IAdaptersToLinearGraph(state, graph, SDXL_DENOISE_LATENTS);

  // NSFW & watermark - must be last thing added to graph
  if (state.system.shouldUseNSFWChecker) {
    // must add before watermarker!
    addNSFWCheckerToGraph(state, graph, CANVAS_OUTPUT);
  }

  if (state.system.shouldUseWatermarker) {
    // must add after nsfw checker!
    addWatermarkerToGraph(state, graph, CANVAS_OUTPUT);
  }

  addLinearUIOutputNode(state, graph);

  return graph;
};
