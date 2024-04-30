import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import {
  type ImageDTO,
  type ImageToLatentsInvocation,
  isNonRefinerMainModelConfig,
  type NonNullableGraph,
} from 'services/api/types';

import { addControlNetToLinearGraph } from './addControlNetToLinearGraph';
import { addIPAdapterToLinearGraph } from './addIPAdapterToLinearGraph';
import { addNSFWCheckerToGraph } from './addNSFWCheckerToGraph';
import { addSDXLLoRAsToGraph } from './addSDXLLoRAstoGraph';
import { addSDXLRefinerToGraph } from './addSDXLRefinerToGraph';
import { addSeamlessToLinearGraph } from './addSeamlessToLinearGraph';
import { addT2IAdaptersToLinearGraph } from './addT2IAdapterToLinearGraph';
import { addVAEToGraph } from './addVAEToGraph';
import { addWatermarkerToGraph } from './addWatermarkerToGraph';
import {
  CANVAS_OUTPUT,
  IMAGE_TO_LATENTS,
  IMG2IMG_RESIZE,
  LATENTS_TO_IMAGE,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH,
  SDXL_DENOISE_LATENTS,
  SDXL_MODEL_LOADER,
  SDXL_REFINER_SEAMLESS,
  SEAMLESS,
} from './constants';
import { getBoardField, getIsIntermediate, getSDXLStylePrompts } from './graphBuilderUtils';
import { addCoreMetadataNode, getModelMetadataField } from './metadata';

/**
 * Builds the Canvas tab's Image to Image graph.
 */
export const buildCanvasSDXLImageToImageGraph = async (
  state: RootState,
  initialImage: ImageDTO
): Promise<NonNullableGraph> => {
  const log = logger('nodes');
  const {
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
    img2imgStrength: strength,
  } = state.generation;
  const { positivePrompt, negativePrompt } = state.controlLayers.present;

  const { refinerModel, refinerStart } = state.sdxl;

  // The bounding box determines width and height, not the width and height params
  const { width, height } = state.canvas.boundingBoxDimensions;

  const { scaledBoundingBoxDimensions, boundingBoxScaleMethod } = state.canvas;

  const fp32 = vaePrecision === 'fp32';
  const is_intermediate = true;
  const isUsingScaledDimensions = ['auto', 'manual'].includes(boundingBoxScaleMethod);

  if (!model) {
    log.error('No model found in state');
    throw new Error('No model found in state');
  }

  // Model Loader ID
  let modelLoaderNodeId = SDXL_MODEL_LOADER;

  const use_cpu = shouldUseCpuNoise;

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
    id: SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH,
    nodes: {
      [modelLoaderNodeId]: {
        type: 'sdxl_model_loader',
        id: modelLoaderNodeId,
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
        is_intermediate,
        use_cpu,
        seed,
        width: !isUsingScaledDimensions ? width : scaledBoundingBoxDimensions.width,
        height: !isUsingScaledDimensions ? height : scaledBoundingBoxDimensions.height,
      },
      [IMAGE_TO_LATENTS]: {
        type: 'i2l',
        id: IMAGE_TO_LATENTS,
        is_intermediate,
        fp32,
      },
      [SDXL_DENOISE_LATENTS]: {
        type: 'denoise_latents',
        id: SDXL_DENOISE_LATENTS,
        is_intermediate,
        cfg_scale,
        cfg_rescale_multiplier,
        scheduler,
        steps,
        denoising_start: refinerModel ? Math.min(refinerStart, 1 - strength) : 1 - strength,
        denoising_end: refinerModel ? refinerStart : 1,
      },
    },
    edges: [
      // Connect Model Loader To UNet & CLIP
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
      // Connect Everything to Denoise Latents
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
      {
        source: {
          node_id: IMAGE_TO_LATENTS,
          field: 'latents',
        },
        destination: {
          node_id: SDXL_DENOISE_LATENTS,
          field: 'latents',
        },
      },
    ],
  };

  // Decode Latents To Image & Handle Scaled Before Processing
  if (isUsingScaledDimensions) {
    graph.nodes[IMG2IMG_RESIZE] = {
      id: IMG2IMG_RESIZE,
      type: 'img_resize',
      is_intermediate,
      image: initialImage,
      width: scaledBoundingBoxDimensions.width,
      height: scaledBoundingBoxDimensions.height,
    };
    graph.nodes[LATENTS_TO_IMAGE] = {
      id: LATENTS_TO_IMAGE,
      type: 'l2i',
      is_intermediate,
      fp32,
    };
    graph.nodes[CANVAS_OUTPUT] = {
      id: CANVAS_OUTPUT,
      type: 'img_resize',
      is_intermediate: getIsIntermediate(state),
      board: getBoardField(state),
      width: width,
      height: height,
      use_cache: false,
    };

    graph.edges.push(
      {
        source: {
          node_id: IMG2IMG_RESIZE,
          field: 'image',
        },
        destination: {
          node_id: IMAGE_TO_LATENTS,
          field: 'image',
        },
      },
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

    (graph.nodes[IMAGE_TO_LATENTS] as ImageToLatentsInvocation).image = initialImage;

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

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);

  addCoreMetadataNode(
    graph,
    {
      generation_mode: 'img2img',
      cfg_scale,
      cfg_rescale_multiplier,
      width: !isUsingScaledDimensions ? width : scaledBoundingBoxDimensions.width,
      height: !isUsingScaledDimensions ? height : scaledBoundingBoxDimensions.height,
      positive_prompt: positivePrompt,
      negative_prompt: negativePrompt,
      model: getModelMetadataField(modelConfig),
      seed,
      steps,
      rand_device: use_cpu ? 'cpu' : 'cuda',
      scheduler,
      strength,
      init_image: initialImage.image_name,
      positive_style_prompt: positiveStylePrompt,
      negative_style_prompt: negativeStylePrompt,
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
    await addSDXLRefinerToGraph(state, graph, SDXL_DENOISE_LATENTS, modelLoaderNodeId);
    if (seamlessXAxis || seamlessYAxis) {
      modelLoaderNodeId = SDXL_REFINER_SEAMLESS;
    }
  }

  // optionally add custom VAE
  await addVAEToGraph(state, graph, modelLoaderNodeId);

  // add LoRA support
  await addSDXLLoRAsToGraph(state, graph, SDXL_DENOISE_LATENTS, modelLoaderNodeId);

  // add controlnet, mutating `graph`
  await addControlNetToLinearGraph(state, graph, SDXL_DENOISE_LATENTS);

  // Add IP Adapter
  await addIPAdapterToLinearGraph(state, graph, SDXL_DENOISE_LATENTS);
  await addT2IAdaptersToLinearGraph(state, graph, SDXL_DENOISE_LATENTS);

  // NSFW & watermark - must be last thing added to graph
  if (state.system.shouldUseNSFWChecker) {
    // must add before watermarker!
    addNSFWCheckerToGraph(state, graph, CANVAS_OUTPUT);
  }

  if (state.system.shouldUseWatermarker) {
    // must add after nsfw checker!
    addWatermarkerToGraph(state, graph, CANVAS_OUTPUT);
  }

  return graph;
};
