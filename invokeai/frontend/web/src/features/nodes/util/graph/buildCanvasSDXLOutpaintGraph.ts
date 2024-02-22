import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type {
  ImageDTO,
  ImageToLatentsInvocation,
  InfillPatchMatchInvocation,
  InfillTileInvocation,
  NoiseInvocation,
  NonNullableGraph,
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
  INPAINT_CREATE_MASK,
  INPAINT_IMAGE,
  INPAINT_IMAGE_RESIZE_DOWN,
  INPAINT_IMAGE_RESIZE_UP,
  INPAINT_INFILL,
  INPAINT_INFILL_RESIZE_DOWN,
  LATENTS_TO_IMAGE,
  MASK_COMBINE,
  MASK_FROM_ALPHA,
  MASK_RESIZE_DOWN,
  MASK_RESIZE_UP,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  SDXL_CANVAS_OUTPAINT_GRAPH,
  SDXL_DENOISE_LATENTS,
  SDXL_MODEL_LOADER,
  SDXL_REFINER_SEAMLESS,
  SEAMLESS,
} from './constants';
import { getBoardField, getIsIntermediate, getSDXLStylePrompts } from './graphBuilderUtils';

/**
 * Builds the Canvas tab's Outpaint graph.
 */
export const buildCanvasSDXLOutpaintGraph = (
  state: RootState,
  canvasInitImage: ImageDTO,
  canvasMaskImage?: ImageDTO
): NonNullableGraph => {
  const log = logger('nodes');
  const {
    positivePrompt,
    negativePrompt,
    model,
    cfgScale: cfg_scale,
    cfgRescaleMultiplier: cfg_rescale_multiplier,
    scheduler,
    steps,
    img2imgStrength: strength,
    seed,
    vaePrecision,
    shouldUseCpuNoise,
    infillTileSize,
    infillPatchmatchDownscaleSize,
    infillMethod,
    seamlessXAxis,
    seamlessYAxis,
    canvasCoherenceMode,
    canvasCoherenceMinDenoise,
    canvasCoherenceEdgeSize,
  } = state.generation;

  const { refinerModel, refinerStart } = state.sdxl;

  if (!model) {
    log.error('No model found in state');
    throw new Error('No model found in state');
  }

  // The bounding box determines width and height, not the width and height params
  const { width, height } = state.canvas.boundingBoxDimensions;

  // We may need to set the inpaint width and height to scale the image
  const { scaledBoundingBoxDimensions, boundingBoxScaleMethod } = state.canvas;

  const fp32 = vaePrecision === 'fp32';
  const is_intermediate = true;
  const isUsingScaledDimensions = ['auto', 'manual'].includes(boundingBoxScaleMethod);

  let modelLoaderNodeId = SDXL_MODEL_LOADER;

  const use_cpu = shouldUseCpuNoise;

  // Construct Style Prompt
  const { positiveStylePrompt, negativeStylePrompt } = getSDXLStylePrompts(state);

  const graph: NonNullableGraph = {
    id: SDXL_CANVAS_OUTPAINT_GRAPH,
    nodes: {
      [SDXL_MODEL_LOADER]: {
        type: 'sdxl_model_loader',
        id: SDXL_MODEL_LOADER,
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
      [MASK_FROM_ALPHA]: {
        type: 'tomask',
        id: MASK_FROM_ALPHA,
        is_intermediate,
        image: canvasInitImage,
      },
      [MASK_COMBINE]: {
        type: 'mask_combine',
        id: MASK_COMBINE,
        is_intermediate,
        mask2: canvasMaskImage,
      },
      [INPAINT_IMAGE]: {
        type: 'i2l',
        id: INPAINT_IMAGE,
        is_intermediate,
        fp32,
      },
      [NOISE]: {
        type: 'noise',
        id: NOISE,
        use_cpu,
        seed,
        is_intermediate,
      },
      [INPAINT_CREATE_MASK]: {
        type: 'create_gradient_mask',
        id: INPAINT_CREATE_MASK,
        is_intermediate,
        coherence_mode: canvasCoherenceMode,
        edge_radius: canvasCoherenceEdgeSize,
        minimum_denoise: canvasCoherenceMinDenoise,
      },
      [SDXL_DENOISE_LATENTS]: {
        type: 'denoise_latents',
        id: SDXL_DENOISE_LATENTS,
        is_intermediate,
        steps: steps,
        cfg_scale: cfg_scale,
        cfg_rescale_multiplier,
        scheduler: scheduler,
        denoising_start: refinerModel ? Math.min(refinerStart, 1 - strength) : 1 - strength,
        denoising_end: refinerModel ? refinerStart : 1,
      },

      [LATENTS_TO_IMAGE]: {
        type: 'l2i',
        id: LATENTS_TO_IMAGE,
        is_intermediate,
        fp32,
      },
      [CANVAS_OUTPUT]: {
        type: 'iai_canvas_paste_back',
        id: CANVAS_OUTPUT,
        is_intermediate: getIsIntermediate(state),
        board: getBoardField(state),
        use_cache: false,
      },
    },
    edges: [
      // Connect Model Loader To UNet and CLIP
      {
        source: {
          node_id: SDXL_MODEL_LOADER,
          field: 'unet',
        },
        destination: {
          node_id: SDXL_DENOISE_LATENTS,
          field: 'unet',
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
      // Connect Infill Result To Inpaint Image
      {
        source: {
          node_id: INPAINT_INFILL,
          field: 'image',
        },
        destination: {
          node_id: INPAINT_IMAGE,
          field: 'image',
        },
      },
      // Combine Mask from Init Image with User Painted Mask
      {
        source: {
          node_id: MASK_FROM_ALPHA,
          field: 'image',
        },
        destination: {
          node_id: MASK_COMBINE,
          field: 'mask1',
        },
      },
      // Plug Everything Into Inpaint Node
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
          node_id: INPAINT_IMAGE,
          field: 'latents',
        },
        destination: {
          node_id: SDXL_DENOISE_LATENTS,
          field: 'latents',
        },
      },
      // Create Inpaint Mask
      {
        source: {
          node_id: isUsingScaledDimensions ? MASK_RESIZE_UP : MASK_COMBINE,
          field: 'image',
        },
        destination: {
          node_id: INPAINT_CREATE_MASK,
          field: 'mask',
        },
      },
      {
        source: {
          node_id: INPAINT_CREATE_MASK,
          field: 'denoise_mask',
        },
        destination: {
          node_id: SDXL_DENOISE_LATENTS,
          field: 'denoise_mask',
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
    ],
  };

  // Add Infill Nodes
  if (infillMethod === 'patchmatch') {
    graph.nodes[INPAINT_INFILL] = {
      type: 'infill_patchmatch',
      id: INPAINT_INFILL,
      is_intermediate,
      downscale: infillPatchmatchDownscaleSize,
    };
  }

  if (infillMethod === 'lama') {
    graph.nodes[INPAINT_INFILL] = {
      type: 'infill_lama',
      id: INPAINT_INFILL,
      is_intermediate,
    };
  }

  if (infillMethod === 'cv2') {
    graph.nodes[INPAINT_INFILL] = {
      type: 'infill_cv2',
      id: INPAINT_INFILL,
      is_intermediate,
    };
  }

  if (infillMethod === 'tile') {
    graph.nodes[INPAINT_INFILL] = {
      type: 'infill_tile',
      id: INPAINT_INFILL,
      is_intermediate,
      tile_size: infillTileSize,
    };
  }

  // Handle Scale Before Processing
  if (isUsingScaledDimensions) {
    const scaledWidth: number = scaledBoundingBoxDimensions.width;
    const scaledHeight: number = scaledBoundingBoxDimensions.height;

    // Add Scaling Nodes
    graph.nodes[INPAINT_IMAGE_RESIZE_UP] = {
      type: 'img_resize',
      id: INPAINT_IMAGE_RESIZE_UP,
      is_intermediate,
      width: scaledWidth,
      height: scaledHeight,
      image: canvasInitImage,
    };
    graph.nodes[MASK_RESIZE_UP] = {
      type: 'img_resize',
      id: MASK_RESIZE_UP,
      is_intermediate,
      width: scaledWidth,
      height: scaledHeight,
    };
    graph.nodes[INPAINT_IMAGE_RESIZE_DOWN] = {
      type: 'img_resize',
      id: INPAINT_IMAGE_RESIZE_DOWN,
      is_intermediate,
      width: width,
      height: height,
    };
    graph.nodes[INPAINT_INFILL_RESIZE_DOWN] = {
      type: 'img_resize',
      id: INPAINT_INFILL_RESIZE_DOWN,
      is_intermediate,
      width: width,
      height: height,
    };
    graph.nodes[MASK_RESIZE_DOWN] = {
      type: 'img_resize',
      id: MASK_RESIZE_DOWN,
      is_intermediate,
      width: width,
      height: height,
    };

    (graph.nodes[NOISE] as NoiseInvocation).width = scaledWidth;
    (graph.nodes[NOISE] as NoiseInvocation).height = scaledHeight;

    // Connect Nodes
    graph.edges.push(
      // Scale Inpaint Image
      {
        source: {
          node_id: INPAINT_IMAGE_RESIZE_UP,
          field: 'image',
        },
        destination: {
          node_id: INPAINT_INFILL,
          field: 'image',
        },
      },
      // Take combined mask and resize
      {
        source: {
          node_id: MASK_COMBINE,
          field: 'image',
        },
        destination: {
          node_id: MASK_RESIZE_UP,
          field: 'image',
        },
      },
      // Resize Results Down
      {
        source: {
          node_id: LATENTS_TO_IMAGE,
          field: 'image',
        },
        destination: {
          node_id: INPAINT_IMAGE_RESIZE_DOWN,
          field: 'image',
        },
      },
      {
        source: {
          node_id: MASK_RESIZE_UP,
          field: 'image',
        },
        destination: {
          node_id: MASK_RESIZE_DOWN,
          field: 'image',
        },
      },
      {
        source: {
          node_id: INPAINT_INFILL,
          field: 'image',
        },
        destination: {
          node_id: INPAINT_INFILL_RESIZE_DOWN,
          field: 'image',
        },
      },
      // Paste Back
      {
        source: {
          node_id: INPAINT_INFILL_RESIZE_DOWN,
          field: 'image',
        },
        destination: {
          node_id: CANVAS_OUTPUT,
          field: 'source_image',
        },
      },
      {
        source: {
          node_id: LATENTS_TO_IMAGE,
          field: 'image',
        },
        destination: {
          node_id: CANVAS_OUTPUT,
          field: 'target_image',
        },
      },
      {
        source: {
          node_id: MASK_RESIZE_DOWN,
          field: 'image',
        },
        destination: {
          node_id: CANVAS_OUTPUT,
          field: 'mask',
        },
      }
    );
  } else {
    // Add Images To Nodes
    graph.nodes[INPAINT_INFILL] = {
      ...(graph.nodes[INPAINT_INFILL] as InfillTileInvocation | InfillPatchMatchInvocation),
      image: canvasInitImage,
    };

    (graph.nodes[NOISE] as NoiseInvocation).width = width;
    (graph.nodes[NOISE] as NoiseInvocation).height = height;

    graph.nodes[INPAINT_IMAGE] = {
      ...(graph.nodes[INPAINT_IMAGE] as ImageToLatentsInvocation),
      image: canvasInitImage,
    };

    graph.edges.push(
      {
        source: {
          node_id: INPAINT_INFILL,
          field: 'image',
        },
        destination: {
          node_id: CANVAS_OUTPUT,
          field: 'source_image',
        },
      },
      {
        source: {
          node_id: LATENTS_TO_IMAGE,
          field: 'image',
        },
        destination: {
          node_id: CANVAS_OUTPUT,
          field: 'target_image',
        },
      },
      {
        source: {
          node_id: MASK_COMBINE,
          field: 'image',
        },
        destination: {
          node_id: CANVAS_OUTPUT,
          field: 'mask',
        },
      }
    );
  }

  // Add Seamless To Graph
  if (seamlessXAxis || seamlessYAxis) {
    addSeamlessToLinearGraph(state, graph, modelLoaderNodeId);
    modelLoaderNodeId = SEAMLESS;
  }

  // Add Refiner if enabled
  if (refinerModel) {
    addSDXLRefinerToGraph(state, graph, SDXL_DENOISE_LATENTS, modelLoaderNodeId, canvasInitImage);
    if (seamlessXAxis || seamlessYAxis) {
      modelLoaderNodeId = SDXL_REFINER_SEAMLESS;
    }
  }

  // Add VAE
  addVAEToGraph(state, graph, modelLoaderNodeId);

  // add LoRA support
  addSDXLLoRAsToGraph(state, graph, SDXL_DENOISE_LATENTS, modelLoaderNodeId);

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

  return graph;
};
