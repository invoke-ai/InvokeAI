import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type {
  CreateDenoiseMaskInvocation,
  ImageBlurInvocation,
  ImageDTO,
  ImageToLatentsInvocation,
  MaskEdgeInvocation,
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
  CANVAS_COHERENCE_DENOISE_LATENTS,
  CANVAS_COHERENCE_INPAINT_CREATE_MASK,
  CANVAS_COHERENCE_MASK_EDGE,
  CANVAS_COHERENCE_NOISE,
  CANVAS_COHERENCE_NOISE_INCREMENT,
  CANVAS_OUTPUT,
  INPAINT_CREATE_MASK,
  INPAINT_IMAGE,
  INPAINT_IMAGE_RESIZE_DOWN,
  INPAINT_IMAGE_RESIZE_UP,
  LATENTS_TO_IMAGE,
  MASK_BLUR,
  MASK_RESIZE_DOWN,
  MASK_RESIZE_UP,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  SDXL_CANVAS_INPAINT_GRAPH,
  SDXL_DENOISE_LATENTS,
  SDXL_MODEL_LOADER,
  SDXL_REFINER_SEAMLESS,
  SEAMLESS,
} from './constants';
import { getBoardField, getIsIntermediate, getSDXLStylePrompts } from './graphBuilderUtils';

/**
 * Builds the Canvas tab's Inpaint graph.
 */
export const buildCanvasSDXLInpaintGraph = (
  state: RootState,
  canvasInitImage: ImageDTO,
  canvasMaskImage: ImageDTO
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
    seed,
    vaePrecision,
    shouldUseCpuNoise,
    maskBlur,
    maskBlurMethod,
    canvasCoherenceMode,
    canvasCoherenceSteps,
    canvasCoherenceStrength,
    seamlessXAxis,
    seamlessYAxis,
    img2imgStrength: strength,
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
    id: SDXL_CANVAS_INPAINT_GRAPH,
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
      [MASK_BLUR]: {
        type: 'img_blur',
        id: MASK_BLUR,
        is_intermediate,
        radius: maskBlur,
        blur_type: maskBlurMethod,
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
        type: 'create_denoise_mask',
        id: INPAINT_CREATE_MASK,
        is_intermediate,
        fp32,
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
      [CANVAS_COHERENCE_NOISE]: {
        type: 'noise',
        id: CANVAS_COHERENCE_NOISE,
        use_cpu,
        seed: seed + 1,
        is_intermediate,
      },
      [CANVAS_COHERENCE_NOISE_INCREMENT]: {
        type: 'add',
        id: CANVAS_COHERENCE_NOISE_INCREMENT,
        b: 1,
        is_intermediate,
      },
      [CANVAS_COHERENCE_DENOISE_LATENTS]: {
        type: 'denoise_latents',
        id: CANVAS_COHERENCE_DENOISE_LATENTS,
        is_intermediate,
        steps: canvasCoherenceSteps,
        cfg_scale: cfg_scale,
        cfg_rescale_multiplier,
        scheduler: scheduler,
        denoising_start: 1 - canvasCoherenceStrength,
        denoising_end: 1,
      },
      [LATENTS_TO_IMAGE]: {
        type: 'l2i',
        id: LATENTS_TO_IMAGE,
        is_intermediate,
        fp32,
      },
      [CANVAS_OUTPUT]: {
        type: 'color_correct',
        id: CANVAS_OUTPUT,
        is_intermediate: getIsIntermediate(state),
        board: getBoardField(state),
        reference: canvasInitImage,
        use_cache: false,
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
      // Connect everything to Inpaint
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
          node_id: MASK_BLUR,
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
      // Canvas Refine
      {
        source: {
          node_id: modelLoaderNodeId,
          field: 'unet',
        },
        destination: {
          node_id: CANVAS_COHERENCE_DENOISE_LATENTS,
          field: 'unet',
        },
      },
      {
        source: {
          node_id: POSITIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: CANVAS_COHERENCE_DENOISE_LATENTS,
          field: 'positive_conditioning',
        },
      },
      {
        source: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: CANVAS_COHERENCE_DENOISE_LATENTS,
          field: 'negative_conditioning',
        },
      },
      {
        source: {
          node_id: CANVAS_COHERENCE_NOISE,
          field: 'noise',
        },
        destination: {
          node_id: CANVAS_COHERENCE_DENOISE_LATENTS,
          field: 'noise',
        },
      },
      {
        source: {
          node_id: SDXL_DENOISE_LATENTS,
          field: 'latents',
        },
        destination: {
          node_id: CANVAS_COHERENCE_DENOISE_LATENTS,
          field: 'latents',
        },
      },
      // Decode Inpainted Latents To Image
      {
        source: {
          node_id: CANVAS_COHERENCE_DENOISE_LATENTS,
          field: 'latents',
        },
        destination: {
          node_id: LATENTS_TO_IMAGE,
          field: 'latents',
        },
      },
    ],
  };

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
      image: canvasMaskImage,
    };
    graph.nodes[INPAINT_IMAGE_RESIZE_DOWN] = {
      type: 'img_resize',
      id: INPAINT_IMAGE_RESIZE_DOWN,
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
    (graph.nodes[CANVAS_COHERENCE_NOISE] as NoiseInvocation).width = scaledWidth;
    (graph.nodes[CANVAS_COHERENCE_NOISE] as NoiseInvocation).height = scaledHeight;

    // Connect Nodes
    graph.edges.push(
      // Scale Inpaint Image and Mask
      {
        source: {
          node_id: INPAINT_IMAGE_RESIZE_UP,
          field: 'image',
        },
        destination: {
          node_id: INPAINT_IMAGE,
          field: 'image',
        },
      },
      {
        source: {
          node_id: MASK_RESIZE_UP,
          field: 'image',
        },
        destination: {
          node_id: MASK_BLUR,
          field: 'image',
        },
      },
      {
        source: {
          node_id: INPAINT_IMAGE_RESIZE_UP,
          field: 'image',
        },
        destination: {
          node_id: INPAINT_CREATE_MASK,
          field: 'image',
        },
      },
      // Color Correct The Inpainted Result
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
          node_id: INPAINT_IMAGE_RESIZE_DOWN,
          field: 'image',
        },
        destination: {
          node_id: CANVAS_OUTPUT,
          field: 'image',
        },
      },
      {
        source: {
          node_id: MASK_BLUR,
          field: 'image',
        },
        destination: {
          node_id: MASK_RESIZE_DOWN,
          field: 'image',
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
    (graph.nodes[NOISE] as NoiseInvocation).width = width;
    (graph.nodes[NOISE] as NoiseInvocation).height = height;
    (graph.nodes[CANVAS_COHERENCE_NOISE] as NoiseInvocation).width = width;
    (graph.nodes[CANVAS_COHERENCE_NOISE] as NoiseInvocation).height = height;

    graph.nodes[INPAINT_IMAGE] = {
      ...(graph.nodes[INPAINT_IMAGE] as ImageToLatentsInvocation),
      image: canvasInitImage,
    };
    graph.nodes[MASK_BLUR] = {
      ...(graph.nodes[MASK_BLUR] as ImageBlurInvocation),
      image: canvasMaskImage,
    };
    graph.nodes[INPAINT_CREATE_MASK] = {
      ...(graph.nodes[INPAINT_CREATE_MASK] as CreateDenoiseMaskInvocation),
      image: canvasInitImage,
    };

    graph.edges.push(
      // Color Correct The Inpainted Result
      {
        source: {
          node_id: LATENTS_TO_IMAGE,
          field: 'image',
        },
        destination: {
          node_id: CANVAS_OUTPUT,
          field: 'image',
        },
      },
      {
        source: {
          node_id: MASK_BLUR,
          field: 'image',
        },
        destination: {
          node_id: CANVAS_OUTPUT,
          field: 'mask',
        },
      }
    );
  }

  // Handle Coherence Mode
  if (canvasCoherenceMode !== 'unmasked') {
    // Create Mask If Coherence Mode Is Not Full
    graph.nodes[CANVAS_COHERENCE_INPAINT_CREATE_MASK] = {
      type: 'create_denoise_mask',
      id: CANVAS_COHERENCE_INPAINT_CREATE_MASK,
      is_intermediate,
      fp32,
    };

    // Handle Image Input For Mask Creation
    if (isUsingScaledDimensions) {
      graph.edges.push({
        source: {
          node_id: INPAINT_IMAGE_RESIZE_UP,
          field: 'image',
        },
        destination: {
          node_id: CANVAS_COHERENCE_INPAINT_CREATE_MASK,
          field: 'image',
        },
      });
    } else {
      graph.nodes[CANVAS_COHERENCE_INPAINT_CREATE_MASK] = {
        ...(graph.nodes[CANVAS_COHERENCE_INPAINT_CREATE_MASK] as CreateDenoiseMaskInvocation),
        image: canvasInitImage,
      };
    }

    // Create Mask If Coherence Mode Is Mask
    if (canvasCoherenceMode === 'mask') {
      if (isUsingScaledDimensions) {
        graph.edges.push({
          source: {
            node_id: MASK_RESIZE_UP,
            field: 'image',
          },
          destination: {
            node_id: CANVAS_COHERENCE_INPAINT_CREATE_MASK,
            field: 'mask',
          },
        });
      } else {
        graph.nodes[CANVAS_COHERENCE_INPAINT_CREATE_MASK] = {
          ...(graph.nodes[CANVAS_COHERENCE_INPAINT_CREATE_MASK] as CreateDenoiseMaskInvocation),
          mask: canvasMaskImage,
        };
      }
    }

    // Create Mask Edge If Coherence Mode Is Edge
    if (canvasCoherenceMode === 'edge') {
      graph.nodes[CANVAS_COHERENCE_MASK_EDGE] = {
        type: 'mask_edge',
        id: CANVAS_COHERENCE_MASK_EDGE,
        is_intermediate,
        edge_blur: maskBlur,
        edge_size: maskBlur * 2,
        low_threshold: 100,
        high_threshold: 200,
      };

      // Handle Scaled Dimensions For Mask Edge
      if (isUsingScaledDimensions) {
        graph.edges.push({
          source: {
            node_id: MASK_RESIZE_UP,
            field: 'image',
          },
          destination: {
            node_id: CANVAS_COHERENCE_MASK_EDGE,
            field: 'image',
          },
        });
      } else {
        graph.nodes[CANVAS_COHERENCE_MASK_EDGE] = {
          ...(graph.nodes[CANVAS_COHERENCE_MASK_EDGE] as MaskEdgeInvocation),
          image: canvasMaskImage,
        };
      }

      graph.edges.push({
        source: {
          node_id: CANVAS_COHERENCE_MASK_EDGE,
          field: 'image',
        },
        destination: {
          node_id: CANVAS_COHERENCE_INPAINT_CREATE_MASK,
          field: 'mask',
        },
      });
    }

    // Plug Denoise Mask To Coherence Denoise Latents
    graph.edges.push({
      source: {
        node_id: CANVAS_COHERENCE_INPAINT_CREATE_MASK,
        field: 'denoise_mask',
      },
      destination: {
        node_id: CANVAS_COHERENCE_DENOISE_LATENTS,
        field: 'denoise_mask',
      },
    });
  }

  // Add Seamless To Graph
  if (seamlessXAxis || seamlessYAxis) {
    addSeamlessToLinearGraph(state, graph, modelLoaderNodeId);
    modelLoaderNodeId = SEAMLESS;
  }

  // Add Refiner if enabled
  if (refinerModel) {
    addSDXLRefinerToGraph(
      state,
      graph,
      CANVAS_COHERENCE_DENOISE_LATENTS,
      modelLoaderNodeId,
      canvasInitImage,
      canvasMaskImage
    );
    if (seamlessXAxis || seamlessYAxis) {
      modelLoaderNodeId = SDXL_REFINER_SEAMLESS;
    }
  }

  // optionally add custom VAE
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
