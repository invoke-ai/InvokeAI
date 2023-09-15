import { logger } from 'app/logging/logger';
import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { initialGenerationState } from 'features/parameters/store/generationSlice';
import {
  DenoiseLatentsInvocation,
  ONNXTextToLatentsInvocation,
} from 'services/api/types';
import { addControlNetToLinearGraph } from './addControlNetToLinearGraph';
import { addDynamicPromptsToGraph } from './addDynamicPromptsToGraph';
import { addLoRAsToGraph } from './addLoRAsToGraph';
import { addNSFWCheckerToGraph } from './addNSFWCheckerToGraph';
import { addSeamlessToLinearGraph } from './addSeamlessToLinearGraph';
import { addVAEToGraph } from './addVAEToGraph';
import { addWatermarkerToGraph } from './addWatermarkerToGraph';
import {
  CANVAS_OUTPUT,
  CANVAS_TEXT_TO_IMAGE_GRAPH,
  CLIP_SKIP,
  DENOISE_LATENTS,
  LATENTS_TO_IMAGE,
  MAIN_MODEL_LOADER,
  METADATA_ACCUMULATOR,
  NEGATIVE_CONDITIONING,
  NOISE,
  ONNX_MODEL_LOADER,
  POSITIVE_CONDITIONING,
  SEAMLESS,
} from './constants';

/**
 * Builds the Canvas tab's Text to Image graph.
 */
export const buildCanvasTextToImageGraph = (
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
    vaePrecision,
    clipSkip,
    shouldUseCpuNoise,
    shouldUseNoiseSettings,
    seamlessXAxis,
    seamlessYAxis,
  } = state.generation;

  // The bounding box determines width and height, not the width and height params
  const { width, height } = state.canvas.boundingBoxDimensions;

  const {
    scaledBoundingBoxDimensions,
    boundingBoxScaleMethod,
    shouldAutoSave,
  } = state.canvas;

  const fp32 = vaePrecision === 'fp32';

  const isUsingScaledDimensions = ['auto', 'manual'].includes(
    boundingBoxScaleMethod
  );

  if (!model) {
    log.error('No model found in state');
    throw new Error('No model found in state');
  }

  const use_cpu = shouldUseNoiseSettings
    ? shouldUseCpuNoise
    : initialGenerationState.shouldUseCpuNoise;

  const isUsingOnnxModel = model.model_type === 'onnx';

  let modelLoaderNodeId = isUsingOnnxModel
    ? ONNX_MODEL_LOADER
    : MAIN_MODEL_LOADER;

  const modelLoaderNodeType = isUsingOnnxModel
    ? 'onnx_model_loader'
    : 'main_model_loader';

  const t2lNode: DenoiseLatentsInvocation | ONNXTextToLatentsInvocation =
    isUsingOnnxModel
      ? {
          type: 't2l_onnx',
          id: DENOISE_LATENTS,
          is_intermediate: true,
          cfg_scale,
          scheduler,
          steps,
        }
      : {
          type: 'denoise_latents',
          id: DENOISE_LATENTS,
          is_intermediate: true,
          cfg_scale,
          scheduler,
          steps,
          denoising_start: 0,
          denoising_end: 1,
        };

  /**
   * The easiest way to build linear graphs is to do it in the node editor, then copy and paste the
   * full graph here as a template. Then use the parameters from app state and set friendlier node
   * ids.
   *
   * The only thing we need extra logic for is handling randomized seed, control net, and for img2img,
   * the `fit` param. These are added to the graph at the end.
   */

  // copy-pasted graph from node editor, filled in with state values & friendly node ids
  // TODO: Actually create the graph correctly for ONNX
  const graph: NonNullableGraph = {
    id: CANVAS_TEXT_TO_IMAGE_GRAPH,
    nodes: {
      [modelLoaderNodeId]: {
        type: modelLoaderNodeType,
        id: modelLoaderNodeId,
        is_intermediate: true,
        model,
      },
      [CLIP_SKIP]: {
        type: 'clip_skip',
        id: CLIP_SKIP,
        is_intermediate: true,
        skipped_layers: clipSkip,
      },
      [POSITIVE_CONDITIONING]: {
        type: isUsingOnnxModel ? 'prompt_onnx' : 'compel',
        id: POSITIVE_CONDITIONING,
        is_intermediate: true,
        prompt: positivePrompt,
      },
      [NEGATIVE_CONDITIONING]: {
        type: isUsingOnnxModel ? 'prompt_onnx' : 'compel',
        id: NEGATIVE_CONDITIONING,
        is_intermediate: true,
        prompt: negativePrompt,
      },
      [NOISE]: {
        type: 'noise',
        id: NOISE,
        is_intermediate: true,
        width: !isUsingScaledDimensions
          ? width
          : scaledBoundingBoxDimensions.width,
        height: !isUsingScaledDimensions
          ? height
          : scaledBoundingBoxDimensions.height,
        use_cpu,
      },
      [t2lNode.id]: t2lNode,
    },
    edges: [
      // Connect Model Loader to UNet & CLIP Skip
      {
        source: {
          node_id: modelLoaderNodeId,
          field: 'unet',
        },
        destination: {
          node_id: DENOISE_LATENTS,
          field: 'unet',
        },
      },
      {
        source: {
          node_id: modelLoaderNodeId,
          field: 'clip',
        },
        destination: {
          node_id: CLIP_SKIP,
          field: 'clip',
        },
      },
      // Connect CLIP Skip to Conditioning
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
      // Connect everything to Denoise Latents
      {
        source: {
          node_id: POSITIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: DENOISE_LATENTS,
          field: 'positive_conditioning',
        },
      },
      {
        source: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: DENOISE_LATENTS,
          field: 'negative_conditioning',
        },
      },
      {
        source: {
          node_id: NOISE,
          field: 'noise',
        },
        destination: {
          node_id: DENOISE_LATENTS,
          field: 'noise',
        },
      },
    ],
  };

  // Decode Latents To Image & Handle Scaled Before Processing
  if (isUsingScaledDimensions) {
    graph.nodes[LATENTS_TO_IMAGE] = {
      id: LATENTS_TO_IMAGE,
      type: isUsingOnnxModel ? 'l2i_onnx' : 'l2i',
      is_intermediate: true,
      fp32,
    };

    graph.nodes[CANVAS_OUTPUT] = {
      id: CANVAS_OUTPUT,
      type: 'img_resize',
      is_intermediate: !shouldAutoSave,
      width: width,
      height: height,
    };

    graph.edges.push(
      {
        source: {
          node_id: DENOISE_LATENTS,
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
      type: isUsingOnnxModel ? 'l2i_onnx' : 'l2i',
      id: CANVAS_OUTPUT,
      is_intermediate: !shouldAutoSave,
      fp32,
    };

    graph.edges.push({
      source: {
        node_id: DENOISE_LATENTS,
        field: 'latents',
      },
      destination: {
        node_id: CANVAS_OUTPUT,
        field: 'latents',
      },
    });
  }

  // add metadata accumulator, which is only mostly populated - some fields are added later
  graph.nodes[METADATA_ACCUMULATOR] = {
    id: METADATA_ACCUMULATOR,
    type: 'metadata_accumulator',
    generation_mode: 'txt2img',
    cfg_scale,
    width: !isUsingScaledDimensions ? width : scaledBoundingBoxDimensions.width,
    height: !isUsingScaledDimensions
      ? height
      : scaledBoundingBoxDimensions.height,
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
      node_id: CANVAS_OUTPUT,
      field: 'metadata',
    },
  });

  // Add Seamless To Graph
  if (seamlessXAxis || seamlessYAxis) {
    addSeamlessToLinearGraph(state, graph, modelLoaderNodeId);
    modelLoaderNodeId = SEAMLESS;
  }

  // optionally add custom VAE
  addVAEToGraph(state, graph, modelLoaderNodeId);

  // add LoRA support
  addLoRAsToGraph(state, graph, DENOISE_LATENTS, modelLoaderNodeId);

  // add dynamic prompts - also sets up core iteration and seed
  addDynamicPromptsToGraph(state, graph);

  // add controlnet, mutating `graph`
  addControlNetToLinearGraph(state, graph, DENOISE_LATENTS);

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
