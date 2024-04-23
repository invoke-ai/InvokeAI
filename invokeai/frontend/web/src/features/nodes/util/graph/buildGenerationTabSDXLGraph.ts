import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { addControlLayersToGraph } from 'features/nodes/util/graph/addControlLayersToGraph';
import { isNonRefinerMainModelConfig, type NonNullableGraph } from 'services/api/types';

import { addSDXLLoRAsToGraph } from './addSDXLLoRAstoGraph';
import { addSDXLRefinerToGraph } from './addSDXLRefinerToGraph';
import { addSeamlessToLinearGraph } from './addSeamlessToLinearGraph';
import { addVAEToGraph } from './addVAEToGraph';
import {
  LATENTS_TO_IMAGE,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  SDXL_CONTROL_LAYERS_GRAPH,
  SDXL_DENOISE_LATENTS,
  SDXL_MODEL_LOADER,
  SDXL_REFINER_SEAMLESS,
  SEAMLESS,
} from './constants';
import { getBoardField, getIsIntermediate, getSDXLStylePrompts } from './graphBuilderUtils';
import { addCoreMetadataNode, getModelMetadataField } from './metadata';

export const buildGenerationTabSDXLGraph = async (state: RootState): Promise<NonNullableGraph> => {
  const log = logger('nodes');
  const {
    model,
    cfgScale: cfg_scale,
    cfgRescaleMultiplier: cfg_rescale_multiplier,
    scheduler,
    seed,
    steps,
    shouldUseCpuNoise,
    vaePrecision,
    seamlessXAxis,
    seamlessYAxis,
  } = state.generation;
  const { positivePrompt, negativePrompt } = state.controlLayers.present;
  const { width, height } = state.controlLayers.present.size;

  const { refinerModel, refinerStart } = state.sdxl;

  const use_cpu = shouldUseCpuNoise;

  if (!model) {
    log.error('No model found in state');
    throw new Error('No model found in state');
  }

  const fp32 = vaePrecision === 'fp32';
  const is_intermediate = true;

  // Construct Style Prompt
  const { positiveStylePrompt, negativeStylePrompt } = getSDXLStylePrompts(state);

  // Model Loader ID
  let modelLoaderNodeId = SDXL_MODEL_LOADER;

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
    id: SDXL_CONTROL_LAYERS_GRAPH,
    nodes: {
      [modelLoaderNodeId]: {
        type: 'sdxl_model_loader',
        id: modelLoaderNodeId,
        model,
        is_intermediate,
      },
      [POSITIVE_CONDITIONING]: {
        type: 'sdxl_compel_prompt',
        id: POSITIVE_CONDITIONING,
        prompt: positivePrompt,
        style: positiveStylePrompt,
        is_intermediate,
      },
      [NEGATIVE_CONDITIONING]: {
        type: 'sdxl_compel_prompt',
        id: NEGATIVE_CONDITIONING,
        prompt: negativePrompt,
        style: negativeStylePrompt,
        is_intermediate,
      },
      [NOISE]: {
        type: 'noise',
        id: NOISE,
        seed,
        width,
        height,
        use_cpu,
        is_intermediate,
      },
      [SDXL_DENOISE_LATENTS]: {
        type: 'denoise_latents',
        id: SDXL_DENOISE_LATENTS,
        cfg_scale,
        cfg_rescale_multiplier,
        scheduler,
        steps,
        denoising_start: 0,
        denoising_end: refinerModel ? refinerStart : 1,
        is_intermediate,
      },
      [LATENTS_TO_IMAGE]: {
        type: 'l2i',
        id: LATENTS_TO_IMAGE,
        fp32,
        is_intermediate: getIsIntermediate(state),
        board: getBoardField(state),
        use_cache: false,
      },
    },
    edges: [
      // Connect Model Loader to UNet, VAE & CLIP
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
      // Decode Denoised Latents To Image
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

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);

  addCoreMetadataNode(
    graph,
    {
      generation_mode: 'txt2img',
      cfg_scale,
      cfg_rescale_multiplier,
      height,
      width,
      positive_prompt: positivePrompt,
      negative_prompt: negativePrompt,
      model: getModelMetadataField(modelConfig),
      seed,
      steps,
      rand_device: use_cpu ? 'cpu' : 'cuda',
      scheduler,
      positive_style_prompt: positiveStylePrompt,
      negative_style_prompt: negativeStylePrompt,
    },
    LATENTS_TO_IMAGE
  );

  // Add Seamless To Graph
  if (seamlessXAxis || seamlessYAxis) {
    addSeamlessToLinearGraph(state, graph, modelLoaderNodeId);
    modelLoaderNodeId = SEAMLESS;
  }

  // Add Refiner if enabled
  if (refinerModel) {
    await addSDXLRefinerToGraph(state, graph, SDXL_DENOISE_LATENTS);
    if (seamlessXAxis || seamlessYAxis) {
      modelLoaderNodeId = SDXL_REFINER_SEAMLESS;
    }
  }

  // add LoRA support
  await addSDXLLoRAsToGraph(state, graph, SDXL_DENOISE_LATENTS, modelLoaderNodeId);

  await addControlLayersToGraph(state, graph, SDXL_DENOISE_LATENTS);

  // optionally add custom VAE
  await addVAEToGraph(state, graph, modelLoaderNodeId);

  return graph;
};
