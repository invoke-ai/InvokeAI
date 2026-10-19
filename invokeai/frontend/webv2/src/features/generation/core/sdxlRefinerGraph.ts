/** The SDXL Refiner's second denoise stage, grafted between the base denoise and the VAE decode. */

import type { BackendGraphContract, BackendInvocationContract } from '@features/generation/core/contracts';

import type { GenerateSettings, MainModelConfig } from './types';

import { coerceSchedulerForGraph } from './baseGenerationPolicies';
import { addEdge, addNode, toModelIdentifier } from './graphBuilder';
import { SDXL_REFINER_COMPEL_PROMPT_NODE_TYPE } from './prompt/compel';

export const REFINER_DENOISE_NODE_ID = 'refiner_denoise_latents';

interface RefinerStageInput {
  graph: BackendGraphContract;
  settings: GenerateSettings;
  /** The SDXL main model; the refiner shares its scheduler set. */
  model: MainModelConfig;
  refiner: MainModelConfig;
  /** The base denoise, whose latents the refiner finishes; `getBaseDenoisingRange` already ends it at the hand-over. */
  denoise: BackendInvocationContract;
  positivePrompt: BackendInvocationContract;
  negativePrompt: BackendInvocationContract;
  /** Set when the base graph tiles seamlessly; the refiner UNet gets the same wrapping. */
  seamless: BackendInvocationContract | null;
}

/** Returns the refiner denoise, which replaces the base denoise as the decode's latents source. */
export const addSDXLRefinerStage = ({
  denoise,
  graph,
  model,
  negativePrompt,
  positivePrompt,
  refiner,
  seamless,
  settings,
}: RefinerStageInput): BackendInvocationContract => {
  const scheduler = coerceSchedulerForGraph(model, settings.refinerScheduler);
  const modelLoader = addNode(graph, {
    id: 'refiner_model_loader',
    model: toModelIdentifier(refiner),
    type: 'sdxl_refiner_model_loader',
  });
  const posCond = addNode(graph, {
    aesthetic_score: settings.refinerPositiveAestheticScore,
    id: 'refiner_pos_cond',
    type: SDXL_REFINER_COMPEL_PROMPT_NODE_TYPE,
  });
  const negCond = addNode(graph, {
    aesthetic_score: settings.refinerNegativeAestheticScore,
    id: 'refiner_neg_cond',
    type: SDXL_REFINER_COMPEL_PROMPT_NODE_TYPE,
  });
  const refinerDenoise = addNode(graph, {
    cfg_scale: settings.refinerCfgScale,
    denoising_end: 1,
    denoising_start: settings.refinerStart,
    id: REFINER_DENOISE_NODE_ID,
    // HiDiffusion is a UNet-level patch, so the refiner pass runs with the same settings as the base pass.
    hidiffusion: denoise.hidiffusion,
    hidiffusion_raunet: denoise.hidiffusion_raunet,
    hidiffusion_window_attn: denoise.hidiffusion_window_attn,
    hidiffusion_t1_ratio: denoise.hidiffusion_t1_ratio,
    hidiffusion_t2_ratio: denoise.hidiffusion_t2_ratio,
    scheduler,
    steps: settings.refinerSteps,
    type: 'denoise_latents',
  });

  if (seamless) {
    const refinerSeamless = addNode(graph, {
      id: 'refiner_seamless',
      seamless_x: seamless.seamless_x,
      seamless_y: seamless.seamless_y,
      type: 'seamless',
    });

    addEdge(graph, modelLoader, 'unet', refinerSeamless, 'unet');
    addEdge(graph, refinerSeamless, 'unet', refinerDenoise, 'unet');
  } else {
    addEdge(graph, modelLoader, 'unet', refinerDenoise, 'unet');
  }

  // The refiner only has the second text encoder; the prompts condition it as style text.
  addEdge(graph, modelLoader, 'clip2', posCond, 'clip2');
  addEdge(graph, modelLoader, 'clip2', negCond, 'clip2');
  addEdge(graph, positivePrompt, 'value', posCond, 'style');
  addEdge(graph, negativePrompt, 'value', negCond, 'style');
  addEdge(graph, posCond, 'conditioning', refinerDenoise, 'positive_conditioning');
  addEdge(graph, negCond, 'conditioning', refinerDenoise, 'negative_conditioning');
  addEdge(graph, denoise, 'latents', refinerDenoise, 'latents');

  return refinerDenoise;
};

export const getRefinerMetadata = (settings: GenerateSettings, model: MainModelConfig, refiner: MainModelConfig) => ({
  refiner_cfg_scale: settings.refinerCfgScale,
  refiner_model: toModelIdentifier(refiner),
  refiner_negative_aesthetic_score: settings.refinerNegativeAestheticScore,
  refiner_positive_aesthetic_score: settings.refinerPositiveAestheticScore,
  refiner_scheduler: coerceSchedulerForGraph(model, settings.refinerScheduler),
  refiner_start: settings.refinerStart,
  refiner_steps: settings.refinerSteps,
});
