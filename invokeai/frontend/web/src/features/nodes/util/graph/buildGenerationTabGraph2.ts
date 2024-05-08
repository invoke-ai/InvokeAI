import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import { isInitialImageLayer, isRegionalGuidanceLayer } from 'features/controlLayers/store/controlLayersSlice';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { addGenerationTabControlLayers } from 'features/nodes/util/graph/addGenerationTabControlLayers';
import { addGenerationTabLoRAs } from 'features/nodes/util/graph/addGenerationTabLoRAs';
import { addGenerationTabSeamless } from 'features/nodes/util/graph/addGenerationTabSeamless';
import { addGenerationTabVAE } from 'features/nodes/util/graph/addGenerationTabVAE';
import type { GraphType } from 'features/nodes/util/graph/Graph';
import { Graph } from 'features/nodes/util/graph/Graph';
import { getBoardField } from 'features/nodes/util/graph/graphBuilderUtils';
import { MetadataUtil } from 'features/nodes/util/graph/MetadataUtil';
import { isNonRefinerMainModelConfig } from 'services/api/types';

import { addHrfToGraph } from './addHrfToGraph';
import { addNSFWCheckerToGraph } from './addNSFWCheckerToGraph';
import { addWatermarkerToGraph } from './addWatermarkerToGraph';
import {
  CLIP_SKIP,
  CONTROL_LAYERS_GRAPH,
  DENOISE_LATENTS,
  LATENTS_TO_IMAGE,
  MAIN_MODEL_LOADER,
  NEGATIVE_CONDITIONING,
  NEGATIVE_CONDITIONING_COLLECT,
  NOISE,
  POSITIVE_CONDITIONING,
  POSITIVE_CONDITIONING_COLLECT,
} from './constants';
import { getModelMetadataField } from './metadata';

const log = logger('nodes');
export const buildGenerationTabGraph2 = async (state: RootState): Promise<GraphType> => {
  const {
    model,
    cfgScale: cfg_scale,
    cfgRescaleMultiplier: cfg_rescale_multiplier,
    scheduler,
    steps,
    clipSkip: skipped_layers,
    shouldUseCpuNoise,
    vaePrecision,
    seed,
  } = state.generation;
  const { positivePrompt, negativePrompt } = state.controlLayers.present;
  const { width, height } = state.controlLayers.present.size;

  if (!model) {
    log.error('No model found in state');
    throw new Error('No model found in state');
  }

  const g = new Graph(CONTROL_LAYERS_GRAPH);
  const modelLoader = g.addNode({
    type: 'main_model_loader',
    id: MAIN_MODEL_LOADER,
    model,
  });
  const clipSkip = g.addNode({
    type: 'clip_skip',
    id: CLIP_SKIP,
    skipped_layers,
  });
  const posCond = g.addNode({
    type: 'compel',
    id: POSITIVE_CONDITIONING,
    prompt: positivePrompt,
  });
  const posCondCollect = g.addNode({
    type: 'collect',
    id: POSITIVE_CONDITIONING_COLLECT,
  });
  const negCond = g.addNode({
    type: 'compel',
    id: NEGATIVE_CONDITIONING,
    prompt: negativePrompt,
  });
  const negCondCollect = g.addNode({
    type: 'collect',
    id: NEGATIVE_CONDITIONING_COLLECT,
  });
  const noise = g.addNode({
    type: 'noise',
    id: NOISE,
    seed,
    width,
    height,
    use_cpu: shouldUseCpuNoise,
  });
  const denoise = g.addNode({
    type: 'denoise_latents',
    id: DENOISE_LATENTS,
    cfg_scale,
    cfg_rescale_multiplier,
    scheduler,
    steps,
    denoising_start: 0,
    denoising_end: 1,
  });
  const l2i = g.addNode({
    type: 'l2i',
    id: LATENTS_TO_IMAGE,
    fp32: vaePrecision === 'fp32',
    board: getBoardField(state),
    // This is the terminal node and must always save to gallery.
    is_intermediate: false,
    use_cache: false,
  });

  g.addEdge(modelLoader, 'unet', denoise, 'unet');
  g.addEdge(modelLoader, 'clip', clipSkip, 'clip');
  g.addEdge(clipSkip, 'clip', posCond, 'clip');
  g.addEdge(clipSkip, 'clip', negCond, 'clip');
  g.addEdge(posCond, 'conditioning', posCondCollect, 'item');
  g.addEdge(negCond, 'conditioning', negCondCollect, 'item');
  g.addEdge(posCondCollect, 'collection', denoise, 'positive_conditioning');
  g.addEdge(negCondCollect, 'collection', denoise, 'negative_conditioning');
  g.addEdge(noise, 'noise', denoise, 'noise');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);

  MetadataUtil.add(g, {
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
    rand_device: shouldUseCpuNoise ? 'cpu' : 'cuda',
    scheduler,
    clip_skip: skipped_layers,
  });
  MetadataUtil.setMetadataReceivingNode(g, l2i);
  g.validate();

  const seamless = addGenerationTabSeamless(state, g, denoise, modelLoader);
  g.validate();
  addGenerationTabVAE(state, g, modelLoader, l2i, i2l, seamless);
  g.validate();
  addGenerationTabLoRAs(state, g, denoise, seamless ?? modelLoader, clipSkip, posCond, negCond);
  g.validate();

  const addedLayers = await addGenerationTabControlLayers(
    state,
    g,
    denoise,
    posCond,
    negCond,
    posCondCollect,
    negCondCollect,
    noise
  );
  g.validate();

  // High resolution fix.
  const shouldUseHRF = !addedLayers.some((l) => isInitialImageLayer(l) || isRegionalGuidanceLayer(l));
  if (state.hrf.hrfEnabled && !shouldUseHRF) {
    addHrfToGraph(state, graph);
  }

  // NSFW & watermark - must be last thing added to graph
  if (state.system.shouldUseNSFWChecker) {
    // must add before watermarker!
    addNSFWCheckerToGraph(state, graph);
  }

  if (state.system.shouldUseWatermarker) {
    // must add after nsfw checker!
    addWatermarkerToGraph(state, graph);
  }

  return g.getGraph();
};
