import type { RootState } from 'app/store/store';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { addGenerationTabControlLayers } from 'features/nodes/util/graph/addGenerationTabControlLayers';
import { addGenerationTabNSFWChecker } from 'features/nodes/util/graph/addGenerationTabNSFWChecker';
import { addGenerationTabSDXLLoRAs } from 'features/nodes/util/graph/addGenerationTabSDXLLoRAs';
import { addGenerationTabSDXLRefiner } from 'features/nodes/util/graph/addGenerationTabSDXLRefiner';
import { addGenerationTabSeamless } from 'features/nodes/util/graph/addGenerationTabSeamless';
import { addGenerationTabWatermarker } from 'features/nodes/util/graph/addGenerationTabWatermarker';
import { Graph } from 'features/nodes/util/graph/Graph';
import type { Invocation, NonNullableGraph } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import {
  LATENTS_TO_IMAGE,
  NEGATIVE_CONDITIONING,
  NEGATIVE_CONDITIONING_COLLECT,
  NOISE,
  POSITIVE_CONDITIONING,
  POSITIVE_CONDITIONING_COLLECT,
  SDXL_CONTROL_LAYERS_GRAPH,
  SDXL_DENOISE_LATENTS,
  SDXL_MODEL_LOADER,
  VAE_LOADER,
} from './constants';
import { getBoardField, getSDXLStylePrompts } from './graphBuilderUtils';
import { getModelMetadataField } from './metadata';

export const buildGenerationTabSDXLGraph = async (state: RootState): Promise<NonNullableGraph> => {
  const {
    model,
    cfgScale: cfg_scale,
    cfgRescaleMultiplier: cfg_rescale_multiplier,
    scheduler,
    seed,
    steps,
    shouldUseCpuNoise,
    vaePrecision,
    vae,
  } = state.generation;
  const { positivePrompt, negativePrompt } = state.controlLayers.present;
  const { width, height } = state.controlLayers.present.size;

  const { refinerModel, refinerStart } = state.sdxl;

  assert(model, 'No model found in state');

  const { positiveStylePrompt, negativeStylePrompt } = getSDXLStylePrompts(state);

  const g = new Graph(SDXL_CONTROL_LAYERS_GRAPH);
  const modelLoader = g.addNode({
    type: 'sdxl_model_loader',
    id: SDXL_MODEL_LOADER,
    model,
  });
  const posCond = g.addNode({
    type: 'sdxl_compel_prompt',
    id: POSITIVE_CONDITIONING,
    prompt: positivePrompt,
    style: positiveStylePrompt,
  });
  const posCondCollect = g.addNode({
    type: 'collect',
    id: POSITIVE_CONDITIONING_COLLECT,
  });
  const negCond = g.addNode({
    type: 'sdxl_compel_prompt',
    id: NEGATIVE_CONDITIONING,
    prompt: negativePrompt,
    style: negativeStylePrompt,
  });
  const negCondCollect = g.addNode({
    type: 'collect',
    id: NEGATIVE_CONDITIONING_COLLECT,
  });
  const noise = g.addNode({ type: 'noise', id: NOISE, seed, width, height, use_cpu: shouldUseCpuNoise });
  const denoise = g.addNode({
    type: 'denoise_latents',
    id: SDXL_DENOISE_LATENTS,
    cfg_scale,
    cfg_rescale_multiplier,
    scheduler,
    steps,
    denoising_start: 0,
    denoising_end: refinerModel ? refinerStart : 1,
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
  const vaeLoader =
    vae?.base === model.base
      ? g.addNode({
          type: 'vae_loader',
          id: VAE_LOADER,
          vae_model: vae,
        })
      : null;

  let imageOutput: Invocation<'l2i'> | Invocation<'img_nsfw'> | Invocation<'img_watermark'> = l2i;

  g.addEdge(modelLoader, 'unet', denoise, 'unet');
  g.addEdge(modelLoader, 'clip', posCond, 'clip');
  g.addEdge(modelLoader, 'clip', negCond, 'clip');
  g.addEdge(modelLoader, 'clip2', posCond, 'clip2');
  g.addEdge(modelLoader, 'clip2', negCond, 'clip2');
  g.addEdge(posCond, 'conditioning', posCondCollect, 'item');
  g.addEdge(negCond, 'conditioning', negCondCollect, 'item');
  g.addEdge(posCondCollect, 'collection', denoise, 'positive_conditioning');
  g.addEdge(negCondCollect, 'collection', denoise, 'negative_conditioning');
  g.addEdge(noise, 'noise', denoise, 'noise');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);

  g.upsertMetadata({
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
    positive_style_prompt: positiveStylePrompt,
    negative_style_prompt: negativeStylePrompt,
    vae: vae ?? undefined,
  });

  const seamless = addGenerationTabSeamless(state, g, denoise, modelLoader, vaeLoader);

  addGenerationTabSDXLLoRAs(state, g, denoise, modelLoader, seamless, posCond, negCond);

  // We might get the VAE from the main model, custom VAE, or seamless node.
  const vaeSource = seamless ?? vaeLoader ?? modelLoader;
  g.addEdge(vaeSource, 'vae', l2i, 'vae');

  // Add Refiner if enabled
  if (refinerModel) {
    await addGenerationTabSDXLRefiner(state, g, denoise, modelLoader, seamless, posCond, negCond, l2i);
  }

  await addGenerationTabControlLayers(
    state,
    g,
    denoise,
    posCond,
    negCond,
    posCondCollect,
    negCondCollect,
    noise,
    vaeSource
  );

  if (state.system.shouldUseNSFWChecker) {
    imageOutput = addGenerationTabNSFWChecker(g, imageOutput);
  }

  if (state.system.shouldUseWatermarker) {
    imageOutput = addGenerationTabWatermarker(g, imageOutput);
  }

  g.setMetadataReceivingNode(imageOutput);
  return g.getGraph();
};
