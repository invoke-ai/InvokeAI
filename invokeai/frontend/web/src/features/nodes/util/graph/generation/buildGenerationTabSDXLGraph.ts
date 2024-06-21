import type { RootState } from 'app/store/store';
import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
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
} from 'features/nodes/util/graph/constants';
import { addControlAdapters } from 'features/nodes/util/graph/generation/addControlAdapters';
import { addIPAdapters } from 'features/nodes/util/graph/generation/addIPAdapters';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addSDXLLoRas } from 'features/nodes/util/graph/generation/addSDXLLoRAs';
import { addSDXLRefiner } from 'features/nodes/util/graph/generation/addSDXLRefiner';
import { addSeamless } from 'features/nodes/util/graph/generation/addSeamless';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { getBoardField, getSDXLStylePrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import type { Invocation, NonNullableGraph } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import { addRegions } from './addRegions';

export const buildGenerationTabSDXLGraph = async (
  state: RootState,
  manager: KonvaNodeManager
): Promise<NonNullableGraph> => {
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
    positivePrompt,
    negativePrompt,
    refinerModel,
    refinerStart,
    img2imgStrength,
  } = state.canvasV2.params;
  const { width, height } = state.canvasV2.bbox;

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
  const i2l = g.addNode({ type: 'i2l', id: 'i2l' });
  const denoise = g.addNode({
    type: 'denoise_latents',
    id: SDXL_DENOISE_LATENTS,
    cfg_scale,
    cfg_rescale_multiplier,
    scheduler,
    steps,
    denoising_start: refinerModel ? Math.min(refinerStart, 1 - img2imgStrength) : 1 - img2imgStrength,
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
  g.addEdge(i2l, 'latents', denoise, 'latents');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'sdxl');

  g.upsertMetadata({
    generation_mode: 'sdxl_txt2img',
    cfg_scale,
    cfg_rescale_multiplier,
    height,
    width,
    positive_prompt: positivePrompt,
    negative_prompt: negativePrompt,
    model: Graph.getModelMetadataField(modelConfig),
    seed,
    steps,
    rand_device: shouldUseCpuNoise ? 'cpu' : 'cuda',
    scheduler,
    positive_style_prompt: positiveStylePrompt,
    negative_style_prompt: negativeStylePrompt,
    vae: vae ?? undefined,
  });

  const seamless = addSeamless(state, g, denoise, modelLoader, vaeLoader);

  addSDXLLoRas(state, g, denoise, modelLoader, seamless, posCond, negCond);

  // We might get the VAE from the main model, custom VAE, or seamless node.
  const vaeSource = seamless ?? vaeLoader ?? modelLoader;
  g.addEdge(vaeSource, 'vae', l2i, 'vae');
  g.addEdge(vaeSource, 'vae', i2l, 'vae');

  // Add Refiner if enabled
  if (refinerModel) {
    await addSDXLRefiner(state, g, denoise, seamless, posCond, negCond, l2i);
  }

  const _addedCAs = addControlAdapters(state.canvasV2.controlAdapters.entities, g, denoise, modelConfig.base);
  const _addedIPAs = addIPAdapters(state.canvasV2.ipAdapters.entities, g, denoise, modelConfig.base);
  const _addedRegions = await addRegions(
    manager,
    state.canvasV2.regions.entities,
    g,
    state.canvasV2.document,
    state.canvasV2.bbox,
    modelConfig.base,
    denoise,
    posCond,
    negCond,
    posCondCollect,
    negCondCollect
  );
  const { image_name } = await manager.util.getImageSourceImage({ bbox: state.canvasV2.bbox, preview: true });
  await manager.util.getInpaintMaskImage({ bbox: state.canvasV2.bbox, preview: true });
  i2l.image = { image_name };

  if (state.system.shouldUseNSFWChecker) {
    imageOutput = addNSFWChecker(g, imageOutput);
  }

  if (state.system.shouldUseWatermarker) {
    imageOutput = addWatermarker(g, imageOutput);
  }

  g.setMetadataReceivingNode(imageOutput);
  return g.getGraph();
};
