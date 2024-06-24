import type { RootState } from 'app/store/store';
import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
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
  VAE_LOADER,
} from 'features/nodes/util/graph/constants';
import { addControlAdapters } from 'features/nodes/util/graph/generation/addControlAdapters';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
// import { addHRF } from 'features/nodes/util/graph/generation/addHRF';
import { addIPAdapters } from 'features/nodes/util/graph/generation/addIPAdapters';
import { addLoRAs } from 'features/nodes/util/graph/generation/addLoRAs';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
import { addSeamless } from 'features/nodes/util/graph/generation/addSeamless';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import type { GraphType } from 'features/nodes/util/graph/generation/Graph';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { getBoardField, getSizes } from 'features/nodes/util/graph/graphBuilderUtils';
import type { Invocation } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import { addRegions } from './addRegions';

export const buildSD1Graph = async (state: RootState, manager: KonvaNodeManager): Promise<GraphType> => {
  const generationMode = manager.util.getGenerationMode();

  const { bbox, params } = state.canvasV2;

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
    vae,
    positivePrompt,
    negativePrompt,
  } = params;

  assert(model, 'No model found in state');

  const { originalSize, scaledSize } = getSizes(bbox);

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
    width: scaledSize.width,
    height: scaledSize.height,
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
  });
  const vaeLoader =
    vae?.base === model.base
      ? g.addNode({
          type: 'vae_loader',
          id: VAE_LOADER,
          vae_model: vae,
        })
      : null;

  let imageOutput: Invocation<'l2i' | 'img_nsfw' | 'img_watermark' | 'img_resize' | 'canvas_paste_back'> = l2i;

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
  assert(modelConfig.base === 'sd-1' || modelConfig.base === 'sd-2');

  g.upsertMetadata({
    generation_mode: 'txt2img',
    cfg_scale,
    cfg_rescale_multiplier,
    width: scaledSize.width,
    height: scaledSize.height,
    positive_prompt: positivePrompt,
    negative_prompt: negativePrompt,
    model: Graph.getModelMetadataField(modelConfig),
    seed,
    steps,
    rand_device: shouldUseCpuNoise ? 'cpu' : 'cuda',
    scheduler,
    clip_skip: skipped_layers,
    vae: vae ?? undefined,
  });

  const seamless = addSeamless(state, g, denoise, modelLoader, vaeLoader);

  addLoRAs(state, g, denoise, modelLoader, seamless, clipSkip, posCond, negCond);

  // We might get the VAE from the main model, custom VAE, or seamless node.
  const vaeSource: Invocation<'main_model_loader' | 'seamless' | 'vae_loader'> = seamless ?? vaeLoader ?? modelLoader;
  g.addEdge(vaeSource, 'vae', l2i, 'vae');

  if (generationMode === 'txt2img') {
    addTextToImage(g, l2i, imageOutput, originalSize, scaledSize);
  } else if (generationMode === 'img2img') {
    addImageToImage(
      g,
      manager,
      l2i,
      denoise,
      vaeSource,
      imageOutput,
      originalSize,
      scaledSize,
      bbox,
      params.img2imgStrength
    );
  } else if (generationMode === 'inpaint') {
    const { compositing } = state.canvasV2;
    addInpaint(
      g,
      manager,
      l2i,
      denoise,
      vaeSource,
      modelLoader,
      imageOutput,
      originalSize,
      scaledSize,
      bbox,
      compositing,
      params.img2imgStrength,
      vaePrecision
    );
  } else if (generationMode === 'outpaint') {
    const { compositing } = state.canvasV2;
    addOutpaint(
      g,
      manager,
      l2i,
      denoise,
      vaeSource,
      modelLoader,
      imageOutput,
      originalSize,
      scaledSize,
      bbox,
      compositing,
      params.img2imgStrength,
      vaePrecision
    );
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

  // const isHRFAllowed = !addedLayers.some((l) => isInitialImageLayer(l) || isRegionalGuidanceLayer(l));
  // if (isHRFAllowed && state.hrf.hrfEnabled) {
  //   imageOutput = addHRF(state, g, denoise, noise, l2i, vaeSource);
  // }

  if (state.system.shouldUseNSFWChecker) {
    imageOutput = addNSFWChecker(g, imageOutput);
  }

  if (state.system.shouldUseWatermarker) {
    imageOutput = addWatermarker(g, imageOutput);
  }

  // This is the terminal node and must always save to gallery.
  imageOutput.is_intermediate = false;
  imageOutput.use_cache = false;

  g.setMetadataReceivingNode(imageOutput);
  return g.getGraph();
};
