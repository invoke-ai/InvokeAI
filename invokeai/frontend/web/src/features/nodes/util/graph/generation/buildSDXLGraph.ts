import type { RootState } from 'app/store/store';
import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import {
  CANVAS_OUTPUT,
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
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addIPAdapters } from 'features/nodes/util/graph/generation/addIPAdapters';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
import { addSDXLLoRAs } from 'features/nodes/util/graph/generation/addSDXLLoRAs';
import { addSDXLRefiner } from 'features/nodes/util/graph/generation/addSDXLRefiner';
import { addSeamless } from 'features/nodes/util/graph/generation/addSeamless';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { getBoardField, getSDXLStylePrompts, getSizes } from 'features/nodes/util/graph/graphBuilderUtils';
import type { Invocation, NonNullableGraph } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import { addRegions } from './addRegions';

export const buildSDXLGraph = async (state: RootState, manager: KonvaNodeManager): Promise<NonNullableGraph> => {
  const generationMode = manager.getGenerationMode();

  const { bbox, params } = state.canvasV2;

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
  } = params;

  assert(model, 'No model found in state');

  const { originalSize, scaledSize } = getSizes(bbox);

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
  });
  const vaeLoader =
    vae?.base === model.base
      ? g.addNode({
          type: 'vae_loader',
          id: VAE_LOADER,
          vae_model: vae,
        })
      : null;

  let canvasOutput: Invocation<'l2i' | 'img_nsfw' | 'img_watermark' | 'img_resize' | 'canvas_paste_back'> = l2i;

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
  assert(modelConfig.base === 'sdxl');

  g.upsertMetadata({
    generation_mode: 'sdxl_txt2img',
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
    positive_style_prompt: positiveStylePrompt,
    negative_style_prompt: negativeStylePrompt,
    vae: vae ?? undefined,
  });

  const seamless = addSeamless(state, g, denoise, modelLoader, vaeLoader);

  addSDXLLoRAs(state, g, denoise, modelLoader, seamless, posCond, negCond);

  // We might get the VAE from the main model, custom VAE, or seamless node.
  const vaeSource = seamless ?? vaeLoader ?? modelLoader;
  g.addEdge(vaeSource, 'vae', l2i, 'vae');

  // Add Refiner if enabled
  if (refinerModel) {
    await addSDXLRefiner(state, g, denoise, seamless, posCond, negCond, l2i);
  }

  if (generationMode === 'txt2img') {
    canvasOutput = addTextToImage(g, l2i, originalSize, scaledSize);
  } else if (generationMode === 'img2img') {
    canvasOutput = await addImageToImage(
      g,
      manager,
      l2i,
      denoise,
      vaeSource,
      originalSize,
      scaledSize,
      bbox,
      refinerModel ? Math.min(refinerStart, 1 - params.img2imgStrength) : 1 - params.img2imgStrength
    );
  } else if (generationMode === 'inpaint') {
    const { compositing } = state.canvasV2;
    canvasOutput = await addInpaint(
      g,
      manager,
      l2i,
      denoise,
      vaeSource,
      modelLoader,
      originalSize,
      scaledSize,
      bbox,
      compositing,
      refinerModel ? Math.min(refinerStart, 1 - params.img2imgStrength) : 1 - params.img2imgStrength,
      vaePrecision
    );
  } else if (generationMode === 'outpaint') {
    const { compositing } = state.canvasV2;
    canvasOutput = await addOutpaint(
      g,
      manager,
      l2i,
      denoise,
      vaeSource,
      modelLoader,
      originalSize,
      scaledSize,
      bbox,
      compositing,
      refinerModel ? Math.min(refinerStart, 1 - params.img2imgStrength) : 1 - params.img2imgStrength,
      vaePrecision
    );
  }

  const _addedCAs = addControlAdapters(state.canvasV2.controlAdapters.entities, g, denoise, modelConfig.base);
  const _addedIPAs = addIPAdapters(state.canvasV2.ipAdapters.entities, g, denoise, modelConfig.base);
  const _addedRegions = await addRegions(
    manager,
    state.canvasV2.regions.entities,
    g,
    state.canvasV2.bbox,
    modelConfig.base,
    denoise,
    posCond,
    negCond,
    posCondCollect,
    negCondCollect
  );

  if (state.system.shouldUseNSFWChecker) {
    canvasOutput = addNSFWChecker(g, canvasOutput);
  }

  if (state.system.shouldUseWatermarker) {
    canvasOutput = addWatermarker(g, canvasOutput);
  }

  // This is the terminal node and must always save to gallery.
  g.updateNode(canvasOutput, {
    id: CANVAS_OUTPUT,
    is_intermediate: false,
    use_cache: false,
    board: getBoardField(state),
  });

  g.setMetadataReceivingNode(canvasOutput);
  return g.getGraph();
};
