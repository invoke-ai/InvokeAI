import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectCanvasSettingsSlice } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { getBoardField, getPresetModifiedPrompts, getSizes } from 'features/nodes/util/graph/graphBuilderUtils';
import type { Invocation } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

const log = logger('system');

export const buildFLUXGraph = async (
  state: RootState,
  manager: CanvasManager
): Promise<{ g: Graph; noise: Invocation<'noise' | 'flux_denoise'>; posCond: Invocation<'flux_text_encoder'> }> => {
  const generationMode = manager.compositor.getGenerationMode();
  log.debug({ generationMode }, 'Building FLUX graph');

  const params = selectParamsSlice(state);
  const canvasSettings = selectCanvasSettingsSlice(state);
  const canvas = selectCanvasSlice(state);

  const { bbox } = canvas;

  const { originalSize, scaledSize } = getSizes(bbox);

  const { model, guidance, seed, steps, fluxVAE, t5EncoderModel, clipEmbedModel, img2imgStrength } = params;

  assert(model, 'No model found in state');
  assert(t5EncoderModel, 'No T5 Encoder model found in state');
  assert(clipEmbedModel, 'No CLIP Embed model found in state');
  assert(fluxVAE, 'No FLUX VAE model found in state');

  const { positivePrompt } = getPresetModifiedPrompts(state);

  const g = new Graph(getPrefixedId('flux_graph'));
  const modelLoader = g.addNode({
    type: 'flux_model_loader',
    id: getPrefixedId('flux_model_loader'),
    model,
    t5_encoder_model: t5EncoderModel,
    clip_embed_model: clipEmbedModel,
    vae_model: fluxVAE,
  });

  const posCond = g.addNode({
    type: 'flux_text_encoder',
    id: getPrefixedId('flux_text_encoder'),
    prompt: positivePrompt,
  });

  const noise = g.addNode({
    type: 'flux_denoise',
    id: getPrefixedId('flux_denoise'),
    guidance,
    num_steps: steps,
    seed,
    denoising_start: 0, // denoising_start should be 0 when latents are not provided
    denoising_end: 1,
    width: scaledSize.width,
    height: scaledSize.height,
  });

  const l2i = g.addNode({
    type: 'flux_vae_decode',
    id: getPrefixedId('flux_vae_decode'),
  });

  let canvasOutput: Invocation<
    'l2i' | 'img_nsfw' | 'img_watermark' | 'img_resize' | 'canvas_v2_mask_and_crop' | 'flux_vae_decode'
  > = l2i;

  g.addEdge(modelLoader, 'transformer', noise, 'transformer');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');

  g.addEdge(modelLoader, 'clip', posCond, 'clip');
  g.addEdge(modelLoader, 't5_encoder', posCond, 't5_encoder');
  g.addEdge(modelLoader, 'max_seq_len', posCond, 't5_max_seq_len');

  g.addEdge(posCond, 'conditioning', noise, 'positive_text_conditioning');

  g.addEdge(noise, 'latents', l2i, 'latents');

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'flux');

  g.upsertMetadata({
    generation_mode: 'flux_txt2img',
    guidance,
    width: scaledSize.width,
    height: scaledSize.height,
    positive_prompt: positivePrompt,
    model: Graph.getModelMetadataField(modelConfig),
    seed,
    steps,
    vae: fluxVAE,
    t5_encoder: t5EncoderModel,
    clip_embed_model: clipEmbedModel,
  });

  if (generationMode === 'txt2img') {
    canvasOutput = addTextToImage(g, l2i, originalSize, scaledSize);
  } else if (generationMode === 'img2img') {
    canvasOutput = await addImageToImage(
      g,
      manager,
      l2i,
      noise,
      modelLoader,
      originalSize,
      scaledSize,
      bbox,
      1 - img2imgStrength,
      false
    );
  } else if (generationMode === 'inpaint') {
    canvasOutput = await addInpaint(
      state,
      g,
      manager,
      l2i,
      noise,
      modelLoader,
      modelLoader,
      originalSize,
      scaledSize,
      1 - img2imgStrength,
      false
    );
  } else if (generationMode === 'outpaint') {
    canvasOutput = await addOutpaint(
      state,
      g,
      manager,
      l2i,
      noise,
      modelLoader,
      modelLoader,
      originalSize,
      scaledSize,
      1 - img2imgStrength,
      false
    );
  }

  if (state.system.shouldUseNSFWChecker) {
    canvasOutput = addNSFWChecker(g, canvasOutput);
  }

  if (state.system.shouldUseWatermarker) {
    canvasOutput = addWatermarker(g, canvasOutput);
  }

  const shouldSaveToGallery = !canvasSettings.sendToCanvas || canvasSettings.autoSave;

  g.updateNode(canvasOutput, {
    id: getPrefixedId('canvas_output'),
    is_intermediate: !shouldSaveToGallery,
    use_cache: false,
    board: getBoardField(state),
  });

  g.setMetadataReceivingNode(canvasOutput);
  return { g, noise, posCond };
};
