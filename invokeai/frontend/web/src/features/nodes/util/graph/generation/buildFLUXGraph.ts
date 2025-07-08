import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasMetadata, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { isFluxKontextReferenceImageConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import { addFLUXFill } from 'features/nodes/util/graph/generation/addFLUXFill';
import { addFLUXLoRAs } from 'features/nodes/util/graph/generation/addFLUXLoRAs';
import { addFLUXReduxes } from 'features/nodes/util/graph/generation/addFLUXRedux';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
import { addRegions } from 'features/nodes/util/graph/generation/addRegions';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { selectCanvasOutputFields } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn, ImageOutputNodes } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { t } from 'i18next';
import type { Invocation } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

import { addControlLoRA, addControlNets } from './addControlAdapters';
import { addIPAdapters } from './addIPAdapters';

const log = logger('system');

export const buildFLUXGraph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;
  log.debug({ generationMode, manager: manager?.id }, 'Building FLUX graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'flux', 'Selected model is not a FLUX model');

  const params = selectParamsSlice(state);
  const canvas = selectCanvasSlice(state);
  const refImages = selectRefImagesSlice(state);

  const { guidance: baseGuidance, steps, fluxVAE, t5EncoderModel, clipEmbedModel } = params;

  assert(t5EncoderModel, 'No T5 Encoder model found in state');
  assert(clipEmbedModel, 'No CLIP Embed model found in state');
  assert(fluxVAE, 'No FLUX VAE model found in state');

  const isFLUXFill = model.variant === 'inpaint';
  let guidance = baseGuidance;
  if (isFLUXFill) {
    // FLUX Fill doesn't work with Text to Image or Image to Image generation modes. Well, technically, it does, but
    // the outputs are garbagio.
    //
    // Unfortunately, we do not know the generation mode until the user clicks Invoke, so this is the first place we
    // can check for this incompatibility.
    //
    // We are opting to fail loudly instead of produce garbage images, hence this being an assert.
    //
    // The message in this assert will be shown in a toast to the user, so we are using the translation system for it.
    //
    // The other asserts above are just for sanity & type check and should never be hit, so they do not have
    // translations.
    if (generationMode === 'txt2img' || generationMode === 'img2img') {
      throw new UnsupportedGenerationModeError(t('toast.fluxFillIncompatibleWithT2IAndI2I'));
    }

    // FLUX Fill wants much higher guidance values than normal FLUX - silently "fix" the value for the user.
    // TODO(psyche): Figure out a way to alert the user that this is happening - maybe return warnings from the graph
    // builder and toast them?
    guidance = 30;
  }

  const isFluxKontextDev = model.name?.toLowerCase().includes('kontext');
  if (isFluxKontextDev) {
    if (generationMode !== 'txt2img') {
      throw new UnsupportedGenerationModeError(t('toast.fluxKontextIncompatibleGenerationMode'));
    }
  }

  const g = new Graph(getPrefixedId('flux_graph'));

  const modelLoader = g.addNode({
    type: 'flux_model_loader',
    id: getPrefixedId('flux_model_loader'),
    model,
    t5_encoder_model: t5EncoderModel,
    clip_embed_model: clipEmbedModel,
    vae_model: fluxVAE,
  });

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const posCond = g.addNode({
    type: 'flux_text_encoder',
    id: getPrefixedId('flux_text_encoder'),
  });
  const posCondCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('pos_cond_collect'),
  });

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });
  const denoise = g.addNode({
    type: 'flux_denoise',
    id: getPrefixedId('flux_denoise'),
    guidance,
    num_steps: steps,
  });

  const l2i = g.addNode({
    type: 'flux_vae_decode',
    id: getPrefixedId('flux_vae_decode'),
  });

  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'vae', denoise, 'controlnet_vae');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');

  g.addEdge(modelLoader, 'clip', posCond, 'clip');
  g.addEdge(modelLoader, 't5_encoder', posCond, 't5_encoder');
  g.addEdge(modelLoader, 'max_seq_len', posCond, 't5_max_seq_len');

  g.addEdge(positivePrompt, 'value', posCond, 'prompt');
  g.addEdge(posCond, 'conditioning', posCondCollect, 'item');
  g.addEdge(posCondCollect, 'collection', denoise, 'positive_text_conditioning');

  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  addFLUXLoRAs(state, g, denoise, modelLoader, posCond);

  g.upsertMetadata({
    guidance,
    model: Graph.getModelMetadataField(model),
    steps,
    vae: fluxVAE,
    t5_encoder: t5EncoderModel,
    clip_embed_model: clipEmbedModel,
  });
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  if (isFluxKontextDev) {
    const validFLUXKontextConfigs = selectRefImagesSlice(state)
      .entities.filter((entity) => entity.isEnabled)
      .filter((entity) => isFluxKontextReferenceImageConfig(entity.config))
      .filter((entity) => getGlobalReferenceImageWarnings(entity, model).length === 0);

    // FLUX Kontext supports only a single conditioning image - we'll just take the first one.
    // In the future, we can explore concatenating multiple conditioning images in image or latent space.
    const firstValidFLUXKontextConfig = validFLUXKontextConfigs[0];

    if (firstValidFLUXKontextConfig) {
      const { image } = firstValidFLUXKontextConfig.config;

      assert(image, 'getGlobalReferenceImageWarnings checks if the image is there, this should never raise');

      const kontextConditioning = g.addNode({
        type: 'flux_kontext',
        id: getPrefixedId('flux_kontext'),
        image,
      });
      g.addEdge(kontextConditioning, 'kontext_cond', denoise, 'kontext_conditioning');
      g.upsertMetadata({ ref_images: [firstValidFLUXKontextConfig] }, 'merge');
    }
  }

  let canvasOutput: Invocation<ImageOutputNodes> = l2i;

  if (isFLUXFill && (generationMode === 'inpaint' || generationMode === 'outpaint')) {
    assert(manager !== null);
    canvasOutput = await addFLUXFill({
      g,
      state,
      manager,
      l2i,
      denoise,
    });
  } else if (generationMode === 'txt2img') {
    canvasOutput = addTextToImage({
      g,
      state,
      denoise,
      l2i,
    });
    g.upsertMetadata({ generation_mode: 'flux_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'flux_vae_encode',
      id: getPrefixedId('flux_vae_encode'),
    });
    canvasOutput = await addImageToImage({
      g,
      state,
      manager,
      l2i,
      i2l,
      denoise,
      vaeSource: modelLoader,
    });
    g.upsertMetadata({ generation_mode: 'flux_img2img' });
  } else if (generationMode === 'inpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'flux_vae_encode',
      id: getPrefixedId('flux_vae_encode'),
    });
    canvasOutput = await addInpaint({
      g,
      state,
      manager,
      l2i,
      i2l,
      denoise,
      vaeSource: modelLoader,
      modelLoader,
      seed,
    });
    g.upsertMetadata({ generation_mode: 'flux_inpaint' });
  } else if (generationMode === 'outpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'flux_vae_encode',
      id: getPrefixedId('flux_vae_encode'),
    });
    canvasOutput = await addOutpaint({
      g,
      state,
      manager,
      l2i,
      i2l,
      denoise,
      vaeSource: modelLoader,
      modelLoader,
      seed,
    });
    g.upsertMetadata({ generation_mode: 'flux_outpaint' });
  } else {
    assert<Equals<typeof generationMode, never>>(false);
  }

  if (manager !== null) {
    const controlNetCollector = g.addNode({
      type: 'collect',
      id: getPrefixedId('control_net_collector'),
    });
    const controlNetResult = await addControlNets({
      manager,
      entities: canvas.controlLayers.entities,
      g,
      rect: canvas.bbox.rect,
      collector: controlNetCollector,
      model,
    });
    if (controlNetResult.addedControlNets > 0) {
      g.addEdge(controlNetCollector, 'collection', denoise, 'control');
    } else {
      g.deleteNode(controlNetCollector.id);
    }

    await addControlLoRA({
      manager,
      entities: canvas.controlLayers.entities,
      g,
      rect: canvas.bbox.rect,
      denoise,
      model,
    });
  }

  const ipAdapterCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('ip_adapter_collector'),
  });
  const ipAdapterResult = addIPAdapters({
    entities: refImages.entities,
    g,
    collector: ipAdapterCollect,
    model,
  });

  let totalIPAdaptersAdded = ipAdapterResult.addedIPAdapters;

  const fluxReduxCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('ip_adapter_collector'),
  });
  const fluxReduxResult = addFLUXReduxes({
    entities: refImages.entities,
    g,
    collector: fluxReduxCollect,
    model,
  });
  let totalReduxesAdded = fluxReduxResult.addedFLUXReduxes;

  if (manager !== null) {
    const regionsResult = await addRegions({
      manager,
      regions: canvas.regionalGuidance.entities,
      g,
      bbox: canvas.bbox.rect,
      model,
      posCond,
      negCond: null,
      posCondCollect,
      negCondCollect: null,
      ipAdapterCollect,
      fluxReduxCollect,
    });

    totalIPAdaptersAdded += regionsResult.reduce((acc, r) => acc + r.addedIPAdapters, 0);
    totalReduxesAdded += regionsResult.reduce((acc, r) => acc + r.addedFLUXReduxes, 0);
  }

  if (totalIPAdaptersAdded > 0) {
    g.addEdge(ipAdapterCollect, 'collection', denoise, 'ip_adapter');
  } else {
    g.deleteNode(ipAdapterCollect.id);
  }

  if (totalReduxesAdded > 0) {
    g.addEdge(fluxReduxCollect, 'collection', denoise, 'redux_conditioning');
  } else {
    g.deleteNode(fluxReduxCollect.id);
  }

  // TODO: Add FLUX Reduxes to denoise node like we do for ipa

  if (state.system.shouldUseNSFWChecker) {
    canvasOutput = addNSFWChecker(g, canvasOutput);
  }

  if (state.system.shouldUseWatermarker) {
    canvasOutput = addWatermarker(g, canvasOutput);
  }

  g.updateNode(canvasOutput, selectCanvasOutputFields(state));

  if (selectActiveTab(state) === 'canvas') {
    g.upsertMetadata(selectCanvasMetadata(state));
  }

  g.setMetadataReceivingNode(canvasOutput);

  return {
    g,
    seed,
    positivePrompt,
  };
};
