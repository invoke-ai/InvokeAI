import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  selectKleinQwen3EncoderModel,
  selectKleinVaeModel,
  selectMainModelConfig,
  selectParamsSlice,
} from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasMetadata, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { isFluxKontextReferenceImageConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import { zImageField } from 'features/nodes/types/common';
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
  assert(model.base === 'flux' || model.base === 'flux2', 'Selected model is not a FLUX model');

  const params = selectParamsSlice(state);
  const canvas = selectCanvasSlice(state);
  const refImages = selectRefImagesSlice(state);

  const { guidance: baseGuidance, steps, fluxScheduler, fluxVAE, t5EncoderModel, clipEmbedModel } = params;

  // Flux2 (Klein) uses Qwen3 instead of CLIP+T5
  // VAE and Qwen3 encoder can be extracted from the main Diffusers model or selected separately
  const isFlux2 = model.base === 'flux2';
  const kleinVaeModel = selectKleinVaeModel(state);
  const kleinQwen3EncoderModel = selectKleinQwen3EncoderModel(state);

  if (!isFlux2) {
    assert(t5EncoderModel, 'No T5 Encoder model found in state');
    assert(clipEmbedModel, 'No CLIP Embed model found in state');
    assert(fluxVAE, 'No FLUX VAE model found in state');
  }

  const isFLUXFill = model.variant === 'dev_fill';
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

  // Create model loader and text encoder nodes based on variant
  // Klein uses Qwen3 instead of CLIP+T5
  let modelLoader: Invocation<'flux_model_loader'> | Invocation<'flux2_klein_model_loader'>;
  let posCond: Invocation<'flux_text_encoder'> | Invocation<'flux2_klein_text_encoder'>;
  let denoise: Invocation<'flux_denoise'> | Invocation<'flux2_denoise'>;
  let posCondCollect: Invocation<'collect'> | null = null;

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });

  // Use appropriate VAE decode node based on model type
  // FLUX.2 Klein uses a 32-channel VAE (AutoencoderKLFlux2)
  let l2i: Invocation<'flux_vae_decode'> | Invocation<'flux2_vae_decode'>;
  if (isFlux2) {
    l2i = g.addNode({
      type: 'flux2_vae_decode',
      id: getPrefixedId('flux2_vae_decode'),
    });
  } else {
    l2i = g.addNode({
      type: 'flux_vae_decode',
      id: getPrefixedId('flux_vae_decode'),
    });
  }

  if (isFlux2) {
    // Flux2 Klein: Use Qwen3-based model loader, text encoder, and dedicated denoise node
    // VAE and Qwen3 encoder can be extracted from the main Diffusers model or selected separately
    modelLoader = g.addNode({
      type: 'flux2_klein_model_loader',
      id: getPrefixedId('flux2_klein_model_loader'),
      model,
      // Optional: Use separately selected VAE and Qwen3 encoder models
      vae_model: kleinVaeModel ?? undefined,
      qwen3_encoder_model: kleinQwen3EncoderModel ?? undefined,
    });

    posCond = g.addNode({
      type: 'flux2_klein_text_encoder',
      id: getPrefixedId('flux2_klein_text_encoder'),
    });

    denoise = g.addNode({
      type: 'flux2_denoise',
      id: getPrefixedId('flux2_denoise'),
      num_steps: steps,
      scheduler: fluxScheduler,
    });

    // Klein: Connect Qwen3 encoder outputs
    const kleinLoader = modelLoader as Invocation<'flux2_klein_model_loader'>;
    const kleinCond = posCond as Invocation<'flux2_klein_text_encoder'>;
    g.addEdge(kleinLoader, 'qwen3_encoder', kleinCond, 'qwen3_encoder');
    g.addEdge(kleinLoader, 'max_seq_len', kleinCond, 'max_seq_len');
    g.addEdge(kleinLoader, 'transformer', denoise, 'transformer');
    g.addEdge(kleinLoader, 'vae', denoise, 'vae'); // VAE needed for BN statistics
    g.addEdge(kleinLoader, 'vae', l2i, 'vae');
    g.addEdge(positivePrompt, 'value', kleinCond, 'prompt');
    g.addEdge(kleinCond, 'conditioning', denoise, 'positive_text_conditioning');
  } else {
    // Standard FLUX: Use CLIP+T5 model loader and text encoder
    modelLoader = g.addNode({
      type: 'flux_model_loader',
      id: getPrefixedId('flux_model_loader'),
      model,
      t5_encoder_model: t5EncoderModel!,
      clip_embed_model: clipEmbedModel!,
      vae_model: fluxVAE,
    });

    posCond = g.addNode({
      type: 'flux_text_encoder',
      id: getPrefixedId('flux_text_encoder'),
    });

    denoise = g.addNode({
      type: 'flux_denoise',
      id: getPrefixedId('flux_denoise'),
      guidance,
      num_steps: steps,
      scheduler: fluxScheduler,
    });

    posCondCollect = g.addNode({
      type: 'collect',
      id: getPrefixedId('pos_cond_collect'),
    });

    // Standard FLUX: Connect CLIP and T5 encoder outputs
    const fluxLoader = modelLoader as Invocation<'flux_model_loader'>;
    const fluxCond = posCond as Invocation<'flux_text_encoder'>;
    g.addEdge(fluxLoader, 'clip', fluxCond, 'clip');
    g.addEdge(fluxLoader, 't5_encoder', fluxCond, 't5_encoder');
    g.addEdge(fluxLoader, 'max_seq_len', fluxCond, 't5_max_seq_len');
    g.addEdge(fluxLoader, 'transformer', denoise, 'transformer');
    g.addEdge(fluxLoader, 'vae', denoise, 'controlnet_vae');
    g.addEdge(fluxLoader, 'vae', l2i, 'vae');
    g.addEdge(positivePrompt, 'value', fluxCond, 'prompt');
    g.addEdge(fluxCond, 'conditioning', posCondCollect, 'item');
    g.addEdge(posCondCollect, 'collection', denoise, 'positive_text_conditioning');
  }

  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  // Metadata
  if (isFlux2) {
    // VAE and Qwen3 encoder can come from the main model or be selected separately
    const flux2Metadata: Record<string, unknown> = {
      model: Graph.getModelMetadataField(model),
      steps,
      scheduler: fluxScheduler,
    };
    if (kleinVaeModel) {
      flux2Metadata.vae = kleinVaeModel;
    }
    if (kleinQwen3EncoderModel) {
      flux2Metadata.qwen3_encoder = kleinQwen3EncoderModel;
    }
    g.upsertMetadata(flux2Metadata);
  } else {
    g.upsertMetadata({
      guidance,
      model: Graph.getModelMetadataField(model),
      steps,
      scheduler: fluxScheduler,
      vae: fluxVAE,
      t5_encoder: t5EncoderModel,
      clip_embed_model: clipEmbedModel,
    });
  }
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  let canvasOutput: Invocation<ImageOutputNodes> = l2i;

  // Flux2 Klein only supports txt2img for now
  if (isFlux2) {
    if (generationMode !== 'txt2img') {
      throw new UnsupportedGenerationModeError(t('toast.flux2OnlySupportsT2I'));
    }
    canvasOutput = addTextToImage({
      g,
      state,
      denoise,
      l2i,
    });
    g.upsertMetadata({ generation_mode: 'flux2_txt2img' });
  } else {
    // Standard FLUX path with all features
    const fluxDenoise = denoise as Invocation<'flux_denoise'>;
    const fluxModelLoader = modelLoader as Invocation<'flux_model_loader'>;
    const fluxPosCond = posCond as Invocation<'flux_text_encoder'>;
    const fluxL2i = l2i as Invocation<'flux_vae_decode'>;

    // Only add FLUX LoRAs for non-Klein models
    addFLUXLoRAs(state, g, fluxDenoise, fluxModelLoader, fluxPosCond);

    if (isFluxKontextDev) {
      const validFLUXKontextConfigs = selectRefImagesSlice(state)
        .entities.filter((entity) => entity.isEnabled)
        .filter((entity) => isFluxKontextReferenceImageConfig(entity.config))
        .filter((entity) => getGlobalReferenceImageWarnings(entity, model).length === 0);

      if (validFLUXKontextConfigs.length > 0) {
        const fluxKontextCollect = g.addNode({
          type: 'collect',
          id: getPrefixedId('flux_kontext_collect'),
        });
        for (const { config } of validFLUXKontextConfigs) {
          const kontextImagePrep = g.addNode({
            id: getPrefixedId('flux_kontext_image_prep'),
            type: 'flux_kontext_image_prep',
            images: [zImageField.parse(config.image?.crop?.image ?? config.image?.original.image)],
          });
          const kontextConditioning = g.addNode({
            type: 'flux_kontext',
            id: getPrefixedId('flux_kontext'),
          });
          g.addEdge(kontextImagePrep, 'image', kontextConditioning, 'image');
          g.addEdge(kontextConditioning, 'kontext_cond', fluxKontextCollect, 'item');
        }
        g.addEdge(fluxKontextCollect, 'collection', fluxDenoise, 'kontext_conditioning');

        g.upsertMetadata({ ref_images: [validFLUXKontextConfigs] }, 'merge');
      }
    }

    if (isFLUXFill && (generationMode === 'inpaint' || generationMode === 'outpaint')) {
      assert(manager !== null);
      canvasOutput = await addFLUXFill({
        g,
        state,
        manager,
        l2i: fluxL2i,
        denoise: fluxDenoise,
      });
    } else if (generationMode === 'txt2img') {
      canvasOutput = addTextToImage({
        g,
        state,
        denoise: fluxDenoise,
        l2i: fluxL2i,
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
        l2i: fluxL2i,
        i2l,
        denoise: fluxDenoise,
        vaeSource: fluxModelLoader,
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
        l2i: fluxL2i,
        i2l,
        denoise: fluxDenoise,
        vaeSource: fluxModelLoader,
        modelLoader: fluxModelLoader,
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
        l2i: fluxL2i,
        i2l,
        denoise: fluxDenoise,
        vaeSource: fluxModelLoader,
        modelLoader: fluxModelLoader,
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
        g.addEdge(controlNetCollector, 'collection', fluxDenoise, 'control');
      } else {
        g.deleteNode(controlNetCollector.id);
      }

      await addControlLoRA({
        manager,
        entities: canvas.controlLayers.entities,
        g,
        rect: canvas.bbox.rect,
        denoise: fluxDenoise,
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
      id: getPrefixedId('flux_redux_collector'),
    });
    const fluxReduxResult = addFLUXReduxes({
      entities: refImages.entities,
      g,
      collector: fluxReduxCollect,
      model,
    });
    let totalReduxesAdded = fluxReduxResult.addedFLUXReduxes;

    // Use posCondCollect from the else block (only exists for standard FLUX, not FLUX.2 Klein)
    if (manager !== null && posCondCollect !== null) {
      const regionsResult = await addRegions({
        manager,
        regions: canvas.regionalGuidance.entities,
        g,
        bbox: canvas.bbox.rect,
        model,
        posCond: fluxPosCond,
        negCond: null,
        posCondCollect: posCondCollect,
        negCondCollect: null,
        ipAdapterCollect,
        fluxReduxCollect,
      });

      totalIPAdaptersAdded += regionsResult.reduce((acc, r) => acc + r.addedIPAdapters, 0);
      totalReduxesAdded += regionsResult.reduce((acc, r) => acc + r.addedFLUXReduxes, 0);
    }

    if (totalIPAdaptersAdded > 0) {
      g.addEdge(ipAdapterCollect, 'collection', fluxDenoise, 'ip_adapter');
    } else {
      g.deleteNode(ipAdapterCollect.id);
    }

    if (totalReduxesAdded > 0) {
      g.addEdge(fluxReduxCollect, 'collection', fluxDenoise, 'redux_conditioning');
    } else {
      g.deleteNode(fluxReduxCollect.id);
    }
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
