import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasMetadata, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { addControlNets, addT2IAdapters } from 'features/nodes/util/graph/generation/addControlAdapters';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addIPAdapters } from 'features/nodes/util/graph/generation/addIPAdapters';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
import {
  addPidDecode,
  addPidImageToImageNative,
  buildPidDecodeChain,
} from 'features/nodes/util/graph/generation/addPidDecode';
import { addSDXLLoRAs } from 'features/nodes/util/graph/generation/addSDXLLoRAs';
import { addSDXLRefiner } from 'features/nodes/util/graph/generation/addSDXLRefiner';
import { addSeamless } from 'features/nodes/util/graph/generation/addSeamless';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  getOriginalAndScaledSizesForOtherModes,
  getOriginalAndScaledSizesForTextToImage,
  selectCanvasOutputFields,
} from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn, ImageOutputNodes } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { t } from 'i18next';
import type { Invocation } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

import { addRegions } from './addRegions';

const log = logger('system');

export const buildSDXLGraph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building SDXL graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'sdxl', 'Selected model is not a SDXL Kontext model');

  const params = selectParamsSlice(state);
  const canvas = selectCanvasSlice(state);
  const refImages = selectRefImagesSlice(state);

  const {
    cfgScale: cfg_scale,
    cfgRescaleMultiplier: cfg_rescale_multiplier,
    scheduler,
    steps,
    shouldUseCpuNoise,
    colorCompensation,
    vaePrecision,
    vae,
    refinerModel,
    pidMode,
  } = params;

  const fp32 = vaePrecision === 'fp32';
  const compensation = colorCompensation ? 'SDXL' : 'None';
  const g = new Graph(getPrefixedId('sdxl_graph'));

  const modelLoader = g.addNode({
    type: 'sdxl_model_loader',
    id: getPrefixedId('sdxl_model_loader'),
    model,
  });

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const negativePrompt = g.addNode({
    id: getPrefixedId('negative_prompt'),
    type: 'string',
  });
  const posCond = g.addNode({
    type: 'sdxl_compel_prompt',
    id: getPrefixedId('pos_cond'),
  });
  const posCondCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('pos_cond_collect'),
  });

  const negCond = g.addNode({
    type: 'sdxl_compel_prompt',
    id: getPrefixedId('neg_cond'),
  });
  const negCondCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('neg_cond_collect'),
  });

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });
  const noise = g.addNode({
    type: 'noise',
    id: getPrefixedId('noise'),
    use_cpu: shouldUseCpuNoise,
  });
  const denoise = g.addNode({
    type: 'denoise_latents',
    id: getPrefixedId('denoise_latents'),
    cfg_scale,
    cfg_rescale_multiplier,
    scheduler,
    steps,
  });
  const l2i = g.addNode({
    type: 'l2i',
    id: getPrefixedId('l2i'),
    fp32,
  });
  const vaeLoader =
    vae?.base === model.base
      ? g.addNode({
          type: 'vae_loader',
          id: getPrefixedId('vae'),
          vae_model: vae,
        })
      : null;

  g.addEdge(modelLoader, 'unet', denoise, 'unet');
  g.addEdge(modelLoader, 'clip', posCond, 'clip');
  g.addEdge(modelLoader, 'clip', negCond, 'clip');
  g.addEdge(modelLoader, 'clip2', posCond, 'clip2');
  g.addEdge(modelLoader, 'clip2', negCond, 'clip2');

  g.addEdge(positivePrompt, 'value', posCond, 'prompt');
  g.addEdge(positivePrompt, 'value', posCond, 'style');
  g.addEdge(negativePrompt, 'value', negCond, 'prompt');
  g.addEdge(negativePrompt, 'value', negCond, 'style');

  g.addEdge(posCond, 'conditioning', posCondCollect, 'item');
  g.addEdge(posCondCollect, 'collection', denoise, 'positive_conditioning');

  g.addEdge(negCond, 'conditioning', negCondCollect, 'item');
  g.addEdge(negCondCollect, 'collection', denoise, 'negative_conditioning');

  g.addEdge(seed, 'value', noise, 'seed');
  g.addEdge(noise, 'noise', denoise, 'noise');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  g.upsertMetadata({
    cfg_scale,
    cfg_rescale_multiplier,
    model: Graph.getModelMetadataField(model),
    steps,
    rand_device: shouldUseCpuNoise ? 'cpu' : 'cuda',
    scheduler,
    vae: vae ?? undefined,
  });
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');
  g.addEdgeToMetadata(negativePrompt, 'value', 'negative_prompt');

  const seamless = addSeamless(state, g, denoise, modelLoader, vaeLoader);

  addSDXLLoRAs(state, g, denoise, modelLoader, seamless, posCond, negCond);

  // We might get the VAE from the main model, custom VAE, or seamless node.
  const vaeSource = seamless ?? vaeLoader ?? modelLoader;
  g.addEdge(vaeSource, 'vae', l2i, 'vae');

  if (pidMode !== 'off') {
    // PiD replaces the VAE decode with a 4x super-res decode. It is not compatible with the SDXL refiner
    // (the refiner runs its own denoise + decode path) - block that combination for now.
    if (refinerModel) {
      throw new UnsupportedGenerationModeError(t('toast.pidUnsupportedMode'));
    }
    // Inpaint/outpaint are not wired for PiD yet - only txt2img and img2img are supported (Fit and Native).
    if (generationMode === 'inpaint' || generationMode === 'outpaint') {
      throw new UnsupportedGenerationModeError(t('toast.pidUnsupportedMode'));
    }
    // PiD decodes at 4x the generation resolution. "Scale Before Processing" (Canvas) would silently inflate
    // the generation size to the model optimal, blowing up the decode - require it off (scaled == original).
    const { originalSize, scaledSize } = getOriginalAndScaledSizesForTextToImage(state);
    if (scaledSize.width !== originalSize.width || scaledSize.height !== originalSize.height) {
      throw new UnsupportedGenerationModeError(t('toast.pidScaleBeforeProcessingOff'));
    }
  }

  // Add Refiner if enabled (never reached together with PiD - the guard above throws first).
  if (refinerModel) {
    await addSDXLRefiner(state, g, denoise, seamless, posCond, negCond, l2i);
  }

  let canvasOutput: Invocation<ImageOutputNodes> = l2i;

  if (generationMode === 'txt2img') {
    if (pidMode !== 'off') {
      // PiD replaces the VAE decode entirely - drop the unused l2i (and its edges). SDXL's VAE source is wired
      // so sdxl_pid_decode can read scaling_factor / shift_factor from it.
      g.deleteNode(l2i.id);
      canvasOutput = addPidDecode({
        g,
        state,
        mode: pidMode,
        denoise,
        noise,
        decodeNodeType: 'sdxl_pid_decode',
        vaeSource,
        positivePrompt,
        seed,
      });
    } else {
      canvasOutput = addTextToImage({
        g,
        state,
        noise,
        denoise,
        l2i,
      });
    }
    g.upsertMetadata({ generation_mode: 'sdxl_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'i2l',
      id: getPrefixedId('i2l'),
      fp32,
      color_compensation: compensation,
    });
    if (pidMode === 'native') {
      // PiD replaces the VAE decode. Native: the bbox is the 4x target - generate at bbox / 4, PiD decodes
      // straight back up to the bbox (no downscale), so the full result composites onto the canvas region.
      g.deleteNode(l2i.id);
      canvasOutput = await addPidImageToImageNative({
        g,
        state,
        manager,
        denoise,
        noise,
        decodeNodeType: 'sdxl_pid_decode',
        i2l,
        vaeSource,
        positivePrompt,
        seed,
      });
    } else if (pidMode === 'fit') {
      // PiD replaces the VAE decode. Fit: generate at the bbox, PiD decodes 4x, then downscale back to the bbox.
      g.deleteNode(l2i.id);
      const { originalSize } = getOriginalAndScaledSizesForOtherModes(state);
      const pidDecode = buildPidDecodeChain({
        g,
        state,
        denoise,
        noise,
        decodeNodeType: 'sdxl_pid_decode',
        vaeSource,
        positivePrompt,
        seed,
        mode: 'fit',
        fitSize: originalSize,
      });
      canvasOutput = await addImageToImage({
        g,
        state,
        manager,
        l2i: pidDecode,
        i2l,
        noise,
        denoise,
        vaeSource,
      });
    } else {
      canvasOutput = await addImageToImage({
        g,
        state,
        manager,
        l2i,
        i2l,
        noise,
        denoise,
        vaeSource,
      });
    }
    g.upsertMetadata({ generation_mode: 'sdxl_img2img' });
  } else if (generationMode === 'inpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'i2l',
      id: getPrefixedId('i2l'),
      fp32,
      color_compensation: compensation,
    });
    canvasOutput = await addInpaint({
      g,
      state,
      manager,
      l2i,
      i2l,
      noise,
      denoise,
      vaeSource,
      modelLoader,
      seed,
    });
    g.upsertMetadata({ generation_mode: 'sdxl_inpaint' });
  } else if (generationMode === 'outpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'i2l',
      id: getPrefixedId('i2l'),
      fp32,
      color_compensation: compensation,
    });
    canvasOutput = await addOutpaint({
      g,
      state,
      manager,
      l2i,
      i2l,
      noise,
      denoise,
      vaeSource,
      modelLoader,
      seed,
    });
    g.upsertMetadata({ generation_mode: 'sdxl_outpaint' });
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

    const t2iAdapterCollector = g.addNode({
      type: 'collect',
      id: getPrefixedId('t2i_adapter_collector'),
    });
    const t2iAdapterResult = await addT2IAdapters({
      manager,
      entities: canvas.controlLayers.entities,
      g,
      rect: canvas.bbox.rect,
      collector: t2iAdapterCollector,
      model,
    });
    if (t2iAdapterResult.addedT2IAdapters > 0) {
      g.addEdge(t2iAdapterCollector, 'collection', denoise, 't2i_adapter');
    } else {
      g.deleteNode(t2iAdapterCollector.id);
    }
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

  if (manager !== null) {
    const regionsResult = await addRegions({
      manager,
      regions: canvas.regionalGuidance.entities,
      g,
      bbox: canvas.bbox.rect,
      model,
      posCond,
      negCond,
      posCondCollect,
      negCondCollect,
      ipAdapterCollect,
      fluxReduxCollect: null,
    });
    totalIPAdaptersAdded += regionsResult.reduce((acc, r) => acc + r.addedIPAdapters, 0);
  }

  if (totalIPAdaptersAdded > 0) {
    g.addEdge(ipAdapterCollect, 'collection', denoise, 'ip_adapter');
  } else {
    g.deleteNode(ipAdapterCollect.id);
  }

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
    negativePrompt,
  };
};
