import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasMetadata } from 'features/controlLayers/store/selectors';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
import {
  addPidDecode,
  addPidImageToImageNative,
  buildPidDecodeChain,
} from 'features/nodes/util/graph/generation/addPidDecode';
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

const log = logger('system');

export const buildSD3Graph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building SD3 graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model found in state');
  assert(model.base === 'sd-3');

  const params = selectParamsSlice(state);

  const { cfgScale: cfg_scale, steps, vae, t5EncoderModel, clipLEmbedModel, clipGEmbedModel, pidMode } = params;

  const g = new Graph(getPrefixedId('sd3_graph'));

  const modelLoader = g.addNode({
    type: 'sd3_model_loader',
    id: getPrefixedId('sd3_model_loader'),
    model,
    t5_encoder_model: t5EncoderModel,
    clip_l_model: clipLEmbedModel,
    clip_g_model: clipGEmbedModel,
    vae_model: vae,
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
    type: 'sd3_text_encoder',
    id: getPrefixedId('pos_cond'),
  });

  const negCond = g.addNode({
    type: 'sd3_text_encoder',
    id: getPrefixedId('neg_cond'),
  });

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });
  const denoise = g.addNode({
    type: 'sd3_denoise',
    id: getPrefixedId('sd3_denoise'),
    cfg_scale,
    steps,
    denoising_start: 0,
    denoising_end: 1,
  });
  const l2i = g.addNode({
    type: 'sd3_l2i',
    id: getPrefixedId('l2i'),
  });

  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'clip_l', posCond, 'clip_l');
  g.addEdge(modelLoader, 'clip_l', negCond, 'clip_l');
  g.addEdge(modelLoader, 'clip_g', posCond, 'clip_g');
  g.addEdge(modelLoader, 'clip_g', negCond, 'clip_g');
  g.addEdge(modelLoader, 't5_encoder', posCond, 't5_encoder');
  g.addEdge(modelLoader, 't5_encoder', negCond, 't5_encoder');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');

  g.addEdge(positivePrompt, 'value', posCond, 'prompt');
  g.addEdge(negativePrompt, 'value', negCond, 'prompt');
  g.addEdge(posCond, 'conditioning', denoise, 'positive_conditioning');
  g.addEdge(negCond, 'conditioning', denoise, 'negative_conditioning');

  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  g.upsertMetadata({
    cfg_scale,
    model: Graph.getModelMetadataField(model),
    steps,
    vae: vae ?? undefined,
  });
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');
  g.addEdgeToMetadata(negativePrompt, 'value', 'negative_prompt');

  let canvasOutput: Invocation<ImageOutputNodes> = l2i;

  if (pidMode !== 'off') {
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

  if (generationMode === 'txt2img') {
    if (pidMode !== 'off') {
      // PiD replaces the VAE decode entirely - drop the unused l2i (and its edges). sd3_pid_decode has no vae
      // input (fixed SD3 constants), so no vaeSource is passed.
      g.deleteNode(l2i.id);
      canvasOutput = addPidDecode({
        g,
        state,
        mode: pidMode,
        denoise,
        decodeNodeType: 'sd3_pid_decode',
        positivePrompt,
        seed,
      });
    } else {
      canvasOutput = addTextToImage({
        g,
        state,
        denoise,
        l2i,
      });
    }
    g.upsertMetadata({ generation_mode: 'sd3_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'sd3_i2l',
      id: getPrefixedId('sd3_i2l'),
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
        decodeNodeType: 'sd3_pid_decode',
        i2l,
        vaeSource: modelLoader,
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
        decodeNodeType: 'sd3_pid_decode',
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
        denoise,
        vaeSource: modelLoader,
      });
    } else {
      canvasOutput = await addImageToImage({
        g,
        state,
        manager,
        l2i,
        i2l,
        denoise,
        vaeSource: modelLoader,
      });
    }
    g.upsertMetadata({ generation_mode: 'sd3_img2img' });
  } else if (generationMode === 'inpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'sd3_i2l',
      id: getPrefixedId('sd3_i2l'),
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
    g.upsertMetadata({ generation_mode: 'sd3_inpaint' });
  } else if (generationMode === 'outpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'sd3_i2l',
      id: getPrefixedId('sd3_i2l'),
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
    g.upsertMetadata({ generation_mode: 'sd3_outpaint' });
  } else {
    assert<Equals<typeof generationMode, never>>(false);
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
