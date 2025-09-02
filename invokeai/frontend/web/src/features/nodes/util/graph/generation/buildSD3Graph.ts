import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectSanitizedCanvasMetadata } from 'features/controlLayers/store/selectors';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { selectCanvasOutputFields, selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn, ImageOutputNodes } from 'features/nodes/util/graph/types';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
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

  const { cfgScale: cfg_scale, steps, vae, t5EncoderModel, clipLEmbedModel, clipGEmbedModel } = params;

  const prompts = selectPresetModifiedPrompts(state);

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
  const posCond = g.addNode({
    type: 'sd3_text_encoder',
    id: getPrefixedId('pos_cond'),
  });

  const negCond = g.addNode({
    type: 'sd3_text_encoder',
    id: getPrefixedId('neg_cond'),
    prompt: prompts.negative,
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
  g.addEdge(posCond, 'conditioning', denoise, 'positive_conditioning');
  g.addEdge(negCond, 'conditioning', denoise, 'negative_conditioning');

  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  g.upsertMetadata({
    cfg_scale,
    negative_prompt: prompts.negative,
    model: Graph.getModelMetadataField(model),
    steps,
    vae: vae ?? undefined,
  });
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  let canvasOutput: Invocation<ImageOutputNodes> = l2i;

  if (generationMode === 'txt2img') {
    canvasOutput = addTextToImage({
      g,
      state,
      denoise,
      l2i,
    });
    g.upsertMetadata({ generation_mode: 'sd3_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'sd3_i2l',
      id: getPrefixedId('sd3_i2l'),
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
    const canvasMetadata = selectSanitizedCanvasMetadata(state);
    if (canvasMetadata) {
      g.upsertMetadata(canvasMetadata);
    }
  }

  g.setMetadataReceivingNode(canvasOutput);
  return {
    g,
    seed,
    positivePrompt,
  };
};
