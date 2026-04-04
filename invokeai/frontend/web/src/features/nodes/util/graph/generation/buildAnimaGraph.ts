import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  selectAnimaQwen3EncoderModel,
  selectAnimaScheduler,
  selectAnimaT5EncoderModel,
  selectAnimaVaeModel,
  selectMainModelConfig,
  selectParamsSlice,
} from 'features/controlLayers/store/paramsSlice';
import { selectCanvasMetadata, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { addAnimaLoRAs } from 'features/nodes/util/graph/generation/addAnimaLoRAs';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
import { addRegions } from 'features/nodes/util/graph/generation/addRegions';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { selectCanvasOutputFields, selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn, ImageOutputNodes } from 'features/nodes/util/graph/types';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { Invocation } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const log = logger('system');

export const buildAnimaGraph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building Anima graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'anima', 'Selected model is not an Anima model');

  // Get Anima component models
  const animaVaeModel = selectAnimaVaeModel(state);
  const animaQwen3EncoderModel = selectAnimaQwen3EncoderModel(state);
  const animaT5EncoderModel = selectAnimaT5EncoderModel(state);
  const animaScheduler = selectAnimaScheduler(state);

  // Validate required component models
  assert(
    animaVaeModel !== null,
    'No VAE model selected for Anima. Set a compatible VAE (Wan 2.1 QwenImage or FLUX VAE).'
  );
  assert(animaQwen3EncoderModel !== null, 'No Qwen3 Encoder model selected for Anima. Set a Qwen3 0.6B encoder model.');
  assert(animaT5EncoderModel !== null, 'No T5 Encoder model selected for Anima. Set a T5-XXL encoder model.');

  const params = selectParamsSlice(state);
  const { cfgScale: guidance_scale, steps } = params;

  const prompts = selectPresetModifiedPrompts(state);

  const g = new Graph(getPrefixedId('anima_graph'));

  const modelLoader = g.addNode({
    type: 'anima_model_loader',
    id: getPrefixedId('anima_model_loader'),
    model,
    vae_model: animaVaeModel,
    qwen3_encoder_model: animaQwen3EncoderModel,
    t5_encoder_model: animaT5EncoderModel,
  });

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const posCond = g.addNode({
    type: 'anima_text_encoder',
    id: getPrefixedId('pos_prompt'),
  });
  // Collect node for regional prompting support
  const posCondCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('pos_cond_collect'),
  });

  // Anima supports negative conditioning when guidance_scale > 1
  let negCond: Invocation<'anima_text_encoder'> | null = null;
  let negCondCollect: Invocation<'collect'> | null = null;
  if (guidance_scale > 1) {
    negCond = g.addNode({
      type: 'anima_text_encoder',
      id: getPrefixedId('neg_prompt'),
      prompt: prompts.negative,
    });
    negCondCollect = g.addNode({
      type: 'collect',
      id: getPrefixedId('neg_cond_collect'),
    });
  }

  // Placeholder collect node for IP adapters (not supported for Anima but needed for addRegions)
  const ipAdapterCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('ip_adapter_collect'),
  });

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });
  const denoise = g.addNode({
    type: 'anima_denoise',
    id: getPrefixedId('denoise_latents'),
    guidance_scale,
    steps,
    scheduler: animaScheduler,
  });
  const l2i = g.addNode({
    type: 'anima_l2i',
    id: getPrefixedId('l2i'),
  });

  // Connect model loader outputs
  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'qwen3_encoder', posCond, 'qwen3_encoder');
  g.addEdge(modelLoader, 't5_encoder', posCond, 't5_encoder');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');

  // Connect positive prompt through collector for regional support
  g.addEdge(positivePrompt, 'value', posCond, 'prompt');
  g.addEdge(posCond, 'conditioning', posCondCollect, 'item');
  g.addEdge(posCondCollect, 'collection', denoise, 'positive_conditioning');

  // Connect negative conditioning if guidance_scale > 1
  if (negCond !== null && negCondCollect !== null) {
    g.addEdge(modelLoader, 'qwen3_encoder', negCond, 'qwen3_encoder');
    g.addEdge(modelLoader, 't5_encoder', negCond, 't5_encoder');
    g.addEdge(negCond, 'conditioning', negCondCollect, 'item');
    g.addEdge(negCondCollect, 'collection', denoise, 'negative_conditioning');
  }

  // Connect seed and denoiser to L2I
  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'anima');

  g.upsertMetadata({
    cfg_scale: guidance_scale,
    negative_prompt: prompts.negative,
    model: Graph.getModelMetadataField(modelConfig),
    steps,
    scheduler: animaScheduler,
    vae: animaVaeModel ?? undefined,
    qwen3_encoder: animaQwen3EncoderModel ?? undefined,
    t5_encoder: animaT5EncoderModel ?? undefined,
  });
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  // Add LoRAs before regions so regional nodes copy the LoRA-patched encoder edges
  addAnimaLoRAs(state, g, denoise, modelLoader, posCond, negCond);

  // Add regional guidance if canvas manager is available
  const canvas = selectCanvasSlice(state);
  if (manager !== null) {
    await addRegions({
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
      fluxReduxCollect: null, // Not supported for Anima
    });
  }

  // IP Adapters are not supported for Anima, so delete the unused collector
  g.deleteNode(ipAdapterCollect.id);

  let canvasOutput: Invocation<ImageOutputNodes> = l2i;

  if (generationMode === 'txt2img') {
    canvasOutput = addTextToImage({
      g,
      state,
      denoise,
      l2i,
    });
    g.upsertMetadata({ generation_mode: 'anima_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'anima_i2l',
      id: getPrefixedId('anima_i2l'),
    });

    canvasOutput = await addImageToImage({
      g,
      state,
      manager,
      denoise,
      l2i,
      i2l,
      vaeSource: modelLoader,
    });
    g.upsertMetadata({ generation_mode: 'anima_img2img' });
  } else if (generationMode === 'inpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'anima_i2l',
      id: getPrefixedId('anima_i2l'),
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
    g.upsertMetadata({ generation_mode: 'anima_inpaint' });
  } else if (generationMode === 'outpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'anima_i2l',
      id: getPrefixedId('anima_i2l'),
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
    g.upsertMetadata({ generation_mode: 'anima_outpaint' });
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
  };
};
