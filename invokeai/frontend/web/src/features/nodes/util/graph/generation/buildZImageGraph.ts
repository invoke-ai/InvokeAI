import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  selectMainModelConfig,
  selectParamsSlice,
  selectZImageQwen3EncoderModel,
  selectZImageQwen3SourceModel,
  selectZImageVaeModel,
} from 'features/controlLayers/store/paramsSlice';
import { selectCanvasMetadata, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { addZImageControl } from 'features/nodes/util/graph/generation/addControlAdapters';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
import { addRegions } from 'features/nodes/util/graph/generation/addRegions';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { addZImageLoRAs } from 'features/nodes/util/graph/generation/addZImageLoRAs';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { selectCanvasOutputFields, selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn, ImageOutputNodes } from 'features/nodes/util/graph/types';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { Invocation } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const log = logger('system');

export const buildZImageGraph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building Z-Image graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'z-image', 'Selected model is not a Z-Image model');

  // Get Z-Image component models
  const zImageVaeModel = selectZImageVaeModel(state);
  const zImageQwen3EncoderModel = selectZImageQwen3EncoderModel(state);
  const zImageQwen3SourceModel = selectZImageQwen3SourceModel(state);

  // Validate that we have the required models
  const hasVaeSource = zImageVaeModel !== null || zImageQwen3SourceModel !== null;
  const hasQwen3Source = zImageQwen3EncoderModel !== null || zImageQwen3SourceModel !== null;
  assert(hasVaeSource, 'No VAE source: Set either VAE or Qwen3 Source model');
  assert(hasQwen3Source, 'No Qwen3 Encoder source: Set either Qwen3 Encoder or Qwen3 Source model');

  const params = selectParamsSlice(state);

  // Z-Image-Turbo uses guidance_scale (stored as cfgScale), defaults to 1.0 for no CFG
  // (1.0 means no CFG effect, matching FLUX convention)
  const { cfgScale: guidance_scale, steps } = params;

  const prompts = selectPresetModifiedPrompts(state);

  const g = new Graph(getPrefixedId('z_image_graph'));

  const modelLoader = g.addNode({
    type: 'z_image_model_loader',
    id: getPrefixedId('z_image_model_loader'),
    model,
    vae_model: zImageVaeModel ?? undefined,
    qwen3_encoder_model: zImageQwen3EncoderModel ?? undefined,
    qwen3_source_model: zImageQwen3SourceModel ?? undefined,
  });

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const posCond = g.addNode({
    type: 'z_image_text_encoder',
    id: getPrefixedId('pos_prompt'),
  });
  // Collect node for regional prompting support
  const posCondCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('pos_cond_collect'),
  });

  // Z-Image supports negative conditioning when guidance_scale > 0
  // Only create negative conditioning nodes if guidance is used
  let negCond: Invocation<'z_image_text_encoder'> | null = null;
  let negCondCollect: Invocation<'collect'> | null = null;
  if (guidance_scale > 0) {
    negCond = g.addNode({
      type: 'z_image_text_encoder',
      id: getPrefixedId('neg_prompt'),
      prompt: prompts.negative,
    });
    negCondCollect = g.addNode({
      type: 'collect',
      id: getPrefixedId('neg_cond_collect'),
    });
  }

  // Placeholder collect node for IP adapters (not supported for Z-Image but needed for addRegions)
  const ipAdapterCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('ip_adapter_collect'),
  });

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });
  const denoise = g.addNode({
    type: 'z_image_denoise',
    id: getPrefixedId('denoise_latents'),
    guidance_scale,
    steps,
  });
  const l2i = g.addNode({
    type: 'z_image_l2i',
    id: getPrefixedId('l2i'),
  });

  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'qwen3_encoder', posCond, 'qwen3_encoder');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');
  // Connect VAE to denoise for control image encoding
  g.addEdge(modelLoader, 'vae', denoise, 'vae');

  g.addEdge(positivePrompt, 'value', posCond, 'prompt');
  // Connect positive conditioning through collector for regional support
  g.addEdge(posCond, 'conditioning', posCondCollect, 'item');
  g.addEdge(posCondCollect, 'collection', denoise, 'positive_conditioning');

  // Connect negative conditioning if guidance_scale > 0
  if (negCond !== null && negCondCollect !== null) {
    g.addEdge(modelLoader, 'qwen3_encoder', negCond, 'qwen3_encoder');
    g.addEdge(negCond, 'conditioning', negCondCollect, 'item');
    g.addEdge(negCondCollect, 'collection', denoise, 'negative_conditioning');
  }

  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  // Add Z-Image LoRAs if any are enabled
  addZImageLoRAs(state, g, denoise, modelLoader, posCond, negCond);

  // Add regional guidance if canvas manager is available
  const canvas = selectCanvasSlice(state);
  if (manager !== null) {
    // Add Z-Image Control layers if any are enabled
    const rect = canvas.bbox.rect;
    await addZImageControl({
      manager,
      entities: canvas.controlLayers.entities,
      g,
      rect,
      denoise,
    });

    // Add regional guidance
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
      fluxReduxCollect: null, // Not supported for Z-Image
    });
  }

  // IP Adapters are not supported for Z-Image, so delete the unused collector
  g.deleteNode(ipAdapterCollect.id);

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'z-image');

  g.upsertMetadata({
    cfg_scale: guidance_scale,
    negative_prompt: prompts.negative,
    model: Graph.getModelMetadataField(modelConfig),
    steps,
    vae: zImageVaeModel ?? undefined,
    qwen3_encoder: zImageQwen3EncoderModel ?? undefined,
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
    g.upsertMetadata({ generation_mode: 'z_image_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'z_image_i2l',
      id: getPrefixedId('z_image_i2l'),
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
    g.upsertMetadata({ generation_mode: 'z_image_img2img' });
  } else if (generationMode === 'inpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'z_image_i2l',
      id: getPrefixedId('z_image_i2l'),
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
    g.upsertMetadata({ generation_mode: 'z_image_inpaint' });
  } else if (generationMode === 'outpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'z_image_i2l',
      id: getPrefixedId('z_image_i2l'),
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
    g.upsertMetadata({ generation_mode: 'z_image_outpaint' });
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
