import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasMetadata } from 'features/controlLayers/store/selectors';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addKrea2LoRAs } from 'features/nodes/util/graph/generation/addKrea2LoRAs';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
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

export const buildKrea2Graph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building Krea-2 graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'krea-2', 'Selected model is not a Krea-2 model');
  // The VAE (Qwen-Image VAE) and Qwen3-VL encoder are extracted from the diffusers pipeline.
  assert(model.format === 'diffusers', 'Krea-2 currently requires a Diffusers-format model');

  const params = selectParamsSlice(state);
  // Krea-2-Turbo uses the standard CFG convention; cfg_scale defaults to 1.0 (no CFG) for the distilled model.
  const {
    cfgScale: cfg_scale,
    steps,
    krea2RebalanceEnabled,
    krea2RebalanceMultiplier,
    krea2RebalanceWeights,
    krea2SeedVarianceEnabled,
    krea2SeedVarianceStrength,
    krea2SeedVarianceRandomizePercent,
  } = params;

  const prompts = selectPresetModifiedPrompts(state);

  const g = new Graph(getPrefixedId('krea2_graph'));

  const modelLoader = g.addNode({
    type: 'krea2_model_loader',
    id: getPrefixedId('krea2_model_loader'),
    model,
  });

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const posCond = g.addNode({
    type: 'krea2_text_encoder',
    id: getPrefixedId('pos_prompt'),
  });

  // Krea-2 supports negative conditioning only when CFG is enabled (cfg_scale > 1).
  let negCond: Invocation<'krea2_text_encoder'> | null = null;
  if (cfg_scale > 1) {
    negCond = g.addNode({
      type: 'krea2_text_encoder',
      id: getPrefixedId('neg_prompt'),
      prompt: prompts.negative,
    });
  }

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });
  const denoise = g.addNode({
    type: 'krea2_denoise',
    id: getPrefixedId('denoise_latents'),
    cfg_scale,
    steps,
  });
  // Krea-2 decodes with the Qwen-Image VAE, so reuse the Qwen-Image latents-to-image node.
  const l2i = g.addNode({
    type: 'qwen_image_l2i',
    id: getPrefixedId('l2i'),
  });

  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'qwen3_vl_encoder', posCond, 'qwen3_vl_encoder');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');

  g.addEdge(positivePrompt, 'value', posCond, 'prompt');

  // Optional conditioning enhancers between the text encoder and denoise. Both default OFF (params), so
  // by default the conditioning flows straight through and stock Krea-2 behaviour is unchanged. Order:
  // rebalance (scale the signal toward the prompt) first, then seed variance (perturb for variety).
  if (krea2RebalanceEnabled) {
    const rebalance = g.addNode({
      type: 'krea2_conditioning_rebalance',
      id: getPrefixedId('krea2_rebalance'),
      multiplier: krea2RebalanceMultiplier,
      per_layer_weights: krea2RebalanceWeights,
    });
    g.addEdge(posCond, 'conditioning', rebalance, 'conditioning');

    if (krea2SeedVarianceEnabled && krea2SeedVarianceStrength > 0) {
      const seedVariance = g.addNode({
        type: 'krea2_seed_variance',
        id: getPrefixedId('krea2_seed_variance'),
        strength: krea2SeedVarianceStrength,
        randomize_percent: krea2SeedVarianceRandomizePercent,
      });
      g.addEdge(rebalance, 'conditioning', seedVariance, 'conditioning');
      g.addEdge(seed, 'value', seedVariance, 'variance_seed');
      g.addEdge(seedVariance, 'conditioning', denoise, 'positive_conditioning');
    } else {
      g.addEdge(rebalance, 'conditioning', denoise, 'positive_conditioning');
    }
  } else if (krea2SeedVarianceEnabled && krea2SeedVarianceStrength > 0) {
    const seedVariance = g.addNode({
      type: 'krea2_seed_variance',
      id: getPrefixedId('krea2_seed_variance'),
      strength: krea2SeedVarianceStrength,
      randomize_percent: krea2SeedVarianceRandomizePercent,
    });
    g.addEdge(posCond, 'conditioning', seedVariance, 'conditioning');
    g.addEdge(seed, 'value', seedVariance, 'variance_seed');
    g.addEdge(seedVariance, 'conditioning', denoise, 'positive_conditioning');
  } else {
    g.addEdge(posCond, 'conditioning', denoise, 'positive_conditioning');
  }

  if (negCond !== null) {
    g.addEdge(modelLoader, 'qwen3_vl_encoder', negCond, 'qwen3_vl_encoder');
    g.addEdge(negCond, 'conditioning', denoise, 'negative_conditioning');
  }

  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  // Apply any enabled Krea-2 LoRAs (reroutes transformer + Qwen3-VL encoder through the collection loader).
  addKrea2LoRAs(state, g, denoise, modelLoader, posCond, negCond);

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'krea-2');

  g.upsertMetadata({
    cfg_scale,
    model: Graph.getModelMetadataField(modelConfig),
    steps,
  });
  // Only record a negative prompt when CFG is enabled (cfg_scale > 1). Krea-2-Turbo runs with CFG
  // disabled by default, in which case there is no negative conditioning - recording it would surface a
  // spurious (often empty) negative prompt on metadata recall.
  if (cfg_scale > 1) {
    g.upsertMetadata({ negative_prompt: prompts.negative });
  }
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  let canvasOutput: Invocation<ImageOutputNodes> = l2i;

  if (generationMode === 'txt2img') {
    canvasOutput = addTextToImage({ g, state, denoise, l2i });
    g.upsertMetadata({ generation_mode: 'krea2_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager !== null);
    const i2l = g.addNode({ type: 'qwen_image_i2l', id: getPrefixedId('qwen_image_i2l') });
    canvasOutput = await addImageToImage({ g, state, manager, denoise, l2i, i2l, vaeSource: modelLoader });
    g.upsertMetadata({ generation_mode: 'krea2_img2img' });
  } else if (generationMode === 'inpaint') {
    assert(manager !== null);
    const i2l = g.addNode({ type: 'qwen_image_i2l', id: getPrefixedId('qwen_image_i2l') });
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
    g.upsertMetadata({ generation_mode: 'krea2_inpaint' });
  } else if (generationMode === 'outpaint') {
    assert(manager !== null);
    const i2l = g.addNode({ type: 'qwen_image_i2l', id: getPrefixedId('qwen_image_i2l') });
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
    g.upsertMetadata({ generation_mode: 'krea2_outpaint' });
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
