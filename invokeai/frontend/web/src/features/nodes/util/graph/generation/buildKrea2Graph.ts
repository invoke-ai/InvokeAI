import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasMetadata, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { isKrea2ReferenceImageConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { zImageField } from 'features/nodes/types/common';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addKrea2LoRAs } from 'features/nodes/util/graph/generation/addKrea2LoRAs';
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

export const buildKrea2Graph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building Krea-2 graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'krea-2', 'Selected model is not a Krea-2 model');

  const params = selectParamsSlice(state);
  // Krea-2-Turbo uses the standard CFG convention; cfg_scale defaults to 1.0 (no CFG) for the distilled model.
  const {
    cfgScale: cfg_scale,
    steps,
    krea2VaeModel,
    krea2Qwen3VlEncoderModel,
    krea2RebalanceEnabled,
    krea2RebalanceMultiplier,
    krea2RebalanceWeights,
    krea2SeedVarianceEnabled,
    krea2SeedVarianceStrength,
    krea2SeedVarianceRandomizePercent,
  } = params;

  // Krea-2 has no source field: a non-diffusers transformer (single-file checkpoint / GGUF) has no
  // bundled VAE or encoder, so both standalone submodels must be selected. (Also enforced in readiness.)
  if (model.format !== 'diffusers') {
    assert(krea2VaeModel, 'Krea-2 non-diffusers models require a VAE to be selected');
    assert(krea2Qwen3VlEncoderModel, 'Krea-2 non-diffusers models require a Qwen3-VL encoder to be selected');
  }

  const prompts = selectPresetModifiedPrompts(state);

  const g = new Graph(getPrefixedId('krea2_graph'));

  const modelLoader = g.addNode({
    type: 'krea2_model_loader',
    id: getPrefixedId('krea2_model_loader'),
    model,
    // Optional standalone submodels (used when the transformer is a single-file checkpoint/GGUF). When
    // unset, the loader extracts the VAE / Qwen3-VL encoder from the diffusers model.
    vae_model: krea2VaeModel ?? undefined,
    qwen3_vl_encoder_model: krea2Qwen3VlEncoderModel ?? undefined,
  });

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const posCond = g.addNode({
    type: 'krea2_text_encoder',
    id: getPrefixedId('pos_prompt'),
  });
  const posCondCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('pos_cond_collect'),
  });
  const ipAdapterCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('ip_adapter_collect'),
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

  type Krea2ConditioningSource = Invocation<
    'krea2_text_encoder' | 'krea2_conditioning_rebalance' | 'krea2_seed_variance'
  >;
  const addConditioningEnhancers = (conditioning: Invocation<'krea2_text_encoder'>): Krea2ConditioningSource => {
    let conditioningSource: Krea2ConditioningSource = conditioning;
    if (krea2RebalanceEnabled) {
      const rebalance = g.addNode({
        type: 'krea2_conditioning_rebalance',
        id: getPrefixedId('krea2_rebalance'),
        multiplier: krea2RebalanceMultiplier,
        per_layer_weights: krea2RebalanceWeights,
      });
      g.addEdge(conditioningSource, 'conditioning', rebalance, 'conditioning');
      conditioningSource = rebalance;
    }
    if (krea2SeedVarianceEnabled && krea2SeedVarianceStrength > 0) {
      const seedVariance = g.addNode({
        type: 'krea2_seed_variance',
        id: getPrefixedId('krea2_seed_variance'),
        strength: krea2SeedVarianceStrength,
        randomize_percent: krea2SeedVarianceRandomizePercent,
      });
      g.addEdge(conditioningSource, 'conditioning', seedVariance, 'conditioning');
      g.addEdge(seed, 'value', seedVariance, 'variance_seed');
      conditioningSource = seedVariance;
    }
    return conditioningSource;
  };

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
  const positiveConditioningSource = addConditioningEnhancers(posCond);
  g.addEdge(positiveConditioningSource, 'conditioning', posCondCollect, 'item');
  g.addEdge(posCondCollect, 'collection', denoise, 'positive_conditioning');

  if (negCond !== null) {
    g.addEdge(modelLoader, 'qwen3_vl_encoder', negCond, 'qwen3_vl_encoder');
    g.addEdge(negCond, 'conditioning', denoise, 'negative_conditioning');
  }

  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  // Apply any enabled Krea-2 LoRAs (reroutes transformer + Qwen3-VL encoder through the collection loader).
  addKrea2LoRAs(state, g, denoise, modelLoader, posCond, negCond);

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
      negCondCollect: null,
      ipAdapterCollect,
      fluxReduxCollect: null,
      transformRegionalPositiveConditioning: (conditioning) => {
        assert(conditioning.type === 'krea2_text_encoder');
        return addConditioningEnhancers(conditioning);
      },
    });
  }
  // Krea-2 does not support *regional* reference-image adapters. Global style reference is handled below.
  g.deleteNode(ipAdapterCollect.id);

  // Global style reference: training-free style transfer via shared-KV reference attention. There is no
  // adapter model, and the technique supports exactly one reference, so consume the first valid entity.
  const styleRefEntity = selectRefImagesSlice(state).entities.find(
    (entity) =>
      entity.isEnabled &&
      isKrea2ReferenceImageConfig(entity.config) &&
      entity.config.image !== null &&
      getGlobalReferenceImageWarnings(entity, model).length === 0
  );
  if (styleRefEntity && isKrea2ReferenceImageConfig(styleRefEntity.config) && styleRefEntity.config.image) {
    const { image, styleStrength } = styleRefEntity.config;
    const styleReference = g.addNode({
      type: 'krea2_style_reference',
      id: getPrefixedId('krea2_style_reference'),
      image: zImageField.parse(image.crop?.image ?? image.original.image),
      // The reference's image tokens are appended to the target's, so both must be the same size.
      width: denoise.width,
      height: denoise.height,
      style_strength: styleStrength,
    });
    g.addEdge(modelLoader, 'vae', styleReference, 'vae');
    g.addEdge(styleReference, 'style_reference', denoise, 'style_reference');
    g.upsertMetadata({ krea2_style_strength: styleStrength });
  }

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'krea-2');

  g.upsertMetadata({
    cfg_scale,
    model: Graph.getModelMetadataField(modelConfig),
    steps,
    // Standalone submodels (used for single-file / GGUF transformers) - recorded so they recall.
    vae: krea2VaeModel ?? undefined,
    qwen3_vl_encoder: krea2Qwen3VlEncoderModel ?? undefined,
    // Conditioning enhancer settings (default off) - recorded so they recall.
    krea2_seed_variance_enabled: krea2SeedVarianceEnabled,
    krea2_seed_variance_strength: krea2SeedVarianceStrength,
    krea2_seed_variance_randomize_percent: krea2SeedVarianceRandomizePercent,
    krea2_rebalance_enabled: krea2RebalanceEnabled,
    krea2_rebalance_multiplier: krea2RebalanceMultiplier,
    krea2_rebalance_weights: krea2RebalanceWeights,
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
