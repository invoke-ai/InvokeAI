import type {
  BackendGraphContract,
  BackendInvocationContract,
  GenerationProjectSettings,
  ResultDestination,
} from '@features/generation/core/contracts';

import type {
  CompiledGenerateGraph,
  ComponentModelConfig,
  GenerateLora,
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateReferenceImageAsset,
  GenerateReferenceImageConfig,
  GenerateSettings,
  MainModelConfig,
} from './types';

import {
  coerceSchedulerForGraph,
  getExternalProviderNodeType,
  getFlux2DiffusersComponentSource,
  getGenerationDimensions,
  getGenerationValidationReasons,
  type SupportedGenerateBase,
} from './baseGenerationPolicies';
import {
  getCompatibleDiffusersComponentSource,
  isAnimaQwen3Encoder,
  isAnimaVae,
  isFlux2Qwen3EncoderForModel,
  isNonAnimaQwen3Encoder,
  isVaeForBases,
} from './componentCompatibility';
import {
  addEdge,
  addMetadata,
  addNode,
  createId,
  getActiveCompatibleLoras,
  toModelIdentifier,
  toGraphContract,
} from './graphBuilder';
import { getEffectiveReferenceImage } from './referenceImage';
import { SEED_MAX } from './settings';

const getCompatibleComponentSource = (
  settings: GenerateSettings,
  model: MainModelConfig
): MainModelConfig | undefined => getCompatibleDiffusersComponentSource(model, settings.componentSourceModel);

const getCompatibleVae = (settings: GenerateSettings, bases: readonly string[]) =>
  settings.vae && isVaeForBases(bases)(settings.vae) ? settings.vae : null;

const toImageField = (image: GenerateReferenceImageAsset) => ({
  image_name: getEffectiveReferenceImage(image).image_name,
});

const getEnabledReferenceImages = <T extends GenerateReferenceImageConfig['type']>(
  settings: GenerateSettings,
  type: T
): Array<GenerateReferenceImage & { config: Extract<GenerateReferenceImageConfig, { type: T }> }> =>
  settings.referenceImages.filter(
    (
      referenceImage
    ): referenceImage is GenerateReferenceImage & { config: Extract<GenerateReferenceImageConfig, { type: T }> } =>
      referenceImage.isEnabled && referenceImage.config.type === type && referenceImage.config.image !== null
  );

const addReferenceImageMetadata = (
  graph: BackendGraphContract,
  outputNode: BackendInvocationContract,
  settings: GenerateSettings
) => {
  const enabled = settings.referenceImages.filter(
    (referenceImage) => referenceImage.isEnabled && referenceImage.config.image
  );

  if (enabled.length === 0) {
    return;
  }

  const metadata = Object.values(graph.nodes).find((node) => node.type === 'core_metadata');

  if (metadata) {
    metadata.ref_images = enabled;
    return;
  }

  outputNode.ref_images = enabled;
};

const addIPAdapterReferenceImages = (
  graph: BackendGraphContract,
  settings: GenerateSettings,
  model: MainModelConfig,
  denoise: BackendInvocationContract
) => {
  const refs = getEnabledReferenceImages(settings, 'ip_adapter').filter((ref) => ref.config.model?.base === model.base);

  if (refs.length === 0) {
    return;
  }

  const collector = addNode(graph, { id: createId('ip_adapter_collector'), type: 'collect' });

  for (const ref of refs) {
    const { beginEndStepPct, clipVisionModel, image, method, model: adapterModel, weight } = ref.config;

    if (!image || !adapterModel) {
      continue;
    }

    const node = addNode(graph, {
      begin_step_percent: beginEndStepPct[0],
      clip_vision_model: clipVisionModel,
      end_step_percent: beginEndStepPct[1],
      id: createId('ip_adapter'),
      image: toImageField(image),
      ip_adapter_model: adapterModel,
      method,
      type: model.base === 'flux' ? 'flux_ip_adapter' : 'ip_adapter',
      weight,
    });

    addEdge(graph, node, 'ip_adapter', collector, 'item');
  }

  addEdge(graph, collector, 'collection', denoise, 'ip_adapter');
};

const FLUX_REDUX_INFLUENCE = {
  lowest: { downsampling_factor: 5, weight: 1 },
  low: { downsampling_factor: 4, weight: 1 },
  medium: { downsampling_factor: 3, weight: 1 },
  high: { downsampling_factor: 2, weight: 1 },
  highest: { downsampling_factor: 1, weight: 1 },
} as const;

const addFluxReduxReferenceImages = (
  graph: BackendGraphContract,
  settings: GenerateSettings,
  model: MainModelConfig,
  denoise: BackendInvocationContract
) => {
  const refs = getEnabledReferenceImages(settings, 'flux_redux').filter((ref) => ref.config.model?.base === model.base);

  if (refs.length === 0) {
    return;
  }

  const collector = addNode(graph, { id: createId('flux_redux_collector'), type: 'collect' });

  for (const ref of refs) {
    const { image, imageInfluence, model: reduxModel } = ref.config;

    if (!image || !reduxModel) {
      continue;
    }

    const node = addNode(graph, {
      id: createId('flux_redux'),
      image: toImageField(image),
      redux_model: reduxModel,
      type: 'flux_redux',
      ...FLUX_REDUX_INFLUENCE[imageInfluence],
    });

    addEdge(graph, node, 'redux_cond', collector, 'item');
  }

  addEdge(graph, collector, 'collection', denoise, 'redux_conditioning');
};

const addFluxKontextReferenceImages = (
  graph: BackendGraphContract,
  settings: GenerateSettings,
  denoise: BackendInvocationContract,
  type: 'flux2_reference_image' | 'flux_kontext_reference_image'
) => {
  const refs = getEnabledReferenceImages(settings, type);

  if (refs.length === 0) {
    return;
  }

  let prevCollect: BackendInvocationContract | null = null;

  for (const ref of refs) {
    const { image } = ref.config;

    if (!image) {
      continue;
    }

    const kontext = addNode(graph, {
      id: createId('flux_kontext'),
      image: toImageField(image),
      type: 'flux_kontext',
    });
    const collect = addNode(graph, {
      id: createId(type === 'flux2_reference_image' ? 'flux2_kontext_collect' : 'flux_kontext_collect'),
      type: 'collect',
    });

    addEdge(graph, kontext, 'kontext_cond', collect, 'item');
    if (prevCollect) {
      addEdge(graph, prevCollect, 'collection', collect, 'collection');
    }
    prevCollect = collect;
  }

  if (prevCollect) {
    addEdge(graph, prevCollect, 'collection', denoise, 'kontext_conditioning');
  }
};

const getDiffusersSource = (settings: GenerateSettings, model: MainModelConfig): MainModelConfig | undefined =>
  getCompatibleComponentSource(settings, model) ?? (model.format === 'diffusers' ? model : undefined);

const requireComponent = <T extends ComponentModelConfig | null>(value: T, label: string): NonNullable<T> => {
  if (!value) {
    throw new Error(`${label} is required for this model family.`);
  }

  return value;
};

const addLoraSelectors = (
  graph: BackendGraphContract,
  loras: GenerateLora[],
  loaderType: string
): BackendInvocationContract => {
  const collector = addNode(graph, { id: createId('lora_collector'), type: 'collect' });
  const loader = addNode(graph, { id: createId(loaderType), type: loaderType });

  addEdge(graph, collector, 'collection', loader, 'loras');

  for (const lora of loras) {
    const selector = addNode(graph, {
      id: createId('lora_selector'),
      lora: toModelIdentifier(lora.model),
      type: 'lora_selector',
      weight: lora.weight,
    });

    addEdge(graph, selector, 'lora', collector, 'item');
  }

  return loader;
};

export const addLoraCollectionLoader = (
  graph: BackendGraphContract,
  loras: GenerateLora[],
  model: GenerateModelConfig,
  sources: {
    clip: BackendInvocationContract;
    clip2?: BackendInvocationContract;
    unet: BackendInvocationContract;
  }
): BackendInvocationContract => {
  const loaderType = model.base === 'sdxl' ? 'sdxl_lora_collection_loader' : 'lora_collection_loader';
  const loader = addLoraSelectors(graph, loras, loaderType);

  addEdge(graph, sources.unet, 'unet', loader, 'unet');
  addEdge(graph, sources.clip, 'clip', loader, 'clip');

  if (sources.clip2) {
    addEdge(graph, sources.clip2, 'clip2', loader, 'clip2');
  }

  return loader;
};

const addTransformerLoraCollectionLoader = (
  graph: BackendGraphContract,
  loras: GenerateLora[],
  loaderType: string,
  modelLoader: BackendInvocationContract,
  fields: string[]
): BackendInvocationContract => {
  const loader = addLoraSelectors(graph, loras, loaderType);

  for (const field of fields) {
    addEdge(graph, modelLoader, field, loader, field);
  }

  return loader;
};

const buildSDGraph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): BackendGraphContract => {
  const graph: BackendGraphContract = { edges: [], id: createId(`${model.base}_graph`), nodes: {} };
  const modelLoaderType = model.base === 'sdxl' ? 'sdxl_model_loader' : 'main_model_loader';
  const compelType = model.base === 'sdxl' ? 'sdxl_compel_prompt' : 'compel';
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const scheduler = coerceSchedulerForGraph(model, settings.scheduler);
  const colorCompensation = settings.colorCompensation ? 'SDXL' : 'None';

  const positivePrompt = addNode(graph, { id: 'positive_prompt', type: 'string' });
  const negativePrompt = addNode(graph, { id: 'negative_prompt', type: 'string' });
  const seed = addNode(graph, { id: 'seed', type: 'integer' });
  const modelLoader = addNode(graph, { id: 'model_loader', model, type: modelLoaderType });
  const posCond = addNode(graph, { id: 'pos_cond', type: compelType });
  const negCond = addNode(graph, { id: 'neg_cond', type: compelType });
  const posCondCollect = addNode(graph, { id: 'pos_cond_collect', type: 'collect' });
  const negCondCollect = addNode(graph, { id: 'neg_cond_collect', type: 'collect' });
  const noise = addNode(graph, {
    height: settings.height,
    id: 'noise',
    type: 'noise',
    use_cpu: projectSettings.useCpuNoise,
    width: settings.width,
  });
  const denoise = addNode(graph, {
    cfg_rescale_multiplier: settings.cfgRescaleMultiplier,
    cfg_scale: settings.cfgScale,
    denoising_end: 1,
    denoising_start: 0,
    id: 'denoise_latents',
    scheduler,
    steps: settings.steps,
    type: 'denoise_latents',
  });
  const output = addNode(graph, {
    color_compensation: colorCompensation,
    fp32: settings.vaePrecision === 'fp32',
    id: 'canvas_output',
    is_intermediate: outputIsIntermediate,
    type: 'l2i',
    use_cache: false,
  });
  // A VAE override only applies when it matches the main model's architecture.
  const vaeLoader =
    settings.vae && settings.vae.base === model.base
      ? addNode(graph, { id: 'vae_loader', type: 'vae_loader', vae_model: settings.vae })
      : null;
  const seamless =
    settings.seamlessXAxis || settings.seamlessYAxis
      ? addNode(graph, {
          id: 'seamless',
          seamless_x: settings.seamlessXAxis,
          seamless_y: settings.seamlessYAxis,
          type: 'seamless',
        })
      : null;
  let unetSource: BackendInvocationContract = seamless ?? modelLoader;
  let clipSource: BackendInvocationContract = modelLoader;
  let clip2Source: BackendInvocationContract | undefined;

  if (model.base === 'sdxl') {
    clip2Source = modelLoader;
    addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
    addEdge(graph, positivePrompt, 'value', posCond, 'style');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
    addEdge(graph, negativePrompt, 'value', negCond, 'style');
  } else {
    const clipSkip = addNode(graph, { id: 'clip_skip', skipped_layers: settings.clipSkip, type: 'clip_skip' });

    addEdge(graph, modelLoader, 'clip', clipSkip, 'clip');
    clipSource = clipSkip;
    addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
  }

  if (activeLoras.length > 0) {
    const loraLoader = addLoraCollectionLoader(graph, activeLoras, model, {
      clip: clipSource,
      clip2: clip2Source,
      unet: unetSource,
    });

    unetSource = loraLoader;
    clipSource = loraLoader;
    clip2Source = model.base === 'sdxl' ? loraLoader : undefined;
  }

  if (model.base === 'sdxl') {
    addEdge(graph, clipSource, 'clip', posCond, 'clip');
    addEdge(graph, clip2Source ?? modelLoader, 'clip2', posCond, 'clip2');
    addEdge(graph, clipSource, 'clip', negCond, 'clip');
    addEdge(graph, clip2Source ?? modelLoader, 'clip2', negCond, 'clip2');
  } else {
    addEdge(graph, clipSource, 'clip', posCond, 'clip');
    addEdge(graph, clipSource, 'clip', negCond, 'clip');
  }

  addEdge(graph, posCond, 'conditioning', posCondCollect, 'item');
  addEdge(graph, posCondCollect, 'collection', denoise, 'positive_conditioning');
  addEdge(graph, negCond, 'conditioning', negCondCollect, 'item');
  addEdge(graph, negCondCollect, 'collection', denoise, 'negative_conditioning');
  addEdge(graph, seed, 'value', noise, 'seed');

  // Seamless slots in between the model loader and the denoise/decode nodes,
  // patching both the UNet and the (possibly overridden) VAE.
  if (seamless) {
    addEdge(graph, modelLoader, 'unet', seamless, 'unet');
    addEdge(graph, vaeLoader ?? modelLoader, 'vae', seamless, 'vae');
  }

  addEdge(graph, unetSource, 'unet', denoise, 'unet');
  addEdge(graph, noise, 'noise', denoise, 'noise');
  addIPAdapterReferenceImages(graph, settings, model, denoise);
  addEdge(graph, denoise, 'latents', output, 'latents');
  addEdge(graph, seamless ?? vaeLoader ?? modelLoader, 'vae', output, 'vae');
  addMetadata(graph, output, settings, model, model.base === 'sdxl' ? 'sdxl_txt2img' : 'txt2img', projectSettings, {
    scheduler,
  });
  addReferenceImageMetadata(graph, output, settings);

  return graph;
};

const addPromptAndSeedNodes = (graph: BackendGraphContract) => ({
  negativePrompt: addNode(graph, { id: 'negative_prompt', type: 'string' }),
  positivePrompt: addNode(graph, { id: 'positive_prompt', type: 'string' }),
  seed: addNode(graph, { id: 'seed', type: 'integer' }),
});

const addImageOutputNode = (graph: BackendGraphContract, type: string, outputIsIntermediate: boolean) =>
  addNode(graph, { id: 'canvas_output', is_intermediate: outputIsIntermediate, type, use_cache: false });

const buildSD3Graph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): BackendGraphContract => {
  const graph: BackendGraphContract = { edges: [], id: createId('sd3_graph'), nodes: {} };
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const modelLoader = addNode(graph, {
    clip_g_model: settings.clipGEmbedModel ?? undefined,
    clip_l_model: settings.clipLEmbedModel ?? undefined,
    id: 'model_loader',
    model,
    t5_encoder_model: settings.t5EncoderModel ?? undefined,
    type: 'sd3_model_loader',
    vae_model: settings.vae ?? undefined,
  });
  const posCond = addNode(graph, { id: 'pos_cond', type: 'sd3_text_encoder' });
  const negCond = addNode(graph, { id: 'neg_cond', type: 'sd3_text_encoder' });
  const denoise = addNode(graph, {
    cfg_scale: settings.cfgScale,
    denoising_end: 1,
    denoising_start: 0,
    height: settings.height,
    id: 'denoise_latents',
    steps: settings.steps,
    type: 'sd3_denoise',
    width: settings.width,
  });
  const output = addImageOutputNode(graph, 'sd3_l2i', outputIsIntermediate);

  addEdge(graph, modelLoader, 'transformer', denoise, 'transformer');
  addEdge(graph, modelLoader, 'clip_l', posCond, 'clip_l');
  addEdge(graph, modelLoader, 'clip_l', negCond, 'clip_l');
  addEdge(graph, modelLoader, 'clip_g', posCond, 'clip_g');
  addEdge(graph, modelLoader, 'clip_g', negCond, 'clip_g');
  addEdge(graph, modelLoader, 't5_encoder', posCond, 't5_encoder');
  addEdge(graph, modelLoader, 't5_encoder', negCond, 't5_encoder');
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');
  addEdge(graph, negCond, 'conditioning', denoise, 'negative_conditioning');
  addEdge(graph, seed, 'value', denoise, 'seed');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addMetadata(graph, output, settings, model, 'sd3_txt2img', projectSettings, {
    scheduler: undefined,
    vae: settings.vae ?? undefined,
  });

  return graph;
};

const buildFluxGraph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): BackendGraphContract => {
  if (model.variant === 'dev_fill') {
    throw new Error('FLUX Fill models do not support text-to-image generation.');
  }

  const graph: BackendGraphContract = { edges: [], id: createId('flux_graph'), nodes: {} };
  const { positivePrompt } = addPromptAndSeedNodes(graph);
  const t5EncoderModel = requireComponent(settings.t5EncoderModel, 'T5 Encoder');
  const clipEmbedModel = requireComponent(settings.clipEmbedModel, 'CLIP Embed');
  const vaeModel = requireComponent(getCompatibleVae(settings, ['flux']), 'FLUX VAE');
  const scheduler = coerceSchedulerForGraph(model, settings.scheduler);
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const modelLoader = addNode(graph, {
    clip_embed_model: clipEmbedModel,
    id: 'model_loader',
    model,
    t5_encoder_model: t5EncoderModel,
    type: 'flux_model_loader',
    vae_model: vaeModel,
  });
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'flux_lora_collection_loader', modelLoader, [
        'transformer',
        'clip',
        't5_encoder',
      ])
    : modelLoader;
  const posCond = addNode(graph, { id: 'pos_cond', type: 'flux_text_encoder' });
  const posCondCollect = addNode(graph, { id: 'pos_cond_collect', type: 'collect' });
  const denoise = addNode(graph, {
    denoising_end: 1,
    denoising_start: 0,
    guidance: settings.cfgScale,
    height: settings.height,
    id: 'denoise_latents',
    num_steps: settings.steps,
    scheduler,
    type: 'flux_denoise',
    width: settings.width,
  });
  const output = addImageOutputNode(graph, 'flux_vae_decode', outputIsIntermediate);

  addEdge(graph, loraSource, 'clip', posCond, 'clip');
  addEdge(graph, loraSource, 't5_encoder', posCond, 't5_encoder');
  addEdge(graph, modelLoader, 'max_seq_len', posCond, 't5_max_seq_len');
  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, modelLoader, 'vae', denoise, 'controlnet_vae');
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', posCondCollect, 'item');
  addEdge(graph, posCondCollect, 'collection', denoise, 'positive_text_conditioning');
  addEdge(graph, graph.nodes.seed, 'value', denoise, 'seed');
  if (model.name.toLowerCase().includes('kontext')) {
    addFluxKontextReferenceImages(graph, settings, denoise, 'flux_kontext_reference_image');
  }
  addIPAdapterReferenceImages(graph, settings, model, denoise);
  addFluxReduxReferenceImages(graph, settings, model, denoise);
  addEdge(graph, denoise, 'latents', output, 'latents');
  addMetadata(graph, output, settings, model, 'flux_txt2img', projectSettings, {
    clip_embed_model: clipEmbedModel,
    guidance: settings.cfgScale,
    scheduler,
    t5_encoder: t5EncoderModel,
    vae: vaeModel,
  });
  addReferenceImageMetadata(graph, output, settings);

  return graph;
};

const buildFlux2Graph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): BackendGraphContract => {
  const sourceModel = getFlux2DiffusersComponentSource(model, settings);
  const hasBundledComponents = model.format === 'diffusers' || sourceModel;
  const qwen3EncoderModel =
    settings.qwen3EncoderModel && isFlux2Qwen3EncoderForModel(model)(settings.qwen3EncoderModel)
      ? settings.qwen3EncoderModel
      : null;
  const vaeModel = getCompatibleVae(settings, ['flux2']);

  if (!hasBundledComponents && (!vaeModel || !qwen3EncoderModel)) {
    throw new Error('FLUX.2 non-Diffusers models require a VAE and Qwen3 Encoder, or a Diffusers component source.');
  }

  const graph: BackendGraphContract = { edges: [], id: createId('flux2_graph'), nodes: {} };
  const { positivePrompt } = addPromptAndSeedNodes(graph);
  const scheduler = coerceSchedulerForGraph(model, settings.scheduler);
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model,
    qwen3_encoder_model: qwen3EncoderModel ?? undefined,
    qwen3_source_model: sourceModel,
    type: 'flux2_klein_model_loader',
    vae_model: vaeModel ?? undefined,
  });
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'flux2_klein_lora_collection_loader', modelLoader, [
        'transformer',
        'qwen3_encoder',
      ])
    : modelLoader;
  const posCond = addNode(graph, { id: 'pos_cond', type: 'flux2_klein_text_encoder' });
  const denoise = addNode(graph, {
    cfg_scale: 1,
    denoising_end: 1,
    denoising_start: 0,
    guidance: settings.cfgScale,
    height: settings.height,
    id: 'denoise_latents',
    num_steps: settings.steps,
    scheduler,
    type: 'flux2_denoise',
    width: settings.width,
  });
  const output = addImageOutputNode(graph, 'flux2_vae_decode', outputIsIntermediate);

  addEdge(graph, loraSource, 'qwen3_encoder', posCond, 'qwen3_encoder');
  addEdge(graph, modelLoader, 'max_seq_len', posCond, 'max_seq_len');
  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, modelLoader, 'vae', denoise, 'vae');
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_text_conditioning');
  addEdge(graph, graph.nodes.seed, 'value', denoise, 'seed');
  addFluxKontextReferenceImages(graph, settings, denoise, 'flux2_reference_image');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addMetadata(graph, output, settings, model, 'flux2_txt2img', projectSettings, {
    qwen3_encoder: qwen3EncoderModel ?? undefined,
    qwen3_source: sourceModel,
    scheduler,
    vae: vaeModel ?? undefined,
  });
  addReferenceImageMetadata(graph, output, settings);

  return graph;
};

const buildCogView4Graph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): BackendGraphContract => {
  const graph: BackendGraphContract = { edges: [], id: createId('cogview4_graph'), nodes: {} };
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const modelLoader = addNode(graph, { id: 'model_loader', model, type: 'cogview4_model_loader' });
  const posCond = addNode(graph, { id: 'pos_cond', type: 'cogview4_text_encoder' });
  const negCond = addNode(graph, { id: 'neg_cond', type: 'cogview4_text_encoder' });
  const denoise = addNode(graph, {
    cfg_scale: settings.cfgScale,
    height: settings.height,
    id: 'denoise_latents',
    steps: settings.steps,
    type: 'cogview4_denoise',
    width: settings.width,
  });
  const output = addImageOutputNode(graph, 'cogview4_l2i', outputIsIntermediate);

  addEdge(graph, modelLoader, 'transformer', denoise, 'transformer');
  addEdge(graph, modelLoader, 'glm_encoder', posCond, 'glm_encoder');
  addEdge(graph, modelLoader, 'glm_encoder', negCond, 'glm_encoder');
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');
  addEdge(graph, negCond, 'conditioning', denoise, 'negative_conditioning');
  addEdge(graph, seed, 'value', denoise, 'seed');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addMetadata(graph, output, settings, model, 'cogview4_txt2img', projectSettings, { scheduler: undefined });

  return graph;
};

const buildQwenImageGraph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): BackendGraphContract => {
  const sourceModel = getCompatibleComponentSource(settings, model);
  const hasBundledComponents = model.format === 'diffusers' || sourceModel;
  const vaeModel = getCompatibleVae(settings, ['qwen-image']);

  if (!hasBundledComponents && (!vaeModel || !settings.qwenVLEncoderModel)) {
    throw new Error(
      'Qwen Image non-Diffusers models require a VAE and Qwen VL Encoder, or a Diffusers component source.'
    );
  }

  const graph: BackendGraphContract = { edges: [], id: createId('qwen_image_graph'), nodes: {} };
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const useCfg = settings.cfgScale > 1;
  const modelLoader = addNode(graph, {
    component_source: sourceModel,
    id: 'model_loader',
    model,
    qwen_vl_encoder_model: settings.qwenVLEncoderModel ?? undefined,
    type: 'qwen_image_model_loader',
    vae_model: vaeModel ?? undefined,
  });
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'qwen_image_lora_collection_loader', modelLoader, [
        'transformer',
      ])
    : modelLoader;
  const posCond = addNode(graph, { id: 'pos_cond', type: 'qwen_image_text_encoder' });
  const negCond = useCfg ? addNode(graph, { id: 'neg_cond', type: 'qwen_image_text_encoder' }) : null;
  const denoise = addNode(graph, {
    cfg_scale: settings.cfgScale,
    height: settings.height,
    id: 'denoise_latents',
    steps: settings.steps,
    type: 'qwen_image_denoise',
    width: settings.width,
  });
  const output = addImageOutputNode(graph, 'qwen_image_l2i', outputIsIntermediate);

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, modelLoader, 'qwen_vl_encoder', posCond, 'qwen_vl_encoder');
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');

  if (negCond) {
    addEdge(graph, modelLoader, 'qwen_vl_encoder', negCond, 'qwen_vl_encoder');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
    addEdge(graph, negCond, 'conditioning', denoise, 'negative_conditioning');
  }

  addEdge(graph, seed, 'value', denoise, 'seed');
  const qwenReferenceImages =
    model.variant === 'edit' ? getEnabledReferenceImages(settings, 'qwen_image_reference_image') : [];

  if (qwenReferenceImages.length > 0) {
    let prevCollect: BackendInvocationContract | null = null;

    for (const { config } of qwenReferenceImages) {
      if (!config.image) {
        continue;
      }

      const imageNode = addNode(graph, {
        id: createId('qwen_ref_img'),
        image: toImageField(config.image),
        type: 'image',
      });
      const collectNode = addNode(graph, { id: createId('qwen_ref_img_collect'), type: 'collect' });

      addEdge(graph, imageNode, 'image', collectNode, 'item');
      if (prevCollect) {
        addEdge(graph, prevCollect, 'collection', collectNode, 'collection');
      }
      prevCollect = collectNode;
    }

    if (prevCollect) {
      addEdge(graph, prevCollect, 'collection', posCond, 'reference_images');
    }

    const firstImage = qwenReferenceImages[0]?.config.image;
    if (firstImage) {
      const refI2l = addNode(graph, { id: createId('qwen_ref_i2l'), type: 'qwen_image_i2l' });
      const refImageNode = addNode(graph, {
        id: createId('qwen_ref_img_for_vae'),
        image: toImageField(firstImage),
        type: 'image',
      });

      addEdge(graph, refImageNode, 'image', refI2l, 'image');
      addEdge(graph, modelLoader, 'vae', refI2l, 'vae');
      addEdge(graph, refI2l, 'latents', denoise, 'reference_latents');
    }
  }
  addEdge(graph, denoise, 'latents', output, 'latents');
  addMetadata(graph, output, settings, model, 'qwen_image_txt2img', projectSettings, {
    qwen_image_component_source: sourceModel,
    qwen_image_qwen_vl_encoder: settings.qwenVLEncoderModel ?? undefined,
    qwen_image_vae: vaeModel ?? undefined,
    scheduler: undefined,
  });
  addReferenceImageMetadata(graph, output, settings);

  return graph;
};

const buildZImageGraph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): BackendGraphContract => {
  const sourceModel = getDiffusersSource(settings, model);
  const qwen3EncoderModel =
    settings.qwen3EncoderModel && isNonAnimaQwen3Encoder(settings.qwen3EncoderModel)
      ? settings.qwen3EncoderModel
      : null;
  const vaeModel = getCompatibleVae(settings, ['flux']);

  if (!sourceModel && (!vaeModel || !qwen3EncoderModel)) {
    throw new Error('Z-Image models require a VAE and Qwen3 Encoder, or a Diffusers component source.');
  }

  const graph: BackendGraphContract = { edges: [], id: createId('z_image_graph'), nodes: {} };
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const scheduler = coerceSchedulerForGraph(model, settings.scheduler);
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const useCfg = settings.cfgScale > 1;
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model,
    qwen3_encoder_model: qwen3EncoderModel ?? undefined,
    qwen3_source_model: sourceModel,
    type: 'z_image_model_loader',
    vae_model: vaeModel ?? undefined,
  });
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'z_image_lora_collection_loader', modelLoader, [
        'transformer',
        'qwen3_encoder',
      ])
    : modelLoader;
  const posCond = addNode(graph, { id: 'pos_cond', type: 'z_image_text_encoder' });
  const posCondCollect = addNode(graph, { id: 'pos_cond_collect', type: 'collect' });
  const negCond = useCfg ? addNode(graph, { id: 'neg_cond', type: 'z_image_text_encoder' }) : null;
  const negCondCollect = useCfg ? addNode(graph, { id: 'neg_cond_collect', type: 'collect' }) : null;
  const denoise = addNode(graph, {
    denoising_end: 1,
    denoising_start: 0,
    guidance_scale: settings.cfgScale,
    height: settings.height,
    id: 'denoise_latents',
    scheduler,
    steps: settings.steps,
    type: 'z_image_denoise',
    width: settings.width,
  });
  const output = addImageOutputNode(graph, 'z_image_l2i', outputIsIntermediate);

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, loraSource, 'qwen3_encoder', posCond, 'qwen3_encoder');
  addEdge(graph, modelLoader, 'vae', denoise, 'vae');
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', posCondCollect, 'item');
  addEdge(graph, posCondCollect, 'collection', denoise, 'positive_conditioning');

  if (negCond && negCondCollect) {
    addEdge(graph, loraSource, 'qwen3_encoder', negCond, 'qwen3_encoder');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
    addEdge(graph, negCond, 'conditioning', negCondCollect, 'item');
    addEdge(graph, negCondCollect, 'collection', denoise, 'negative_conditioning');
  }

  addEdge(graph, seed, 'value', denoise, 'seed');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addMetadata(graph, output, settings, model, 'z_image_txt2img', projectSettings, {
    qwen3_encoder: qwen3EncoderModel ?? undefined,
    qwen3_source: sourceModel,
    scheduler,
    vae: vaeModel ?? undefined,
  });

  return graph;
};

const buildAnimaGraph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): BackendGraphContract => {
  const vaeModel = requireComponent(settings.vae && isAnimaVae(settings.vae) ? settings.vae : null, 'Anima VAE');
  const qwen3EncoderModel = requireComponent(
    settings.qwen3EncoderModel && isAnimaQwen3Encoder(settings.qwen3EncoderModel) ? settings.qwen3EncoderModel : null,
    'Qwen3 Encoder'
  );
  const graph: BackendGraphContract = { edges: [], id: createId('anima_graph'), nodes: {} };
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const scheduler = coerceSchedulerForGraph(model, settings.scheduler);
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const useCfg = settings.cfgScale > 1;
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model,
    qwen3_encoder_model: qwen3EncoderModel,
    type: 'anima_model_loader',
    vae_model: vaeModel,
  });
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'anima_lora_collection_loader', modelLoader, [
        'transformer',
        'qwen3_encoder',
      ])
    : modelLoader;
  const posCond = addNode(graph, { id: 'pos_cond', type: 'anima_text_encoder' });
  const posCondCollect = addNode(graph, { id: 'pos_cond_collect', type: 'collect' });
  const negCond = useCfg ? addNode(graph, { id: 'neg_cond', type: 'anima_text_encoder' }) : null;
  const negCondCollect = useCfg ? addNode(graph, { id: 'neg_cond_collect', type: 'collect' }) : null;
  const denoise = addNode(graph, {
    denoising_end: 1,
    denoising_start: 0,
    guidance_scale: settings.cfgScale,
    height: settings.height,
    id: 'denoise_latents',
    scheduler,
    steps: settings.steps,
    type: 'anima_denoise',
    width: settings.width,
  });
  const output = addImageOutputNode(graph, 'anima_l2i', outputIsIntermediate);

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, loraSource, 'qwen3_encoder', posCond, 'qwen3_encoder');
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', posCondCollect, 'item');
  addEdge(graph, posCondCollect, 'collection', denoise, 'positive_conditioning');

  if (negCond && negCondCollect) {
    addEdge(graph, loraSource, 'qwen3_encoder', negCond, 'qwen3_encoder');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
    addEdge(graph, negCond, 'conditioning', negCondCollect, 'item');
    addEdge(graph, negCondCollect, 'collection', denoise, 'negative_conditioning');
  }

  addEdge(graph, seed, 'value', denoise, 'seed');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addMetadata(graph, output, settings, model, 'anima_txt2img', projectSettings, {
    qwen3_encoder: qwen3EncoderModel,
    scheduler,
    vae: vaeModel,
  });

  return graph;
};

const buildExternalGraph = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): BackendGraphContract => {
  if (model.type !== 'external_image_generator') {
    throw new Error('Selected model is not an external image generator.');
  }

  if (model.capabilities?.modes && !model.capabilities.modes.includes('txt2img')) {
    throw new Error(`${model.name} does not support text-to-image generation.`);
  }

  const providerId = typeof model.provider_id === 'string' ? model.provider_id : '';
  const nodeType = getExternalProviderNodeType(providerId);

  if (!nodeType) {
    throw new Error(`No invocation node registered for external provider '${providerId}'.`);
  }

  const graph: BackendGraphContract = { edges: [], id: createId('external_graph'), nodes: {} };
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const output = addNode(graph, {
    height: settings.height,
    id: 'canvas_output',
    image_size: null,
    is_intermediate: outputIsIntermediate,
    mode: 'txt2img',
    model,
    num_images: 1,
    type: nodeType,
    use_cache: false,
    width: settings.width,
  });

  addEdge(graph, positivePrompt, 'value', output, 'prompt');
  if (model.capabilities?.supports_negative_prompt === true) {
    addEdge(graph, negativePrompt, 'value', output, 'negative_prompt');
  }

  if (model.capabilities?.supports_seed === true) {
    addEdge(graph, seed, 'value', output, 'seed');
  }
  if (model.capabilities?.supports_reference_images === true) {
    const referenceImages = getEnabledReferenceImages(settings, 'external_reference_image')
      .map((referenceImage) => referenceImage.config.image)
      .filter((image): image is GenerateReferenceImageAsset => image !== null)
      .map(toImageField);

    if (referenceImages.length > 0) {
      (output as Record<string, unknown>).reference_images = referenceImages;
    }
  }
  addMetadata(graph, output, settings, model, null, projectSettings, { scheduler: undefined });
  addReferenceImageMetadata(graph, output, settings);

  return graph;
};

type GenerateGraphBuilder = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
) => BackendGraphContract;

export const GRAPH_BUILDERS = {
  'sd-1': buildSDGraph,
  'sd-2': buildSDGraph,
  sdxl: buildSDGraph,
  'sd-3': buildSD3Graph,
  flux: buildFluxGraph,
  flux2: buildFlux2Graph,
  cogview4: buildCogView4Graph,
  'qwen-image': buildQwenImageGraph,
  'z-image': buildZImageGraph,
  anima: buildAnimaGraph,
} satisfies Record<SupportedGenerateBase, GenerateGraphBuilder>;

export const compileGenerateGraph = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  destination: ResultDestination,
  projectSettings: GenerationProjectSettings
): CompiledGenerateGraph => {
  const validationReasons = getGenerationValidationReasons(model, settings);

  if (validationReasons.length > 0) {
    throw new Error(validationReasons[0]);
  }

  const outputIsIntermediate = destination === 'canvas';
  const backendGraph = (() => {
    if (model.type === 'external_image_generator') {
      return buildExternalGraph(settings, model, outputIsIntermediate, projectSettings);
    }

    const builder = GRAPH_BUILDERS[model.base as SupportedGenerateBase];

    if (!builder) {
      const dimensionGrid = getGenerationDimensions(model).grid;

      throw new Error(`${model.name} dimensions must be multiples of ${dimensionGrid}.`);
    }

    return builder(settings, model, outputIsIntermediate, projectSettings);
  })();

  return {
    backendGraph,
    graph: toGraphContract(backendGraph, `${model.name} txt2img`),
    negativePromptNodeId: 'negative_prompt',
    positivePromptNodeId: 'positive_prompt',
    seedNodeId: 'seed',
  };
};

export const resolveGenerateSeed = (settings: GenerateSettings): number =>
  settings.shouldRandomizeSeed ? Math.floor(Math.random() * SEED_MAX) : settings.seed;

export const generateSeedSequence = (start: number, count: number): number[] => {
  const seedCount = Math.max(1, Math.round(count));
  const seeds: number[] = [];

  for (let index = 0; index < seedCount; index += 1) {
    seeds.push((start + index) % SEED_MAX);
  }

  return seeds;
};
