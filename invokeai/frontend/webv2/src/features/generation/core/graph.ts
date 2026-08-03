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
  isKrea2Vae,
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
import { addPidDecode, getPidDenoiseSize, getPidMetadata, shouldUsePidDecode } from './pidGraph';
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
  const supportsHiDiffusion = model.base === 'sd-1' || model.base === 'sdxl';

  const positivePrompt = addNode(graph, { id: 'positive_prompt', type: 'string' });
  const negativePrompt = addNode(graph, { id: 'negative_prompt', type: 'string' });
  const seed = addNode(graph, { id: 'seed', type: 'integer' });
  const modelLoader = addNode(graph, { id: 'model_loader', model, type: modelLoaderType });
  const posCond = addNode(graph, { id: 'pos_cond', type: compelType });
  const negCond = addNode(graph, { id: 'neg_cond', type: compelType });
  const posCondCollect = addNode(graph, { id: 'pos_cond_collect', type: 'collect' });
  const negCondCollect = addNode(graph, { id: 'neg_cond_collect', type: 'collect' });
  const sdDenoiseSize = getDenoiseSize(settings, model);
  const noise = addNode(graph, {
    height: sdDenoiseSize.height,
    id: 'noise',
    type: 'noise',
    use_cpu: projectSettings.useCpuNoise,
    width: sdDenoiseSize.width,
  });
  const denoise = addNode(graph, {
    cfg_rescale_multiplier: settings.cfgRescaleMultiplier,
    cfg_scale: settings.cfgScale,
    denoising_end: 1,
    denoising_start: 0,
    id: 'denoise_latents',
    ...(supportsHiDiffusion
      ? {
          hidiffusion: settings.hiDiffusionEnabled,
          hidiffusion_raunet: settings.hiDiffusionRauNetEnabled,
          hidiffusion_window_attn: settings.hiDiffusionWindowAttentionEnabled,
          ...(settings.hiDiffusionEnabled
            ? {
                hidiffusion_t1_ratio: settings.hiDiffusionT1Ratio,
                hidiffusion_t2_ratio: settings.hiDiffusionT2Ratio,
              }
            : {}),
        }
      : {}),
    scheduler,
    steps: settings.steps,
    type: 'denoise_latents',
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
  // Created here rather than with the other nodes because the decode's VAE source is
  // only settled once the seamless / VAE-override nodes above are known.
  const output = addDecodeOutput({
    denoise,
    graph,
    l2iProps: { color_compensation: colorCompensation, fp32: settings.vaePrecision === 'fp32' },
    l2iType: 'l2i',
    model,
    outputIsIntermediate,
    positivePrompt,
    seed,
    settings,
    vaeSource: seamless ?? vaeLoader ?? modelLoader,
  });

  addMetadata(graph, output, settings, model, model.base === 'sdxl' ? 'sdxl_txt2img' : 'txt2img', projectSettings, {
    ...(supportsHiDiffusion
      ? {
          hidiffusion: settings.hiDiffusionEnabled,
          hidiffusion_raunet: settings.hiDiffusionRauNetEnabled,
          hidiffusion_t1_ratio: settings.hiDiffusionT1Ratio,
          hidiffusion_t2_ratio: settings.hiDiffusionT2Ratio,
          hidiffusion_window_attn: settings.hiDiffusionWindowAttentionEnabled,
        }
      : {}),
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

/**
 * The decode that turns the denoised latents into the output image: either the base's
 * ordinary VAE decode, or PiD's 4x super-resolution decode when PiD is on.
 *
 * Both branches wire `latents` (and the VAE where the node needs it) and return the
 * terminal `canvas_output` node, so callers attach metadata to the result either way.
 */
const addDecodeOutput = ({
  graph,
  settings,
  model,
  l2iType,
  l2iProps,
  denoise,
  positivePrompt,
  seed,
  vaeSource,
  vaeField = 'vae',
  outputIsIntermediate,
}: {
  graph: BackendGraphContract;
  settings: GenerateSettings;
  model: MainModelConfig;
  /** The base's ordinary VAE-decode node type, used when PiD is off. */
  l2iType: string;
  /** Extra props for the ordinary decode node; ignored by the PiD chain, which has its own. */
  l2iProps?: Record<string, unknown>;
  denoise: BackendInvocationContract;
  positivePrompt: BackendInvocationContract;
  seed: BackendInvocationContract;
  vaeSource?: BackendInvocationContract;
  vaeField?: string;
  outputIsIntermediate: boolean;
}): BackendInvocationContract => {
  if (shouldUsePidDecode(settings, model.base)) {
    return addPidDecode({
      base: model.base,
      denoise,
      graph,
      outputIsIntermediate,
      positivePrompt,
      seed,
      settings,
      vaeField,
      vaeSource,
    });
  }

  const output = addNode(graph, {
    id: 'canvas_output',
    is_intermediate: outputIsIntermediate,
    type: l2iType,
    use_cache: false,
    ...l2iProps,
  });

  if (vaeSource) {
    addEdge(graph, vaeSource, vaeField, output, 'vae');
  }
  addEdge(graph, denoise, 'latents', output, 'latents');

  return output;
};

/**
 * The width/height to denoise at. Identical to the requested size unless PiD is in
 * native mode, where the requested size is the 4x decode target and generation runs at
 * a quarter of it.
 */
const getDenoiseSize = (settings: GenerateSettings, model: MainModelConfig): { width: number; height: number } =>
  // Deliberately the model's OWN grid (the default `off` mode), not the PiD-scaled grid
  // the dimension fields present: requested / 4 must land on the model's grid.
  getPidDenoiseSize(settings, model.base, getGenerationDimensions(model).grid);

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
  const sd3DenoiseSize = getDenoiseSize(settings, model);
  const denoise = addNode(graph, {
    cfg_scale: settings.cfgScale,
    denoising_end: 1,
    denoising_start: 0,
    height: sd3DenoiseSize.height,
    id: 'denoise_latents',
    steps: settings.steps,
    type: 'sd3_denoise',
    width: sd3DenoiseSize.width,
  });
  const output = addDecodeOutput({
    denoise,
    graph,
    l2iType: 'sd3_l2i',
    model,
    outputIsIntermediate,
    positivePrompt,
    seed,
    settings,
    vaeSource: modelLoader,
  });

  addEdge(graph, modelLoader, 'transformer', denoise, 'transformer');
  addEdge(graph, modelLoader, 'clip_l', posCond, 'clip_l');
  addEdge(graph, modelLoader, 'clip_l', negCond, 'clip_l');
  addEdge(graph, modelLoader, 'clip_g', posCond, 'clip_g');
  addEdge(graph, modelLoader, 'clip_g', negCond, 'clip_g');
  addEdge(graph, modelLoader, 't5_encoder', posCond, 't5_encoder');
  addEdge(graph, modelLoader, 't5_encoder', negCond, 't5_encoder');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');
  addEdge(graph, negCond, 'conditioning', denoise, 'negative_conditioning');
  addEdge(graph, seed, 'value', denoise, 'seed');
  addMetadata(graph, output, settings, model, 'sd3_txt2img', projectSettings, {
    scheduler: undefined,
    vae: settings.vae ?? undefined,
    ...(shouldUsePidDecode(settings, model.base) ? getPidMetadata(settings) : {}),
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
  const { positivePrompt, seed } = addPromptAndSeedNodes(graph);
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
  const denoiseSize = getDenoiseSize(settings, model);
  const denoise = addNode(graph, {
    denoising_end: 1,
    denoising_start: 0,
    guidance: settings.cfgScale,
    height: denoiseSize.height,
    id: 'denoise_latents',
    num_steps: settings.steps,
    scheduler,
    type: 'flux_denoise',
    width: denoiseSize.width,
  });
  const output = addDecodeOutput({
    denoise,
    graph,
    l2iType: 'flux_vae_decode',
    model,
    outputIsIntermediate,
    positivePrompt,
    seed,
    settings,
    vaeSource: modelLoader,
  });

  addEdge(graph, loraSource, 'clip', posCond, 'clip');
  addEdge(graph, loraSource, 't5_encoder', posCond, 't5_encoder');
  addEdge(graph, modelLoader, 'max_seq_len', posCond, 't5_max_seq_len');
  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, modelLoader, 'vae', denoise, 'controlnet_vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', posCondCollect, 'item');
  addEdge(graph, posCondCollect, 'collection', denoise, 'positive_text_conditioning');
  addEdge(graph, graph.nodes.seed, 'value', denoise, 'seed');
  if (model.name.toLowerCase().includes('kontext')) {
    addFluxKontextReferenceImages(graph, settings, denoise, 'flux_kontext_reference_image');
  }
  addIPAdapterReferenceImages(graph, settings, model, denoise);
  addFluxReduxReferenceImages(graph, settings, model, denoise);
  addMetadata(graph, output, settings, model, 'flux_txt2img', projectSettings, {
    clip_embed_model: clipEmbedModel,
    guidance: settings.cfgScale,
    scheduler,
    t5_encoder: t5EncoderModel,
    vae: vaeModel,
    ...(shouldUsePidDecode(settings, model.base) ? getPidMetadata(settings) : {}),
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
  const posCondCollect = addNode(graph, { id: 'pos_cond_collect', type: 'collect' });
  const flux2DenoiseSize = getDenoiseSize(settings, model);
  const denoise = addNode(graph, {
    cfg_scale: 1,
    denoising_end: 1,
    denoising_start: 0,
    guidance: settings.cfgScale,
    height: flux2DenoiseSize.height,
    id: 'denoise_latents',
    num_steps: settings.steps,
    scheduler,
    type: 'flux2_denoise',
    width: flux2DenoiseSize.width,
  });
  const output = addDecodeOutput({
    denoise,
    graph,
    l2iType: 'flux2_vae_decode',
    model,
    outputIsIntermediate,
    positivePrompt,
    seed: graph.nodes.seed,
    settings,
    vaeSource: modelLoader,
  });

  addEdge(graph, loraSource, 'qwen3_encoder', posCond, 'qwen3_encoder');
  addEdge(graph, modelLoader, 'max_seq_len', posCond, 'max_seq_len');
  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, modelLoader, 'vae', denoise, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', posCondCollect, 'item');
  addEdge(graph, posCondCollect, 'collection', denoise, 'positive_text_conditioning');
  addEdge(graph, graph.nodes.seed, 'value', denoise, 'seed');
  addFluxKontextReferenceImages(graph, settings, denoise, 'flux2_reference_image');
  addMetadata(graph, output, settings, model, 'flux2_txt2img', projectSettings, {
    qwen3_encoder: qwen3EncoderModel ?? undefined,
    qwen3_source: sourceModel,
    scheduler,
    vae: vaeModel ?? undefined,
    ...(shouldUsePidDecode(settings, model.base) ? getPidMetadata(settings) : {}),
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
  const qwenDenoiseSize = getDenoiseSize(settings, model);
  const denoise = addNode(graph, {
    cfg_scale: settings.cfgScale,
    height: qwenDenoiseSize.height,
    id: 'denoise_latents',
    steps: settings.steps,
    type: 'qwen_image_denoise',
    width: qwenDenoiseSize.width,
  });
  const output = addDecodeOutput({
    denoise,
    graph,
    l2iType: 'qwen_image_l2i',
    model,
    outputIsIntermediate,
    positivePrompt,
    seed,
    settings,
    vaeSource: modelLoader,
  });

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, modelLoader, 'qwen_vl_encoder', posCond, 'qwen_vl_encoder');
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
  addMetadata(graph, output, settings, model, 'qwen_image_txt2img', projectSettings, {
    qwen_image_component_source: sourceModel,
    qwen_image_qwen_vl_encoder: settings.qwenVLEncoderModel ?? undefined,
    qwen_image_vae: vaeModel ?? undefined,
    scheduler: undefined,
    ...(shouldUsePidDecode(settings, model.base) ? getPidMetadata(settings) : {}),
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
  const zImageDenoiseSize = getDenoiseSize(settings, model);
  const denoise = addNode(graph, {
    denoising_end: 1,
    denoising_start: 0,
    guidance_scale: settings.cfgScale,
    height: zImageDenoiseSize.height,
    id: 'denoise_latents',
    scheduler,
    steps: settings.steps,
    type: 'z_image_denoise',
    width: zImageDenoiseSize.width,
  });
  const output = addDecodeOutput({
    denoise,
    graph,
    l2iType: 'z_image_l2i',
    model,
    outputIsIntermediate,
    positivePrompt,
    seed,
    settings,
    vaeSource: modelLoader,
  });

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, loraSource, 'qwen3_encoder', posCond, 'qwen3_encoder');
  addEdge(graph, modelLoader, 'vae', denoise, 'vae');
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
  addMetadata(graph, output, settings, model, 'z_image_txt2img', projectSettings, {
    qwen3_encoder: qwen3EncoderModel ?? undefined,
    qwen3_source: sourceModel,
    scheduler,
    vae: vaeModel ?? undefined,
    ...(shouldUsePidDecode(settings, model.base) ? getPidMetadata(settings) : {}),
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

const buildKrea2Graph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): BackendGraphContract => {
  // Krea-2 has no component-source concept: a non-diffusers transformer (single-file checkpoint
  // or GGUF) bundles neither VAE nor encoder, so both must be selected. A diffusers model
  // carries them, and the loader extracts them when these are omitted.
  const isDiffusers = model.format === 'diffusers';
  // Same base set as the picker (`isKrea2Vae`) and the backend loader: a Qwen-Image VAE
  // installed under the `anima` base is still the VAE Krea-2 wants. Narrowing it here would
  // silently drop a VAE the user had legitimately selected.
  const vaeModel = settings.vae && isKrea2Vae(settings.vae) ? settings.vae : null;
  const qwen3VlEncoderModel = settings.qwen3VLEncoderModel;

  if (!isDiffusers) {
    requireComponent(vaeModel, 'Krea-2 VAE');
    requireComponent(qwen3VlEncoderModel, 'Qwen3-VL Encoder');
  }

  const graph: BackendGraphContract = { edges: [], id: createId('krea2_graph'), nodes: {} };
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const activeLoras = getActiveCompatibleLoras(settings, model);
  // Krea-2 accepts negative conditioning only with CFG on; Krea-2-Turbo defaults to CFG off.
  const useCfg = settings.cfgScale > 1;
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model,
    qwen3_vl_encoder_model: qwen3VlEncoderModel ?? undefined,
    type: 'krea2_model_loader',
    vae_model: vaeModel ?? undefined,
  });
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'krea2_lora_collection_loader', modelLoader, [
        'transformer',
        'qwen3_vl_encoder',
      ])
    : modelLoader;
  const posCond = addNode(graph, { id: 'pos_cond', type: 'krea2_text_encoder' });
  const negCond = useCfg ? addNode(graph, { id: 'neg_cond', type: 'krea2_text_encoder' }) : null;
  const denoise = addNode(graph, {
    cfg_scale: settings.cfgScale,
    height: settings.height,
    id: 'denoise_latents',
    steps: settings.steps,
    type: 'krea2_denoise',
    width: settings.width,
  });
  // Krea-2 decodes with the Qwen-Image VAE, so it reuses that family's latents-to-image node.
  const output = addImageOutputNode(graph, 'qwen_image_l2i', outputIsIntermediate);

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, loraSource, 'qwen3_vl_encoder', posCond, 'qwen3_vl_encoder');
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');

  // Optional conditioning enhancers, both default-off, chained between the encoder and denoise:
  // rebalance first (scale the signal toward the prompt), then seed variance (perturb it).
  let positiveConditioningSource = posCond;

  if (settings.krea2RebalanceEnabled) {
    const rebalance = addNode(graph, {
      id: 'krea2_rebalance',
      multiplier: settings.krea2RebalanceMultiplier,
      per_layer_weights: settings.krea2RebalanceWeights,
      type: 'krea2_conditioning_rebalance',
    });

    addEdge(graph, positiveConditioningSource, 'conditioning', rebalance, 'conditioning');
    positiveConditioningSource = rebalance;
  }

  if (settings.krea2SeedVarianceEnabled && settings.krea2SeedVarianceStrength > 0) {
    const seedVariance = addNode(graph, {
      id: 'krea2_seed_variance',
      randomize_percent: settings.krea2SeedVarianceRandomizePercent,
      strength: settings.krea2SeedVarianceStrength,
      type: 'krea2_seed_variance',
    });

    addEdge(graph, positiveConditioningSource, 'conditioning', seedVariance, 'conditioning');
    addEdge(graph, seed, 'value', seedVariance, 'variance_seed');
    positiveConditioningSource = seedVariance;
  }

  addEdge(graph, positiveConditioningSource, 'conditioning', denoise, 'positive_conditioning');

  if (negCond) {
    addEdge(graph, loraSource, 'qwen3_vl_encoder', negCond, 'qwen3_vl_encoder');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
    addEdge(graph, negCond, 'conditioning', denoise, 'negative_conditioning');
  }

  addEdge(graph, seed, 'value', denoise, 'seed');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addMetadata(graph, output, settings, model, 'krea2_txt2img', projectSettings, {
    krea2_rebalance_enabled: settings.krea2RebalanceEnabled,
    krea2_rebalance_multiplier: settings.krea2RebalanceMultiplier,
    krea2_rebalance_weights: settings.krea2RebalanceWeights,
    krea2_seed_variance_enabled: settings.krea2SeedVarianceEnabled,
    krea2_seed_variance_randomize_percent: settings.krea2SeedVarianceRandomizePercent,
    krea2_seed_variance_strength: settings.krea2SeedVarianceStrength,
    qwen3_vl_encoder: qwen3VlEncoderModel ?? undefined,
    vae: vaeModel ?? undefined,
  });

  return graph;
};

const buildIdeogram4Graph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): BackendGraphContract => {
  const graph: BackendGraphContract = { edges: [], id: createId('ideogram4_graph'), nodes: {} };
  // Ideogram 4 is positive-only: there is no negative conditioning input on its denoise node.
  const { positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const modelLoader = addNode(graph, { id: 'model_loader', model, type: 'ideogram4_model_loader' });
  // The caption builder turns the prompt (plus optional colour terms) into Ideogram's own
  // caption format before encoding.
  const captionBuilder = addNode(graph, {
    color_palette: settings.ideogram4ColorPalette,
    id: 'caption_builder',
    type: 'ideogram4_caption_builder',
  });
  const posCond = addNode(graph, { id: 'pos_cond', type: 'ideogram4_text_encoder' });
  const denoise = addNode(graph, {
    height: settings.height,
    id: 'denoise_latents',
    // Steps, guidance and mu are preset-derived unless explicitly overridden — sending null
    // would override the preset with nothing, so unset values are omitted entirely.
    ...(settings.ideogram4Steps === null ? {} : { steps: settings.ideogram4Steps }),
    ...(settings.ideogram4GuidanceScale === null ? {} : { guidance_scale: settings.ideogram4GuidanceScale }),
    ...(settings.ideogram4Mu === null ? {} : { mu: settings.ideogram4Mu }),
    sampler_preset: settings.ideogram4SamplerPreset,
    type: 'ideogram4_denoise',
    width: settings.width,
  });
  const output = addImageOutputNode(graph, 'ideogram4_l2i', outputIsIntermediate);

  addEdge(graph, modelLoader, 'transformer', denoise, 'transformer');
  addEdge(graph, modelLoader, 'qwen3_encoder', posCond, 'qwen3_encoder');
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, positivePrompt, 'value', captionBuilder, 'prompt');
  addEdge(graph, captionBuilder, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');
  addEdge(graph, seed, 'value', denoise, 'seed');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addMetadata(graph, output, settings, model, 'ideogram4_txt2img', projectSettings, {
    ideogram4_color_palette: settings.ideogram4ColorPalette,
    ideogram4_guidance_scale: settings.ideogram4GuidanceScale ?? undefined,
    ideogram4_mu: settings.ideogram4Mu ?? undefined,
    ideogram4_sampler_preset: settings.ideogram4SamplerPreset,
    ideogram4_steps: settings.ideogram4Steps ?? undefined,
  });

  return graph;
};

const buildWanGraph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): BackendGraphContract => {
  // A GGUF Wan main carries only the transformer, so the VAE and UMT5-XXL encoder come from
  // standalone models or a Diffusers component source.
  const sourceModel = getDiffusersSource(settings, model);
  const vaeModel = getCompatibleVae(settings, ['wan']);
  const wanT5EncoderModel = settings.wanT5EncoderModel;

  if (!sourceModel && (!vaeModel || !wanT5EncoderModel)) {
    throw new Error('Wan models require a VAE and Wan T5 Encoder, or a Diffusers component source.');
  }

  const graph: BackendGraphContract = { edges: [], id: createId('wan_graph'), nodes: {} };
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const modelLoader = addNode(graph, {
    component_source: sourceModel,
    id: 'model_loader',
    model,
    // The second A14B expert. Omitted, the high-noise expert covers the whole schedule.
    transformer_low_noise_model: settings.wanLowNoiseModel ?? undefined,
    type: 'wan_model_loader',
    vae_model: vaeModel ?? undefined,
    wan_t5_encoder_model: wanT5EncoderModel ?? undefined,
  });
  // Wan LoRAs patch only the transformer; the text encoder is untouched.
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'wan_lora_collection_loader', modelLoader, ['transformer'])
    : modelLoader;
  const posCond = addNode(graph, { id: 'pos_cond', type: 'wan_text_encoder' });
  const negCond = addNode(graph, { id: 'neg_cond', type: 'wan_text_encoder' });
  const denoise = addNode(graph, {
    guidance_scale: settings.cfgScale,
    // Null lets the backend reuse guidance_scale for the low-noise half of the schedule.
    ...(settings.wanGuidanceScaleLowNoise === null
      ? {}
      : { guidance_scale_low_noise: settings.wanGuidanceScaleLowNoise }),
    height: settings.height,
    id: 'denoise_latents',
    steps: settings.steps,
    type: 'wan_denoise',
    width: settings.width,
  });
  // Single-frame image output, not video: `wan_l2i` rather than `wan_l2v`.
  const output = addImageOutputNode(graph, 'wan_l2i', outputIsIntermediate);

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, modelLoader, 'wan_t5_encoder', posCond, 'wan_t5_encoder');
  addEdge(graph, modelLoader, 'wan_t5_encoder', negCond, 'wan_t5_encoder');
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');
  addEdge(graph, negCond, 'conditioning', denoise, 'negative_conditioning');
  addEdge(graph, seed, 'value', denoise, 'seed');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addMetadata(graph, output, settings, model, 'wan_txt2img', projectSettings, {
    guidance_scale_low_noise: settings.wanGuidanceScaleLowNoise ?? undefined,
    transformer_low_noise: settings.wanLowNoiseModel ?? undefined,
    vae: vaeModel ?? undefined,
    wan_component_source: sourceModel,
    wan_t5_encoder: wanT5EncoderModel ?? undefined,
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
  'ideogram-4': buildIdeogram4Graph,
  'krea-2': buildKrea2Graph,
  anima: buildAnimaGraph,
  wan: buildWanGraph,
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
