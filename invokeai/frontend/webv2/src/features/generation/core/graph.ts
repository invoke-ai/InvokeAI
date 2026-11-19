import type {
  BackendGraphContract,
  BackendInvocationContract,
  GenerationProjectSettings,
  ResultDestination,
} from '@features/generation/core/contracts';

import { SEED_MAX } from '@platform/core/seed';

import type { FamilyGraph, FamilyPipeline, FamilyPipelineInput } from './familyGraph';
import type {
  CompiledGenerateGraph,
  ComponentModelConfig,
  GenerateLora,
  GenerateModelConfig,
  GenerateReferenceImageAsset,
  GenerateSettings,
  MainModelConfig,
} from './types';

import {
  coerceSchedulerForGraph,
  getExternalModeReason,
  getExternalProviderNodeType,
  getFlux2ComponentSource,
  getGenerationDimensions,
  getGenerationValidationReasons,
  isStrengthOnlyIPAdapterBase,
  type SupportedGenerateBase,
} from './baseGenerationPolicies';
import {
  getCompatibleDiffusersComponentSource,
  isAnimaQwen35Encoder,
  isAnimaQwen3Encoder,
  isBundledFlux2Main,
  isBundledMainForBase,
  isErnieImageMistralEncoder,
  isGgufQwen3Vl4bEncoder,
  isGgufQwen3Vl8bEncoder,
  isFlux2MistralEncoder,
  isFlux2Qwen3EncoderForModel,
  isQwen3Vl4bVisionTower,
  isQwen3Vl8bEncoder,
  isQwen3Vl8bVisionTower,
  isMossImageQwen35Encoder,
  isQwen3Vl4bEncoder,
  isVisionTextLlm,
  isSelfContainedSDNQFlux1Pipeline,
  isVaeCompatibleWithGenerateModel,
  isZImageQwen3Encoder,
  type GenerateComponentFilter,
} from './componentCompatibility';
import {
  getExternalProviderInvocationOptions,
  getExternalProviderMetadata,
  getExternalReferenceImages,
  resolveExternalImageSize,
} from './externalProvider';
import {
  addEdge,
  addMetadata,
  addNode,
  addOutputPostprocessors,
  createId,
  getActiveCompatibleLoras,
  toModelIdentifier,
  toGraphContract,
} from './graphBuilder';
import { addKrea2ConditioningEnhancers } from './krea2Conditioning';
import { addPidDecode, getPidDenoiseSize, getPidMetadata, shouldUsePidDecode } from './pidGraph';
import { getCompelPromptNodeType } from './prompt/compel';
import { getEffectiveReferenceImage, getEnabledReferenceImages } from './referenceImage';
import { getActiveRefiner, getBaseDenoisingRange } from './sdxlRefiner';
import { addSDXLRefinerStage, getRefinerMetadata } from './sdxlRefinerGraph';

const getCompatibleComponentSource = (
  settings: Pick<GenerateSettings, 'componentSourceModel'>,
  model: MainModelConfig
): MainModelConfig | undefined => getCompatibleDiffusersComponentSource(model, settings.componentSourceModel);

// The picker's rule, not a copy of it: a base list here once dropped a VAE the picker offered.
const getCompatibleVae = (settings: Pick<GenerateSettings, 'vae'>, model: MainModelConfig) =>
  settings.vae && isVaeCompatibleWithGenerateModel(model, settings.vae) ? settings.vae : null;

/** Filter stale optional selections that required-slot validation skips. */
const getCompatibleComponent = (
  component: ComponentModelConfig | null,
  filter: GenerateComponentFilter
): ComponentModelConfig | null => (component && filter(component) ? component : null);

/** FLUX.3 and MOSS-Image denoise in the FLUX.2 VAE's raw latent layout, so they encode with it too. */
const FLUX2_ENCODE_TYPE = 'flux2_vae_encode';
/** Qwen-Image's VAE encode; Krea-2 shares the Qwen-Image VAE. Canvas and reference-image encodes use it alike. */
const QWEN_IMAGE_ENCODE_TYPE = 'qwen_image_i2l';
const QWEN_IMAGE_2_1_ENCODE_TYPE = 'qwen_image_2_1_i2l';

const toImageField = (image: GenerateReferenceImageAsset) => ({
  image_name: getEffectiveReferenceImage(image).image_name,
});

/** Recorded even when empty, on the `core_metadata` node `addMetadata` added: an empty list tells recall to clear. */
const addReferenceImageMetadata = (graph: BackendGraphContract, settings: GenerateSettings) => {
  const metadata = Object.values(graph.nodes).find((node) => node.type === 'core_metadata');

  if (metadata) {
    metadata.ref_images = settings.referenceImages.filter(
      (referenceImage) => referenceImage.isEnabled && referenceImage.config.image
    );
  }
};

const addIPAdapterReferenceImages = (
  graph: BackendGraphContract,
  settings: GenerateSettings,
  model: MainModelConfig,
  denoise: BackendInvocationContract
): BackendInvocationContract | null => {
  const strengthOnly = isStrengthOnlyIPAdapterBase(model.base);
  // Anima drops a strength-0 adapter anyway; sending one would still encode the image, or install the encoder.
  const refs = getEnabledReferenceImages(settings.referenceImages, 'ip_adapter').filter(
    (ref) => ref.config.model?.base === model.base && !(strengthOnly && ref.config.weight === 0)
  );

  if (refs.length === 0) {
    return null;
  }

  const collector = addNode(graph, { id: createId('ip_adapter_collector'), type: 'collect' });

  for (const ref of refs) {
    const { beginEndStepPct, clipVisionModel, image, method, model: adapterModel, weight } = ref.config;

    if (!image || !adapterModel) {
      continue;
    }

    const shared = {
      begin_step_percent: beginEndStepPct[0],
      end_step_percent: beginEndStepPct[1],
      image: toImageField(image),
      ip_adapter_model: adapterModel,
    };
    const node = strengthOnly
      ? addNode(graph, { ...shared, id: createId('anima_ip_adapter'), strength: weight, type: 'anima_ip_adapter' })
      : addNode(graph, {
          ...shared,
          clip_vision_model: clipVisionModel,
          id: createId('ip_adapter'),
          method,
          type: model.base === 'flux' ? 'flux_ip_adapter' : 'ip_adapter',
          weight,
        });

    addEdge(graph, node, 'ip_adapter', collector, 'item');
  }

  addEdge(graph, collector, 'collection', denoise, 'ip_adapter');
  return collector;
};

export const FLUX_REDUX_INFLUENCE = {
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
): BackendInvocationContract | null => {
  const refs = getEnabledReferenceImages(settings.referenceImages, 'flux_redux').filter(
    (ref) => ref.config.model?.base === model.base
  );

  if (refs.length === 0) {
    return null;
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
  return collector;
};

const addFluxKontextReferenceImages = (
  graph: BackendGraphContract,
  settings: GenerateSettings,
  denoise: BackendInvocationContract,
  type: 'flux2_reference_image' | 'flux_kontext_reference_image'
): BackendInvocationContract | null => {
  const refs = getEnabledReferenceImages(settings.referenceImages, type);

  if (refs.length === 0) {
    return null;
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

  return prevCollect;
};

const getDiffusersSource = (
  settings: Pick<GenerateSettings, 'componentSourceModel'>,
  model: MainModelConfig
): MainModelConfig | undefined =>
  getCompatibleComponentSource(settings, model) ?? (isBundledMainForBase(model.base)(model) ? model : undefined);

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

export const addTransformerLoraCollectionLoader = (
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

type GenerateGraphBuilder = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
) => FamilyGraph;

/** What a decode hook is handed: the latents to decode, the VAE and what PiD conditions its decode on. */
interface DecodeSources {
  denoise: BackendInvocationContract;
  positivePrompt: BackendInvocationContract;
  seed: BackendInvocationContract;
  vaeSource?: BackendInvocationContract;
  vaeField?: string;
}

/** Builds the decode in place of the plain VAE decode and returns the result node. */
type DecodeBuilder = (graph: BackendGraphContract, sources: DecodeSources) => BackendInvocationContract;

/** Generate-only hooks: splices run inside the pipeline; the rest keep Generate's edge order, which collects follow. */
interface PipelineExtras {
  /** Literal fields the denoise node gains: FLUX DyPE, a schedule shift, MOSS-Image's skip layers, HiDiffusion. */
  denoiseFields?: Record<string, unknown>;
  /** The decode Generate builds instead of the VAE decode: PiD. */
  decode?: DecodeBuilder;
}

/** Positive conditioning passed through Generate's enhancers on its way to the collector. */
type ConditioningHook = (
  graph: BackendGraphContract,
  conditioning: BackendInvocationContract,
  seed: BackendInvocationContract
) => BackendInvocationContract;

const addPromptAndSeedNodes = (graph: BackendGraphContract) => ({
  negativePrompt: addNode(graph, { id: 'negative_prompt', type: 'string' }),
  positivePrompt: addNode(graph, { id: 'positive_prompt', type: 'string' }),
  seed: addNode(graph, { id: 'seed', type: 'integer' }),
});

const addImageOutputNode = (graph: BackendGraphContract, type: string, outputIsIntermediate: boolean) =>
  addNode(graph, { id: 'canvas_output', is_intermediate: outputIsIntermediate, type, use_cache: false });

/** A pipeline's output node without its edges, for families that wire it as they go. */
const addPipelineOutputNode = (
  graph: BackendGraphContract,
  input: FamilyPipelineInput,
  type: string,
  { tiles }: { tiles: boolean }
) =>
  addNode(graph, {
    id: input.output.id,
    is_intermediate: input.output.isIntermediate,
    type,
    use_cache: false,
    ...(tiles ? vaeTilingFields(input) : {}),
  });

/** The switch a VAE node with tiling takes, when the input asks for it. */
const vaeTilingFields = (input: FamilyPipelineInput) =>
  input.vaeTileSize === null ? {} : { tile_size: input.vaeTileSize, tiled: true };

/** A pass from 0 sends no range, so Generate's graph is unchanged; Canvas sets the range itself. */
const partialDenoisingRange = (input: FamilyPipelineInput) =>
  input.denoisingStart === 0 ? {} : { denoising_end: 1, denoising_start: input.denoisingStart };

const addVaeDecode = (
  graph: BackendGraphContract,
  output: { id: string; isIntermediate: boolean; tiling: Record<string, unknown> },
  {
    denoise,
    l2iProps,
    l2iType,
    vaeField = 'vae',
    vaeSource,
  }: Omit<DecodeSources, 'positivePrompt' | 'seed'> & { l2iType: string; l2iProps?: Record<string, unknown> }
): BackendInvocationContract => {
  const node = addNode(graph, {
    id: output.id,
    is_intermediate: output.isIntermediate,
    type: l2iType,
    use_cache: false,
    ...l2iProps,
    ...output.tiling,
  });

  if (vaeSource) {
    addEdge(graph, vaeSource, vaeField, node, 'vae');
  }
  addEdge(graph, denoise, 'latents', node, 'latents');

  return node;
};

/** The pipeline's decode: the extra's where Generate passes one, else the VAE decode the input asks for. */
const addPipelineDecode = (
  graph: BackendGraphContract,
  input: FamilyPipelineInput,
  decode: DecodeBuilder | undefined,
  {
    l2iProps,
    l2iType,
    ...sources
  }: DecodeSources & {
    l2iType: string;
    /** Extra props for the VAE decode node; a decode hook builds its own. */
    l2iProps?: Record<string, unknown>;
  }
): BackendInvocationContract =>
  decode
    ? decode(graph, sources)
    : addVaeDecode(
        graph,
        { id: input.output.id, isIntermediate: input.output.isIntermediate, tiling: vaeTilingFields(input) },
        { ...sources, l2iProps, l2iType }
      );

/** Generate's decode hook: PiD where it is on; otherwise none, and the pipeline decodes with the VAE. */
const getGenerateDecode = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean
): DecodeBuilder | undefined =>
  shouldUsePidDecode(settings, model.base)
    ? (graph, sources) => addPidDecode({ ...sources, base: model.base, graph, outputIsIntermediate, settings })
    : undefined;

/** Both decode branches return canvas_output for metadata attachment. */
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
  l2iType: string;
  /** Extra props for the ordinary decode node; ignored by the PiD chain, which has its own. */
  l2iProps?: Record<string, unknown>;
  denoise: BackendInvocationContract;
  positivePrompt: BackendInvocationContract;
  seed: BackendInvocationContract;
  vaeSource?: BackendInvocationContract;
  vaeField?: string;
  outputIsIntermediate: boolean;
}): BackendInvocationContract =>
  getGenerateDecode(
    settings,
    model,
    outputIsIntermediate
  )?.(graph, {
    denoise,
    positivePrompt,
    seed,
    vaeField,
    vaeSource,
  }) ??
  addVaeDecode(
    graph,
    { id: 'canvas_output', isIntermediate: outputIsIntermediate, tiling: {} },
    { denoise, l2iProps, l2iType, vaeField, vaeSource }
  );

/** Native PiD denoises at one quarter of the requested target dimensions. */
const getDenoiseSize = (settings: GenerateSettings, model: MainModelConfig): { width: number; height: number } =>
  // Use the model-native grid for divided dimensions, not the UI's PiD-scaled grid.
  getPidDenoiseSize(settings, model.base, getGenerationDimensions(model).grid);

/** The pipeline a Generate run asks for: the whole schedule, decoded whole into `canvas_output`. */
const toGeneratePipelineInput = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings,
  denoiseSize: { width: number; height: number }
): FamilyPipelineInput => ({
  denoiseSize,
  denoisingStart: 0,
  model,
  multiDiffusion: null,
  output: { id: 'canvas_output', isIntermediate: outputIsIntermediate },
  settings,
  useCpuNoise: projectSettings.useCpuNoise,
  vaeTileSize: null,
});

interface SdPipelineExtras extends PipelineExtras {
  /** The denoise's start and end, which the SDXL refiner splits. */
  denoisingRange?: { denoising_start: number; denoising_end: number };
  /** Seamless tiling, which wraps the UNet and the VAE the graph decodes with. */
  seamless?: { seamless_x: boolean; seamless_y: boolean } | null;
  /** Runs once the denoise is wired and returns the node whose latents are decoded: the refiner's, when it runs. */
  afterDenoise?: (
    graph: BackendGraphContract,
    nodes: {
      denoise: BackendInvocationContract;
      negativePrompt: BackendInvocationContract;
      positivePrompt: BackendInvocationContract;
      seamless: BackendInvocationContract | null;
    }
  ) => BackendInvocationContract;
}

const buildSdPipeline = (input: FamilyPipelineInput, extras: SdPipelineExtras = {}): FamilyPipeline => {
  const { model, multiDiffusion, settings } = input;
  const graph: BackendGraphContract = { edges: [], id: createId(`${model.base}_graph`), nodes: {} };
  const modelLoaderType = model.base === 'sdxl' ? 'sdxl_model_loader' : 'main_model_loader';
  const compelType = getCompelPromptNodeType(model.base);
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const scheduler = coerceSchedulerForGraph(model, settings.scheduler);

  const positivePrompt = addNode(graph, { id: 'positive_prompt', type: 'string' });
  const negativePrompt = addNode(graph, { id: 'negative_prompt', type: 'string' });
  const seed = addNode(graph, { id: 'seed', type: 'integer' });
  const modelLoader = addNode(graph, { id: 'model_loader', model, type: modelLoaderType });
  const posCond = addNode(graph, { id: 'pos_cond', type: compelType });
  const negCond = addNode(graph, { id: 'neg_cond', type: compelType });
  // The multi-diffusion denoise takes one conditioning per side, not a collected list.
  const posCondCollect = multiDiffusion ? null : addNode(graph, { id: 'pos_cond_collect', type: 'collect' });
  const negCondCollect = multiDiffusion ? null : addNode(graph, { id: 'neg_cond_collect', type: 'collect' });
  const noise = addNode(graph, {
    height: input.denoiseSize.height,
    id: 'noise',
    type: 'noise',
    use_cpu: input.useCpuNoise,
    width: input.denoiseSize.width,
  });
  const denoise = addNode(graph, {
    cfg_rescale_multiplier: settings.cfgRescaleMultiplier,
    cfg_scale: settings.cfgScale,
    ...(extras.denoisingRange ?? { denoising_end: 1, denoising_start: input.denoisingStart }),
    id: 'denoise_latents',
    ...extras.denoiseFields,
    scheduler,
    steps: settings.steps,
    ...(multiDiffusion
      ? {
          tile_height: multiDiffusion.tileSize,
          tile_overlap: multiDiffusion.tileOverlap,
          tile_width: multiDiffusion.tileSize,
        }
      : {}),
    type: multiDiffusion ? 'tiled_multi_diffusion_denoise_latents' : 'denoise_latents',
  });
  const vaeOverride = getCompatibleVae(settings, model);
  const vaeLoader = vaeOverride
    ? addNode(graph, { id: 'vae_loader', type: 'vae_loader', vae_model: vaeOverride })
    : null;
  const seamless = extras.seamless ? addNode(graph, { id: 'seamless', ...extras.seamless, type: 'seamless' }) : null;
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

  if (posCondCollect && negCondCollect) {
    addEdge(graph, posCond, 'conditioning', posCondCollect, 'item');
    addEdge(graph, posCondCollect, 'collection', denoise, 'positive_conditioning');
    addEdge(graph, negCond, 'conditioning', negCondCollect, 'item');
    addEdge(graph, negCondCollect, 'collection', denoise, 'negative_conditioning');
  } else {
    addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');
    addEdge(graph, negCond, 'conditioning', denoise, 'negative_conditioning');
  }
  addEdge(graph, seed, 'value', noise, 'seed');

  // Seamless wrapping covers both the selected UNet and overridden VAE.
  if (seamless) {
    addEdge(graph, modelLoader, 'unet', seamless, 'unet');
    addEdge(graph, vaeLoader ?? modelLoader, 'vae', seamless, 'vae');
  }

  addEdge(graph, unetSource, 'unet', denoise, 'unet');
  addEdge(graph, noise, 'noise', denoise, 'noise');
  const decoded = extras.afterDenoise?.(graph, { denoise, negativePrompt, positivePrompt, seamless }) ?? denoise;
  const vaeSource = seamless ?? vaeLoader ?? modelLoader;
  const output = addPipelineDecode(graph, input, extras.decode, {
    denoise: decoded,
    l2iProps: { fp32: settings.vaePrecision === 'fp32' },
    l2iType: 'l2i',
    positivePrompt,
    seed,
    vaeSource,
  });

  return {
    conditioning:
      posCondCollect && negCondCollect
        ? {
            fluxReduxCollector: null,
            ipAdapterCollector: null,
            negative: { collector: negCondCollect, node: negCond },
            positive: posCond,
            positiveCollector: posCondCollect,
          }
        : null,
    denoise,
    denoiseSize: input.denoiseSize,
    graph,
    latents: {
      encodeType: 'i2l',
      fp32: settings.vaePrecision === 'fp32',

      unet: { field: 'unet', node: unetSource },
      vae: { field: 'vae', node: vaeSource },
    },
    layoutPrompt: null,
    output,
    vaeModel: vaeOverride,
  };
};

const buildSDGraph: GenerateGraphBuilder = (settings, model, outputIsIntermediate, projectSettings) => {
  const supportsHiDiffusion = model.base === 'sd-1' || model.base === 'sdxl';
  const scheduler = coerceSchedulerForGraph(model, settings.scheduler);
  // Validation rejects PiD alongside a refiner, so the refiner always feeds the ordinary VAE decode.
  const refiner = getActiveRefiner(settings, model);
  let ipAdapterCollector = null as BackendInvocationContract | null;
  const pipeline = buildSdPipeline(
    toGeneratePipelineInput(settings, model, outputIsIntermediate, projectSettings, getDenoiseSize(settings, model)),
    {
      afterDenoise: (graph, { denoise, negativePrompt, positivePrompt, seamless }) => {
        ipAdapterCollector = addIPAdapterReferenceImages(graph, settings, model, denoise);

        return refiner
          ? addSDXLRefinerStage({ denoise, graph, model, negativePrompt, positivePrompt, refiner, seamless, settings })
          : denoise;
      },
      decode: getGenerateDecode(settings, model, outputIsIntermediate),
      denoiseFields: supportsHiDiffusion
        ? {
            hidiffusion: settings.hiDiffusionEnabled,
            hidiffusion_raunet: settings.hiDiffusionRauNetEnabled,
            hidiffusion_window_attn: settings.hiDiffusionWindowAttentionEnabled,
            ...(settings.hiDiffusionEnabled
              ? {
                  hidiffusion_auto_ratios: settings.hiDiffusionT1Ratio === null && settings.hiDiffusionT2Ratio === null,
                  hidiffusion_t1_ratio: settings.hiDiffusionT1Ratio ?? 0,
                  hidiffusion_t2_ratio: settings.hiDiffusionT2Ratio ?? 0,
                }
              : {}),
          }
        : {},
      denoisingRange: getBaseDenoisingRange(settings, model, 0),
      seamless:
        settings.seamlessXAxis || settings.seamlessYAxis
          ? { seamless_x: settings.seamlessXAxis, seamless_y: settings.seamlessYAxis }
          : null,
    }
  );
  const { graph, output } = pipeline;

  addMetadata(graph, output, settings, model, model.base === 'sdxl' ? 'sdxl_txt2img' : 'txt2img', projectSettings, {
    ...(supportsHiDiffusion
      ? {
          hidiffusion: settings.hiDiffusionEnabled,
          hidiffusion_raunet: settings.hiDiffusionRauNetEnabled,
          hidiffusion_auto_ratios: settings.hiDiffusionT1Ratio === null && settings.hiDiffusionT2Ratio === null,
          hidiffusion_t1_ratio: settings.hiDiffusionT1Ratio,
          hidiffusion_t2_ratio: settings.hiDiffusionT2Ratio,
          hidiffusion_window_attn: settings.hiDiffusionWindowAttentionEnabled,
        }
      : {}),
    ...(shouldUsePidDecode(settings, model.base) ? getPidMetadata(settings) : {}),
    ...(refiner ? getRefinerMetadata(settings, model, refiner) : {}),
    scheduler,
  });
  addReferenceImageMetadata(graph, settings);

  return {
    ...pipeline,
    conditioning: pipeline.conditioning && { ...pipeline.conditioning, ipAdapterCollector },
  };
};

const buildSD3Graph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): FamilyGraph => {
  const graph: BackendGraphContract = { edges: [], id: createId('sd3_graph'), nodes: {} };
  const vaeModel = getCompatibleVae(settings, model);
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const modelLoader = addNode(graph, {
    clip_g_model: settings.clipGEmbedModel ?? undefined,
    clip_l_model: settings.clipLEmbedModel ?? undefined,
    id: 'model_loader',
    model,
    t5_encoder_model: settings.t5EncoderModel ?? undefined,
    type: 'sd3_model_loader',
    vae_model: vaeModel ?? undefined,
  });
  // SD 3 LoRAs train the transformer and, with DreamBooth's text-encoder option, CLIP-L and CLIP-G; T5 takes none.
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'sd3_lora_collection_loader', modelLoader, [
        'transformer',
        'clip_l',
        'clip_g',
      ])
    : modelLoader;
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

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, loraSource, 'clip_l', posCond, 'clip_l');
  addEdge(graph, loraSource, 'clip_l', negCond, 'clip_l');
  addEdge(graph, loraSource, 'clip_g', posCond, 'clip_g');
  addEdge(graph, loraSource, 'clip_g', negCond, 'clip_g');
  addEdge(graph, modelLoader, 't5_encoder', posCond, 't5_encoder');
  addEdge(graph, modelLoader, 't5_encoder', negCond, 't5_encoder');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');
  addEdge(graph, negCond, 'conditioning', denoise, 'negative_conditioning');
  addEdge(graph, seed, 'value', denoise, 'seed');
  addMetadata(graph, output, settings, model, 'sd3_txt2img', projectSettings, {
    clip_g_model: modelLoader.clip_g_model,
    clip_l_model: modelLoader.clip_l_model,
    scheduler: undefined,
    t5_encoder: modelLoader.t5_encoder_model,
    vae: vaeModel ?? undefined,
    ...(shouldUsePidDecode(settings, model.base) ? getPidMetadata(settings) : {}),
  });

  return {
    conditioning: null,
    denoise,
    denoiseSize: { height: denoise.height, width: denoise.width },
    graph,
    latents: { encodeType: 'sd3_i2l', vae: { field: 'vae', node: modelLoader } },
    layoutPrompt: null,
  };
};

/** The node defaults to `off`, so that preset sends nothing; scale and exponent apply only under `manual`. */
const getFluxDypeFields = (
  settings: GenerateSettings
): { dype_preset?: string; dype_scale?: number; dype_exponent?: number } => {
  if (settings.fluxDypePreset === 'off') {
    return {};
  }

  return {
    dype_preset: settings.fluxDypePreset,
    ...(settings.fluxDypePreset === 'manual'
      ? { dype_exponent: settings.fluxDypeExponent, dype_scale: settings.fluxDypeScale }
      : {}),
  };
};

const buildFluxPipeline = (input: FamilyPipelineInput, extras: PipelineExtras = {}) => {
  const { model, settings } = input;

  if (model.variant === 'dev_fill') {
    throw new Error('FLUX Fill models do not support text-to-image generation.');
  }

  const graph: BackendGraphContract = { edges: [], id: createId('flux_graph'), nodes: {} };
  const { positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const hasBundledComponents = isSelfContainedSDNQFlux1Pipeline(model);
  const t5EncoderModel = hasBundledComponents ? null : requireComponent(settings.t5EncoderModel, 'T5 Encoder');
  const clipEmbedModel = hasBundledComponents ? null : requireComponent(settings.clipEmbedModel, 'CLIP Embed');
  const vaeModel = hasBundledComponents ? null : requireComponent(getCompatibleVae(settings, model), 'FLUX VAE');
  const scheduler = coerceSchedulerForGraph(model, settings.scheduler);
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const modelLoader = addNode(graph, {
    clip_embed_model: clipEmbedModel ?? undefined,
    id: 'model_loader',
    model,
    t5_encoder_model: t5EncoderModel ?? undefined,
    type: 'flux_model_loader',
    vae_model: vaeModel ?? undefined,
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
    denoising_start: input.denoisingStart,
    guidance: settings.cfgScale,
    height: input.denoiseSize.height,
    id: 'denoise_latents',
    num_steps: settings.steps,
    scheduler,
    type: 'flux_denoise',
    width: input.denoiseSize.width,
    ...extras.denoiseFields,
  });
  const output = addPipelineDecode(graph, input, extras.decode, {
    denoise,
    l2iType: 'flux_vae_decode',
    positivePrompt,
    seed,
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
  addEdge(graph, seed, 'value', denoise, 'seed');

  return {
    components: { clipEmbedModel, t5EncoderModel },
    conditioning: {
      fluxReduxCollector: null,
      ipAdapterCollector: null,
      negative: null,
      positive: posCond,
      positiveCollector: posCondCollect,
    },
    denoise,
    denoiseSize: input.denoiseSize,
    graph,
    latents: { encodeType: 'flux_vae_encode', vae: { field: 'vae', node: modelLoader } },
    layoutPrompt: null,
    output,
    vaeModel,
  };
};

const buildFluxGraph: GenerateGraphBuilder = (settings, model, outputIsIntermediate, projectSettings) => {
  const dype = getFluxDypeFields(settings);
  const pipeline = buildFluxPipeline(
    toGeneratePipelineInput(settings, model, outputIsIntermediate, projectSettings, getDenoiseSize(settings, model)),
    { decode: getGenerateDecode(settings, model, outputIsIntermediate), denoiseFields: dype }
  );
  const { components, denoise, graph, output, vaeModel } = pipeline;

  // Generation validation has checked that the enabled Kontext references belong to this model.
  addFluxKontextReferenceImages(graph, settings, denoise, 'flux_kontext_reference_image');
  const ipAdapterCollector = addIPAdapterReferenceImages(graph, settings, model, denoise);
  const fluxReduxCollector = addFluxReduxReferenceImages(graph, settings, model, denoise);
  addMetadata(graph, output, settings, model, 'flux_txt2img', projectSettings, {
    clip_embed_model: components.clipEmbedModel ?? undefined,
    guidance: settings.cfgScale,
    ...dype,
    // Recorded even when off, so recall can restore it over a preset the panel has since turned on.
    dype_preset: settings.fluxDypePreset,
    scheduler: coerceSchedulerForGraph(model, settings.scheduler),
    t5_encoder: components.t5EncoderModel ?? undefined,
    vae: vaeModel ?? undefined,
    ...(shouldUsePidDecode(settings, model.base) ? getPidMetadata(settings) : {}),
  });
  addReferenceImageMetadata(graph, settings);

  return { ...pipeline, conditioning: { ...pipeline.conditioning, fluxReduxCollector, ipAdapterCollector } };
};

const buildFlux2Pipeline = (input: FamilyPipelineInput, extras: PipelineExtras = {}) => {
  const { model, settings } = input;
  const isDev = model.variant === 'dev';
  const sourceModel = getFlux2ComponentSource(model, settings);
  const hasBundledComponents = isBundledFlux2Main(model) || sourceModel;
  const mistralEncoderModel =
    settings.mistralEncoderModel && isFlux2MistralEncoder(settings.mistralEncoderModel)
      ? settings.mistralEncoderModel
      : null;
  const qwen3EncoderModel =
    !isDev && settings.qwen3EncoderModel && isFlux2Qwen3EncoderForModel(model)(settings.qwen3EncoderModel)
      ? settings.qwen3EncoderModel
      : null;
  const encoderModel = isDev ? mistralEncoderModel : qwen3EncoderModel;
  const vaeModel = getCompatibleVae(settings, model);

  if (!hasBundledComponents && (!vaeModel || !encoderModel)) {
    throw new Error(
      `This FLUX.2 model needs a VAE and ${isDev ? 'Mistral' : 'Qwen3'} Encoder, or a FLUX.2 pipeline its loader can take them from.`
    );
  }

  const graph: BackendGraphContract = { edges: [], id: createId('flux2_graph'), nodes: {} };
  const { positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const scheduler = coerceSchedulerForGraph(model, settings.scheduler);
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const modelLoader = addNode(
    graph,
    isDev
      ? {
          id: 'model_loader',
          mistral_encoder_model: mistralEncoderModel ?? undefined,
          mistral_source_model: sourceModel,
          model,
          type: 'flux2_dev_model_loader',
          vae_model: vaeModel ?? undefined,
        }
      : {
          id: 'model_loader',
          model,
          qwen3_encoder_model: qwen3EncoderModel ?? undefined,
          qwen3_source_model: sourceModel,
          type: 'flux2_klein_model_loader',
          vae_model: vaeModel ?? undefined,
        }
  );
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(
        graph,
        activeLoras,
        isDev ? 'flux2_dev_lora_collection_loader' : 'flux2_klein_lora_collection_loader',
        modelLoader,
        ['transformer', isDev ? 'mistral_encoder' : 'qwen3_encoder']
      )
    : modelLoader;
  const posCond = addNode(graph, {
    id: 'pos_cond',
    type: isDev ? 'flux2_dev_text_encoder' : 'flux2_klein_text_encoder',
  });
  const posCondCollect = addNode(graph, { id: 'pos_cond_collect', type: 'collect' });
  // CFG stays off: the slider drives the guidance [dev] is distilled on, which Klein ignores.
  const denoise = addNode(graph, {
    cfg_scale: 1,
    denoising_end: 1,
    denoising_start: input.denoisingStart,
    guidance: settings.cfgScale,
    height: input.denoiseSize.height,
    id: 'denoise_latents',
    num_steps: settings.steps,
    scheduler,
    type: 'flux2_denoise',
    width: input.denoiseSize.width,
  });
  const output = addPipelineDecode(graph, input, extras.decode, {
    denoise,
    l2iType: 'flux2_vae_decode',
    positivePrompt,
    seed,
    vaeSource: modelLoader,
  });

  addEdge(
    graph,
    loraSource,
    isDev ? 'mistral_encoder' : 'qwen3_encoder',
    posCond,
    isDev ? 'mistral_encoder' : 'qwen3_encoder'
  );
  addEdge(graph, modelLoader, 'max_seq_len', posCond, 'max_seq_len');
  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  // The denoise normalizes the latents with this VAE's BatchNorm statistics.
  addEdge(graph, modelLoader, 'vae', denoise, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', posCondCollect, 'item');
  addEdge(graph, posCondCollect, 'collection', denoise, 'positive_text_conditioning');
  addEdge(graph, seed, 'value', denoise, 'seed');

  return {
    components: { mistralEncoderModel, qwen3EncoderModel, sourceModel },
    conditioning: {
      fluxReduxCollector: null,
      ipAdapterCollector: null,
      negative: null,
      positive: posCond,
      positiveCollector: posCondCollect,
    },
    denoise,
    denoiseSize: input.denoiseSize,
    graph,
    latents: { encodeType: FLUX2_ENCODE_TYPE, vae: { field: 'vae', node: modelLoader } },
    layoutPrompt: null,
    output,
    vaeModel,
  };
};

const buildFlux2Graph: GenerateGraphBuilder = (settings, model, outputIsIntermediate, projectSettings) => {
  const pipeline = buildFlux2Pipeline(
    toGeneratePipelineInput(settings, model, outputIsIntermediate, projectSettings, getDenoiseSize(settings, model)),
    { decode: getGenerateDecode(settings, model, outputIsIntermediate) }
  );
  const { components, denoise, graph, output, vaeModel } = pipeline;

  addFluxKontextReferenceImages(graph, settings, denoise, 'flux2_reference_image');
  addMetadata(graph, output, settings, model, 'flux2_txt2img', projectSettings, {
    ...(model.variant === 'dev'
      ? { mistral_encoder: components.mistralEncoderModel ?? undefined }
      : { qwen3_encoder: components.qwen3EncoderModel ?? undefined, qwen3_source: components.sourceModel }),
    scheduler: coerceSchedulerForGraph(model, settings.scheduler),
    vae: vaeModel ?? undefined,
    ...(shouldUsePidDecode(settings, model.base) ? getPidMetadata(settings) : {}),
  });
  addReferenceImageMetadata(graph, settings);

  return pipeline;
};

const buildCogView4Graph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): FamilyGraph => {
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

  return {
    conditioning: null,
    denoise,
    denoiseSize: { height: denoise.height, width: denoise.width },
    graph,
    latents: { encodeType: 'cogview4_i2l', vae: { field: 'vae', node: modelLoader } },
    layoutPrompt: null,
  };
};

const buildErnieImageGraph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): FamilyGraph => {
  const isDiffusers = model.format === 'diffusers';
  const vaeModel = getCompatibleVae(settings, model);
  const mistralEncoderModel = getCompatibleComponent(settings.mistralEncoderModel, isErnieImageMistralEncoder);

  if (!isDiffusers) {
    requireComponent(mistralEncoderModel, 'Ministral 3B Encoder');
    requireComponent(vaeModel, 'ERNIE-Image VAE');
  }

  const graph: BackendGraphContract = { edges: [], id: createId('ernie_image_graph'), nodes: {} };
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const scheduler = coerceSchedulerForGraph(model, settings.scheduler);
  const useCfg = settings.cfgScale > 1;
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model,
    text_encoder_model: mistralEncoderModel ?? undefined,
    type: 'ernie_image_model_loader',
    // Unload the unused enhancer because it cannot idle-offload.
    use_prompt_enhancer: false,
    vae_model: vaeModel ?? undefined,
  });
  // ERNIE-Image LoRAs train the transformer only; the Ministral encoder field carries none.
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'ernie_image_lora_collection_loader', modelLoader, [
        'transformer',
      ])
    : modelLoader;
  const posCond = addNode(graph, { id: 'pos_cond', type: 'ernie_image_text_encoder' });
  const negCond = useCfg ? addNode(graph, { id: 'neg_cond', type: 'ernie_image_text_encoder' }) : null;
  const denoise = addNode(graph, {
    denoising_end: 1,
    denoising_start: 0,
    guidance_scale: settings.cfgScale,
    height: settings.height,
    id: 'denoise_latents',
    scheduler,
    steps: settings.steps,
    type: 'ernie_image_denoise',
    width: settings.width,
  });
  const output = addImageOutputNode(graph, 'ernie_image_vae_decode', outputIsIntermediate);

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, modelLoader, 'text_encoder', posCond, 'text_encoder');
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');

  if (negCond) {
    addEdge(graph, modelLoader, 'text_encoder', negCond, 'text_encoder');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
    addEdge(graph, negCond, 'conditioning', denoise, 'negative_conditioning');
  }

  addEdge(graph, seed, 'value', denoise, 'seed');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addMetadata(graph, output, settings, model, 'ernie_image_txt2img', projectSettings, {
    mistral_encoder: mistralEncoderModel ?? undefined,
  });

  return {
    conditioning: null,
    denoise,
    denoiseSize: { height: denoise.height, width: denoise.width },
    graph,
    latents: null,
    layoutPrompt: null,
  };
};

interface QwenImagePipelineExtras extends PipelineExtras {
  /** Literal fields each text encoder gains: Generate's encoder quantization. */
  encoderFields?: Record<string, unknown>;
}

const buildQwenImagePipeline = (input: FamilyPipelineInput, extras: QwenImagePipelineExtras = {}) => {
  const { model, settings } = input;
  const sourceModel = getCompatibleComponentSource(settings, model);
  const hasBundledComponents = model.format === 'diffusers' || sourceModel;
  const vaeModel = getCompatibleVae(settings, model);

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
        'qwen_vl_encoder',
      ])
    : modelLoader;
  const posCond = addNode(graph, { id: 'pos_cond', type: 'qwen_image_text_encoder', ...extras.encoderFields });
  const negCond = useCfg
    ? addNode(graph, { id: 'neg_cond', type: 'qwen_image_text_encoder', ...extras.encoderFields })
    : null;
  const denoise = addNode(graph, {
    cfg_scale: settings.cfgScale,
    ...partialDenoisingRange(input),
    height: input.denoiseSize.height,
    id: 'denoise_latents',
    ...extras.denoiseFields,
    steps: settings.steps,
    type: 'qwen_image_denoise',
    width: input.denoiseSize.width,
  });
  const output = addPipelineDecode(graph, input, extras.decode, {
    denoise,
    l2iType: 'qwen_image_l2i',
    positivePrompt,
    seed,
    vaeSource: modelLoader,
  });

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  // The denoise encodes a ControlNet's control and inpaint images with this VAE; without one the input is idle.
  addEdge(graph, modelLoader, 'vae', denoise, 'vae');
  addEdge(graph, loraSource, 'qwen_vl_encoder', posCond, 'qwen_vl_encoder');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');

  if (negCond) {
    addEdge(graph, loraSource, 'qwen_vl_encoder', negCond, 'qwen_vl_encoder');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
    addEdge(graph, negCond, 'conditioning', denoise, 'negative_conditioning');
  }

  addEdge(graph, seed, 'value', denoise, 'seed');

  return {
    components: { sourceModel },
    conditioning: null,
    denoise,
    denoiseSize: input.denoiseSize,
    graph,
    latents: { encodeType: QWEN_IMAGE_ENCODE_TYPE, vae: { field: 'vae', node: modelLoader } },
    layoutPrompt: null,
    output,
    positiveEncoder: posCond,
    vaeModel,
  };
};

const buildQwenImageGraph: GenerateGraphBuilder = (settings, model, outputIsIntermediate, projectSettings) => {
  const pipeline = buildQwenImagePipeline(
    toGeneratePipelineInput(settings, model, outputIsIntermediate, projectSettings, getDenoiseSize(settings, model)),
    {
      decode: getGenerateDecode(settings, model, outputIsIntermediate),
      // Null keeps the node's dynamic shift; a value fixes the schedule (3 for the Lightning LoRAs).
      denoiseFields: settings.qwenImageShift === null ? {} : { shift: settings.qwenImageShift },
      // The encoders default to `none`, so only an actual quantization choice reaches the graph.
      encoderFields:
        settings.qwenImageEncoderQuantization === 'none' ? {} : { quantization: settings.qwenImageEncoderQuantization },
    }
  );
  const { components, denoise, graph, output, positiveEncoder: posCond, vaeModel } = pipeline;
  // The VAE the graph decodes with, which encodes the edit reference too.
  const vae = pipeline.latents.vae;
  const qwenReferenceImages =
    model.variant === 'edit' ? getEnabledReferenceImages(settings.referenceImages, 'qwen_image_reference_image') : [];

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
      const refI2l = addNode(graph, { id: createId('qwen_ref_i2l'), type: QWEN_IMAGE_ENCODE_TYPE });
      const refImageNode = addNode(graph, {
        id: createId('qwen_ref_img_for_vae'),
        image: toImageField(firstImage),
        type: 'image',
      });

      addEdge(graph, refImageNode, 'image', refI2l, 'image');
      addEdge(graph, vae.node, vae.field, refI2l, 'vae');
      addEdge(graph, refI2l, 'latents', denoise, 'reference_latents');
    }
  }
  addMetadata(graph, output, settings, model, 'qwen_image_txt2img', projectSettings, {
    qwen_image_component_source: components.sourceModel,
    qwen_image_quantization: settings.qwenImageEncoderQuantization,
    qwen_image_qwen_vl_encoder: settings.qwenVLEncoderModel ?? undefined,
    qwen_image_shift: settings.qwenImageShift ?? 'auto',
    qwen_image_vae: vaeModel ?? undefined,
    scheduler: undefined,
    ...(shouldUsePidDecode(settings, model.base) ? getPidMetadata(settings) : {}),
  });
  addReferenceImageMetadata(graph, settings);

  return pipeline;
};

interface ConditionedPipelineExtras extends PipelineExtras {
  /** Passes the positive conditioning through Generate's enhancers before its collector. */
  positiveConditioning?: ConditioningHook;
}

const buildZImagePipeline = (input: FamilyPipelineInput, extras: ConditionedPipelineExtras = {}) => {
  const { model, settings } = input;
  const sourceModel = getDiffusersSource(settings, model);
  const qwen3EncoderModel =
    settings.qwen3EncoderModel && isZImageQwen3Encoder(settings.qwen3EncoderModel) ? settings.qwen3EncoderModel : null;
  const vaeModel = getCompatibleVae(settings, model);

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
    denoising_start: input.denoisingStart,
    guidance_scale: settings.cfgScale,
    height: input.denoiseSize.height,
    id: 'denoise_latents',
    scheduler,
    ...extras.denoiseFields,
    steps: settings.steps,
    type: 'z_image_denoise',
    width: input.denoiseSize.width,
  });
  const output = addPipelineDecode(graph, input, extras.decode, {
    denoise,
    l2iType: 'z_image_l2i',
    positivePrompt,
    seed,
    vaeSource: modelLoader,
  });

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, loraSource, 'qwen3_encoder', posCond, 'qwen3_encoder');
  // The denoise encodes a control image itself, with this VAE.
  addEdge(graph, modelLoader, 'vae', denoise, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  const positiveConditioning = extras.positiveConditioning?.(graph, posCond, seed) ?? posCond;
  addEdge(graph, positiveConditioning, 'conditioning', posCondCollect, 'item');
  addEdge(graph, posCondCollect, 'collection', denoise, 'positive_conditioning');

  if (negCond && negCondCollect) {
    addEdge(graph, loraSource, 'qwen3_encoder', negCond, 'qwen3_encoder');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
    addEdge(graph, negCond, 'conditioning', negCondCollect, 'item');
    addEdge(graph, negCondCollect, 'collection', denoise, 'negative_conditioning');
  }

  addEdge(graph, seed, 'value', denoise, 'seed');

  return {
    components: { qwen3EncoderModel, sourceModel },
    conditioning: {
      fluxReduxCollector: null,
      ipAdapterCollector: null,
      negative: negCond && negCondCollect ? { collector: negCondCollect, node: negCond } : null,
      positive: posCond,
      positiveCollector: posCondCollect,
    },
    denoise,
    denoiseSize: input.denoiseSize,
    graph,
    latents: { encodeType: 'z_image_i2l', vae: { field: 'vae', node: modelLoader } },
    layoutPrompt: null,
    output,
    vaeModel,
  };
};

const buildZImageGraph: GenerateGraphBuilder = (settings, model, outputIsIntermediate, projectSettings) => {
  const pipeline = buildZImagePipeline(
    toGeneratePipelineInput(settings, model, outputIsIntermediate, projectSettings, getDenoiseSize(settings, model)),
    {
      decode: getGenerateDecode(settings, model, outputIsIntermediate),
      // Null keeps the node's size-derived shift.
      denoiseFields: settings.zImageShift === null ? {} : { shift: settings.zImageShift },
      // Zero strength would be a no-op node, so it is left out like Krea-2's enhancer.
      positiveConditioning:
        settings.zImageSeedVarianceEnabled && settings.zImageSeedVarianceStrength > 0
          ? (graph, conditioning, seed) => {
              const seedVariance = addNode(graph, {
                id: 'seed_variance_enhancer',
                randomize_percent: settings.zImageSeedVarianceRandomizePercent,
                strength: settings.zImageSeedVarianceStrength,
                type: 'z_image_seed_variance_enhancer',
              });

              addEdge(graph, seed, 'value', seedVariance, 'seed');
              addEdge(graph, conditioning, 'conditioning', seedVariance, 'conditioning');

              return seedVariance;
            }
          : undefined,
    }
  );
  const { components, graph, output, vaeModel } = pipeline;

  addMetadata(graph, output, settings, model, 'z_image_txt2img', projectSettings, {
    qwen3_encoder: components.qwen3EncoderModel ?? undefined,
    qwen3_source: components.sourceModel,
    scheduler: coerceSchedulerForGraph(model, settings.scheduler),
    vae: vaeModel ?? undefined,
    z_image_seed_variance_enabled: settings.zImageSeedVarianceEnabled,
    z_image_seed_variance_randomize_percent: settings.zImageSeedVarianceRandomizePercent,
    z_image_seed_variance_strength: settings.zImageSeedVarianceStrength,
    z_image_shift: settings.zImageShift ?? 'auto',
    ...(shouldUsePidDecode(settings, model.base) ? getPidMetadata(settings) : {}),
  });

  return pipeline;
};

const buildAnimaPipeline = (input: FamilyPipelineInput) => {
  const { model, settings } = input;
  const vaeModel = requireComponent(getCompatibleVae(settings, model), 'Anima VAE');
  const qwen3EncoderModel = requireComponent(
    settings.qwen3EncoderModel && isAnimaQwen3Encoder(settings.qwen3EncoderModel) ? settings.qwen3EncoderModel : null,
    'Qwen3 Encoder'
  );
  const qwen35EncoderModel =
    model.variant === 'anima_qwen35'
      ? requireComponent(getCompatibleComponent(settings.qwen35EncoderModel, isAnimaQwen35Encoder), 'Qwen3.5 Encoder')
      : null;
  const graph: BackendGraphContract = { edges: [], id: createId('anima_graph'), nodes: {} };
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const scheduler = coerceSchedulerForGraph(model, settings.scheduler);
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const useCfg = settings.cfgScale > 1;
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model,
    qwen3_encoder_model: qwen3EncoderModel,
    ...(qwen35EncoderModel ? { qwen3_5_encoder_model: qwen35EncoderModel } : {}),
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
    denoising_start: input.denoisingStart,
    guidance_scale: settings.cfgScale,
    height: input.denoiseSize.height,
    id: 'denoise_latents',
    scheduler,
    steps: settings.steps,
    type: 'anima_denoise',
    width: input.denoiseSize.width,
  });
  // `anima_l2i` has no tiling switch: it tiles on its own once an untiled decode would take most of the card.
  const output = addPipelineOutputNode(graph, input, 'anima_l2i', { tiles: false });

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, loraSource, 'qwen3_encoder', posCond, 'qwen3_encoder');
  if (qwen35EncoderModel) {
    // Straight from the loader: the LoRA collection loader passes Qwen3.5 through untouched and has no output for it.
    addEdge(graph, modelLoader, 'qwen3_5_encoder', posCond, 'qwen3_5_encoder');
  }
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', posCondCollect, 'item');
  addEdge(graph, posCondCollect, 'collection', denoise, 'positive_conditioning');

  if (negCond && negCondCollect) {
    addEdge(graph, loraSource, 'qwen3_encoder', negCond, 'qwen3_encoder');
    if (qwen35EncoderModel) {
      addEdge(graph, modelLoader, 'qwen3_5_encoder', negCond, 'qwen3_5_encoder');
    }
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
    addEdge(graph, negCond, 'conditioning', negCondCollect, 'item');
    addEdge(graph, negCondCollect, 'collection', denoise, 'negative_conditioning');
  }

  addEdge(graph, seed, 'value', denoise, 'seed');
  addEdge(graph, denoise, 'latents', output, 'latents');

  return {
    components: { qwen35EncoderModel, qwen3EncoderModel },
    conditioning: {
      fluxReduxCollector: null,
      ipAdapterCollector: null,
      negative: negCond && negCondCollect ? { collector: negCondCollect, node: negCond } : null,
      positive: posCond,
      positiveCollector: posCondCollect,
    },
    denoise,
    denoiseSize: input.denoiseSize,
    graph,
    latents: { encodeType: 'anima_i2l', vae: { field: 'vae', node: modelLoader } },
    layoutPrompt: null,
    output,
    vaeModel,
  };
};

const buildAnimaGraph: GenerateGraphBuilder = (settings, model, outputIsIntermediate, projectSettings) => {
  const pipeline = buildAnimaPipeline(
    toGeneratePipelineInput(settings, model, outputIsIntermediate, projectSettings, {
      height: settings.height,
      width: settings.width,
    })
  );
  const { components, denoise, graph, output, vaeModel } = pipeline;
  const ipAdapterCollector = addIPAdapterReferenceImages(graph, settings, model, denoise);

  addMetadata(graph, output, settings, model, 'anima_txt2img', projectSettings, {
    qwen3_encoder: components.qwen3EncoderModel,
    ...(components.qwen35EncoderModel ? { qwen3_5_encoder: components.qwen35EncoderModel } : {}),
    scheduler: coerceSchedulerForGraph(model, settings.scheduler),
    vae: vaeModel,
  });
  addReferenceImageMetadata(graph, settings);

  return { ...pipeline, conditioning: { ...pipeline.conditioning, ipAdapterCollector } };
};

/** Each reference reaches encoder and denoise in the same order; chained collects keep it, a single one would not. */
const addKrea2EditReferences = (
  graph: BackendGraphContract,
  {
    denoise,
    prompts,
    referenceImages,
    referenceKvCache,
    vaeSource,
  }: {
    denoise: BackendInvocationContract;
    prompts: readonly BackendInvocationContract[];
    referenceImages: readonly { image_name: string }[];
    /** Unset leaves the denoise default. */
    referenceKvCache: boolean | undefined;
    vaeSource: BackendInvocationContract;
  }
) => {
  for (const prompt of prompts) {
    prompt.reference_images = referenceImages.map((image) => ({ ...image }));
  }

  let previousCollect: BackendInvocationContract | null = null;

  for (const image of referenceImages) {
    // `reference` sizes the image as ai-toolkit trains edit LoRAs, so width and height stay unset.
    const encode = addNode(graph, {
      id: createId('krea2_ref_i2l'),
      image,
      reference: true,
      type: QWEN_IMAGE_ENCODE_TYPE,
    });
    const collect = addNode(graph, { id: createId('krea2_ref_latents_collect'), type: 'collect' });

    addEdge(graph, vaeSource, 'vae', encode, 'vae');
    addEdge(graph, encode, 'latents', collect, 'item');
    if (previousCollect) {
      addEdge(graph, previousCollect, 'collection', collect, 'collection');
    }
    previousCollect = collect;
  }

  if (previousCollect) {
    addEdge(graph, previousCollect, 'collection', denoise, 'reference_latents');
  }
  if (referenceKvCache !== undefined) {
    denoise.reference_kv_cache = referenceKvCache;
  }
};

interface Krea2PipelineExtras extends ConditionedPipelineExtras {
  /** The GGUF encoder's vision tower for edit references; the loader refuses one beside any other encoder. */
  visionTower?: ComponentModelConfig | null;
}

const buildKrea2Pipeline = (input: FamilyPipelineInput, extras: Krea2PipelineExtras = {}) => {
  const { model, settings } = input;
  const isDiffusers = model.format === 'diffusers';
  const vaeModel = getCompatibleVae(settings, model);
  const qwen3VlEncoderModel = getCompatibleComponent(settings.qwen3VLEncoderModel, isQwen3Vl4bEncoder);
  const qwen3VlVisionTowerModel =
    qwen3VlEncoderModel && isGgufQwen3Vl4bEncoder(qwen3VlEncoderModel)
      ? getCompatibleComponent(extras.visionTower ?? null, isQwen3Vl4bVisionTower)
      : null;

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
    qwen3_vl_vision_tower_model: qwen3VlVisionTowerModel ?? undefined,
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
  const posCondCollect = addNode(graph, { id: 'pos_cond_collect', type: 'collect' });
  const negCond = useCfg ? addNode(graph, { id: 'neg_cond', type: 'krea2_text_encoder' }) : null;
  const negCondCollect = useCfg ? addNode(graph, { id: 'neg_cond_collect', type: 'collect' }) : null;
  // `krea2_denoise` picks its own shift, fixed for Turbo and by size for Raw, so the graph sends none.
  const denoise = addNode(graph, {
    cfg_scale: settings.cfgScale,
    ...partialDenoisingRange(input),
    height: input.denoiseSize.height,
    id: 'denoise_latents',
    steps: settings.steps,
    type: 'krea2_denoise',
    width: input.denoiseSize.width,
  });
  // Krea-2 decodes with the Qwen-Image VAE, so it reuses that family's latents-to-image node.
  const output = addPipelineOutputNode(graph, input, 'qwen_image_l2i', { tiles: true });

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, loraSource, 'qwen3_vl_encoder', posCond, 'qwen3_vl_encoder');
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');

  const positiveConditioning = extras.positiveConditioning?.(graph, posCond, seed) ?? posCond;
  addEdge(graph, positiveConditioning, 'conditioning', posCondCollect, 'item');
  addEdge(graph, posCondCollect, 'collection', denoise, 'positive_conditioning');

  if (negCond && negCondCollect) {
    addEdge(graph, loraSource, 'qwen3_vl_encoder', negCond, 'qwen3_vl_encoder');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
    addEdge(graph, negCond, 'conditioning', negCondCollect, 'item');
    addEdge(graph, negCondCollect, 'collection', denoise, 'negative_conditioning');
  }

  addEdge(graph, seed, 'value', denoise, 'seed');
  addEdge(graph, denoise, 'latents', output, 'latents');

  return {
    components: { loadsLoras: activeLoras.length > 0, qwen3VlEncoderModel, qwen3VlVisionTowerModel },
    conditioning: {
      fluxReduxCollector: null,
      ipAdapterCollector: null,
      negative: negCond && negCondCollect ? { collector: negCondCollect, node: negCond } : null,
      positive: posCond,
      positiveCollector: posCondCollect,
    },
    denoise,
    denoiseSize: input.denoiseSize,
    graph,
    latents: { encodeType: QWEN_IMAGE_ENCODE_TYPE, vae: { field: 'vae', node: modelLoader } },
    layoutPrompt: null,
    output,
    vaeModel,
  };
};

const buildKrea2Graph: GenerateGraphBuilder = (settings, model, outputIsIntermediate, projectSettings) => {
  const pipeline = buildKrea2Pipeline(
    toGeneratePipelineInput(settings, model, outputIsIntermediate, projectSettings, {
      height: settings.height,
      width: settings.width,
    }),
    {
      positiveConditioning: (graph, conditioning, seed) =>
        addKrea2ConditioningEnhancers({ conditioning, graph, idPrefix: 'krea2', seed, settings }),
      visionTower: settings.qwen3VLVisionTowerModel,
    }
  );
  const { components, conditioning, denoise, graph, output, vaeModel } = pipeline;
  // The VAE the graph decodes with, which encodes the references too.
  const vaeSource = pipeline.latents.vae.node;
  // Validation keeps edit and style references apart; the denoise refuses the two together.
  const editReferenceImages = getEnabledReferenceImages(settings.referenceImages, 'krea2_edit_reference_image').flatMap(
    ({ config }) => (config.image ? [toImageField(config.image)] : [])
  );
  // Shared-KV reference attention takes exactly one reference; zero strength is a full bypass, so it is not run.
  const styleReference =
    editReferenceImages.length > 0
      ? undefined
      : getEnabledReferenceImages(settings.referenceImages, 'krea2_reference_image').find(
          (referenceImage) => referenceImage.config.styleStrength > 0
        );

  // Only an edit LoRA trained with `kv_cache` reads the cache; without one the stored switch is kept but not sent.
  const referenceKvCache =
    editReferenceImages.length > 0 && components.loadsLoras ? settings.krea2ReferenceKvCache : undefined;

  if (editReferenceImages.length > 0) {
    addKrea2EditReferences(graph, {
      denoise,
      prompts: conditioning.negative ? [conditioning.positive, conditioning.negative.node] : [conditioning.positive],
      referenceImages: editReferenceImages,
      referenceKvCache,
      vaeSource,
    });
  } else if (styleReference?.config.image) {
    const styleReferenceNode = addNode(graph, {
      // The reference's image tokens are appended to the target's, so both must be the same size.
      height: settings.height,
      id: 'krea2_style_reference',
      image: toImageField(styleReference.config.image),
      style_strength: styleReference.config.styleStrength,
      type: 'krea2_style_reference',
      width: settings.width,
    });

    addEdge(graph, vaeSource, 'vae', styleReferenceNode, 'vae');
    addEdge(graph, styleReferenceNode, 'style_reference', denoise, 'style_reference');
  }
  addMetadata(graph, output, settings, model, 'krea2_txt2img', projectSettings, {
    krea2_rebalance_enabled: settings.krea2RebalanceEnabled,
    krea2_rebalance_multiplier: settings.krea2RebalanceMultiplier,
    krea2_rebalance_weights: settings.krea2RebalanceWeights,
    krea2_seed_variance_enabled: settings.krea2SeedVarianceEnabled,
    krea2_seed_variance_randomize_percent: settings.krea2SeedVarianceRandomizePercent,
    krea2_seed_variance_strength: settings.krea2SeedVarianceStrength,
    ...(styleReference ? { krea2_style_strength: styleReference.config.styleStrength } : {}),
    ...(referenceKvCache === undefined ? {} : { krea2_reference_kv_cache: referenceKvCache }),
    qwen3_vl_encoder: components.qwen3VlEncoderModel ?? undefined,
    qwen3_vl_vision_tower: components.qwen3VlVisionTowerModel ?? undefined,
    vae: vaeModel ?? undefined,
  });
  addReferenceImageMetadata(graph, settings);

  return pipeline;
};

const buildQwenImage21Graph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): FamilyGraph => {
  // For single-file and GGUF transformers the generation policy requires VAE and encoder before this runs.
  const vaeModel = getCompatibleVae(settings, model);
  const qwen3VlEncoderModel = getCompatibleComponent(settings.qwen3VLEncoderModel, isQwen3Vl8bEncoder);
  const qwen3VlVisionTowerModel =
    qwen3VlEncoderModel && isGgufQwen3Vl8bEncoder(qwen3VlEncoderModel)
      ? getCompatibleComponent(settings.qwen3VLVisionTowerModel, isQwen3Vl8bVisionTower)
      : null;

  const graph: BackendGraphContract = { edges: [], id: createId('qwen_image_2_1_graph'), nodes: {} };
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  // True CFG runs only above 1, the model's default being off; a negative prompt has no effect without it.
  const useCfg = settings.cfgScale > 1;
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model,
    qwen3_vl_encoder_model: qwen3VlEncoderModel ?? undefined,
    qwen3_vl_vision_tower_model: qwen3VlVisionTowerModel ?? undefined,
    type: 'qwen_image_2_1_model_loader',
    vae_model: vaeModel ?? undefined,
  });
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'qwen_image_2_1_lora_collection_loader', modelLoader, [
        'transformer',
      ])
    : modelLoader;
  const posCond = addNode(graph, { id: 'pos_cond', type: 'qwen_image_2_1_text_encoder' });
  const negCond = useCfg ? addNode(graph, { id: 'neg_cond', type: 'qwen_image_2_1_text_encoder' }) : null;
  const denoise = addNode(graph, {
    cfg_scale: settings.cfgScale,
    height: settings.height,
    id: 'denoise_latents',
    steps: settings.steps,
    type: 'qwen_image_2_1_denoise',
    width: settings.width,
  });
  const output = addImageOutputNode(graph, 'qwen_image_2_1_l2i', outputIsIntermediate);

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, modelLoader, 'qwen3_vl_encoder', posCond, 'qwen3_vl_encoder');
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');
  if (negCond) {
    addEdge(graph, modelLoader, 'qwen3_vl_encoder', negCond, 'qwen3_vl_encoder');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
    addEdge(graph, negCond, 'conditioning', denoise, 'negative_conditioning');
  }
  addEdge(graph, seed, 'value', denoise, 'seed');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addQwenImage21ReferenceImages(graph, settings, {
    conditionings: negCond ? [posCond, negCond] : [posCond],
    denoise,
    modelLoader,
  });
  // The denoise adds its prefix-cache mode to the record, so metadata passes through it.
  addMetadata(graph, denoise, settings, model, 'qwen_image_2_1_txt2img', projectSettings, {
    qwen3_vl_encoder: qwen3VlEncoderModel ?? undefined,
    qwen3_vl_vision_tower: qwen3VlVisionTowerModel ?? undefined,
    vae: vaeModel ?? undefined,
  });
  addEdge(graph, denoise, 'metadata', output, 'metadata');
  addReferenceImageMetadata(graph, settings);

  return {
    conditioning: null,
    denoise,
    denoiseSize: { height: denoise.height, width: denoise.width },
    graph,
    latents: { encodeType: QWEN_IMAGE_2_1_ENCODE_TYPE, vae: { field: 'vae', node: modelLoader } },
    layoutPrompt: null,
  };
};

/** The encoder sees each reference through its vision tower and the denoise places its latents, in one order. */
const addQwenImage21ReferenceImages = (
  graph: BackendGraphContract,
  settings: GenerateSettings,
  nodes: {
    conditionings: BackendInvocationContract[];
    denoise: BackendInvocationContract;
    modelLoader: BackendInvocationContract;
  }
): void => {
  const referenceImages = getEnabledReferenceImages(settings.referenceImages, 'qwen_image_2_1_reference_image');
  if (referenceImages.length === 0) {
    return;
  }

  let imageCollect: BackendInvocationContract | null = null;
  let latentsCollect: BackendInvocationContract | null = null;

  for (const { config } of referenceImages) {
    if (!config.image) {
      continue;
    }

    const imageNode = addNode(graph, {
      id: createId('qwen21_ref_img'),
      image: toImageField(config.image),
      type: 'image',
    });
    const i2l = addNode(graph, { id: createId('qwen21_ref_i2l'), reference: true, type: QWEN_IMAGE_2_1_ENCODE_TYPE });
    addEdge(graph, imageNode, 'image', i2l, 'image');
    addEdge(graph, nodes.modelLoader, 'vae', i2l, 'vae');

    const nextImageCollect = addNode(graph, { id: createId('qwen21_ref_img_collect'), type: 'collect' });
    addEdge(graph, imageNode, 'image', nextImageCollect, 'item');
    if (imageCollect) {
      addEdge(graph, imageCollect, 'collection', nextImageCollect, 'collection');
    }
    imageCollect = nextImageCollect;

    const nextLatentsCollect = addNode(graph, { id: createId('qwen21_ref_latents_collect'), type: 'collect' });
    addEdge(graph, i2l, 'latents', nextLatentsCollect, 'item');
    if (latentsCollect) {
      addEdge(graph, latentsCollect, 'collection', nextLatentsCollect, 'collection');
    }
    latentsCollect = nextLatentsCollect;
  }

  if (!imageCollect || !latentsCollect) {
    return;
  }
  for (const conditioning of nodes.conditionings) {
    addEdge(graph, imageCollect, 'collection', conditioning, 'reference_images');
  }
  addEdge(graph, latentsCollect, 'collection', nodes.denoise, 'reference_latents');
};

interface MossImagePipelineExtras extends PipelineExtras {
  /** The node whose text the positive encoder reads instead of the prompt: Generate's layout prompt. */
  positivePromptSource?: (
    graph: BackendGraphContract,
    positivePrompt: BackendInvocationContract
  ) => BackendInvocationContract;
}

const buildMossImagePipeline = (input: FamilyPipelineInput, extras: MossImagePipelineExtras = {}) => {
  const { model, settings } = input;
  // The filter drops a stale Qwen3.5 4B pick left over from Anima-3.8B, which shares the setting.
  const isDiffusers = model.format === 'diffusers';
  const vaeModel = getCompatibleVae(settings, model);
  const qwen35EncoderModel = getCompatibleComponent(settings.qwen35EncoderModel, isMossImageQwen35Encoder);

  if (!isDiffusers) {
    requireComponent(qwen35EncoderModel, 'Qwen3.5 9B Encoder');
    requireComponent(vaeModel, 'MOSS-Image VAE');
  }

  const graph: BackendGraphContract = { edges: [], id: createId('moss_image_graph'), nodes: {} };
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  // The negative branch runs only above CFG 1, where the node requires its conditioning.
  const useCfg = settings.cfgScale > 1;
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model,
    qwen3_5_encoder_model: qwen35EncoderModel ?? undefined,
    type: 'moss_image_model_loader',
    vae_model: vaeModel ?? undefined,
  });
  // MOSS-Image LoRAs train the transformer only; the VAE and the Qwen3.5 encoder stay on the model loader.
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'moss_image_lora_collection_loader', modelLoader, [
        'transformer',
      ])
    : modelLoader;
  const posCond = addNode(graph, { id: 'pos_cond', type: 'moss_image_text_encoder' });
  const negCond = useCfg ? addNode(graph, { id: 'neg_cond', type: 'moss_image_text_encoder' }) : null;
  const denoise = addNode(graph, {
    cfg_scale: settings.cfgScale,
    denoising_end: 1,
    denoising_start: input.denoisingStart,
    height: input.denoiseSize.height,
    id: 'denoise_latents',
    steps: settings.steps,
    type: 'moss_image_denoise',
    ...extras.denoiseFields,
    width: input.denoiseSize.width,
  });
  // MOSS-Image denoises into the FLUX.2 VAE's raw latent layout, so it decodes, and PiD-decodes, as FLUX.2 does.
  const output = addPipelineDecode(graph, input, extras.decode, {
    denoise,
    l2iType: 'flux2_vae_decode',
    positivePrompt,
    seed,
    vaeSource: modelLoader,
  });

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  // The denoise node reads the VAE's BatchNorm statistics, which define the latent space it works in.
  addEdge(graph, modelLoader, 'vae', denoise, 'vae');
  addEdge(graph, modelLoader, 'qwen3_5_encoder', posCond, 'qwen3_5_encoder');
  const promptSource = extras.positivePromptSource?.(graph, positivePrompt) ?? positivePrompt;
  addEdge(graph, promptSource, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');

  if (negCond) {
    addEdge(graph, modelLoader, 'qwen3_5_encoder', negCond, 'qwen3_5_encoder');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
    addEdge(graph, negCond, 'conditioning', denoise, 'negative_conditioning');
  }

  addEdge(graph, seed, 'value', denoise, 'seed');

  return {
    components: { qwen35EncoderModel },
    conditioning: null,
    denoise,
    denoiseSize: input.denoiseSize,
    graph,
    latents: { encodeType: FLUX2_ENCODE_TYPE, vae: { field: 'vae', node: modelLoader } },
    layoutPrompt: null as BackendInvocationContract | null,
    output,
    vaeModel,
  };
};

const buildMossImageGraph: GenerateGraphBuilder = (settings, model, outputIsIntermediate, projectSettings) => {
  const denoiseSize = getDenoiseSize(settings, model);
  let layoutPrompt = null as BackendInvocationContract | null;
  const pipeline = buildMossImagePipeline(
    toGeneratePipelineInput(settings, model, outputIsIntermediate, projectSettings, denoiseSize),
    {
      decode: getGenerateDecode(settings, model, outputIsIntermediate),
      denoiseFields: { uncond_skip_layers: [...settings.mossImageUncondSkipLayers] },
      // Compiles a layout draft and canvas `regions` at run time; batch prompt variation flows via `positive_prompt`.
      positivePromptSource: (graph, positivePrompt) => {
        layoutPrompt = addNode(graph, {
          height: denoiseSize.height,
          id: 'layout_prompt',
          regions: [],
          type: 'moss_image_layout_prompt',
          width: denoiseSize.width,
        });
        addEdge(graph, positivePrompt, 'value', layoutPrompt, 'prompt');

        return layoutPrompt;
      },
    }
  );
  const { components, graph, output, vaeModel } = pipeline;

  addMetadata(graph, output, settings, model, 'moss_image_txt2img', projectSettings, {
    moss_image_uncond_skip_layers: [...settings.mossImageUncondSkipLayers],
    qwen3_5_encoder: components.qwen35EncoderModel ?? undefined,
    vae: vaeModel ?? undefined,
    ...(shouldUsePidDecode(settings, model.base) ? getPidMetadata(settings) : {}),
  });
  const metadata = Object.values(graph.nodes).find((node) => node.type === 'core_metadata');
  if (metadata && layoutPrompt) {
    addEdge(graph, layoutPrompt, 'value', metadata, 'moss_image_layout_prompt');
  }

  return { ...pipeline, layoutPrompt };
};

const buildFlux3Graph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): FamilyGraph => {
  const isDiffusers = model.format === 'diffusers';
  const vaeModel = getCompatibleVae(settings, model);
  const qwen3VlEncoderModel = getCompatibleComponent(settings.qwen3VLEncoderModel, isQwen3Vl4bEncoder);

  if (!isDiffusers) {
    requireComponent(vaeModel, 'FLUX.3 VAE');
    requireComponent(qwen3VlEncoderModel, 'Qwen3-VL Encoder');
  }

  const graph: BackendGraphContract = { edges: [], id: createId('flux3_graph'), nodes: {} };
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  // Klein Base takes CFG with a negative prompt; Klein (the flow map) samples at CFG 1 and takes neither.
  const useCfg = settings.cfgScale > 1;
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model,
    qwen3_vl_encoder_model: qwen3VlEncoderModel ?? undefined,
    type: 'flux3_model_loader',
    vae_model: vaeModel ?? undefined,
  });
  // FLUX.3 LoRAs change the transformer only. Klein and Klein Base share it, so either variant's LoRAs apply.
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'flux3_lora_collection_loader', modelLoader, [
        'transformer',
      ])
    : modelLoader;
  // Canvas regions become a box table at run time; the prompt upsampler takes this place when on.
  const upsamplerModel = settings.flux3PromptUpsamplerEnabled
    ? requireComponent(
        getCompatibleComponent(settings.flux3PromptUpsamplerModel, isVisionTextLlm),
        'FLUX.3 prompt upsampler model'
      )
    : null;
  const layoutPrompt: BackendInvocationContract = upsamplerModel
    ? addNode(graph, {
        id: 'prompt_upsampler',
        type: 'flux3_prompt_upsampler',
        upsampler_model: toModelIdentifier(upsamplerModel),
      })
    : addNode(graph, { id: 'layout_prompt', type: 'flux3_layout_prompt' });
  const posCond = addNode(graph, { id: 'pos_cond', type: 'flux3_text_encoder' });
  const negCond = useCfg ? addNode(graph, { id: 'neg_cond', type: 'flux3_text_encoder' }) : null;
  const denoiseSize = getDenoiseSize(settings, model);
  const denoise = addNode(graph, {
    cfg_scale: settings.cfgScale,
    denoising_end: 1,
    denoising_start: 0,
    height: denoiseSize.height,
    id: 'denoise_latents',
    steps: settings.steps,
    type: 'flux3_denoise',
    width: denoiseSize.width,
  });
  // FLUX.3 decodes with the FLUX.2 VAE, so it reuses FLUX.2's decode node and PiD decoders.
  const output = addDecodeOutput({
    denoise,
    graph,
    l2iType: 'flux2_vae_decode',
    model,
    outputIsIntermediate,
    positivePrompt,
    seed,
    settings,
    vaeSource: modelLoader,
  });

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, modelLoader, 'qwen3_vl_encoder', posCond, 'qwen3_vl_encoder');
  addEdge(graph, modelLoader, 'vae', denoise, 'vae');
  addEdge(graph, positivePrompt, 'value', layoutPrompt, 'prompt');
  addEdge(graph, layoutPrompt, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');

  if (negCond) {
    addEdge(graph, modelLoader, 'qwen3_vl_encoder', negCond, 'qwen3_vl_encoder');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
    addEdge(graph, negCond, 'conditioning', denoise, 'negative_conditioning');
  }

  addEdge(graph, seed, 'value', denoise, 'seed');
  const referenceCollect = addFluxKontextReferenceImages(graph, settings, denoise, 'flux2_reference_image');
  if (upsamplerModel) {
    // The upsampler derives edit targets from the images it receives; one collect feeds it and the denoiser.
    if (referenceCollect) {
      addEdge(graph, referenceCollect, 'collection', layoutPrompt, 'reference_images');
    }
  } else if (referenceCollect) {
    // With reference images FLUX.3 edits them, and its boxes mark where new content goes in the output.
    layoutPrompt.edit_targets = true;
  }
  const metadata = addMetadata(graph, output, settings, model, 'flux3_txt2img', projectSettings, {
    // FLUX.3 offers no scheduler: its denoise node always runs the reference Euler sampler.
    scheduler: 'euler',
    ...(upsamplerModel ? { flux3_prompt_upsampler: toModelIdentifier(upsamplerModel) } : {}),
    qwen3_vl_encoder: qwen3VlEncoderModel ?? undefined,
    vae: vaeModel ?? undefined,
    ...(shouldUsePidDecode(settings, model.base) ? getPidMetadata(settings) : {}),
  });
  if (upsamplerModel) {
    // Metadata records the caption encoded, not the prompt typed (recall restores `positive_prompt`).
    addEdge(graph, layoutPrompt, 'value', metadata, 'flux3_caption');
  }
  addReferenceImageMetadata(graph, settings);

  return {
    conditioning: null,
    denoise,
    denoiseSize: { height: denoise.height, width: denoise.width },
    graph,
    latents: { encodeType: FLUX2_ENCODE_TYPE, vae: { field: 'vae', node: modelLoader } },
    layoutPrompt,
  };
};

const buildIdeogram4Graph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): FamilyGraph => {
  const isDiffusers = model.format === 'diffusers';
  const unconditionalModel = settings.ideogram4UnconditionalModel;
  const qwen3VlEncoderModel = getCompatibleComponent(settings.qwen3VLEncoderModel, isQwen3Vl8bEncoder);
  const vaeModel = getCompatibleVae(settings, model);

  if (!isDiffusers) {
    requireComponent(unconditionalModel, 'Ideogram 4 unconditional transformer');
    requireComponent(qwen3VlEncoderModel, 'Qwen3-VL Encoder');
    requireComponent(vaeModel, 'Ideogram 4 VAE');
  }

  const graph: BackendGraphContract = { edges: [], id: createId('ideogram4_graph'), nodes: {} };
  // Ideogram 4 is positive-only: there is no negative conditioning input on its denoise node.
  const { positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model,
    qwen3_vl_encoder_model: qwen3VlEncoderModel ?? undefined,
    type: 'ideogram4_model_loader',
    unconditional_model: isDiffusers ? undefined : (unconditionalModel ?? undefined),
    vae_model: vaeModel ?? undefined,
  });
  // LoRAs ride on 'Transformer' only; the denoise node applies them to both branches.
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'ideogram4_lora_collection_loader', modelLoader, [
        'transformer',
      ])
    : modelLoader;
  const captionBuilder = addNode(graph, {
    color_palette: settings.ideogram4ColorPalette,
    id: 'caption_builder',
    type: 'ideogram4_caption_builder',
  });
  const posCond = addNode(graph, { id: 'pos_cond', type: 'ideogram4_text_encoder' });
  const denoise = addNode(graph, {
    height: settings.height,
    id: 'denoise_latents',
    // Omit unset overrides; null would replace preset defaults.
    ...(settings.ideogram4Steps === null ? {} : { steps: settings.ideogram4Steps }),
    ...(settings.ideogram4GuidanceScale === null ? {} : { guidance_scale: settings.ideogram4GuidanceScale }),
    ...(settings.ideogram4Mu === null ? {} : { mu: settings.ideogram4Mu }),
    sampler_preset: settings.ideogram4SamplerPreset,
    type: 'ideogram4_denoise',
    width: settings.width,
  });
  const output = addImageOutputNode(graph, 'ideogram4_l2i', outputIsIntermediate);

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  if (!isDiffusers) {
    // Wire the second branch only for standalone models; bundles already contain both.
    addEdge(graph, modelLoader, 'unconditional_transformer', denoise, 'unconditional_transformer');
  }
  addEdge(graph, modelLoader, 'qwen3_encoder', posCond, 'qwen3_encoder');
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, positivePrompt, 'value', captionBuilder, 'prompt');
  addEdge(graph, captionBuilder, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');
  addEdge(graph, seed, 'value', denoise, 'seed');
  addEdge(graph, denoise, 'latents', output, 'latents');
  const metadata = addMetadata(graph, output, settings, model, 'ideogram4_txt2img', projectSettings, {
    ideogram4_color_palette: settings.ideogram4ColorPalette,
    ideogram4_guidance_scale: settings.ideogram4GuidanceScale ?? null,
    ideogram4_mu: settings.ideogram4Mu ?? null,
    ideogram4_sampler_preset: settings.ideogram4SamplerPreset,
    ideogram4_steps: settings.ideogram4Steps ?? null,
    ideogram4_unconditional_model: isDiffusers ? undefined : (unconditionalModel ?? undefined),
    qwen3_vl_encoder: qwen3VlEncoderModel ?? undefined,
    vae: vaeModel ?? undefined,
  });
  // Record the caption the encoder saw; the positive prompt alone does not show regions or palette.
  addEdge(graph, captionBuilder, 'value', metadata, 'ideogram4_caption');

  return {
    conditioning: null,
    denoise,
    denoiseSize: { height: denoise.height, width: denoise.width },
    graph,
    latents: null,
    layoutPrompt: captionBuilder,
  };
};

const buildWanGraph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings
): FamilyGraph => {
  const sourceModel = getDiffusersSource(settings, model);
  const vaeModel = getCompatibleVae(settings, model);
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
    vae: vaeModel ?? undefined,
    wan_component_source: sourceModel,
    wan_guidance_scale_low_noise: settings.wanGuidanceScaleLowNoise ?? null,
    wan_t5_encoder_model: wanT5EncoderModel ?? undefined,
    wan_transformer_low_noise: settings.wanLowNoiseModel ?? undefined,
  });

  return {
    conditioning: null,
    denoise,
    denoiseSize: { height: denoise.height, width: denoise.width },
    graph,
    latents: { encodeType: 'wan_i2l', vae: { field: 'vae', node: modelLoader } },
    layoutPrompt: null,
  };
};

export type ExternalCanvasInputs = { mode: 'txt2img' } | { mode: 'img2img'; initImageName: string };

/** The Canvas keep-mask matches no provider's edit-mask contract, so inpaint is not compiled. */
export const buildExternalGraph = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: GenerationProjectSettings,
  canvas: ExternalCanvasInputs | null = null
): BackendGraphContract => {
  if (model.type !== 'external_image_generator') {
    throw new Error('Selected model is not an external image generator.');
  }

  const mode = canvas?.mode ?? 'txt2img';
  const modeReason = getExternalModeReason(model, mode);

  if (modeReason) {
    throw new Error(modeReason);
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
    image_size: resolveExternalImageSize(settings, model),
    is_intermediate: outputIsIntermediate,
    mode,
    model,
    num_images: 1,
    type: nodeType,
    use_cache: false,
    width: settings.width,
    ...getExternalProviderInvocationOptions(settings, model),
    ...(canvas?.mode === 'img2img' ? { init_image: { image_name: canvas.initImageName } } : {}),
  });

  addEdge(graph, positivePrompt, 'value', output, 'prompt');
  if (model.capabilities?.supports_negative_prompt === true) {
    addEdge(graph, negativePrompt, 'value', output, 'negative_prompt');
  }

  if (model.capabilities?.supports_seed === true) {
    addEdge(graph, seed, 'value', output, 'seed');
  }
  const referenceImages = getExternalReferenceImages(settings, model).map(toImageField);

  if (referenceImages.length > 0) {
    (output as Record<string, unknown>).reference_images = referenceImages;
  }
  addMetadata(graph, output, settings, model, null, projectSettings, {
    scheduler: undefined,
    ...getExternalProviderMetadata(settings, model),
  });
  addReferenceImageMetadata(graph, settings);

  return graph;
};

export const GRAPH_BUILDERS = {
  'sd-1': buildSDGraph,
  'sd-2': buildSDGraph,
  sdxl: buildSDGraph,
  'sd-3': buildSD3Graph,
  flux: buildFluxGraph,
  flux2: buildFlux2Graph,
  cogview4: buildCogView4Graph,
  'ernie-image': buildErnieImageGraph,
  'qwen-image': buildQwenImageGraph,
  'z-image': buildZImageGraph,
  'ideogram-4': buildIdeogram4Graph,
  'krea-2': buildKrea2Graph,
  'qwen-image-2-1': buildQwenImage21Graph,
  flux3: buildFlux3Graph,
  anima: buildAnimaGraph,
  'moss-image': buildMossImageGraph,
  wan: buildWanGraph,
} satisfies Record<SupportedGenerateBase, GenerateGraphBuilder>;

/** The families Upscale's detail pass can run: Generate's pipeline builders, called without Generate's extras. */
const IMAGE_TO_IMAGE_PIPELINES = {
  anima: buildAnimaPipeline,
  flux: buildFluxPipeline,
  flux2: buildFlux2Pipeline,
  'krea-2': buildKrea2Pipeline,
  'moss-image': buildMossImagePipeline,
  'qwen-image': buildQwenImagePipeline,
  'sd-1': buildSdPipeline,
  sdxl: buildSdPipeline,
  'z-image': buildZImagePipeline,
} satisfies Partial<Record<SupportedGenerateBase, (input: FamilyPipelineInput) => FamilyPipeline>>;

export type ImageToImagePipelineBase = keyof typeof IMAGE_TO_IMAGE_PIPELINES;

/** A family's image pipeline as another feature asks for it; throws for a family that has none. */
export const buildImageToImagePipeline = (input: FamilyPipelineInput): FamilyPipeline => {
  const { base } = input.model;

  if (!Object.hasOwn(IMAGE_TO_IMAGE_PIPELINES, base)) {
    throw new Error(`${input.model.name} has no image-to-image pipeline.`);
  }

  return IMAGE_TO_IMAGE_PIPELINES[base as ImageToImagePipelineBase](input);
};

/** The builder for a base, or null. Own keys only: a base from the API must not reach `Object.prototype` members. */
export const getFamilyGraphBuilder = (base: string): GenerateGraphBuilder | null =>
  Object.hasOwn(GRAPH_BUILDERS, base) ? GRAPH_BUILDERS[base as SupportedGenerateBase] : null;

export const compileGenerateGraph = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  destination: ResultDestination,
  projectSettings: GenerationProjectSettings,
  randDevice = projectSettings.useCpuNoise ? 'cpu' : 'cuda'
): CompiledGenerateGraph => {
  const validationReasons = getGenerationValidationReasons(model, settings);

  if (validationReasons.length > 0) {
    throw new Error(validationReasons[0]);
  }

  const outputIsIntermediate = destination === 'canvas';
  const runtimeProjectSettings = { ...projectSettings, randDevice };
  const backendGraph = (() => {
    if (model.type === 'external_image_generator') {
      return buildExternalGraph(settings, model, outputIsIntermediate, runtimeProjectSettings);
    }

    const builder = getFamilyGraphBuilder(model.base);

    if (!builder) {
      const dimensionGrid = getGenerationDimensions(model).grid;

      throw new Error(`${model.name} dimensions must be multiples of ${dimensionGrid}.`);
    }

    const { graph } = builder(settings, model, outputIsIntermediate, runtimeProjectSettings);
    // External providers return finished images; the filters apply to every locally decoded family.
    addOutputPostprocessors(graph, runtimeProjectSettings, outputIsIntermediate);

    return graph;
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
  settings.seedMode === 'random' ? Math.floor(Math.random() * SEED_MAX) : settings.seed;
