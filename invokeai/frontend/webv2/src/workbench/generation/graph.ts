import type {
  BackendGraphContract,
  BackendGraphEdgeContract,
  BackendInvocationContract,
  ProjectSettings,
  ResultDestination,
} from '@workbench/types';

import type { CompiledGenerateGraph, GenerateSettings, MainModelConfig, SupportedGenerateBase } from './types';

import { clampDimension, deriveAspectRatioId, getOptimalDimension, SEED_MAX } from './settings';

const now = (): string => new Date().toISOString();

const createId = (prefix: string): string =>
  `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;

const getNumber = (value: number | null | undefined, fallback: number): number =>
  Number.isFinite(value) && value !== null && value !== undefined ? value : fallback;

export const isSupportedGenerateModel = <T extends { base: string; type: string }>(
  model: T
): model is T & { base: SupportedGenerateBase; type: 'main' } =>
  model.type === 'main' && (model.base === 'sdxl' || model.base === 'sd-1' || model.base === 'sd-2');

export const getDefaultGenerateSettings = (model?: MainModelConfig): GenerateSettings => {
  const defaults = model?.default_settings;
  const optimalDimension = getOptimalDimension(model?.base ?? 'sdxl');
  const width = clampDimension(getNumber(defaults?.width, optimalDimension));
  const height = clampDimension(getNumber(defaults?.height, optimalDimension));

  return {
    aspectRatioId: deriveAspectRatioId(width, height),
    aspectRatioIsLocked: false,
    aspectRatioValue: height > 0 ? width / height : 1,
    batchCount: 1,
    cfgRescaleMultiplier: getNumber(defaults?.cfg_rescale_multiplier, 0),
    cfgScale: getNumber(defaults?.cfg_scale, 7),
    clipSkip: 0,
    height,
    modelKey: model?.key ?? '',
    negativePrompt: '',
    positivePrompt: '',
    scheduler: defaults?.scheduler ?? 'euler_a',
    seamlessXAxis: false,
    seamlessYAxis: false,
    seed: Math.floor(Math.random() * SEED_MAX),
    shouldRandomizeSeed: true,
    steps: Math.max(1, Math.round(getNumber(defaults?.steps, 30))),
    vae: null,
    vaePrecision: defaults?.vae_precision === 'fp16' ? 'fp16' : 'fp32',
    width,
  };
};

export const getSettingsWithModelDefaults = (settings: GenerateSettings, model: MainModelConfig): GenerateSettings => {
  const modelDefaults = getDefaultGenerateSettings(model);

  return {
    ...settings,
    aspectRatioId: modelDefaults.aspectRatioId,
    aspectRatioIsLocked: false,
    aspectRatioValue: modelDefaults.aspectRatioValue,
    cfgRescaleMultiplier: modelDefaults.cfgRescaleMultiplier,
    cfgScale: modelDefaults.cfgScale,
    height: modelDefaults.height,
    modelKey: model.key,
    scheduler: modelDefaults.scheduler,
    steps: modelDefaults.steps,
    // VAE overrides are base-specific; the widget re-resolves the model's default VAE after a switch.
    vae: null,
    vaePrecision: modelDefaults.vaePrecision,
    width: modelDefaults.width,
  };
};

const addNode = <T extends BackendInvocationContract>(graph: BackendGraphContract, node: T): T => {
  graph.nodes[node.id] = {
    is_intermediate: true,
    use_cache: true,
    ...node,
  };

  return graph.nodes[node.id] as T;
};

const addEdge = (
  graph: BackendGraphContract,
  source: BackendInvocationContract,
  sourceField: string,
  destination: BackendInvocationContract,
  destinationField: string
): BackendGraphEdgeContract => {
  const edge = {
    destination: { field: destinationField, node_id: destination.id },
    source: { field: sourceField, node_id: source.id },
  };
  graph.edges.push(edge);

  return edge;
};

const addMetadata = (
  graph: BackendGraphContract,
  outputNode: BackendInvocationContract,
  settings: GenerateSettings,
  model: MainModelConfig,
  generationMode: string,
  projectSettings: Pick<ProjectSettings, 'useCpuNoise'>
) => {
  const metadata = addNode(graph, {
    cfg_scale: settings.cfgScale,
    cfg_rescale_multiplier: settings.cfgRescaleMultiplier,
    generation_mode: generationMode,
    height: settings.height,
    id: createId('core_metadata'),
    model,
    rand_device: projectSettings.useCpuNoise ? 'cpu' : 'cuda',
    scheduler: settings.scheduler,
    seamless_x: settings.seamlessXAxis,
    seamless_y: settings.seamlessYAxis,
    steps: settings.steps,
    type: 'core_metadata',
    vae: settings.vae ?? undefined,
    width: settings.width,
    ...(model.base === 'sdxl' ? {} : { clip_skip: settings.clipSkip }),
  });

  addEdge(graph, graph.nodes.seed, 'value', metadata, 'seed');
  addEdge(graph, graph.nodes.positive_prompt, 'value', metadata, 'positive_prompt');
  addEdge(graph, graph.nodes.negative_prompt, 'value', metadata, 'negative_prompt');
  addEdge(graph, metadata, 'metadata', outputNode, 'metadata');
};

const buildSDGraph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean,
  projectSettings: Pick<ProjectSettings, 'useCpuNoise'>
): BackendGraphContract => {
  const graph: BackendGraphContract = { edges: [], id: createId(`${model.base}_graph`), nodes: {} };
  const modelLoaderType = model.base === 'sdxl' ? 'sdxl_model_loader' : 'main_model_loader';
  const compelType = model.base === 'sdxl' ? 'sdxl_compel_prompt' : 'compel';

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
    scheduler: settings.scheduler,
    steps: settings.steps,
    type: 'denoise_latents',
  });
  const output = addNode(graph, {
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

  if (model.base === 'sdxl') {
    addEdge(graph, modelLoader, 'clip', posCond, 'clip');
    addEdge(graph, modelLoader, 'clip2', posCond, 'clip2');
    addEdge(graph, modelLoader, 'clip', negCond, 'clip');
    addEdge(graph, modelLoader, 'clip2', negCond, 'clip2');
    addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
    addEdge(graph, positivePrompt, 'value', posCond, 'style');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
    addEdge(graph, negativePrompt, 'value', negCond, 'style');
  } else {
    const clipSkip = addNode(graph, { id: 'clip_skip', skipped_layers: settings.clipSkip, type: 'clip_skip' });

    addEdge(graph, modelLoader, 'clip', clipSkip, 'clip');
    addEdge(graph, clipSkip, 'clip', posCond, 'clip');
    addEdge(graph, clipSkip, 'clip', negCond, 'clip');
    addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
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
    addEdge(graph, seamless, 'unet', denoise, 'unet');
  } else {
    addEdge(graph, modelLoader, 'unet', denoise, 'unet');
  }

  addEdge(graph, noise, 'noise', denoise, 'noise');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addEdge(graph, seamless ?? vaeLoader ?? modelLoader, 'vae', output, 'vae');
  addMetadata(graph, output, settings, model, model.base === 'sdxl' ? 'sdxl_txt2img' : 'txt2img', projectSettings);

  return graph;
};

const toGraphContract = (backendGraph: BackendGraphContract, label: string) => ({
  backendGraph,
  edges: backendGraph.edges.map((edge, index) => ({
    id: `edge-${index}`,
    sourceField: edge.source.field,
    sourceNodeId: edge.source.node_id,
    targetField: edge.destination.field,
    targetNodeId: edge.destination.node_id,
  })),
  id: backendGraph.id,
  label,
  nodes: Object.values(backendGraph.nodes).map(({ id, type, ...inputs }) => ({ id, inputs, type })),
  updatedAt: now(),
  version: 1 as const,
});

export const compileGenerateGraph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  destination: ResultDestination,
  projectSettings: Pick<ProjectSettings, 'useCpuNoise'>
): CompiledGenerateGraph => {
  if (!isSupportedGenerateModel(model)) {
    throw new Error(`${model.base} generation is not wired into the V7 Generate widget yet.`);
  }

  const backendGraph = buildSDGraph(settings, model, destination === 'canvas', projectSettings);

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
