import type {
  BackendGraphContract,
  BackendGraphEdgeContract,
  BackendInvocationContract,
  ResultDestination,
} from '../types';
import type { CompiledGenerateGraph, GenerateSettings, MainModelConfig, SupportedGenerateBase } from './types';

const now = (): string => new Date().toISOString();

const createId = (prefix: string): string =>
  `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;

const getNumber = (value: number | null | undefined, fallback: number): number =>
  Number.isFinite(value) && value !== null && value !== undefined ? value : fallback;

const clampDimension = (value: number): number => Math.max(64, Math.round(value / 8) * 8);

export const isSupportedGenerateModel = (
  model: MainModelConfig
): model is MainModelConfig & { base: SupportedGenerateBase } =>
  model.type === 'main' && (model.base === 'sdxl' || model.base === 'sd-1' || model.base === 'sd-2');

export const getDefaultGenerateSettings = (model?: MainModelConfig): GenerateSettings => {
  const defaults = model?.default_settings;

  return {
    batchCount: 1,
    cfgRescaleMultiplier: getNumber(defaults?.cfg_rescale_multiplier, 0),
    cfgScale: getNumber(defaults?.cfg_scale, 7),
    height: clampDimension(getNumber(defaults?.height, 1024)),
    modelKey: model?.key ?? '',
    negativePrompt: '',
    positivePrompt: '',
    scheduler: defaults?.scheduler ?? 'euler_a',
    seed: Math.floor(Math.random() * 4_294_967_295),
    shouldRandomizeSeed: true,
    steps: Math.max(1, Math.round(getNumber(defaults?.steps, 30))),
    width: clampDimension(getNumber(defaults?.width, 1024)),
  };
};

export const getSettingsWithModelDefaults = (settings: GenerateSettings, model: MainModelConfig): GenerateSettings => {
  const modelDefaults = getDefaultGenerateSettings(model);

  return {
    ...settings,
    cfgRescaleMultiplier: modelDefaults.cfgRescaleMultiplier,
    cfgScale: modelDefaults.cfgScale,
    height: modelDefaults.height,
    modelKey: model.key,
    scheduler: modelDefaults.scheduler,
    steps: modelDefaults.steps,
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
  generationMode: string
) => {
  const metadata = addNode(graph, {
    cfg_scale: settings.cfgScale,
    cfg_rescale_multiplier: settings.cfgRescaleMultiplier,
    generation_mode: generationMode,
    height: settings.height,
    id: createId('core_metadata'),
    model,
    rand_device: 'cuda',
    scheduler: settings.scheduler,
    steps: settings.steps,
    type: 'core_metadata',
    width: settings.width,
  });

  addEdge(graph, graph.nodes.seed, 'value', metadata, 'seed');
  addEdge(graph, graph.nodes.positive_prompt, 'value', metadata, 'positive_prompt');
  addEdge(graph, graph.nodes.negative_prompt, 'value', metadata, 'negative_prompt');
  addEdge(graph, metadata, 'metadata', outputNode, 'metadata');
};

const buildSDGraph = (
  settings: GenerateSettings,
  model: MainModelConfig,
  outputIsIntermediate: boolean
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
    use_cpu: false,
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
    fp32: false,
    id: 'canvas_output',
    is_intermediate: outputIsIntermediate,
    type: 'l2i',
    use_cache: false,
  });

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
    const clipSkip = addNode(graph, { id: 'clip_skip', skipped_layers: 0, type: 'clip_skip' });

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
  addEdge(graph, modelLoader, 'unet', denoise, 'unet');
  addEdge(graph, noise, 'noise', denoise, 'noise');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addMetadata(graph, output, settings, model, model.base === 'sdxl' ? 'sdxl_txt2img' : 'txt2img');

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
  destination: ResultDestination
): CompiledGenerateGraph => {
  if (!isSupportedGenerateModel(model)) {
    throw new Error(`${model.base} generation is not wired into the V7 Generate widget yet.`);
  }

  const backendGraph = buildSDGraph(settings, model, destination === 'canvas');

  return {
    backendGraph,
    graph: toGraphContract(backendGraph, `${model.name} txt2img`),
    negativePromptNodeId: 'negative_prompt',
    positivePromptNodeId: 'positive_prompt',
    seedNodeId: 'seed',
  };
};

export const resolveGenerateSeed = (settings: GenerateSettings): number =>
  settings.shouldRandomizeSeed ? Math.floor(Math.random() * 4_294_967_295) : settings.seed;
