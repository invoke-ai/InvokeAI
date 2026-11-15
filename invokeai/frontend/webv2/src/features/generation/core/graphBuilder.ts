import type {
  BackendGraphContract,
  BackendGraphEdgeContract,
  BackendInvocationContract,
  GenerationProjectSettings,
} from '@features/generation/core/contracts';

import type { GenerateLora, GenerateModelConfig, GenerateSettings } from './types';

import { coerceSchedulerForGraph } from './baseGenerationPolicies';
import { isLoraCompatibleWithModel } from './settings';

const now = (): string => new Date().toISOString();

export const createId = (prefix: string): string =>
  `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;

export const addNode = <T extends BackendInvocationContract>(graph: BackendGraphContract, node: T): T => {
  graph.nodes[node.id] = {
    is_intermediate: true,
    use_cache: true,
    ...node,
  };

  return graph.nodes[node.id] as T;
};

export const addEdge = (
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

/** Rename every matching node ID and edge reference. */
export const renameNode = (graph: BackendGraphContract, oldId: string, newId: string): BackendInvocationContract => {
  const node = graph.nodes[oldId];
  if (!node) {
    throw new Error(`Could not find the "${oldId}" node to rename.`);
  }
  delete graph.nodes[oldId];
  node.id = newId;
  graph.nodes[newId] = node;
  for (const edge of graph.edges) {
    if (edge.source.node_id === oldId) {
      edge.source.node_id = newId;
    }
    if (edge.destination.node_id === oldId) {
      edge.destination.node_id = newId;
    }
  }
  return node;
};

/**
 * Makes `output` the graph's `canvas_output`, the node result hydration follows. The node that held the id is renamed
 * to `retiredId` and made intermediate, and its metadata edge moves to `output` so the saved record stays on the node
 * that emits the final image. Edges into `output` are the caller's; the retired node keeps its edges under its new id.
 */
export const replaceCanvasOutput = (
  graph: BackendGraphContract,
  retiredId: string,
  output: { type: string; [key: string]: unknown }
): { output: BackendInvocationContract; retired: BackendInvocationContract } => {
  const retired = renameNode(graph, 'canvas_output', retiredId);
  retired.is_intermediate = true;

  const next = addNode(graph, { ...output, id: 'canvas_output' });
  const metadataEdge = graph.edges.find(
    (edge) => edge.destination.node_id === retiredId && edge.destination.field === 'metadata'
  );
  if (metadataEdge) {
    metadataEdge.destination.node_id = next.id;
  }

  return { output: next, retired };
};

/**
 * Chains the project's optional output filters after the image the graph ends on. Each filter in turn takes over
 * `canvas_output`, so the last one is the durable result and the unfiltered image becomes intermediate.
 */
export const addOutputPostprocessors = (
  graph: BackendGraphContract,
  projectSettings: Pick<GenerationProjectSettings, 'useNsfwChecker' | 'useWatermark'>,
  outputIsIntermediate: boolean
): void => {
  const filters = [
    ...(projectSettings.useNsfwChecker ? ['img_nsfw'] : []),
    ...(projectSettings.useWatermark ? ['img_watermark'] : []),
  ];
  let retiredId = 'canvas_unfiltered_output';

  for (const type of filters) {
    const { output, retired } = replaceCanvasOutput(graph, retiredId, {
      is_intermediate: outputIsIntermediate,
      type,
      use_cache: false,
    });

    addEdge(graph, retired, 'image', output, 'image');
    retiredId = `canvas_${type}`;
  }
};

export const getActiveCompatibleLoras = (
  settings: Pick<GenerateSettings, 'loras'>,
  model: GenerateModelConfig
): GenerateLora[] => settings.loras.filter((lora) => lora.isEnabled && isLoraCompatibleWithModel(lora.model, model));

export const toModelIdentifier = (
  model: { base: string; key: string; name: string; type: string } & Record<string, unknown>
) => ({
  base: model.base,
  key: model.key,
  name: model.name,
  type: model.type,
  ...(typeof model.hash === 'string' ? { hash: model.hash } : {}),
  ...(typeof model.submodel_type === 'string' ? { submodel_type: model.submodel_type } : {}),
});

/**
 * The LoRAs a graph loads, as metadata records them. Read from the graph rather than the settings, so a record names
 * only concepts that reached inference; a family whose graph has no LoRA loader loads none.
 */
export const getLoadedLoras = (graph: BackendGraphContract) =>
  Object.values(graph.nodes)
    .filter((node) => node.type === 'lora_selector')
    .map((node) => ({ model: node.lora, weight: node.weight }));

/**
 * The component-model fields of the loader nodes a graph can hold, and the metadata key recall reads each from. A
 * loader field missing here reaches inference unrecorded, so a builder that adds one extends this table.
 * The main loaders' own `model` is the main model, which the record holds as `model`.
 */
export const RECORDED_COMPONENT_FIELDS: Readonly<Record<string, Readonly<Record<string, readonly string[]>>>> = {
  anima_model_loader: {
    qwen3_5_encoder_model: ['qwen3_5_encoder'],
    qwen3_encoder_model: ['qwen3_encoder'],
    vae_model: ['vae'],
  },
  ernie_image_model_loader: { text_encoder_model: ['mistral_encoder'], vae_model: ['vae'] },
  flux2_dev_model_loader: {
    mistral_encoder_model: ['mistral_encoder'],
    mistral_source_model: ['mistral_source'],
    vae_model: ['vae'],
  },
  flux2_klein_model_loader: {
    qwen3_encoder_model: ['qwen3_encoder'],
    qwen3_source_model: ['qwen3_source'],
    vae_model: ['vae'],
  },
  flux3_model_loader: { qwen3_vl_encoder_model: ['qwen3_vl_encoder'], vae_model: ['vae'] },
  flux3_prompt_upsampler: { upsampler_model: ['flux3_prompt_upsampler'] },
  flux_model_loader: {
    clip_embed_model: ['clip_embed_model'],
    t5_encoder_model: ['t5_encoder'],
    vae_model: ['vae'],
  },
  gemma2_encoder_loader: { gemma2_model: ['gemma2_encoder'] },
  ideogram4_model_loader: {
    qwen3_vl_encoder_model: ['qwen3_vl_encoder'],
    unconditional_model: ['ideogram4_unconditional_model'],
    vae_model: ['vae'],
  },
  krea2_model_loader: {
    qwen3_vl_encoder_model: ['qwen3_vl_encoder'],
    qwen3_vl_vision_tower_model: ['qwen3_vl_vision_tower'],
    vae_model: ['vae'],
  },
  moss_image_model_loader: { qwen3_5_encoder_model: ['qwen3_5_encoder'], vae_model: ['vae'] },
  pid_decoder_loader: { pid_decoder_model: ['pid_decoder'] },
  qwen_image_2_1_model_loader: {
    qwen3_vl_encoder_model: ['qwen3_vl_encoder'],
    qwen3_vl_vision_tower_model: ['qwen3_vl_vision_tower'],
    vae_model: ['vae'],
  },
  qwen_image_model_loader: {
    component_source: ['qwen_image_component_source'],
    qwen_vl_encoder_model: ['qwen_image_qwen_vl_encoder'],
    // Older recall reads the VAE of a Qwen-Image record from the alias.
    vae_model: ['vae', 'qwen_image_vae'],
  },
  sd3_model_loader: {
    clip_g_model: ['clip_g_model'],
    clip_l_model: ['clip_l_model'],
    t5_encoder_model: ['t5_encoder'],
    vae_model: ['vae'],
  },
  sdxl_refiner_model_loader: { model: ['refiner_model'] },
  vae_loader: { vae_model: ['vae'] },
  wan_model_loader: {
    component_source: ['wan_component_source'],
    transformer_low_noise_model: ['wan_transformer_low_noise'],
    vae_model: ['vae'],
    wan_t5_encoder_model: ['wan_t5_encoder_model'],
  },
  z_image_model_loader: {
    qwen3_encoder_model: ['qwen3_encoder'],
    qwen3_source_model: ['qwen3_source'],
    vae_model: ['vae'],
  },
};

/**
 * The component models a graph loads, under the metadata keys recall reads. Read from the graph rather than the
 * settings, so a record names only what reached a loader; a loader field without a value records nothing.
 */
export const getRecordedComponents = (graph: BackendGraphContract): Record<string, unknown> => {
  const components: Record<string, unknown> = {};

  for (const node of Object.values(graph.nodes)) {
    for (const [field, keys] of Object.entries(RECORDED_COMPONENT_FIELDS[node.type] ?? {})) {
      const value = node[field];

      if (value !== undefined && value !== null) {
        for (const key of keys) {
          components[key] = value;
        }
      }
    }
  }

  return components;
};

export const addMetadata = (
  graph: BackendGraphContract,
  outputNode: BackendInvocationContract,
  settings: GenerateSettings,
  model: GenerateModelConfig,
  generationMode: string | null,
  projectSettings: GenerationProjectSettings & { randDevice?: string },
  extras: Record<string, unknown> = {}
): BackendInvocationContract => {
  const loadedLoras = getLoadedLoras(graph);
  const scheduler = coerceSchedulerForGraph(model, settings.scheduler);
  const metadata = addNode(graph, {
    cfg_scale: settings.cfgScale,
    cfg_rescale_multiplier: settings.cfgRescaleMultiplier,
    ...(generationMode ? { generation_mode: generationMode } : {}),
    height: settings.height,
    id: createId('core_metadata'),
    model,
    rand_device: projectSettings.randDevice ?? (projectSettings.useCpuNoise ? 'cpu' : 'cuda'),
    scheduler,
    seamless_x: settings.seamlessXAxis,
    seamless_y: settings.seamlessYAxis,
    steps: settings.steps,
    type: 'core_metadata',
    width: settings.width,
    ...extras,
    ...getRecordedComponents(graph),
    ...(loadedLoras.length ? { loras: loadedLoras } : {}),
    ...(model.base === 'sdxl' ? {} : { clip_skip: settings.clipSkip }),
  });

  addEdge(graph, graph.nodes.seed, 'value', metadata, 'seed');
  addEdge(graph, graph.nodes.positive_prompt, 'value', metadata, 'positive_prompt');
  if (graph.nodes.negative_prompt) {
    addEdge(graph, graph.nodes.negative_prompt, 'value', metadata, 'negative_prompt');
  }
  addEdge(graph, metadata, 'metadata', outputNode, 'metadata');
  return metadata;
};

export const toGraphContract = (backendGraph: BackendGraphContract, label: string) => ({
  backendGraph,
  edges: backendGraph.edges.map((edge, index) => ({
    id: `edge-${index}`,
    sourceField: edge.source.field,
    sourceNodeId: edge.source.node_id,
    targetField: edge.destination.field,
    targetNodeId: edge.destination.node_id,
    ...(edge.type ? { type: edge.type } : {}),
  })),
  id: backendGraph.id,
  label,
  nodes: Object.values(backendGraph.nodes).map(({ id, type, ...inputs }) => ({ id, inputs, type })),
  updatedAt: now(),
  version: 1 as const,
});
