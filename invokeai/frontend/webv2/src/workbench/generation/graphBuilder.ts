import type {
  BackendGraphContract,
  BackendGraphEdgeContract,
  BackendInvocationContract,
  ProjectSettings,
} from '@workbench/types';

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

export const getActiveCompatibleLoras = (settings: GenerateSettings, model: GenerateModelConfig): GenerateLora[] =>
  settings.loras.filter((lora) => lora.isEnabled && isLoraCompatibleWithModel(lora.model, model));

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

export const addMetadata = (
  graph: BackendGraphContract,
  outputNode: BackendInvocationContract,
  settings: GenerateSettings,
  model: GenerateModelConfig,
  generationMode: string | null,
  projectSettings: Pick<ProjectSettings, 'useCpuNoise'>,
  extras: Record<string, unknown> = {}
) => {
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const scheduler = coerceSchedulerForGraph(model, settings.scheduler);
  const metadata = addNode(graph, {
    cfg_scale: settings.cfgScale,
    cfg_rescale_multiplier: settings.cfgRescaleMultiplier,
    ...(generationMode ? { generation_mode: generationMode } : {}),
    height: settings.height,
    id: createId('core_metadata'),
    model,
    rand_device: projectSettings.useCpuNoise ? 'cpu' : 'cuda',
    scheduler,
    seamless_x: settings.seamlessXAxis,
    seamless_y: settings.seamlessYAxis,
    steps: settings.steps,
    type: 'core_metadata',
    vae: settings.vae ?? undefined,
    width: settings.width,
    ...extras,
    ...(activeLoras.length
      ? {
          loras: activeLoras.map((lora) => ({
            model: toModelIdentifier(lora.model),
            weight: lora.weight,
          })),
        }
      : {}),
    ...(model.base === 'sdxl' ? {} : { clip_skip: settings.clipSkip }),
  });

  addEdge(graph, graph.nodes.seed, 'value', metadata, 'seed');
  addEdge(graph, graph.nodes.positive_prompt, 'value', metadata, 'positive_prompt');
  if (graph.nodes.negative_prompt) {
    addEdge(graph, graph.nodes.negative_prompt, 'value', metadata, 'negative_prompt');
  }
  addEdge(graph, metadata, 'metadata', outputNode, 'metadata');
};

export const toGraphContract = (backendGraph: BackendGraphContract, label: string) => ({
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
