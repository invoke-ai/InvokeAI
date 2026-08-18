import type { BackendGraphContract, GenerationModelCatalogItem, GraphContract, ResultDestination } from './contracts';
import type { GenerateWidgetValues } from './types';

import { getGenerationModelAvailabilityReasons, getGenerationValidationReasons } from './baseGenerationPolicies';
import { compileGenerateGraph } from './graph';
import { resolveGenerateWidgetValues } from './resolveGenerateWidgetValues';
import { normalizeGenerateSettings } from './settings';

/**
 * Preview-only compile: deterministic where the submit path is intentionally
 * random. Ids from `createId` (`prefix_<b36 time>_<6 rand>`) collapse to
 * `prefix`, `prefix_2`, … in insertion order so selection and layout survive
 * live recompiles, and the seed stays the stored literal (the dialog labels it
 * "regenerated each run" when randomization is on).
 */

const NO_SUPPORTED_MODEL = 'Generate needs a supported model before it can be invoked.';
// `getDefaultGenerateSettings` (the fallback `resolveGenerateWidgetValues` reaches for
// when `storedValues` doesn't parse — first run, cleared storage, corrupted state) sets
// `seed: Math.floor(Math.random() * SEED_MAX)`. A stored seed is always kept verbatim;
// this placeholder only stands in for the synthesized-default case, so the preview is
// deterministic even before Generate settings have ever been persisted.
const UNINITIALIZED_SEED_PLACEHOLDER = 0;

export interface GeneratePreviewInput {
  destination: ResultDestination;
  models: readonly GenerationModelCatalogItem[];
  storedValues: unknown;
  useCpuNoise: boolean;
}

export type GeneratePreviewResult =
  | { status: 'ready'; graph: GraphContract; settings: GenerateWidgetValues }
  | { status: 'invalid'; reasons: string[] };

const CREATED_ID_PATTERN = /^(.+)_[a-z0-9]{7,9}_[a-z0-9]{4,6}$/;

export const stabilizeBackendGraphIds = (graph: BackendGraphContract): BackendGraphContract => {
  const rename = new Map<string, string>();
  const used = new Set<string>();

  for (const id of Object.keys(graph.nodes)) {
    const match = CREATED_ID_PATTERN.exec(id);
    const base = match ? match[1] : id;
    let candidate = base;
    let suffix = 2;

    while (used.has(candidate)) {
      candidate = `${base}_${suffix}`;
      suffix += 1;
    }

    used.add(candidate);
    rename.set(id, candidate);
  }

  return {
    edges: graph.edges.map((edge) => ({
      destination: {
        field: edge.destination.field,
        node_id: rename.get(edge.destination.node_id) ?? edge.destination.node_id,
      },
      source: { field: edge.source.field, node_id: rename.get(edge.source.node_id) ?? edge.source.node_id },
    })),
    id: 'generate-preview',
    nodes: Object.fromEntries(
      Object.entries(graph.nodes).map(([id, node]) => {
        const nextId = rename.get(id) ?? id;

        return [nextId, { ...node, id: nextId }];
      })
    ),
  };
};

const toPreviewContract = (backendGraph: BackendGraphContract, label: string): GraphContract => ({
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
  updatedAt: '',
  version: 1,
});

export const compileGeneratePreviewGraph = (input: GeneratePreviewInput): GeneratePreviewResult => {
  const resolved = resolveGenerateWidgetValues({ models: input.models, storedValues: input.storedValues });

  if (!resolved) {
    return { reasons: [NO_SUPPORTED_MODEL], status: 'invalid' };
  }

  const storedSettingsParsed = normalizeGenerateSettings(input.storedValues) !== null;
  const settings = storedSettingsParsed
    ? resolved.values
    : { ...resolved.values, seed: UNINITIALIZED_SEED_PLACEHOLDER };
  const reasons = [
    ...getGenerationValidationReasons(settings.model, settings),
    ...getGenerationModelAvailabilityReasons(settings.model, settings, input.models),
  ];

  if (reasons.length > 0) {
    return { reasons, status: 'invalid' };
  }

  try {
    const compiled = compileGenerateGraph(settings, settings.model, input.destination, {
      useCpuNoise: input.useCpuNoise,
    });
    // The submit path leaves the seed node's `value` unset and injects the
    // resolved seed via queue batch data instead (see `promptBatch.ts`). The
    // preview has no batch step, so bake the literal, unresolved seed in here.
    const seedNode = compiled.backendGraph.nodes[compiled.seedNodeId];
    const backendGraphWithSeed: BackendGraphContract = seedNode
      ? {
          ...compiled.backendGraph,
          nodes: { ...compiled.backendGraph.nodes, [compiled.seedNodeId]: { ...seedNode, value: settings.seed } },
        }
      : compiled.backendGraph;
    const backendGraph = stabilizeBackendGraphIds(backendGraphWithSeed);

    return { graph: toPreviewContract(backendGraph, compiled.graph.label), settings, status: 'ready' };
  } catch (error) {
    return { reasons: [error instanceof Error ? error.message : String(error)], status: 'invalid' };
  }
};
