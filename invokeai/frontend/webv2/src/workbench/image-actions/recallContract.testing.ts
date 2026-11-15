/** Read what a compiled graph records, for the recall contracts. */

import type { CompiledGenerateGraph, GenerateSettings } from '@features/generation/contracts';

/** The declared keys of a generated schema type, without its index signature. */
export type KnownKeys<T> = keyof {
  [K in keyof T as string extends K ? never : number extends K ? never : K]: T[K];
};

const RUNTIME_PREFIX = '<runtime:';

/** A value a node computes only when the graph runs, so the record cannot be known when it compiles. */
export const isRuntimeValue = (value: unknown): value is string =>
  typeof value === 'string' && value.startsWith(RUNTIME_PREFIX);

/** The type of the node that computes a runtime value; null for a value the record carries as compiled. */
export const runtimeSourceOf = (value: unknown): string | null =>
  isRuntimeValue(value) ? value.slice(RUNTIME_PREFIX.length, -1) : null;

/**
 * The record a compiled graph stores: the metadata node's serialized fields without its node envelope, plus what its
 * edges carry. The queue fills the prompts and the seed from the batch; any other source is named as a runtime value.
 * Serializing drops fields a builder set to `undefined`, as the backend never receives them.
 */
export const recordOf = (
  compiled: Pick<
    CompiledGenerateGraph,
    'backendGraph' | 'negativePromptNodeId' | 'positivePromptNodeId' | 'seedNodeId'
  >,
  batch: Pick<GenerateSettings, 'negativePrompt' | 'positivePrompt' | 'seed'>
): Record<string, unknown> => {
  const { backendGraph } = compiled;
  const node = Object.values(backendGraph.nodes).find((candidate) => candidate.type === 'core_metadata');

  if (!node) {
    throw new Error(`${backendGraph.id} records no metadata.`);
  }

  const {
    id: _id,
    is_intermediate: _isIntermediate,
    type: _type,
    use_cache: _useCache,
    ...record
  } = JSON.parse(JSON.stringify(node)) as Record<string, unknown>;
  const batchValues: Record<string, unknown> = {
    [compiled.negativePromptNodeId]: batch.negativePrompt,
    [compiled.positivePromptNodeId]: batch.positivePrompt,
    [compiled.seedNodeId]: batch.seed,
  };

  for (const { destination, source } of backendGraph.edges) {
    if (destination.node_id === node.id) {
      record[destination.field] = Object.hasOwn(batchValues, source.node_id)
        ? batchValues[source.node_id]
        : `${RUNTIME_PREFIX}${backendGraph.nodes[source.node_id]!.type}>`;
    }
  }

  return record;
};
