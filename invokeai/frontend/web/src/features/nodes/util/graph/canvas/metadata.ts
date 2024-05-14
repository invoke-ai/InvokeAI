import type { JSONObject } from 'common/types';
import type { ModelIdentifierField } from 'features/nodes/types/common';
import { METADATA } from 'features/nodes/util/graph/constants';
import type { AnyModelConfig, NonNullableGraph, S } from 'services/api/types';

export const addCoreMetadataNode = (
  graph: NonNullableGraph,
  metadata: Partial<S['CoreMetadataInvocation']>,
  nodeId: string
): void => {
  graph.nodes[METADATA] = {
    id: METADATA,
    type: 'core_metadata',
    ...metadata,
  };

  graph.edges.push({
    source: {
      node_id: METADATA,
      field: 'metadata',
    },
    destination: {
      node_id: nodeId,
      field: 'metadata',
    },
  });

  return;
};

export const upsertMetadata = (
  graph: NonNullableGraph,
  metadata: Partial<S['CoreMetadataInvocation']> | JSONObject
): void => {
  const metadataNode = graph.nodes[METADATA] as S['CoreMetadataInvocation'] | undefined;

  if (!metadataNode) {
    return;
  }

  Object.assign(metadataNode, metadata);
};

export const removeMetadata = (graph: NonNullableGraph, key: keyof S['CoreMetadataInvocation']): void => {
  const metadataNode = graph.nodes[METADATA] as S['CoreMetadataInvocation'] | undefined;

  if (!metadataNode) {
    return;
  }

  delete metadataNode[key];
};

export const getHasMetadata = (graph: NonNullableGraph): boolean => {
  const metadataNode = graph.nodes[METADATA] as S['CoreMetadataInvocation'] | undefined;

  return Boolean(metadataNode);
};

export const getModelMetadataField = ({ key, hash, name, base, type }: AnyModelConfig): ModelIdentifierField => ({
  key,
  hash,
  name,
  base,
  type,
});
