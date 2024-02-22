import type { JSONObject } from 'common/types';
import type { CoreMetadataInvocation, NonNullableGraph } from 'services/api/types';

import { METADATA } from './constants';

export const addCoreMetadataNode = (
  graph: NonNullableGraph,
  metadata: Partial<CoreMetadataInvocation>,
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
  metadata: Partial<CoreMetadataInvocation> | JSONObject
): void => {
  const metadataNode = graph.nodes[METADATA] as CoreMetadataInvocation | undefined;

  if (!metadataNode) {
    return;
  }

  Object.assign(metadataNode, metadata);
};

export const removeMetadata = (graph: NonNullableGraph, key: keyof CoreMetadataInvocation): void => {
  const metadataNode = graph.nodes[METADATA] as CoreMetadataInvocation | undefined;

  if (!metadataNode) {
    return;
  }

  delete metadataNode[key];
};

export const getHasMetadata = (graph: NonNullableGraph): boolean => {
  const metadataNode = graph.nodes[METADATA] as CoreMetadataInvocation | undefined;

  return Boolean(metadataNode);
};

export const setMetadataReceivingNode = (graph: NonNullableGraph, nodeId: string) => {
  graph.edges = graph.edges.filter((edge) => edge.source.node_id !== METADATA);

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
};
