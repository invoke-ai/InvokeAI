import { NonNullableGraph } from 'features/nodes/types/types';
import { CoreMetadataInvocation } from 'services/api/types';
import { JsonObject } from 'type-fest';
import { METADATA, SAVE_IMAGE } from './constants';

export const addCoreMetadataNode = (
  graph: NonNullableGraph,
  metadata: Partial<CoreMetadataInvocation>
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
      node_id: SAVE_IMAGE,
      field: 'metadata',
    },
  });

  return;
};

export const upsertMetadata = (
  graph: NonNullableGraph,
  metadata: Partial<CoreMetadataInvocation> | JsonObject
): void => {
  const metadataNode = graph.nodes[METADATA] as
    | CoreMetadataInvocation
    | undefined;

  if (!metadataNode) {
    return;
  }

  Object.assign(metadataNode, metadata);
};

export const removeMetadata = (
  graph: NonNullableGraph,
  key: keyof CoreMetadataInvocation
): void => {
  const metadataNode = graph.nodes[METADATA] as
    | CoreMetadataInvocation
    | undefined;

  if (!metadataNode) {
    return;
  }

  delete metadataNode[key];
};

export const getHasMetadata = (graph: NonNullableGraph): boolean => {
  const metadataNode = graph.nodes[METADATA] as
    | CoreMetadataInvocation
    | undefined;

  return Boolean(metadataNode);
};
