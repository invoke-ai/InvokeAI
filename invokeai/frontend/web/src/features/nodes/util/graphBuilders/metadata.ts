import { NonNullableGraph } from 'features/nodes/types/types';
import { map } from 'lodash-es';
import { MetadataInvocationAsCollection } from 'services/api/types';
import { JsonObject } from 'type-fest';
import {
  BATCH_METADATA,
  BATCH_METADATA_COLLECT,
  MERGE_METADATA,
  METADATA,
  METADATA_COLLECT,
  SAVE_IMAGE,
} from './constants';

export const addMainMetadataNodeToGraph = (
  graph: NonNullableGraph,
  metadata: JsonObject
): void => {
  graph.nodes[METADATA] = {
    id: METADATA,
    type: 'metadata',
    items: map(metadata, (value, label) => ({ label, value })),
  };

  graph.nodes[METADATA_COLLECT] = {
    id: METADATA_COLLECT,
    type: 'collect',
  };

  graph.nodes[MERGE_METADATA] = {
    id: MERGE_METADATA,
    type: 'merge_metadata_dict',
  };

  graph.edges.push({
    source: {
      node_id: METADATA,
      field: 'metadata_dict',
    },
    destination: {
      node_id: METADATA_COLLECT,
      field: 'item',
    },
  });

  graph.edges.push({
    source: {
      node_id: METADATA_COLLECT,
      field: 'collection',
    },
    destination: {
      node_id: MERGE_METADATA,
      field: 'collection',
    },
  });

  graph.edges.push({
    source: {
      node_id: MERGE_METADATA,
      field: 'metadata_dict',
    },
    destination: {
      node_id: SAVE_IMAGE,
      field: 'metadata',
    },
  });

  return;
};

export const addMainMetadata = (
  graph: NonNullableGraph,
  metadata: JsonObject
): void => {
  const metadataNode = graph.nodes[METADATA] as
    | MetadataInvocationAsCollection
    | undefined;

  if (!metadataNode) {
    return;
  }

  metadataNode.items.push(
    ...map(metadata, (value, label) => ({ label, value }))
  );
};

export const removeMetadataFromMainMetadataNode = (
  graph: NonNullableGraph,
  label: string
): void => {
  const metadataNode = graph.nodes[METADATA] as
    | MetadataInvocationAsCollection
    | undefined;

  if (!metadataNode) {
    return;
  }

  metadataNode.items = metadataNode.items.filter(
    (item) => item.label !== label
  );
};

export const addBatchMetadataNodeToGraph = (
  graph: NonNullableGraph,
  itemNodeIds: string[]
) => {
  graph.nodes[BATCH_METADATA] = {
    id: BATCH_METADATA,
    type: 'metadata',
  };
  graph.nodes[BATCH_METADATA_COLLECT] = {
    id: BATCH_METADATA_COLLECT,
    type: 'collect',
  };

  itemNodeIds.forEach((id) => {
    graph.edges.push({
      source: {
        node_id: id,
        field: 'item',
      },
      destination: {
        node_id: BATCH_METADATA_COLLECT,
        field: 'item',
      },
    });
  });

  graph.edges.push({
    source: {
      node_id: BATCH_METADATA_COLLECT,
      field: 'collection',
    },
    destination: {
      node_id: BATCH_METADATA,
      field: 'items',
    },
  });

  graph.edges.push({
    source: {
      node_id: BATCH_METADATA,
      field: 'metadata_dict',
    },
    destination: {
      node_id: METADATA_COLLECT,
      field: 'item',
    },
  });
};
