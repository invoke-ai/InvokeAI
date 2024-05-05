import type { ModelIdentifierField } from 'features/nodes/types/common';
import { METADATA } from 'features/nodes/util/graph/constants';
import { isString, unset } from 'lodash-es';
import type { AnyModelConfig, Invocation } from 'services/api/types';

import type { Graph } from './Graph';

export class MetadataUtil {
  static metadataNodeId = METADATA;

  static getNode(graph: Graph): Invocation<'core_metadata'> {
    return graph.getNode(this.metadataNodeId, 'core_metadata');
  }

  static add(graph: Graph, metadata: Partial<Invocation<'core_metadata'>>): Invocation<'core_metadata'> {
    const metadataNode = graph.getNodeSafe(this.metadataNodeId, 'core_metadata');
    if (!metadataNode) {
      return graph.addNode({
        id: this.metadataNodeId,
        type: 'core_metadata',
        ...metadata,
      });
    } else {
      return graph.updateNode(this.metadataNodeId, 'core_metadata', metadata);
    }
  }

  static remove(graph: Graph, key: string): Invocation<'core_metadata'>;
  static remove(graph: Graph, keys: string[]): Invocation<'core_metadata'>;
  static remove(graph: Graph, keyOrKeys: string | string[]): Invocation<'core_metadata'> {
    const metadataNode = this.getNode(graph);
    if (isString(keyOrKeys)) {
      unset(metadataNode, keyOrKeys);
    } else {
      for (const key of keyOrKeys) {
        unset(metadataNode, key);
      }
    }
    return metadataNode;
  }

  static setMetadataReceivingNode(graph: Graph, nodeId: string): void {
    // We need to break the rules to update metadata - `addEdge` doesn't allow `core_metadata` as a node type
    graph._graph.edges = graph._graph.edges.filter((edge) => edge.source.node_id !== this.metadataNodeId);
    graph.addEdge(this.metadataNodeId, 'metadata', nodeId, 'metadata');
  }

  static getModelMetadataField({ key, hash, name, base, type }: AnyModelConfig): ModelIdentifierField {
    return {
      key,
      hash,
      name,
      base,
      type,
    };
  }
}
