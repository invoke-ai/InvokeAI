import type { ModelIdentifierField } from 'features/nodes/types/common';
import { METADATA } from 'features/nodes/util/graph/constants';
import { isString, unset } from 'lodash-es';
import type {
  AnyInvocation,
  AnyInvocationIncMetadata,
  AnyModelConfig,
  CoreMetadataInvocation,
  S,
} from 'services/api/types';
import { assert } from 'tsafe';

import type { Graph } from './Graph';

const isCoreMetadata = (node: S['Graph']['nodes'][string]): node is CoreMetadataInvocation =>
  node.type === 'core_metadata';

export class MetadataUtil {
  static metadataNodeId = METADATA;

  static getNode(g: Graph): CoreMetadataInvocation {
    const node = g.getNode(this.metadataNodeId) as AnyInvocationIncMetadata;
    assert(isCoreMetadata(node));
    return node;
  }

  static add(g: Graph, metadata: Partial<CoreMetadataInvocation>): CoreMetadataInvocation {
    try {
      const node = g.getNode(this.metadataNodeId) as AnyInvocationIncMetadata;
      assert(isCoreMetadata(node));
      Object.assign(node, metadata);
      return node;
    } catch {
      const metadataNode: CoreMetadataInvocation = {
        id: this.metadataNodeId,
        type: 'core_metadata',
        ...metadata,
      };
      // @ts-expect-error `Graph` excludes `core_metadata` nodes due to its excessively wide typing
      return g.addNode(metadataNode);
    }
  }

  static remove(g: Graph, key: string): CoreMetadataInvocation;
  static remove(g: Graph, keys: string[]): CoreMetadataInvocation;
  static remove(g: Graph, keyOrKeys: string | string[]): CoreMetadataInvocation {
    const metadataNode = this.getNode(g);
    if (isString(keyOrKeys)) {
      unset(metadataNode, keyOrKeys);
    } else {
      for (const key of keyOrKeys) {
        unset(metadataNode, key);
      }
    }
    return metadataNode;
  }

  static setMetadataReceivingNode(g: Graph, node: AnyInvocation): void {
    // @ts-expect-error `Graph` excludes `core_metadata` nodes due to its excessively wide typing
    g.deleteEdgesFrom(this.getNode(g));
    // @ts-expect-error `Graph` excludes `core_metadata` nodes due to its excessively wide typing
    g.addEdge(this.getNode(g), 'metadata', node, 'metadata');
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
