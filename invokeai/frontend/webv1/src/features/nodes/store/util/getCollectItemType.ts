import type { Templates } from 'features/nodes/store/types';
import type { FieldType } from 'features/nodes/types/field';
import type { AnyEdge, AnyNode } from 'features/nodes/types/invocation';

const toItemType = (fieldType: FieldType): FieldType | null => {
  if (fieldType.name === 'CollectionField') {
    return null;
  }
  if (fieldType.cardinality === 'COLLECTION' || fieldType.cardinality === 'SINGLE_OR_COLLECTION') {
    return { ...fieldType, cardinality: 'SINGLE' };
  }
  return fieldType;
};

/**
 * Given a collect node, return the type of the items it collects. The graph is traversed to find the first node and
 * field connected to the collector's `item` input. The field type of that field is returned, else null if there is no
 * input field.
 * @param templates The current invocation templates
 * @param nodes The current nodes
 * @param edges The current edges
 * @param nodeId The collect node's id
 * @returns The type of the items the collect node collects, or null if there is no input field
 */
export const getCollectItemType = (
  templates: Templates,
  nodes: AnyNode[],
  edges: AnyEdge[],
  nodeId: string
): FieldType | null => {
  const getCollectItemTypeInternal = (currentNodeId: string, visited: Set<string>): FieldType | null => {
    if (visited.has(currentNodeId)) {
      return null;
    }
    visited.add(currentNodeId);

    const firstItemEdgeToCollect = edges.find((edge) => edge.target === currentNodeId && edge.targetHandle === 'item');
    if (firstItemEdgeToCollect?.sourceHandle) {
      const node = nodes.find((n) => n.id === firstItemEdgeToCollect.source);
      if (!node) {
        return null;
      }
      const template = templates[node.data.type];
      if (!template) {
        return null;
      }
      const fieldTemplate = template.outputs[firstItemEdgeToCollect.sourceHandle];
      if (!fieldTemplate) {
        return null;
      }
      return toItemType(fieldTemplate.type);
    }

    const firstCollectionEdgeToCollect = edges.find(
      (edge) => edge.target === currentNodeId && edge.targetHandle === 'collection'
    );
    if (!firstCollectionEdgeToCollect?.sourceHandle) {
      return null;
    }
    const sourceNode = nodes.find((n) => n.id === firstCollectionEdgeToCollect.source);
    if (!sourceNode) {
      return null;
    }
    if (sourceNode.data.type === 'collect' && firstCollectionEdgeToCollect.sourceHandle === 'collection') {
      return getCollectItemTypeInternal(sourceNode.id, visited);
    }
    const sourceTemplate = templates[sourceNode.data.type];
    if (!sourceTemplate) {
      return null;
    }
    const sourceFieldTemplate = sourceTemplate.outputs[firstCollectionEdgeToCollect.sourceHandle];
    if (!sourceFieldTemplate) {
      return null;
    }
    return toItemType(sourceFieldTemplate.type);
  };

  const itemType = getCollectItemTypeInternal(nodeId, new Set());
  if (!itemType) {
    return null;
  }
  return itemType;
};
