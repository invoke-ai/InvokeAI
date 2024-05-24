import type { Templates } from 'features/nodes/store/types';
import type { FieldType } from 'features/nodes/types/field';
import type { AnyNode, InvocationNodeEdge } from 'features/nodes/types/invocation';

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
  edges: InvocationNodeEdge[],
  nodeId: string
): FieldType | null => {
  const firstEdgeToCollect = edges.find((edge) => edge.target === nodeId && edge.targetHandle === 'item');
  if (!firstEdgeToCollect?.sourceHandle) {
    return null;
  }
  const node = nodes.find((n) => n.id === firstEdgeToCollect.source);
  if (!node) {
    return null;
  }
  const template = templates[node.data.type];
  if (!template) {
    return null;
  }
  const fieldTemplate = template.outputs[firstEdgeToCollect.sourceHandle];
  if (!fieldTemplate) {
    return null;
  }
  return fieldTemplate.type;
};
