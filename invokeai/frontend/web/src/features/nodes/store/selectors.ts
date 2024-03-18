import type { NodesState } from 'features/nodes/store/types';
import type { FieldInputInstance, FieldInputTemplate, FieldOutputTemplate } from 'features/nodes/types/field';
import type { InvocationNode, InvocationNodeData, InvocationTemplate } from 'features/nodes/types/invocation';
import { isInvocationNode } from 'features/nodes/types/invocation';

export const selectInvocationNode = (nodesSlice: NodesState, nodeId: string): InvocationNode | null => {
  const node = nodesSlice.nodes.find((node) => node.id === nodeId);
  if (!isInvocationNode(node)) {
    return null;
  }
  return node;
};

export const selectNodeData = (nodesSlice: NodesState, nodeId: string): InvocationNodeData | null => {
  return selectInvocationNode(nodesSlice, nodeId)?.data ?? null;
};

export const selectNodeTemplate = (nodesSlice: NodesState, nodeId: string): InvocationTemplate | null => {
  const node = selectInvocationNode(nodesSlice, nodeId);
  if (!node) {
    return null;
  }
  return nodesSlice.templates[node.data.type] ?? null;
};

export const selectFieldInputInstance = (
  nodesSlice: NodesState,
  nodeId: string,
  fieldName: string
): FieldInputInstance | null => {
  const data = selectNodeData(nodesSlice, nodeId);
  return data?.inputs[fieldName] ?? null;
};

export const selectFieldInputTemplate = (
  nodesSlice: NodesState,
  nodeId: string,
  fieldName: string
): FieldInputTemplate | null => {
  const template = selectNodeTemplate(nodesSlice, nodeId);
  return template?.inputs[fieldName] ?? null;
};

export const selectFieldOutputTemplate = (
  nodesSlice: NodesState,
  nodeId: string,
  fieldName: string
): FieldOutputTemplate | null => {
  const template = selectNodeTemplate(nodesSlice, nodeId);
  return template?.outputs[fieldName] ?? null;
};
