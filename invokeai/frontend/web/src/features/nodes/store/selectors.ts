import type { NodesState } from 'features/nodes/store/types';
import type { FieldInputInstance } from 'features/nodes/types/field';
import type { InvocationNode, InvocationNodeData } from 'features/nodes/types/invocation';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { assert } from 'tsafe';

export const selectInvocationNode = (nodesSlice: NodesState, nodeId: string): InvocationNode => {
  const node = nodesSlice.nodes.find((node) => node.id === nodeId);
  assert(isInvocationNode(node), `Node ${nodeId} is not an invocation node`);
  return node;
};

export const selectInvocationNodeType = (nodesSlice: NodesState, nodeId: string): string => {
  const node = selectInvocationNode(nodesSlice, nodeId);
  return node.data.type;
};

export const selectNodeData = (nodesSlice: NodesState, nodeId: string): InvocationNodeData => {
  const node = selectInvocationNode(nodesSlice, nodeId);
  return node.data;
};

export const selectFieldInputInstance = (
  nodesSlice: NodesState,
  nodeId: string,
  fieldName: string
): FieldInputInstance | null => {
  const data = selectNodeData(nodesSlice, nodeId);
  return data?.inputs[fieldName] ?? null;
};

export const selectLastSelectedNode = (nodesSlice: NodesState) => {
  const selectedNodes = nodesSlice.nodes.filter((n) => n.selected);
  if (selectedNodes.length === 1) {
    return selectedNodes[0];
  }
  return null;
};
