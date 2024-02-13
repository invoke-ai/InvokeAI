import type { FieldInputTemplate, FieldOutputTemplate, FieldType } from 'features/nodes/types/field';
import type { AnyNode, InvocationNodeEdge, InvocationTemplate } from 'features/nodes/types/invocation';
import { isInvocationNode } from 'features/nodes/types/invocation';
import type { Connection, Edge, HandleType, Node } from 'reactflow';

import { getIsGraphAcyclic } from './getIsGraphAcyclic';
import { validateSourceAndTargetTypes } from './validateSourceAndTargetTypes';

const isValidConnection = (
  edges: Edge[],
  handleCurrentType: HandleType,
  handleCurrentFieldType: FieldType,
  node: Node,
  handle: FieldInputTemplate | FieldOutputTemplate
) => {
  let isValidConnection = true;
  if (handleCurrentType === 'source') {
    if (
      edges.find((edge) => {
        return edge.target === node.id && edge.targetHandle === handle.name;
      })
    ) {
      isValidConnection = false;
    }
  } else {
    if (
      edges.find((edge) => {
        return edge.source === node.id && edge.sourceHandle === handle.name;
      })
    ) {
      isValidConnection = false;
    }
  }

  if (!validateSourceAndTargetTypes(handleCurrentFieldType, handle.type)) {
    isValidConnection = false;
  }

  return isValidConnection;
};

export const findConnectionToValidHandle = (
  node: AnyNode,
  nodes: AnyNode[],
  edges: InvocationNodeEdge[],
  templates: Record<string, InvocationTemplate>,
  handleCurrentNodeId: string,
  handleCurrentName: string,
  handleCurrentType: HandleType,
  handleCurrentFieldType: FieldType
): Connection | null => {
  if (node.id === handleCurrentNodeId || !isInvocationNode(node)) {
    return null;
  }

  const template = templates[node.data.type];

  if (!template) {
    return null;
  }

  const handles = handleCurrentType === 'source' ? template.inputs : template.outputs;

  //Prioritize handles whos name matches the node we're coming from
  const handle = handles[handleCurrentName];

  if (handle) {
    const sourceID = handleCurrentType === 'source' ? handleCurrentNodeId : node.id;
    const targetID = handleCurrentType === 'source' ? node.id : handleCurrentNodeId;
    const sourceHandle = handleCurrentType === 'source' ? handleCurrentName : handle.name;
    const targetHandle = handleCurrentType === 'source' ? handle.name : handleCurrentName;

    const isGraphAcyclic = getIsGraphAcyclic(sourceID, targetID, nodes, edges);

    const valid = isValidConnection(edges, handleCurrentType, handleCurrentFieldType, node, handle);

    if (isGraphAcyclic && valid) {
      return {
        source: sourceID,
        sourceHandle: sourceHandle,
        target: targetID,
        targetHandle: targetHandle,
      };
    }
  }

  for (const handleName in handles) {
    const handle = handles[handleName];
    if (!handle) {
      continue;
    }

    const sourceID = handleCurrentType === 'source' ? handleCurrentNodeId : node.id;
    const targetID = handleCurrentType === 'source' ? node.id : handleCurrentNodeId;
    const sourceHandle = handleCurrentType === 'source' ? handleCurrentName : handle.name;
    const targetHandle = handleCurrentType === 'source' ? handle.name : handleCurrentName;

    const isGraphAcyclic = getIsGraphAcyclic(sourceID, targetID, nodes, edges);

    const valid = isValidConnection(edges, handleCurrentType, handleCurrentFieldType, node, handle);

    if (isGraphAcyclic && valid) {
      return {
        source: sourceID,
        sourceHandle: sourceHandle,
        target: targetID,
        targetHandle: targetHandle,
      };
    }
  }
  return null;
};
