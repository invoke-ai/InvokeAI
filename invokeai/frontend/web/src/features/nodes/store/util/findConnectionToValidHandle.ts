import type { FieldInputInstance, FieldOutputInstance, FieldType } from 'features/nodes/types/field';
import type { Connection, Edge, HandleType, Node } from 'reactflow';

import { getIsGraphAcyclic } from './getIsGraphAcyclic';
import { validateSourceAndTargetTypes } from './validateSourceAndTargetTypes';

const isValidConnection = (
  edges: Edge[],
  handleCurrentType: HandleType,
  handleCurrentFieldType: FieldType,
  node: Node,
  handle: FieldInputInstance | FieldOutputInstance
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
  node: Node,
  nodes: Node[],
  edges: Edge[],
  handleCurrentNodeId: string,
  handleCurrentName: string,
  handleCurrentType: HandleType,
  handleCurrentFieldType: FieldType
): Connection | null => {
  if (node.id === handleCurrentNodeId) {
    return null;
  }

  const handles = handleCurrentType === 'source' ? node.data.inputs : node.data.outputs;

  //Prioritize handles whos name matches the node we're coming from
  if (handles[handleCurrentName]) {
    const handle = handles[handleCurrentName];

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
