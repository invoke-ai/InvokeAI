import { Connection, HandleType } from 'reactflow';
import { Node, Edge } from 'reactflow';
import { FieldType } from 'features/nodes/types/types';

import { validateSourceAndTargetTypes } from './validateSourceAndTargetTypes';
import { getIsGraphAcyclic } from 'features/nodes/hooks/useIsValidConnection';

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

  const handles =
    handleCurrentType == 'source' ? node.data.inputs : node.data.outputs;
  for (const handleName in handles) {
    const handle = handles[handleName];

    const sourceID =
      handleCurrentType == 'source' ? handleCurrentNodeId : node.id;
    const targetID =
      handleCurrentType == 'source' ? node.id : handleCurrentNodeId;
    const sourceHandle =
      handleCurrentType == 'source' ? handleCurrentName : handle.name;
    const targetHandle =
      handleCurrentType == 'source' ? handle.name : handleCurrentName;

    //TODO: Is there a better way to do this? Some of the logic is duplicated from useIsValidConnection.ts
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
        }) ||
        (handleCurrentFieldType !== 'CollectionItem' &&
          edges.find((edge) => {
            return (
              edge.target === handleCurrentNodeId &&
              edge.targetHandle === handleCurrentName
            );
          }))
      ) {
        isValidConnection = false;
      }
    }

    if (!validateSourceAndTargetTypes(handleCurrentFieldType, handle.type)) {
      isValidConnection = false;
    }

    const isGraphAcyclic = getIsGraphAcyclic(sourceID, targetID, nodes, edges);

    if (isGraphAcyclic && isValidConnection) {
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
