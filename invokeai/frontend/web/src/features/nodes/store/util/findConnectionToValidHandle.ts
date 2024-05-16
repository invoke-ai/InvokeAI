import type { PendingConnection, Templates } from 'features/nodes/store/types';
import { getCollectItemType } from 'features/nodes/store/util/makeIsConnectionValidSelector';
import type { AnyNode, InvocationNode, InvocationNodeEdge, InvocationTemplate } from 'features/nodes/types/invocation';
import { differenceWith, isEqual, map } from 'lodash-es';
import type { Connection } from 'reactflow';
import { assert } from 'tsafe';

import { getIsGraphAcyclic } from './getIsGraphAcyclic';
import { validateSourceAndTargetTypes } from './validateSourceAndTargetTypes';

export const getFirstValidConnection = (
  templates: Templates,
  nodes: AnyNode[],
  edges: InvocationNodeEdge[],
  pendingConnection: PendingConnection,
  candidateNode: InvocationNode,
  candidateTemplate: InvocationTemplate
): Connection | null => {
  if (pendingConnection.node.id === candidateNode.id) {
    // Cannot connect to self
    return null;
  }

  const pendingFieldKind = pendingConnection.fieldTemplate.fieldKind === 'input' ? 'target' : 'source';

  if (pendingFieldKind === 'source') {
    // Connecting from a source to a target
    if (!getIsGraphAcyclic(pendingConnection.node.id, candidateNode.id, nodes, edges)) {
      return null;
    }
    if (candidateNode.data.type === 'collect') {
      // Special handling for collect node - the `item` field takes any number of connections
      return {
        source: pendingConnection.node.id,
        sourceHandle: pendingConnection.fieldTemplate.name,
        target: candidateNode.id,
        targetHandle: 'item',
      };
    }
    // Only one connection per target field is allowed - look for an unconnected target field
    const candidateFields = map(candidateTemplate.inputs);
    const candidateConnectedFields = edges
      .filter((edge) => edge.target === candidateNode.id)
      .map((edge) => {
        // Edges must always have a targetHandle, safe to assert here
        assert(edge.targetHandle);
        return edge.targetHandle;
      });
    const candidateUnconnectedFields = differenceWith(
      candidateFields,
      candidateConnectedFields,
      (field, connectedFieldName) => field.name === connectedFieldName
    );
    const candidateField = candidateUnconnectedFields.find((field) =>
      validateSourceAndTargetTypes(pendingConnection.fieldTemplate.type, field.type)
    );
    if (candidateField) {
      return {
        source: pendingConnection.node.id,
        sourceHandle: pendingConnection.fieldTemplate.name,
        target: candidateNode.id,
        targetHandle: candidateField.name,
      };
    }
  } else {
    // Connecting from a target to a source
    // Ensure we there is not already an edge to the target, except for collect nodes
    const isCollect = pendingConnection.node.data.type === 'collect';
    const isTargetAlreadyConnected = edges.some(
      (e) => e.target === pendingConnection.node.id && e.targetHandle === pendingConnection.fieldTemplate.name
    );
    if (!isCollect && isTargetAlreadyConnected) {
      return null;
    }

    if (!getIsGraphAcyclic(candidateNode.id, pendingConnection.node.id, nodes, edges)) {
      return null;
    }

    // Sources/outputs can have any number of edges, we can take the first matching output field
    let candidateFields = map(candidateTemplate.outputs);
    if (isCollect) {
      // Narrow candidates to same field type as already is connected to the collect node
      const collectItemType = getCollectItemType(templates, nodes, edges, pendingConnection.node.id);
      if (collectItemType) {
        candidateFields = candidateFields.filter((field) => isEqual(field.type, collectItemType));
      }
    }
    const candidateField = candidateFields.find((field) => {
      const isValid = validateSourceAndTargetTypes(field.type, pendingConnection.fieldTemplate.type);
      const isAlreadyConnected = edges.some((e) => e.source === candidateNode.id && e.sourceHandle === field.name);
      return isValid && !isAlreadyConnected;
    });
    if (candidateField) {
      return {
        source: candidateNode.id,
        sourceHandle: candidateField.name,
        target: pendingConnection.node.id,
        targetHandle: pendingConnection.fieldTemplate.name,
      };
    }
  }

  return null;
};
