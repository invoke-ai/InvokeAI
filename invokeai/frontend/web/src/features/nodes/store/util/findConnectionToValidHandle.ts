import type { PendingConnection, Templates } from 'features/nodes/store/types';
import type { FieldInputTemplate, FieldOutputTemplate, FieldType } from 'features/nodes/types/field';
import type { AnyNode, InvocationNode, InvocationNodeEdge, InvocationTemplate } from 'features/nodes/types/invocation';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { differenceWith, map } from 'lodash-es';
import type { Connection, Edge, HandleType, Node } from 'reactflow';
import { assert } from 'tsafe';

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
  templates: Templates,
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

export const getFirstValidConnection = (
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
    // Ensure we there is not already an edge to the target
    if (
      edges.some(
        (e) => e.target === pendingConnection.node.id && e.targetHandle === pendingConnection.fieldTemplate.name
      )
    ) {
      return null;
    }

    if (!getIsGraphAcyclic(candidateNode.id, pendingConnection.node.id, nodes, edges)) {
      return null;
    }

    if (candidateNode.data.type === 'collect') {
      // Special handling for collect node - connect to the `collection` field
      return {
        source: candidateNode.id,
        sourceHandle: 'collection',
        target: pendingConnection.node.id,
        targetHandle: pendingConnection.fieldTemplate.name,
      };
    }
    // Sources/outputs can have any number of edges, we can take the first matching output field
    const candidateFields = map(candidateTemplate.outputs);
    const candidateField = candidateFields.find((field) =>
      validateSourceAndTargetTypes(field.type, pendingConnection.fieldTemplate.type)
    );
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
