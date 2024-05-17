import { createSelector } from '@reduxjs/toolkit';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import type { PendingConnection, Templates } from 'features/nodes/store/types';
import type { FieldType } from 'features/nodes/types/field';
import type { AnyNode, InvocationNodeEdge } from 'features/nodes/types/invocation';
import i18n from 'i18next';
import { isEqual } from 'lodash-es';
import type { HandleType } from 'reactflow';
import { assert } from 'tsafe';

import { getIsGraphAcyclic } from './getIsGraphAcyclic';
import { validateSourceAndTargetTypes } from './validateSourceAndTargetTypes';

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
  const fieldType = template.outputs[firstEdgeToCollect.sourceHandle]?.type ?? null;
  return fieldType;
};

/**
 * NOTE: The logic here must be duplicated in `invokeai/frontend/web/src/features/nodes/hooks/useIsValidConnection.ts`
 * TODO: Figure out how to do this without duplicating all the logic
 */

export const makeConnectionErrorSelector = (
  templates: Templates,
  pendingConnection: PendingConnection | null,
  nodeId: string,
  fieldName: string,
  handleType: HandleType,
  fieldType?: FieldType | null
) => {
  return createSelector(selectNodesSlice, (nodesSlice) => {
    const { nodes, edges } = nodesSlice;

    if (!fieldType) {
      return i18n.t('nodes.noFieldType');
    }

    if (!pendingConnection) {
      return i18n.t('nodes.noConnectionInProgress');
    }

    const connectionNodeId = pendingConnection.node.id;
    const connectionFieldName = pendingConnection.fieldTemplate.name;
    const connectionHandleType = pendingConnection.fieldTemplate.fieldKind === 'input' ? 'target' : 'source';
    const connectionStartFieldType = pendingConnection.fieldTemplate.type;

    if (!connectionHandleType || !connectionNodeId || !connectionFieldName) {
      return i18n.t('nodes.noConnectionData');
    }

    const targetType = handleType === 'target' ? fieldType : connectionStartFieldType;
    const sourceType = handleType === 'source' ? fieldType : connectionStartFieldType;

    if (nodeId === connectionNodeId) {
      return i18n.t('nodes.cannotConnectToSelf');
    }

    if (handleType === connectionHandleType) {
      if (handleType === 'source') {
        return i18n.t('nodes.cannotConnectOutputToOutput');
      }
      return i18n.t('nodes.cannotConnectInputToInput');
    }

    // we have to figure out which is the target and which is the source
    const targetNodeId = handleType === 'target' ? nodeId : connectionNodeId;
    const targetFieldName = handleType === 'target' ? fieldName : connectionFieldName;
    const sourceNodeId = handleType === 'source' ? nodeId : connectionNodeId;
    const sourceFieldName = handleType === 'source' ? fieldName : connectionFieldName;

    if (
      edges.find((edge) => {
        edge.target === targetNodeId &&
          edge.targetHandle === targetFieldName &&
          edge.source === sourceNodeId &&
          edge.sourceHandle === sourceFieldName;
      })
    ) {
      // We already have a connection from this source to this target
      return i18n.t('nodes.cannotDuplicateConnection');
    }

    const targetNode = nodes.find((node) => node.id === targetNodeId);
    assert(targetNode, `Target node not found: ${targetNodeId}`);
    const targetTemplate = templates[targetNode.data.type];
    assert(targetTemplate, `Target template not found: ${targetNode.data.type}`);

    if (targetTemplate.inputs[targetFieldName]?.input === 'direct') {
      return i18n.t('nodes.cannotConnectToDirectInput');
    }

    if (targetNode.data.type === 'collect' && targetFieldName === 'item') {
      // Collect nodes shouldn't mix and match field types
      const collectItemType = getCollectItemType(templates, nodes, edges, targetNode.id);
      if (collectItemType) {
        if (!isEqual(sourceType, collectItemType)) {
          return i18n.t('nodes.cannotMixAndMatchCollectionItemTypes');
        }
      }
    }

    if (
      edges.find((edge) => {
        return edge.target === targetNodeId && edge.targetHandle === targetFieldName;
      }) &&
      // except CollectionItem inputs can have multiples
      targetType.name !== 'CollectionItemField'
    ) {
      return i18n.t('nodes.inputMayOnlyHaveOneConnection');
    }

    if (!validateSourceAndTargetTypes(sourceType, targetType)) {
      return i18n.t('nodes.fieldTypesMustMatch');
    }

    const isGraphAcyclic = getIsGraphAcyclic(
      connectionHandleType === 'source' ? connectionNodeId : nodeId,
      connectionHandleType === 'source' ? nodeId : connectionNodeId,
      nodes,
      edges
    );

    if (!isGraphAcyclic) {
      return i18n.t('nodes.connectionWouldCreateCycle');
    }

    return;
  });
};
