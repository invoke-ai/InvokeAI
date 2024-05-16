import { createSelector } from '@reduxjs/toolkit';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import type { PendingConnection } from 'features/nodes/store/types';
import type { FieldType } from 'features/nodes/types/field';
import i18n from 'i18next';
import type { HandleType } from 'reactflow';

import { getIsGraphAcyclic } from './getIsGraphAcyclic';
import { validateSourceAndTargetTypes } from './validateSourceAndTargetTypes';

/**
 * NOTE: The logic here must be duplicated in `invokeai/frontend/web/src/features/nodes/hooks/useIsValidConnection.ts`
 * TODO: Figure out how to do this without duplicating all the logic
 */

export const makeConnectionErrorSelector = (
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
    const target = handleType === 'target' ? nodeId : connectionNodeId;
    const targetHandle = handleType === 'target' ? fieldName : connectionFieldName;
    const source = handleType === 'source' ? nodeId : connectionNodeId;
    const sourceHandle = handleType === 'source' ? fieldName : connectionFieldName;

    if (
      edges.find((edge) => {
        edge.target === target &&
          edge.targetHandle === targetHandle &&
          edge.source === source &&
          edge.sourceHandle === sourceHandle;
      })
    ) {
      // We already have a connection from this source to this target
      return i18n.t('nodes.cannotDuplicateConnection');
    }

    if (
      edges.find((edge) => {
        return edge.target === target && edge.targetHandle === targetHandle;
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
