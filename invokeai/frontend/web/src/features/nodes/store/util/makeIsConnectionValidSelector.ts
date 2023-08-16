import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { getIsGraphAcyclic } from 'features/nodes/hooks/useIsValidConnection';
import { COLLECTION_TYPES } from 'features/nodes/types/constants';
import { FieldType } from 'features/nodes/types/types';
import { HandleType } from 'reactflow';

export const makeConnectionErrorSelector = (
  nodeId: string,
  fieldName: string,
  handleType: HandleType,
  fieldType?: FieldType
) =>
  createSelector(stateSelector, (state) => {
    if (!fieldType) {
      return 'No field type';
    }

    const { currentConnectionFieldType, connectionStartParams, nodes, edges } =
      state.nodes;

    if (!state.nodes.shouldValidateGraph) {
      // manual override!
      return null;
    }

    if (!connectionStartParams || !currentConnectionFieldType) {
      return 'No connection in progress';
    }

    const {
      handleType: connectionHandleType,
      nodeId: connectionNodeId,
      handleId: connectionFieldName,
    } = connectionStartParams;

    if (!connectionHandleType || !connectionNodeId || !connectionFieldName) {
      return 'No connection data';
    }

    const targetFieldType =
      handleType === 'target' ? fieldType : currentConnectionFieldType;
    const sourceFieldType =
      handleType === 'source' ? fieldType : currentConnectionFieldType;

    if (nodeId === connectionNodeId) {
      return 'Cannot connect to self';
    }

    if (handleType === connectionHandleType) {
      if (handleType === 'source') {
        return 'Cannot connect output to output';
      }
      return 'Cannot connect input to input';
    }

    if (
      fieldType !== currentConnectionFieldType &&
      fieldType !== 'CollectionItem' &&
      currentConnectionFieldType !== 'CollectionItem'
    ) {
      if (
        !(
          COLLECTION_TYPES.includes(targetFieldType) &&
          COLLECTION_TYPES.includes(sourceFieldType)
        )
      ) {
        // except for collection items, field types must match
        return 'Field types must match';
      }
    }

    if (
      handleType === 'target' &&
      edges.find((edge) => {
        return edge.target === nodeId && edge.targetHandle === fieldName;
      }) &&
      // except CollectionItem inputs can have multiples
      targetFieldType !== 'CollectionItem'
    ) {
      return 'Inputs may only have one connection';
    }

    const isGraphAcyclic = getIsGraphAcyclic(
      connectionHandleType === 'source' ? connectionNodeId : nodeId,
      connectionHandleType === 'source' ? nodeId : connectionNodeId,
      nodes,
      edges
    );

    if (!isGraphAcyclic) {
      return 'Connection would create a cycle';
    }

    return null;
  });
