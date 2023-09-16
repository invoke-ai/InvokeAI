import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { getIsGraphAcyclic } from 'features/nodes/hooks/useIsValidConnection';
import {
  COLLECTION_MAP,
  COLLECTION_TYPES,
  POLYMORPHIC_TO_SINGLE_MAP,
  POLYMORPHIC_TYPES,
} from 'features/nodes/types/constants';
import { FieldType } from 'features/nodes/types/types';
import { HandleType } from 'reactflow';
import i18n from 'i18next';

/**
 * NOTE: The logic here must be duplicated in `invokeai/frontend/web/src/features/nodes/hooks/useIsValidConnection.ts`
 * TODO: Figure out how to do this without duplicating all the logic
 */

export const makeConnectionErrorSelector = (
  nodeId: string,
  fieldName: string,
  handleType: HandleType,
  fieldType?: FieldType
) => {
  return createSelector(stateSelector, (state) => {
    if (!fieldType) {
      return i18n.t('nodes.noFieldType');
    }

    const { currentConnectionFieldType, connectionStartParams, nodes, edges } =
      state.nodes;

    if (!connectionStartParams || !currentConnectionFieldType) {
      return i18n.t('nodes.noConnectionInProgress');
    }

    const {
      handleType: connectionHandleType,
      nodeId: connectionNodeId,
      handleId: connectionFieldName,
    } = connectionStartParams;

    if (!connectionHandleType || !connectionNodeId || !connectionFieldName) {
      return i18n.t('nodes.noConnectionData');
    }

    const targetType =
      handleType === 'target' ? fieldType : currentConnectionFieldType;
    const sourceType =
      handleType === 'source' ? fieldType : currentConnectionFieldType;

    if (nodeId === connectionNodeId) {
      return i18n.t('nodes.cannotConnectToSelf');
    }

    if (handleType === connectionHandleType) {
      if (handleType === 'source') {
        return i18n.t('nodes.cannotConnectOutputToOutput');
      }
      return i18n.t('nodes.cannotConnectInputToInput');
    }

    if (
      edges.find((edge) => {
        return edge.target === nodeId && edge.targetHandle === fieldName;
      }) &&
      // except CollectionItem inputs can have multiples
      targetType !== 'CollectionItem'
    ) {
      return i18n.t('nodes.inputMayOnlyHaveOneConnection');
    }

    /**
     * Connection types must be the same for a connection, with exceptions:
     * - CollectionItem can connect to any non-Collection
     * - Non-Collections can connect to CollectionItem
     * - Anything (non-Collections, Collections, Polymorphics) can connect to Polymorphics of the same base type
     * - Generic Collection can connect to any other Collection or Polymorphic
     * - Any Collection can connect to a Generic Collection
     */

    if (sourceType !== targetType) {
      const isCollectionItemToNonCollection =
        sourceType === 'CollectionItem' &&
        !COLLECTION_TYPES.includes(targetType);

      const isNonCollectionToCollectionItem =
        targetType === 'CollectionItem' &&
        !COLLECTION_TYPES.includes(sourceType) &&
        !POLYMORPHIC_TYPES.includes(sourceType);

      const isAnythingToPolymorphicOfSameBaseType =
        POLYMORPHIC_TYPES.includes(targetType) &&
        (() => {
          if (!POLYMORPHIC_TYPES.includes(targetType)) {
            return false;
          }
          const baseType =
            POLYMORPHIC_TO_SINGLE_MAP[
              targetType as keyof typeof POLYMORPHIC_TO_SINGLE_MAP
            ];

          const collectionType =
            COLLECTION_MAP[baseType as keyof typeof COLLECTION_MAP];

          return sourceType === baseType || sourceType === collectionType;
        })();

      const isGenericCollectionToAnyCollectionOrPolymorphic =
        sourceType === 'Collection' &&
        (COLLECTION_TYPES.includes(targetType) ||
          POLYMORPHIC_TYPES.includes(targetType));

      const isCollectionToGenericCollection =
        targetType === 'Collection' && COLLECTION_TYPES.includes(sourceType);

      const isIntToFloat = sourceType === 'integer' && targetType === 'float';

      if (
        !(
          isCollectionItemToNonCollection ||
          isNonCollectionToCollectionItem ||
          isAnythingToPolymorphicOfSameBaseType ||
          isGenericCollectionToAnyCollectionOrPolymorphic ||
          isCollectionToGenericCollection ||
          isIntToFloat
        )
      ) {
        return i18n.t('nodes.fieldTypesMustMatch');
      }
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

    return null;
  });
};
