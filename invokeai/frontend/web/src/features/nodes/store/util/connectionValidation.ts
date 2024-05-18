import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import type { RootState } from 'app/store/store';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import type { NodesState, PendingConnection, Templates } from 'features/nodes/store/types';
import { validateConnectionTypes } from 'features/nodes/store/util/validateConnectionTypes';
import type { FieldType } from 'features/nodes/types/field';
import i18n from 'i18next';
import type { HandleType } from 'reactflow';
import { assert } from 'tsafe';

import { areTypesEqual } from './areTypesEqual';
import { getCollectItemType } from './getCollectItemType';
import { getHasCycles } from './getHasCycles';

/**
 * Creates a selector that validates a pending connection.
 *
 * NOTE: The logic here must be duplicated in `invokeai/frontend/web/src/features/nodes/hooks/useIsValidConnection.ts`
 * TODO: Figure out how to do this without duplicating all the logic
 *
 * @param templates The invocation templates
 * @param pendingConnection The current pending connection (if there is one)
 * @param nodeId The id of the node for which the selector is being created
 * @param fieldName The name of the field for which the selector is being created
 * @param handleType The type of the handle for which the selector is being created
 * @param fieldType The type of the field for which the selector is being created
 * @returns
 */
export const makeConnectionErrorSelector = (
  templates: Templates,
  nodeId: string,
  fieldName: string,
  handleType: HandleType,
  fieldType: FieldType
) => {
  return createMemoizedSelector(
    selectNodesSlice,
    (state: RootState, pendingConnection: PendingConnection | null) => pendingConnection,
    (nodesSlice: NodesState, pendingConnection: PendingConnection | null) => {
      const { nodes, edges } = nodesSlice;

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
          if (!areTypesEqual(sourceType, collectItemType)) {
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

      if (!validateConnectionTypes(sourceType, targetType)) {
        return i18n.t('nodes.fieldTypesMustMatch');
      }

      const hasCycles = getHasCycles(
        connectionHandleType === 'source' ? connectionNodeId : nodeId,
        connectionHandleType === 'source' ? nodeId : connectionNodeId,
        nodes,
        edges
      );

      if (hasCycles) {
        return i18n.t('nodes.connectionWouldCreateCycle');
      }

      return;
    }
  );
};
