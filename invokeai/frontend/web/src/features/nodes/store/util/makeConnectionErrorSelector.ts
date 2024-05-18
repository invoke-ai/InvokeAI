import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import type { RootState } from 'app/store/store';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import type { NodesState, PendingConnection, Templates } from 'features/nodes/store/types';
import { validateConnection } from 'features/nodes/store/util/validateConnection';
import i18n from 'i18next';
import type { Edge, HandleType } from 'reactflow';

/**
 * Creates a selector that validates a pending connection.
 *
 * NOTE: The logic here must be duplicated in `invokeai/frontend/web/src/features/nodes/hooks/useIsValidConnection.ts`
 * TODO: Figure out how to do this without duplicating all the logic
 *
 * @param templates The invocation templates
 * @param nodeId The id of the node for which the selector is being created
 * @param fieldName The name of the field for which the selector is being created
 * @param handleType The type of the handle for which the selector is being created
 * @returns
 */
export const makeConnectionErrorSelector = (
  templates: Templates,
  nodeId: string,
  fieldName: string,
  handleType: HandleType
) => {
  return createMemoizedSelector(
    selectNodesSlice,
    (state: RootState, pendingConnection: PendingConnection | null) => pendingConnection,
    (state: RootState, pendingConnection: PendingConnection | null, edgePendingUpdate: Edge | null) =>
      edgePendingUpdate,
    (nodesSlice: NodesState, pendingConnection: PendingConnection | null, edgePendingUpdate: Edge | null) => {
      const { nodes, edges } = nodesSlice;

      if (!pendingConnection) {
        return i18n.t('nodes.noConnectionInProgress');
      }

      const connectionHandleType = pendingConnection.fieldTemplate.fieldKind === 'input' ? 'target' : 'source';

      if (handleType === connectionHandleType) {
        if (handleType === 'source') {
          return i18n.t('nodes.cannotConnectOutputToOutput');
        }
        return i18n.t('nodes.cannotConnectInputToInput');
      }

      // we have to figure out which is the target and which is the source
      const source = handleType === 'source' ? nodeId : pendingConnection.node.id;
      const sourceHandle = handleType === 'source' ? fieldName : pendingConnection.fieldTemplate.name;
      const target = handleType === 'target' ? nodeId : pendingConnection.node.id;
      const targetHandle = handleType === 'target' ? fieldName : pendingConnection.fieldTemplate.name;

      const validationResult = validateConnection(
        {
          source,
          sourceHandle,
          target,
          targetHandle,
        },
        nodes,
        edges,
        templates,
        edgePendingUpdate
      );

      if (!validationResult.isValid) {
        return i18n.t(validationResult.messageTKey);
      }
    }
  );
};
