import { createSelector } from '@reduxjs/toolkit';
import type { HandleType } from '@xyflow/react';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { NodesState, PendingConnection, Templates } from 'features/nodes/store/types';
import { validateConnection } from 'features/nodes/store/util/validateConnection';
import type { AnyEdge } from 'features/nodes/types/invocation';

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
  handleType: HandleType,
  pendingConnection: PendingConnection | null,
  edgePendingUpdate: AnyEdge | null
) => {
  return createSelector(selectNodesSlice, (nodesSlice: NodesState): string | null => {
    const { nodes, edges } = nodesSlice;

    if (!pendingConnection) {
      return 'nodes.noConnectionInProgress';
    }

    if (handleType === pendingConnection.handleType) {
      if (handleType === 'source') {
        return 'nodes.cannotConnectOutputToOutput';
      }
      return 'nodes.cannotConnectInputToInput';
    }

    // we have to figure out which is the target and which is the source
    const source = handleType === 'source' ? nodeId : pendingConnection.nodeId;
    const sourceHandle = handleType === 'source' ? fieldName : pendingConnection.handleId;
    const target = handleType === 'target' ? nodeId : pendingConnection.nodeId;
    const targetHandle = handleType === 'target' ? fieldName : pendingConnection.handleId;

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

    return validationResult;
  });
};
