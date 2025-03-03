import { useStore } from '@nanostores/react';
import type { IsValidConnection } from '@xyflow/react';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { $edgePendingUpdate, $templates } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { validateConnection } from 'features/nodes/store/util/validateConnection';
import { selectShouldShouldValidateGraph } from 'features/nodes/store/workflowSettingsSlice';
import type { AnyEdge } from 'features/nodes/types/invocation';
import { useCallback } from 'react';

export const useIsValidConnection = (): IsValidConnection<AnyEdge> => {
  const store = useAppStore();
  const templates = useStore($templates);
  const shouldValidateGraph = useAppSelector(selectShouldShouldValidateGraph);
  const isValidConnection = useCallback<IsValidConnection<AnyEdge>>(
    ({ source, sourceHandle, target, targetHandle }) => {
      // Connection must have valid targets
      if (!(source && sourceHandle && target && targetHandle)) {
        return false;
      }
      const edgePendingUpdate = $edgePendingUpdate.get();
      const { nodes, edges } = selectNodesSlice(store.getState());

      const connectionErrorTKey = validateConnection(
        { source, sourceHandle, target, targetHandle },
        nodes,
        edges,
        templates,
        edgePendingUpdate,
        shouldValidateGraph
      );

      return connectionErrorTKey === null;
    },
    [templates, shouldValidateGraph, store]
  );

  return isValidConnection;
};
