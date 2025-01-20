import { useStore } from '@nanostores/react';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { $edgePendingUpdate, $templates } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { validateConnection } from 'features/nodes/store/util/validateConnection';
import { selectShouldShouldValidateGraph } from 'features/nodes/store/workflowSettingsSlice';
import { useCallback } from 'react';
import type { Connection } from 'reactflow';

export const useIsValidConnection = () => {
  const store = useAppStore();
  const templates = useStore($templates);
  const shouldValidateGraph = useAppSelector(selectShouldShouldValidateGraph);
  const isValidConnection = useCallback(
    ({ source, sourceHandle, target, targetHandle }: Connection): boolean => {
      // Connection must have valid targets
      if (!(source && sourceHandle && target && targetHandle)) {
        return false;
      }
      const edgePendingUpdate = $edgePendingUpdate.get();
      const { nodes, edges } = selectNodesSlice(store.getState());

      const validationResult = validateConnection(
        { source, sourceHandle, target, targetHandle },
        nodes,
        edges,
        templates,
        edgePendingUpdate,
        shouldValidateGraph
      );

      return validationResult.isValid;
    },
    [templates, shouldValidateGraph, store]
  );

  return isValidConnection;
};
