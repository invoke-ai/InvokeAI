// TODO: enable this at some point
import { useStore } from '@nanostores/react';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { $edgePendingUpdate, $templates } from 'features/nodes/store/nodesSlice';
import { validateConnection } from 'features/nodes/store/util/validateConnection';
import { useCallback } from 'react';
import type { Connection } from 'reactflow';

/**
 * NOTE: The logic here must be duplicated in `invokeai/frontend/web/src/features/nodes/store/util/makeIsConnectionValidSelector.ts`
 * TODO: Figure out how to do this without duplicating all the logic
 */

export const useIsValidConnection = () => {
  const store = useAppStore();
  const templates = useStore($templates);
  const shouldValidateGraph = useAppSelector((s) => s.workflowSettings.shouldValidateGraph);
  const isValidConnection = useCallback(
    ({ source, sourceHandle, target, targetHandle }: Connection): boolean => {
      // Connection must have valid targets
      if (!(source && sourceHandle && target && targetHandle)) {
        return false;
      }
      const edgePendingUpdate = $edgePendingUpdate.get();
      const { nodes, edges } = store.getState().nodes.present;

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
