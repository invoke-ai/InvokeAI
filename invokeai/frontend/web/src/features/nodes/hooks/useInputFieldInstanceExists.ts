import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectInvocationNodeSafe, selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useInputFieldInstanceExists = (nodeId: string, fieldName: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodesSlice) => {
        const node = selectInvocationNodeSafe(nodesSlice, nodeId);
        if (!node) {
          return false;
        }
        const instance = node.data.inputs[fieldName];
        return Boolean(instance);
      }),
    [fieldName, nodeId]
  );

  const exists = useAppSelector(selector);

  return exists;
};
