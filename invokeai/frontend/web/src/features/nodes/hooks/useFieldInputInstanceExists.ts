import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectInvocationNode, selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useFieldInputInstanceExists = (nodeId: string, fieldName: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodesSlice) => {
        const node = selectInvocationNode(nodesSlice, nodeId);
        const instance = node.data.inputs[fieldName];
        return Boolean(instance);
      }),
    [fieldName, nodeId]
  );

  const exists = useAppSelector(selector);

  return exists;
};
