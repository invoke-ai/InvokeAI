import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodeData, selectNodesSlice } from 'features/nodes/store/selectors';
import type { InvocationNodeData } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useNodeData = (nodeId: string): InvocationNodeData => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        return selectNodeData(nodes, nodeId);
      }),
    [nodeId]
  );

  const nodeData = useAppSelector(selector);

  return nodeData;
};
