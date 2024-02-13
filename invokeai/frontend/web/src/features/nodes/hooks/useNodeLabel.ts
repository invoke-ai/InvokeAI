import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeData } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useNodeLabel = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        return selectNodeData(nodes, nodeId)?.label ?? null;
      }),
    [nodeId]
  );

  const label = useAppSelector(selector);
  return label;
};
