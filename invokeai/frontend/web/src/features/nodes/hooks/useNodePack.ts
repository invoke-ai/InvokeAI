import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodeData, selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useNodePack = (nodeId: string): string | null => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        return selectNodeData(nodes, nodeId)?.nodePack ?? null;
      }),
    [nodeId]
  );

  const nodePack = useAppSelector(selector);
  return nodePack;
};
