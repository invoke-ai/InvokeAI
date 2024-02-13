import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeData } from 'features/nodes/store/selectors';
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
