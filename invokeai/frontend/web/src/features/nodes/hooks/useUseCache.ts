import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodeData, selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useUseCache = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        return selectNodeData(nodes, nodeId)?.useCache ?? false;
      }),
    [nodeId]
  );

  const useCache = useAppSelector(selector);
  return useCache;
};
