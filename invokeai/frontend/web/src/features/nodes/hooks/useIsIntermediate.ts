import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeData } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useIsIntermediate = (nodeId: string): boolean => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        return selectNodeData(nodes, nodeId)?.isIntermediate ?? false;
      }),
    [nodeId]
  );

  const isIntermediate = useAppSelector(selector);
  return isIntermediate;
};
