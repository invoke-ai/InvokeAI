import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodeData, selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useNodeType = (nodeId: string): string => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        return selectNodeData(nodes, nodeId).type;
      }),
    [nodeId]
  );

  const type = useAppSelector(selector);

  return type;
};
