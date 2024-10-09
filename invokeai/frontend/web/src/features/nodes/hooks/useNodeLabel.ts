import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useNodeLabel = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodesSlice) => {
        const node = nodesSlice.nodes.find((node) => node.id === nodeId);
        return node?.data.label;
      }),
    [nodeId]
  );

  const label = useAppSelector(selector);
  return label;
};
