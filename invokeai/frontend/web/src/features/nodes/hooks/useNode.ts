import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectInvocationNode, selectNodesSlice } from 'features/nodes/store/selectors';
import type { InvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useNode = (nodeId: string): InvocationNode => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        return selectInvocationNode(nodes, nodeId);
      }),
    [nodeId]
  );

  const node = useAppSelector(selector);

  return node;
};
