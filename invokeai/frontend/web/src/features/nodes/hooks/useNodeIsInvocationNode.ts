import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNode, selectNodesSlice } from 'features/nodes/store/selectors';
import { isInvocationNode as _isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useNodeIsInvocationNode = (nodeId: string): boolean => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        const node = selectNode(nodes, nodeId);
        return _isInvocationNode(node);
      }),
    [nodeId]
  );

  const isInvocationNode = useAppSelector(selector);

  return isInvocationNode;
};
