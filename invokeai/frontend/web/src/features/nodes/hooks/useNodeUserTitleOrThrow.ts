import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useNodeUserTitleOrThrow = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodesSlice) => {
        const node = nodesSlice.nodes.find((node) => node.id === nodeId);
        assert(isInvocationNode(node), 'Node not found');
        return node.data.label;
      }),
    [nodeId]
  );

  const title = useAppSelector(selector);
  return title;
};
