import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNode, selectNodesSlice } from 'features/nodes/store/selectors';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useInvocationNodeNotes = (nodeId: string): string => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        const node = selectNode(nodes, nodeId);
        assert(isInvocationNode(node), `Node with id ${nodeId} is not an invocation node`);
        return node.data.notes;
      }),
    [nodeId]
  );

  const notes = useAppSelector(selector);

  return notes;
};
