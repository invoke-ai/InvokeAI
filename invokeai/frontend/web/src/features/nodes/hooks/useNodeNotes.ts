import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectInvocationNode, selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useInvocationNodeNotes = (nodeId: string): string => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        const node = selectInvocationNode(nodes, nodeId);
        return node.data.notes;
      }),
    [nodeId]
  );

  const notes = useAppSelector(selector);

  return notes;
};
