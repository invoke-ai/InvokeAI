import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useInputFieldDescription = (nodeId: string, fieldName: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return '';
        }
        return node?.data.inputs[fieldName]?.description ?? '';
      }),
    [fieldName, nodeId]
  );

  const notes = useAppSelector(selector);
  return notes;
};
