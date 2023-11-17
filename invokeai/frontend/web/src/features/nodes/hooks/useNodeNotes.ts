import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useMemo } from 'react';
import { isInvocationNode } from '../types/types';

export const useNodeNotes = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return;
          }
          return node.data.notes;
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const nodeNotes = useAppSelector(selector);

  return nodeNotes;
};
