import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeTemplate } from 'features/nodes/store/selectors';
import type { Classification } from 'features/nodes/types/common';
import { useMemo } from 'react';

export const useNodeClassification = (nodeId: string): Classification | null => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        return selectNodeTemplate(nodes, nodeId)?.classification ?? null;
      }),
    [nodeId]
  );

  const title = useAppSelector(selector);
  return title;
};
