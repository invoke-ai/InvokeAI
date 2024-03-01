import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeTemplate } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useNodeTemplateTitle = (nodeId: string): string | null => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        return selectNodeTemplate(nodes, nodeId)?.title ?? null;
      }),
    [nodeId]
  );

  const title = useAppSelector(selector);
  return title;
};
