import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import type { InvocationTemplate } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useNodeTemplateByType = (type: string): InvocationTemplate | null => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        return nodes.templates[type] ?? null;
      }),
    [type]
  );

  const nodeTemplate = useAppSelector(selector);

  return nodeTemplate;
};
