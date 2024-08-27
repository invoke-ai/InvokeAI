import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useNodeTemplateTitle = (nodeId: string): string | null => {
  const templates = useStore($templates);
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodesSlice) => {
        const node = nodesSlice.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return null;
        }
        const template = templates[node.data.type];
        return template?.title ?? null;
      }),
    [nodeId, templates]
  );
  const title = useAppSelector(selector);
  return title;
};
