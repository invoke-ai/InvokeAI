import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeTemplatesSlice } from 'features/nodes/store/nodeTemplatesSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useNodeTemplateTitle = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, selectNodeTemplatesSlice, (nodes, nodeTemplates) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return false;
        }
        const nodeTemplate = node ? nodeTemplates.templates[node.data.type] : undefined;

        return nodeTemplate?.title;
      }),
    [nodeId]
  );

  const title = useAppSelector(selector);
  return title;
};
