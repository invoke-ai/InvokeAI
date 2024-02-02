import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { compareVersions } from 'compare-versions';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeTemplatesSlice } from 'features/nodes/store/nodeTemplatesSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useDoNodeVersionsMatch = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, selectNodeTemplatesSlice, (nodes, nodeTemplates) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return false;
        }
        const nodeTemplate = nodeTemplates.templates[node?.data.type ?? ''];
        if (!nodeTemplate?.version || !node.data?.version) {
          return false;
        }
        return compareVersions(nodeTemplate.version, node.data.version) === 0;
      }),
    [nodeId]
  );

  const nodeTemplate = useAppSelector(selector);

  return nodeTemplate;
};
