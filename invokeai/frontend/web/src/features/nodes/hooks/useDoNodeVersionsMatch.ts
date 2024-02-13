import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { compareVersions } from 'compare-versions';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeData, selectNodeTemplate } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useDoNodeVersionsMatch = (nodeId: string): boolean => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        const data = selectNodeData(nodes, nodeId);
        const template = selectNodeTemplate(nodes, nodeId);
        if (!template?.version || !data?.version) {
          return false;
        }
        return compareVersions(template.version, data.version) === 0;
      }),
    [nodeId]
  );

  const nodeTemplate = useAppSelector(selector);

  return nodeTemplate;
};
