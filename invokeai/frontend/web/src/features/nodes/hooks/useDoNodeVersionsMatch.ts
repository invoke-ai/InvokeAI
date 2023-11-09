import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { compareVersions } from 'compare-versions';
import { useMemo } from 'react';
import { isInvocationNode } from '../types/types';

export const useDoNodeVersionsMatch = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return false;
          }
          const nodeTemplate = nodes.nodeTemplates[node?.data.type ?? ''];
          if (!nodeTemplate?.version || !node.data?.version) {
            return false;
          }
          return compareVersions(nodeTemplate.version, node.data.version) === 0;
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const nodeTemplate = useAppSelector(selector);

  return nodeTemplate;
};
