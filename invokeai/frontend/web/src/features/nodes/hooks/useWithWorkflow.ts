import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useMemo } from 'react';
import { isInvocationNode } from '../types/invocation';

export const useWithWorkflow = (nodeId: string) => {
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
          if (!nodeTemplate) {
            return false;
          }
          return nodeTemplate.withWorkflow;
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const withWorkflow = useAppSelector(selector);
  return withWorkflow;
};
